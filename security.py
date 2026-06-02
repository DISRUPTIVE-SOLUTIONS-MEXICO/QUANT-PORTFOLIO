"""Security helpers for Quant Portfolio-Kaizen.

Scope:
- Input validation (tickers, file uploads, free-form text fields).
- Per-user rate limiting on expensive backend calls (run_pipeline).
- Defensive HTTP / CSP headers injected as a `<meta>` element.
- Lightweight audit log shared with `auth.py`.

This module does no financial math.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import streamlit as st


# ----------------------------------------------------------------------
# Audit log (shared with auth.py)
# ----------------------------------------------------------------------

_AUDIT_FILE = Path(os.environ.get("QPK_AUDIT_LOG", "audit.jsonl"))


def audit(event: str, *, username: str | None = None, **fields: object) -> None:
    """Append a JSON record to the audit log. Best-effort, never raises."""
    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "event": event,
        "username": username,
        **fields,
    }
    try:
        _AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with _AUDIT_FILE.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
    except Exception:
        pass


# ----------------------------------------------------------------------
# Input validation
# ----------------------------------------------------------------------

_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")
_MAX_TICKER_COUNT = 500
_MAX_FREEFORM_LEN = 4096


def sanitize_ticker(raw: str) -> str | None:
    """Return a normalized ticker or None if it doesn't pass the whitelist."""
    if not raw:
        return None
    candidate = str(raw).strip().upper()
    if not _TICKER_RE.fullmatch(candidate):
        return None
    return candidate


def sanitize_ticker_list(values: Iterable[str], *, cap: int = _MAX_TICKER_COUNT) -> list[str]:
    """Filter and de-duplicate a sequence of ticker strings."""
    seen: dict[str, None] = {}
    for v in values:
        t = sanitize_ticker(v)
        if t and t not in seen:
            seen[t] = None
        if len(seen) >= cap:
            break
    return list(seen.keys())


def sanitize_free_text(raw: str, *, max_len: int = _MAX_FREEFORM_LEN) -> str:
    """Strip control characters and cap length for free-form input.

    NUL bytes and ASCII control chars are removed; HTML/JS payloads are
    rendered harmlessly by the consumer if they keep `unsafe_allow_html=False`,
    which is the default everywhere except our explicit style/banner helpers.
    """
    if raw is None:
        return ""
    text = str(raw)
    # Remove NUL and the rest of the C0 control set except common whitespace.
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text[:max_len]


# ----------------------------------------------------------------------
# Rate limiting (per-user, sliding window)
# ----------------------------------------------------------------------

class RateLimiter:
    """In-memory sliding-window rate limiter keyed by (username, action).

    Notes
    -----
    - State lives in ``st.session_state``. On Streamlit Cloud each user
      session has its own state, so the limiter enforces per-session quotas
      which is the right granularity for "expensive run buttons".
    - The limiter is best-effort: if state is lost on restart, the user
      simply gets a fresh quota. This is acceptable for our threat model
      (preventing accidental DoS, not adversarial abuse at scale).
    """

    def __init__(self, *, capacity: int, window_seconds: int) -> None:
        self.capacity = int(capacity)
        self.window = int(window_seconds)

    def _bucket(self, key: str) -> deque[float]:
        store: dict[str, deque[float]] = st.session_state.setdefault("_rate_buckets", {})
        if key not in store:
            store[key] = deque()
        return store[key]

    def allow(self, key: str) -> tuple[bool, int]:
        """Return (allowed, retry_after_seconds)."""
        now = time.monotonic()
        bucket = self._bucket(key)
        while bucket and bucket[0] <= now - self.window:
            bucket.popleft()
        if len(bucket) >= self.capacity:
            retry = int(self.window - (now - bucket[0])) + 1
            return False, max(retry, 1)
        bucket.append(now)
        return True, 0


# Default policy for the heavy pipeline action.
PIPELINE_LIMIT = RateLimiter(capacity=6, window_seconds=15 * 60)


def enforce_pipeline_quota(username: str) -> bool:
    """Return True if the call is allowed; render an error and return False otherwise."""
    allowed, retry = PIPELINE_LIMIT.allow(f"run_pipeline:{username}")
    if not allowed:
        st.error(
            f"Rate limit reached for the allocation engine. "
            f"Try again in {retry // 60}m {retry % 60}s."
        )
        audit("rate_limit.exceeded", username=username, action="run_pipeline", retry_after=retry)
    return allowed


# ----------------------------------------------------------------------
# HTTP / CSP hardening
# ----------------------------------------------------------------------
#
# Why no meta-CSP here:
# Content-Security-Policy enforced via `<meta http-equiv>` is brittle for
# Streamlit because Streamlit's bundle dynamically injects style/font/blob
# URLs that any restrictive policy will misclassify (the visible symptom is
# Material Symbols rendering as literal text like "keyboard_arrow_right").
# The correct place for CSP is the HTTP response header at the
# platform / reverse-proxy layer — Streamlit Cloud, Render, Fly, and Cloud
# Run all expose this. We keep only the meta tags that are safe and
# meaningful at the document layer.
#
# Recommended platform-level CSP for production (set as response header):
#   Content-Security-Policy: default-src 'self' data: blob:; img-src 'self'
#     data: blob: https:; style-src 'self' 'unsafe-inline'
#     https://fonts.googleapis.com data:; font-src 'self'
#     https://fonts.gstatic.com data: blob:; script-src 'self'
#     'unsafe-inline' 'unsafe-eval' blob:; connect-src 'self' wss: https:;
#     frame-ancestors 'none'; base-uri 'self'; form-action 'self';
#     object-src 'none'

def inject_security_headers() -> None:
    """Inject the document-level security meta tags that browsers respect.

    Notes
    -----
    - ``Referrer-Policy`` and ``robots`` are honored via meta.
    - ``X-Content-Type-Options`` and ``Content-Security-Policy`` should be set
      as HTTP response headers; meta versions are unreliable or unsupported.
    """
    st.markdown(
        '<meta name="referrer" content="strict-origin-when-cross-origin">'
        '<meta name="robots" content="noindex, nofollow">',
        unsafe_allow_html=True,
    )
