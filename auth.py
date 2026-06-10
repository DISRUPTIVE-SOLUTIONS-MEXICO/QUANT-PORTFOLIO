"""Authentication and authorization gate for Quant Portfolio-Kaizen.

Design goals:
- Self-contained: no external service required.
- Zero-cost: works on Streamlit Community Cloud's free tier.
- Defense in depth: bcrypt password hashes, JWT cookies, brute-force lockout,
  audit log, role-based authorization (RBAC).
- Render-only contract: this module owns no financial logic.

Credentials shape (loaded from ``st.secrets["auth"]``):

    [auth]
    cookie_name        = "qpk_auth"
    cookie_key         = "<32+ chars random>"
    cookie_expiry_days = 1
    preauthorized      = []        # optional: emails allowed to self-register

    [auth.users.chris]
    name          = "Chris"
    email         = "chris@example.com"
    password_hash = "$2b$12$..."   # bcrypt hash (see scripts/hash_password.py)
    role          = "admin"        # one of: admin | analyst | viewer

    [auth.users.viewer1]
    name          = "Read-Only User"
    email         = "viewer1@example.com"
    password_hash = "$2b$12$..."
    role          = "viewer"

Role matrix:

    admin    : full app, including Advanced Research and Data Ops.
    analyst  : Overview / Allocation / Risk / Validation / Market Regime /
               Options / Fundamentals / Data Freshness.
    viewer   : Overview / Allocation / Price Path only (read-only).
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import streamlit as st

try:
    import streamlit_authenticator as stauth  # type: ignore
except Exception as exc:  # pragma: no cover - hard dependency.
    raise RuntimeError("streamlit-authenticator is required. Install via `pip install -r requirements.txt`.") from exc


Role = Literal["admin", "analyst", "viewer"]

ROLE_PRIORITY: dict[Role, int] = {"viewer": 1, "analyst": 2, "admin": 3}

SECTION_ACCESS: dict[str, Role] = {
    "overview": "viewer",
    "allocation": "viewer",
    "my-portfolio": "viewer",
    "private-alpha": "analyst",
    "price-path": "viewer",
    "risk": "analyst",
    "validation": "analyst",
    "market-regime": "analyst",
    "options": "analyst",
    "fundamentals": "analyst",
    "data-freshness": "analyst",
    "advanced": "admin",
}

# Brute-force protection: max failed attempts per username before temporary lockout.
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_SECONDS = 15 * 60  # 15 minutes


@dataclass(frozen=True)
class AuthenticatedUser:
    """Immutable identity object passed to the rest of the app."""

    username: str
    name: str
    email: str
    role: Role

    def can_access(self, section_slug: str) -> bool:
        required = SECTION_ACCESS.get(section_slug, "admin")
        return ROLE_PRIORITY.get(self.role, 0) >= ROLE_PRIORITY.get(required, 999)


# ----------------------------------------------------------------------
# Audit log — JSON Lines, append-only.
# ----------------------------------------------------------------------

_AUDIT_FILE = Path(os.environ.get("QPK_AUDIT_LOG", "audit.jsonl"))


def _audit(event: str, *, username: str | None = None, **fields: object) -> None:
    """Append a structured event to the audit log. Best-effort, never raises."""
    record = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "event": event,
        "username": username,
        **fields,
    }
    try:
        _AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with _AUDIT_FILE.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
    except Exception:
        # We never want auth/audit failures to take the app down.
        pass


# ----------------------------------------------------------------------
# Credentials loader.
# ----------------------------------------------------------------------


def _load_credentials_from_secrets() -> dict:
    """Read ``st.secrets["auth"]`` and convert to the shape expected by stauth.

    The library expects::

        credentials = {
            "usernames": {
                "<username>": {
                    "name": "...",
                    "email": "...",
                    "password": "<bcrypt hash>",
                    # optional extras carried by the library: failed_login_attempts, logged_in
                }
            }
        }

    We additionally hold a sidecar mapping ``username -> role`` because the
    library does not have a first-class role concept.
    """
    if "auth" not in st.secrets:
        raise RuntimeError(
            "Missing `[auth]` section in Streamlit secrets. "
            "See `.streamlit/secrets.toml.example` for the expected shape."
        )

    auth_block = st.secrets["auth"]
    raw_users = auth_block.get("users", {}) or {}
    if not raw_users:
        raise RuntimeError("No users configured under [auth.users].")

    credentials: dict[str, dict[str, dict[str, object]]] = {"usernames": {}}
    roles: dict[str, Role] = {}
    for username, info in raw_users.items():
        # Streamlit's `st.secrets` returns ``AttrDict``s (Mapping, not dict).
        # Accept any Mapping subclass; reject scalars / lists.
        if not isinstance(info, Mapping):
            continue
        pw_hash = str(info.get("password_hash", "")).strip()
        if not pw_hash.startswith("$2"):
            # Hard fail rather than silently letting a plaintext password through.
            raise RuntimeError(
                f"User '{username}' must define a bcrypt `password_hash` "
                "(starts with $2). Generate one with scripts/hash_password.py."
            )
        credentials["usernames"][str(username).lower()] = {
            "name": str(info.get("name", username)),
            "email": str(info.get("email", "")),
            "password": pw_hash,
        }
        role = str(info.get("role", "viewer")).lower()
        if role not in ROLE_PRIORITY:
            role = "viewer"
        roles[str(username).lower()] = role  # type: ignore[assignment]

    return {
        "credentials": credentials,
        "roles": roles,
        "cookie_name": str(auth_block.get("cookie_name", "qpk_auth")),
        "cookie_key": str(auth_block.get("cookie_key", "")),
        "cookie_expiry_days": float(auth_block.get("cookie_expiry_days", 1.0)),
        "preauthorized": list(auth_block.get("preauthorized", []) or []),
    }


# ----------------------------------------------------------------------
# Lockout state (per-session)
# ----------------------------------------------------------------------


def _is_locked_out(username: str | None) -> tuple[bool, int]:
    if not username:
        return False, 0
    state = st.session_state.setdefault("_auth_lockouts", {})
    info = state.get(username.lower())
    if not info:
        return False, 0
    if info["count"] < MAX_FAILED_ATTEMPTS:
        return False, 0
    elapsed = time.time() - info["since"]
    if elapsed >= LOCKOUT_SECONDS:
        state.pop(username.lower(), None)
        return False, 0
    return True, int(LOCKOUT_SECONDS - elapsed)


def _register_failed_attempt(username: str | None) -> None:
    if not username:
        return
    state = st.session_state.setdefault("_auth_lockouts", {})
    key = username.lower()
    info = state.get(key) or {"count": 0, "since": time.time()}
    info["count"] += 1
    if info["count"] >= MAX_FAILED_ATTEMPTS:
        info["since"] = time.time()
        _audit("auth.lockout", username=key, attempts=info["count"])
    state[key] = info


def _reset_attempts(username: str | None) -> None:
    if not username:
        return
    state = st.session_state.setdefault("_auth_lockouts", {})
    state.pop(username.lower(), None)


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def require_authentication() -> AuthenticatedUser:
    """Gate the rest of the app behind a login form.

    Returns the authenticated user on success. If the user is not
    authenticated, this function calls ``st.stop()`` after rendering the login
    UI, ensuring no downstream code runs.
    """
    try:
        cfg = _load_credentials_from_secrets()
    except Exception as exc:
        st.error("Authentication is not configured. Contact the administrator.")
        st.caption(str(exc))
        _audit("auth.config_error", error=str(exc))
        st.stop()

    if not cfg["cookie_key"] or len(cfg["cookie_key"]) < 16:
        st.error("Authentication is misconfigured: cookie key is too short or missing.")
        _audit("auth.cookie_key_invalid")
        st.stop()

    # streamlit-authenticator 0.4.x defaults `auto_hash=True`, which would re-hash
    # the bcrypt strings we already store. Disable it: our `password_hash` values
    # in secrets.toml are the canonical bcrypt digests.
    # `preauthorized` was removed from this constructor in 0.4.x (it now lives in
    # `register_user`); do not pass it here.
    try:
        authenticator = stauth.Authenticate(
            credentials=cfg["credentials"],
            cookie_name=cfg["cookie_name"],
            cookie_key=cfg["cookie_key"],
            cookie_expiry_days=cfg["cookie_expiry_days"],
            auto_hash=False,
        )
    except TypeError:
        # Fallback for stauth 0.3.x where the kw was `preauthorized` and
        # `auto_hash` does not exist.
        authenticator = stauth.Authenticate(
            credentials=cfg["credentials"],
            cookie_name=cfg["cookie_name"],
            cookie_key=cfg["cookie_key"],
            cookie_expiry_days=cfg["cookie_expiry_days"],
            preauthorized=cfg["preauthorized"] or None,
        )

    # Lockout guard: do not present the form if the previously tried
    # username is currently locked out.
    last_tried = st.session_state.get("_auth_last_username")
    locked, wait = _is_locked_out(last_tried)
    if locked:
        st.error(f"Too many failed attempts for `{last_tried}`. Retry in {wait // 60}m {wait % 60}s.")
        st.stop()

    # Reserve the branded intro above the form, but remove it once the user is
    # authenticated so it does not duplicate the application header.
    login_brand = st.empty()

    # Render the login form. The library handles cookies + JWT internally and
    # exposes the outcome via st.session_state, so the return value is unused.
    try:
        authenticator.login(location="main", fields={"Form name": "Sign in"})
    except TypeError:
        # Older library signature.
        authenticator.login("Sign in", "main")

    name = st.session_state.get("name")
    auth_status = st.session_state.get("authentication_status")
    username = st.session_state.get("username")

    if not bool(auth_status):
        login_brand.markdown(
            """
            <div class="qpk-login-brand">
                <div class="qpk-kicker">Secure portfolio workspace</div>
                <div class="qpk-login-brand-title">Quant Portfolio-Kaizen</div>
                <div class="qpk-login-brand-copy">
                    Sign in to access your allocation, benchmark evidence, downside diagnostics,
                    and auditable research history.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        login_brand.empty()

    if auth_status is False:
        st.session_state["_auth_last_username"] = username
        _register_failed_attempt(username)
        _audit("auth.failed", username=username)
        st.error("Invalid username or password.")
        st.stop()

    if auth_status is None:
        # User has not submitted credentials yet.
        st.stop()

    # auth_status == True
    _reset_attempts(username)
    role = cfg["roles"].get(str(username).lower(), "viewer")  # type: ignore[assignment]
    user = AuthenticatedUser(
        username=str(username),
        name=str(name or username),
        email=str(cfg["credentials"]["usernames"].get(str(username).lower(), {}).get("email", "")),
        role=role,  # type: ignore[arg-type]
    )

    if not st.session_state.get("_auth_login_logged"):
        _audit("auth.login", username=user.username, role=user.role)
        st.session_state["_auth_login_logged"] = True

    # Streamlit's component reconciliation may briefly retain the unauthenticated
    # brand placeholder after cookie-based login. The marker makes the signed-in
    # state authoritative and prevents duplicate product headers.
    st.markdown(
        """
        <div class="qpk-authenticated-marker" aria-hidden="true"></div>
        <style>
        body:has(.qpk-authenticated-marker) .qpk-login-brand {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Attach a logout button + identity card to the sidebar.
    _render_sidebar_identity(authenticator, user)
    return user


def _render_sidebar_identity(authenticator: stauth.Authenticate, user: AuthenticatedUser) -> None:
    import html as _html

    with st.sidebar:
        st.markdown(
            f'<div role="region" aria-label="Signed-in user" '
            f'style="border:1px solid var(--qpk-line, rgba(148,163,184,0.18));'
            f"border-left:3px solid var(--qpk-accent, #7dd3fc);"
            f"background:rgba(125,211,252,0.05);padding:10px 12px;"
            f'border-radius:4px;margin-bottom:10px;">'
            f'<div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.10em;color:#a8b3c7;">Signed in as</div>'
            f'<div style="font-weight:600;color:#eef3fb;margin-top:2px;">{_html.escape(user.name)}</div>'
            f'<div style="font-size:0.78rem;color:#a8b3c7;">role: {_html.escape(user.role)}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
        try:
            authenticator.logout(button_name="Sign out", location="sidebar")
        except TypeError:
            authenticator.logout("Sign out", "sidebar")


def filter_accessible_sections(user: AuthenticatedUser, sections: Iterable[str]) -> list[str]:
    """Return the subset of section slugs the user is allowed to view."""
    return [s for s in sections if user.can_access(s)]
