"""Governed scraping engine for official sources without an API.

Rules of engagement (enforced in code, documented in SECURITY.md):

- ``robots.txt`` is checked per domain before any fetch.
- A per-domain minimum interval throttles request rates.
- Retries use exponential backoff with jitter.
- The raw HTML snapshot is persisted (hash-keyed) so any parsed dataset can
  be traced back to the exact bytes served — scraping stays reproducible.
"""

from __future__ import annotations

import random
import time
import urllib.parse
import urllib.robotparser

import pandas as pd

from quant_core.data.base import PERSISTENT_CACHE, http_read_text
from quant_core.data.provenance import Provenance, content_sha256

USER_AGENT = "QuantPortfolioKaizen/0.2 (research; contact via repo)"


class GovernedScraper:
    """Polite, reproducible scraper for official public sources."""

    def __init__(
        self,
        *,
        min_interval_seconds: float = 2.0,
        max_retries: int = 3,
        timeout: int = 30,
        user_agent: str = USER_AGENT,
        respect_robots: bool = True,
    ):
        self.min_interval_seconds = float(min_interval_seconds)
        self.max_retries = int(max_retries)
        self.timeout = int(timeout)
        self.user_agent = user_agent
        self.respect_robots = bool(respect_robots)
        self._last_hit: dict[str, float] = {}
        self._robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}

    def _domain(self, url: str) -> str:
        return urllib.parse.urlparse(url).netloc.lower()

    def _robots_allows(self, url: str) -> bool:
        if not self.respect_robots:
            return True
        domain = self._domain(url)
        if domain not in self._robots:
            fresh = urllib.robotparser.RobotFileParser()
            robots_url = f"https://{domain}/robots.txt"
            try:
                fresh.parse(http_read_text(robots_url, user_agent=self.user_agent, timeout=10).splitlines())
                self._robots[domain] = fresh
            except Exception:
                # No reachable robots.txt: default allow (standard convention).
                self._robots[domain] = None
        parser = self._robots[domain]
        return True if parser is None else parser.can_fetch(self.user_agent, url)

    def _throttle(self, domain: str) -> None:
        last = self._last_hit.get(domain, 0.0)
        wait = self.min_interval_seconds - (time.monotonic() - last)
        if wait > 0:
            time.sleep(wait)
        self._last_hit[domain] = time.monotonic()

    def fetch(self, url: str, *, snapshot_namespace: str = "scraper_snapshots") -> tuple[str, Provenance]:
        """Fetch a page politely; persist the raw snapshot for provenance."""
        if not self._robots_allows(url):
            raise PermissionError(f"robots.txt disallows scraping: {url}")
        domain = self._domain(url)
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            self._throttle(domain)
            try:
                html = http_read_text(url, user_agent=self.user_agent, timeout=self.timeout)
                digest = content_sha256(html)
                snapshot = pd.DataFrame({"url": [url], "content_hash": [digest], "html": [html]})
                PERSISTENT_CACHE.set_df(snapshot_namespace, {"url": url, "hash": digest}, snapshot)
                return html, Provenance(
                    source=f"scrape:{domain}",
                    url=url,
                    content_hash=digest,
                    license_note="Official public source; governed scraping (robots-checked, throttled).",
                    rows=1,
                )
            except Exception as exc:
                last_error = exc
                backoff = (2.0**attempt) + random.uniform(0.0, 0.5)  # noqa: S311
                time.sleep(backoff)
        raise RuntimeError(f"scrape failed after {self.max_retries} attempts: {url}") from last_error


def parse_banxico_policy_announcements(html: str) -> pd.DataFrame:
    """Parse Banxico's monetary-policy announcements table from page HTML.

    Fixture-validated parser: selectors target the announcements table with
    date and decision columns. Returns [Date, Decision, Rate] rows; an empty
    frame when the structure is not recognized (parser then needs updating).
    """
    import io

    try:
        tables = pd.read_html(io.StringIO(html))
    except ValueError:
        return pd.DataFrame()
    for table in tables:
        cols = [str(c).strip().lower() for c in table.columns]
        has_date = any("fecha" in c or "date" in c for c in cols)
        has_decision = any("decisi" in c or "anuncio" in c or "decision" in c for c in cols)
        if not (has_date and has_decision):
            continue
        out = table.copy()
        out.columns = cols
        date_col = next(c for c in cols if "fecha" in c or "date" in c)
        decision_col = next(c for c in cols if "decisi" in c or "anuncio" in c or "decision" in c)
        rate_col = next((c for c in cols if "tasa" in c or "rate" in c), None)
        result = pd.DataFrame(
            {
                "Date": pd.to_datetime(out[date_col], dayfirst=True, errors="coerce"),
                "Decision": out[decision_col].astype(str).str.strip(),
            }
        )
        if rate_col:
            result["Rate"] = pd.to_numeric(
                out[rate_col].astype(str).str.replace("%", "", regex=False).str.replace(",", ".", regex=False),
                errors="coerce",
            )
        return result.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return pd.DataFrame()
