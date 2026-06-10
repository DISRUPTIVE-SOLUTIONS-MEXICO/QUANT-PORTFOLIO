"""Point-in-time universe reconstruction and delisting registry helpers.

Survivorship control at zero cost:

1. The live Wikipedia S&P 500 page provides current constituents plus a
   historical changes table (additions/removals with dates) back to ~2000.
2. For as-of dates beyond the changes-table coverage, the Wayback Machine
   availability API (free, no key) locates an archived snapshot of the page
   taken near the requested date, which is then parsed directly.
3. Tickers removed from the index feed a local delisting registry so the
   backtest can apply an explicit delisting-return assumption instead of the
   silently optimistic 0% a forward-filled price implies.
"""

from __future__ import annotations

import io
import urllib.parse

import pandas as pd

from quant_core.data.base import http_read_json, http_read_text

WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
WAYBACK_AVAILABLE_URL = "https://archive.org/wayback/available?url={url}&timestamp={timestamp}"


def wayback_snapshot_url(
    target_url: str,
    asof_date,
    *,
    user_agent: str = "QuantStockPicker/1.0",
) -> str | None:
    """Closest archived snapshot URL for ``target_url`` near ``asof_date``.

    Uses the Internet Archive availability API; returns ``None`` when no
    snapshot exists or the API cannot be reached.
    """
    ts = pd.Timestamp(asof_date).strftime("%Y%m%d")
    query = WAYBACK_AVAILABLE_URL.format(url=urllib.parse.quote(target_url, safe=""), timestamp=ts)
    try:
        data = http_read_json(query, user_agent=user_agent, timeout=30)
    except Exception:
        return None
    snapshot = (data.get("archived_snapshots") or {}).get("closest") or {}
    url = str(snapshot.get("url", "")).strip()
    if not snapshot.get("available") or not url:
        return None
    # Normalize protocol-relative or http snapshots to https.
    if url.startswith("http://"):
        url = "https://" + url[len("http://") :]
    return url


def parse_sp500_constituents_html(html: str) -> pd.DataFrame:
    """Parse the constituents table of a (possibly archived) S&P 500 page."""
    try:
        tables = pd.read_html(io.StringIO(html))
    except ValueError:
        return pd.DataFrame()
    for table in tables:
        cols = {str(c).strip().lower() for c in table.columns}
        if {"symbol"}.issubset(cols) and ({"security"} & cols or {"company"} & cols):
            out = table.copy()
            out.columns = [str(c).strip() for c in out.columns]
            rename = {"Symbol": "Ticker", "Security": "Name", "Company": "Name", "GICS Sector": "Sector"}
            out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})
            keep = [c for c in ("Ticker", "Name", "Sector") if c in out.columns]
            out = out[keep].dropna(subset=["Ticker"])
            out["Ticker"] = out["Ticker"].astype(str).str.strip().str.upper().str.replace(".", "-", regex=False)
            return out.drop_duplicates("Ticker").reset_index(drop=True)
    return pd.DataFrame()


def fetch_sp500_constituents_wayback(
    asof_date,
    *,
    user_agent: str = "QuantStockPicker/1.0",
) -> pd.DataFrame:
    """Point-in-time S&P 500 membership from an archived Wikipedia snapshot."""
    snapshot = wayback_snapshot_url(WIKI_SP500_URL, asof_date, user_agent=user_agent)
    if not snapshot:
        return pd.DataFrame()
    try:
        html = http_read_text(snapshot, user_agent=user_agent, timeout=60)
    except Exception:
        return pd.DataFrame()
    out = parse_sp500_constituents_html(html)
    if out.empty:
        return out
    out["Universe_Source"] = "Wikipedia S&P 500 (Wayback snapshot)"
    out["Source_Status"] = "wayback_snapshot"
    out["Universe_AsOf"] = pd.Timestamp(asof_date).normalize()
    out["Snapshot_URL"] = snapshot
    return out


def build_delisted_registry(
    wiki_removals: pd.DataFrame,
    listed_now: pd.DataFrame,
    extra_symbols: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Combine free evidence sources into a delisting registry.

    ``wiki_removals``: columns [Ticker, Removed_Date] from the S&P changes
    table. ``listed_now``: current listings (NasdaqTrader / SEC tickers) with
    a Ticker column. ``extra_symbols``: optional frame of tickers known to a
    long-history provider (e.g. Stooq) with a Ticker column. A ticker is
    registered when it left the index and no longer appears in current
    listings; ``Recovery_Status`` records whether a price history source is
    known for it.
    """
    removals = pd.DataFrame(wiki_removals).copy()
    if removals.empty or "Ticker" not in removals.columns:
        return pd.DataFrame(columns=["Ticker", "Delist_Date_Approx", "Last_Price_Source", "Recovery_Status"])
    removals["Ticker"] = removals["Ticker"].astype(str).str.strip().str.upper()
    removals = removals[removals["Ticker"].str.len() > 0]
    listed = set()
    if listed_now is not None and not pd.DataFrame(listed_now).empty and "Ticker" in listed_now.columns:
        listed = set(pd.DataFrame(listed_now)["Ticker"].astype(str).str.strip().str.upper())
    recoverable = set()
    if extra_symbols is not None and not pd.DataFrame(extra_symbols).empty and "Ticker" in extra_symbols.columns:
        recoverable = set(pd.DataFrame(extra_symbols)["Ticker"].astype(str).str.strip().str.upper())
    gone = removals[~removals["Ticker"].isin(listed)].copy()
    if gone.empty:
        return pd.DataFrame(columns=["Ticker", "Delist_Date_Approx", "Last_Price_Source", "Recovery_Status"])
    date_col = next((c for c in gone.columns if "date" in str(c).lower()), None)
    gone["Delist_Date_Approx"] = pd.to_datetime(gone[date_col], errors="coerce") if date_col else pd.NaT
    gone["Last_Price_Source"] = gone["Ticker"].map(lambda t: "stooq" if t in recoverable else "")
    gone["Recovery_Status"] = gone["Ticker"].map(
        lambda t: "price_history_available" if t in recoverable else "no_known_free_history"
    )
    out = gone[["Ticker", "Delist_Date_Approx", "Last_Price_Source", "Recovery_Status"]]
    return out.sort_values("Delist_Date_Approx", ascending=False).drop_duplicates("Ticker").reset_index(drop=True)
