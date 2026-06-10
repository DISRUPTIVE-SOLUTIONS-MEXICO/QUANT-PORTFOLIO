"""Free price providers beyond yfinance.

Stooq (https://stooq.com) serves long-history end-of-day CSVs at zero cost,
including a number of delisted US tickers, which makes it both a redundancy
check against Yahoo and a partial survivorship-bias mitigation.
"""

from __future__ import annotations

import io
from collections.abc import Iterable

import pandas as pd

from quant_core.data.base import PERSISTENT_CACHE, http_read_text
from quant_core.data.provenance import Provenance, content_sha256

STOOQ_DAILY_URL = "https://stooq.com/q/d/l/?s={symbol}&i=d"
STOOQ_LICENSE_NOTE = "Stooq free EOD data; personal/research use."


def stooq_symbol(ticker: str) -> str:
    """Map a US ticker to Stooq's symbol convention (lowercase, ``.us`` suffix).

    Class shares use a dash on Stooq just like Yahoo (BRK-B -> brk-b.us), so
    only case and suffix change.
    """
    t = str(ticker).strip().upper().replace(".", "-")
    return f"{t.lower()}.us"


def parse_stooq_csv(raw: str, ticker: str) -> pd.DataFrame:
    """Parse one Stooq daily CSV into a single-column close frame."""
    if not raw or raw.strip().lower().startswith("no data") or "Date" not in raw.splitlines()[0]:
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO(raw))
    if df.empty or "Date" not in df.columns or "Close" not in df.columns:
        return pd.DataFrame()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
    out = df[["Close"]].rename(columns={"Close": str(ticker).strip().upper()})
    return out[out.iloc[:, 0].notna()]


def fetch_stooq_prices(
    tickers: Iterable[str],
    *,
    start=None,
    end=None,
    use_cache: bool = True,
    cache_ttl_hours: int = 24,
    user_agent: str = "QuantStockPicker/1.0",
) -> tuple[pd.DataFrame, list[Provenance]]:
    """Fetch daily closes for ``tickers`` from Stooq's public CSV endpoint.

    Returns the close-price frame plus one provenance record per ticker.
    Failures degrade per ticker (missing column) rather than failing the batch.
    """
    symbols = list(dict.fromkeys([str(t).strip().upper() for t in tickers if str(t).strip()]))
    if not symbols:
        return pd.DataFrame(), []
    payload = {"tickers": symbols, "start": str(start), "end": str(end), "source": "stooq_v1"}
    if use_cache:
        cached = PERSISTENT_CACHE.get_df("prices_stooq", payload, cache_ttl_hours)
        if cached is not None and not cached.empty:
            return cached, [
                Provenance(source="stooq", status="cache", rows=len(cached), license_note=STOOQ_LICENSE_NOTE)
            ]
    frames: list[pd.DataFrame] = []
    provenance: list[Provenance] = []
    for ticker in symbols:
        url = STOOQ_DAILY_URL.format(symbol=stooq_symbol(ticker))
        try:
            raw = http_read_text(url, user_agent=user_agent, timeout=30)
            frame = parse_stooq_csv(raw, ticker)
            status = "ok" if not frame.empty else "empty"
            provenance.append(
                Provenance(
                    source="stooq",
                    url=url,
                    content_hash=content_sha256(raw),
                    license_note=STOOQ_LICENSE_NOTE,
                    rows=len(frame),
                    status=status,
                )
            )
            if not frame.empty:
                frames.append(frame)
        except Exception as exc:  # pragma: no cover - network failure path.
            provenance.append(Provenance(source="stooq", url=url, status=f"error:{exc.__class__.__name__}"))
            continue
    if not frames:
        return pd.DataFrame(), provenance
    prices = pd.concat(frames, axis=1).sort_index()
    if start is not None:
        prices = prices.loc[pd.Timestamp(start) :]
    if end is not None:
        prices = prices.loc[: pd.Timestamp(end)]
    if use_cache and not prices.empty:
        PERSISTENT_CACHE.set_df("prices_stooq", payload, prices)
    return prices, provenance
