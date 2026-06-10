"""Build the local delisting registry from free public evidence.

Combines (a) the Wikipedia S&P 500 historical changes table (removals with
dates), (b) current NasdaqTrader listings, and (c) optionally a probe of
Stooq for recoverable price history, into
``data/universes/delisted_registry.parquet`` (+ CSV for inspection).

Run from the repo root (network required):

    python scripts/build_delisted_registry.py [--probe-stooq] [--max-probe 200]
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant_core.data.prices import fetch_stooq_prices  # noqa: E402
from quant_core.data.universe import WIKI_SP500_URL, build_delisted_registry  # noqa: E402
from quant_stockpicker_core import http_read_text, load_nasdaq_trader_universe  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "universes"


def fetch_wiki_removals() -> pd.DataFrame:
    html = http_read_text(WIKI_SP500_URL, user_agent="QuantStockPicker/1.0", timeout=30)
    tables = pd.read_html(io.StringIO(html))
    if len(tables) < 2:
        return pd.DataFrame()
    changes = tables[1].copy()
    changes.columns = [
        "_".join([str(x) for x in col if str(x) != "nan"]).strip("_") if isinstance(col, tuple) else str(col)
        for col in changes.columns
    ]
    date_col = next((c for c in changes.columns if "Date" in c), None)
    removed_col = next((c for c in changes.columns if "Removed_Ticker" in c or c == "Removed"), None)
    if not date_col or not removed_col:
        return pd.DataFrame()
    out = changes[[date_col, removed_col]].rename(columns={date_col: "Removed_Date", removed_col: "Ticker"})
    out["Removed_Date"] = pd.to_datetime(out["Removed_Date"], errors="coerce")
    out["Ticker"] = out["Ticker"].astype(str).str.strip().str.upper()
    return out[out["Ticker"].str.len().between(1, 6)].dropna(subset=["Removed_Date"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-stooq", action="store_true", help="probe Stooq for recoverable delisted histories")
    parser.add_argument("--max-probe", type=int, default=200, help="max delisted tickers to probe on Stooq")
    args = parser.parse_args()

    removals = fetch_wiki_removals()
    print(f"wiki removals: {len(removals)} rows")
    listed = load_nasdaq_trader_universe(use_cache=True)
    print(f"current listings: {len(listed)} rows")

    extra = pd.DataFrame()
    if args.probe_stooq and not removals.empty:
        candidates = removals[~removals["Ticker"].isin(set(listed.get("Ticker", pd.Series(dtype=str))))]
        probe = candidates["Ticker"].drop_duplicates().head(args.max_probe).tolist()
        print(f"probing stooq for {len(probe)} delisted tickers ...")
        stooq_px, _ = fetch_stooq_prices(probe, use_cache=True, cache_ttl_hours=24 * 30)
        extra = pd.DataFrame({"Ticker": list(stooq_px.columns)})
        print(f"stooq has history for {len(extra)} of them")

    registry = build_delisted_registry(removals, listed, extra_symbols=extra if not extra.empty else None)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    registry.to_parquet(OUT_DIR / "delisted_registry.parquet", index=False)
    registry.to_csv(OUT_DIR / "delisted_registry.csv", index=False)
    print(f"registry written: {len(registry)} tickers -> {OUT_DIR / 'delisted_registry.parquet'}")
    recoverable = (registry["Recovery_Status"] == "price_history_available").mean() if not registry.empty else 0.0
    print(f"recoverable price-history share: {recoverable:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
