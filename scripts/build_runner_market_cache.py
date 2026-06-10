"""Build the market cache contract consumed by run_xcdr_v3_parallel_research.

Produces ``prices_{key}.parquet``, ``volumes_{key}.parquet`` and
``market_{key}.json`` inside the runner cache directory
(``QPK_XCDR3_CACHE_DIR``). The universe is current S&P 500 constituents plus
the benchmark/reference ETFs, trimmed to the most liquid names. All data come
from the zero-cost provider chain (yfinance with Stooq fallback).

Run from the repo root (network required):

    python scripts/build_runner_market_cache.py --period 10y --max-tickers 280
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant_stockpicker_core import (  # noqa: E402
    download_prices,
    download_volume,
    load_sp500_wikipedia_asof,
)
from run_xcdr_v3_parallel_research import BENCHMARKS, REFERENCE_ASSETS  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--period", default="10y", help="price history period (10y/15y/max)")
    parser.add_argument("--max-tickers", type=int, default=280, help="liquidity-ranked universe cap")
    parser.add_argument("--cache-ttl-hours", type=int, default=24)
    parser.add_argument(
        "--cache-dir",
        default=os.getenv("QPK_XCDR3_CACHE_DIR", ""),
        help="runner cache directory (defaults to QPK_XCDR3_CACHE_DIR)",
    )
    parser.add_argument("--chunk-size", type=int, default=120, help="tickers per download batch")
    args = parser.parse_args()
    if not args.cache_dir:
        raise SystemExit("Set --cache-dir or QPK_XCDR3_CACHE_DIR")
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    sp500 = load_sp500_wikipedia_asof(use_cache=True)
    equity = sp500["Ticker"].dropna().astype(str).str.upper().tolist() if not sp500.empty else []
    must_have = list(dict.fromkeys(BENCHMARKS + REFERENCE_ASSETS))
    candidates = list(dict.fromkeys(equity + must_have))
    print(f"universe candidates: {len(candidates)} (sp500={len(equity)}, etfs={len(must_have)})")

    price_chunks, volume_chunks = [], []
    for i in range(0, len(candidates), args.chunk_size):
        chunk = candidates[i : i + args.chunk_size]
        px = download_prices(chunk, period=args.period, use_cache=True, cache_ttl_hours=args.cache_ttl_hours)
        vol = download_volume(chunk, period=args.period, use_cache=True, cache_ttl_hours=args.cache_ttl_hours)
        if not px.empty:
            price_chunks.append(px)
        if not vol.empty:
            volume_chunks.append(vol)
        print(f"chunk {i // args.chunk_size + 1}: prices={px.shape} volumes={vol.shape}")
        time.sleep(1.0)

    if not price_chunks:
        raise SystemExit("No prices downloaded")
    prices = pd.concat(price_chunks, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()].sort_index().ffill()
    volumes = pd.concat(volume_chunks, axis=1) if volume_chunks else pd.DataFrame(index=prices.index)
    volumes = volumes.loc[:, ~volumes.columns.duplicated()].reindex(prices.index)

    # Liquidity trim: keep benchmarks/references always; rank equities by
    # median dollar volume over the last year.
    last_year = prices.tail(252)
    dollar_vol = (last_year * volumes.reindex(columns=prices.columns).tail(252)).median()
    equity_cols = [c for c in prices.columns if c not in must_have]
    ranked = dollar_vol.reindex(equity_cols).dropna().sort_values(ascending=False)
    keep_equity = ranked.head(max(args.max_tickers - len(must_have), 50)).index.tolist()
    keep = list(dict.fromkeys(keep_equity + [c for c in must_have if c in prices.columns]))
    prices = prices[keep]
    volumes = volumes.reindex(columns=keep)
    print(f"final cache universe: {prices.shape[1]} columns x {len(prices)} rows")

    key = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    prices.to_parquet(cache_dir / f"prices_{key}.parquet")
    volumes.to_parquet(cache_dir / f"volumes_{key}.parquet")
    meta = {
        "cached_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "period": args.period,
        "rows": int(len(prices)),
        "columns": int(prices.shape[1]),
        "source_chain": "yfinance+stooq",
    }
    (cache_dir / f"market_{key}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"cache key {key} written to {cache_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
