from __future__ import annotations

import argparse
from datetime import datetime

import pandas as pd

from quant_stockpicker_core import (
    DEFAULT_SIDE_BOOM_TICKERS,
    download_prices,
    download_volume,
    fetch_forex_factory_calendar,
    fetch_interbank_reference_rates,
    geopolitical_thermometer,
    global_yield_curve_discrete_history,
    global_yield_curve_snapshot,
    market_regime,
    market_sentiment_sem,
)


DEFAULT_TICKERS = """
AAPL MSFT NVDA META GOOGL AMZN ORCL CRM AMD QCOM
JPM BAC WFC GS MS BLK SCHW C
XOM CVX COP SLB EOG MPC
LLY JNJ MRK ABBV AMGN TMO DHR ISRG VRTX GEHC REGN
PG KO PEP WMT COST MDLZ
HD LOW MCD NKE SBUX BKNG
NEE DUK SO XLU VST CEG SMR ED
EQIX DLR PLD AMT CCI
SPY QQQ ACWI VT EWW XLK XLV XLU XLE XLF
"""


def parse_tickers(text: str) -> list[str]:
    return sorted({t.strip().upper().replace(".", "-") for t in text.replace(",", " ").split() if t.strip()})


def main() -> int:
    parser = argparse.ArgumentParser(description="Prewarm zero-cost Quant Portfolio-Kaizen public-data cache.")
    parser.add_argument("--period", default="5y", help="Yahoo price period to cache.")
    parser.add_argument("--ttl-hours", type=int, default=24, help="Cache TTL used by the app.")
    parser.add_argument("--country", default="United States", help="Primary macro/rate country.")
    parser.add_argument("--include-geopolitical", action="store_true", help="Refresh GDELT/RSS geopolitical thermometer.")
    parser.add_argument("--tickers", default="", help="Optional extra tickers separated by spaces or commas.")
    args = parser.parse_args()

    started = datetime.now()
    tickers = parse_tickers(DEFAULT_TICKERS + " " + " ".join(DEFAULT_SIDE_BOOM_TICKERS) + " " + args.tickers)
    print(f"[{started:%Y-%m-%d %H:%M:%S}] Prewarming {len(tickers)} tickers, period={args.period}, country={args.country}")

    prices = download_prices(tickers, period=args.period, use_cache=True, cache_ttl_hours=args.ttl_hours)
    print(f"prices: {prices.shape}")
    volumes = download_volume(tickers, period=args.period, use_cache=True, cache_ttl_hours=args.ttl_hours)
    print(f"volumes: {volumes.shape}")

    if not prices.empty:
        macro, latest = market_regime(prices, country=args.country, use_cache=True, cache_ttl_hours=args.ttl_hours)
        print(f"macro: {macro.shape}; latest regime={latest.get('Regime_Hawkish_Dovish', 'n/a')}/{latest.get('Regime_Bull_Bear', 'n/a')}")
        global_curves = global_yield_curve_snapshot(prices, use_cache=True, cache_ttl_hours=args.ttl_hours)
        print(f"global_curves: {global_curves.shape}")
        global_history = global_yield_curve_discrete_history(prices, use_cache=True, cache_ttl_hours=args.ttl_hours)
        print(f"global_rate_history: {global_history.shape}")

    interbank = fetch_interbank_reference_rates(
        start=pd.Timestamp.today() - pd.Timedelta(days=365 * 3),
        end=pd.Timestamp.today(),
        use_cache=True,
        cache_ttl_hours=args.ttl_hours,
    )
    print(f"interbank_reference_rates: {interbank.shape}")
    ff = fetch_forex_factory_calendar(use_cache=True, cache_ttl_hours=24)
    print(f"forex_factory_calendar: {ff.shape}")
    ff_risk = pd.DataFrame()
    try:
        from quant_stockpicker_core import forex_factory_event_risk

        ff_risk = forex_factory_event_risk(ff)
    except Exception:
        ff_risk = pd.DataFrame()

    geo = {"summary": pd.DataFrame(), "articles": pd.DataFrame(), "country_heatmap": pd.DataFrame()}
    if args.include_geopolitical:
        geo = geopolitical_thermometer(use_cache=True, cache_ttl_hours=24)
        print(
            "geopolitical: "
            f"summary={geo.get('summary', pd.DataFrame()).shape}, "
            f"articles={geo.get('articles', pd.DataFrame()).shape}, "
            f"country_heatmap={geo.get('country_heatmap', pd.DataFrame()).shape}"
        )
    if not prices.empty:
        sem = market_sentiment_sem(
            prices,
            macro=macro if "macro" in locals() else pd.DataFrame(),
            forex_event_risk=ff_risk,
            geopolitical_summary=geo.get("summary", pd.DataFrame()),
            benchmark="SPY",
        )
        print(
            "market_sentiment_sem: "
            f"timeline={sem.get('timeline', pd.DataFrame()).shape}, "
            f"loadings={sem.get('loadings', pd.DataFrame()).shape}"
        )

    elapsed = datetime.now() - started
    print(f"done in {elapsed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
