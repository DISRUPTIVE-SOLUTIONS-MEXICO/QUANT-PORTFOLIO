from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from quant_stockpicker_core import RunConfig, run_pipeline
from supabase_store import _json_safe, save_run_to_supabase


DEFAULT_CLOUD_TICKERS = """
AAPL MSFT NVDA META GOOGL AMZN ORCL CRM AMD QCOM
JPM BAC WFC GS MS BLK SCHW C
XOM CVX COP SLB EOG MPC
LLY JNJ MRK ABBV AMGN TMO DHR ISRG VRTX GEHC REGN
PG KO PEP WMT COST MDLZ
HD LOW MCD NKE SBUX BKNG
CAT DE HON GE RTX LMT UPS UNP
LIN APD SHW ECL NEM FCX
NEE DUK SO AEP XEL VST CEG SMR ED
PLD AMT EQIX DLR O WELL
SPY QQQ IWM DIA ACWI VT EEM VEA
"""


def parse_tickers(raw: str) -> tuple[str, ...]:
    tokens = raw.replace(",", " ").replace("\n", " ").split()
    return tuple(dict.fromkeys(t.strip().upper().replace(".", "-") for t in tokens if t.strip()))


def build_cloud_config(args: argparse.Namespace) -> RunConfig:
    rigorous = args.mode == "rigorous"
    tickers = parse_tickers(DEFAULT_CLOUD_TICKERS + " " + args.tickers)
    use_sec_edgar = args.use_sec_edgar or rigorous
    use_options_snapshot = args.use_options_snapshot or rigorous
    use_forex_factory = args.use_forex_factory or rigorous
    return RunConfig(
        tickers=tickers[: args.max_tickers],
        benchmark_ticker=args.benchmark,
        price_period=args.period,
        top_n=args.top_n,
        preselect_n=args.preselect_n,
        min_chunk=5,
        max_chunk=8 if rigorous else 5,
        max_combos=10_000 if rigorous else 300,
        max_names_per_sector=3 if rigorous else 2,
        max_weight=0.20,
        sector_weight_cap=0.35,
        weight_objective=args.objective,
        compute_mode=args.mode,
        use_persistent_cache=True,
        cache_ttl_hours=args.ttl_hours,
        max_workers=args.workers,
        rate_country=args.country,
        use_sec_edgar=use_sec_edgar,
        sec_user_agent=args.sec_user_agent,
        use_sec_nlp=rigorous and use_sec_edgar,
        sec_nlp_max_tickers=20 if rigorous else 8,
        use_options_snapshot=use_options_snapshot,
        option_expiries=2 if rigorous else 1,
        use_garch=rigorous,
        garch_candidate_n=20 if rigorous else 8,
        validation_bootstrap_samples=256 if rigorous else 16,
        reality_check_samples=256 if rigorous else 16,
        cpcv_folds=4 if rigorous else 2,
        use_gdelt=args.include_geopolitical,
        use_forex_factory_calendar=use_forex_factory,
        use_latent_macro_regime=rigorous,
        use_kaizen_bandit=False,
        sortino_multistarts=6 if rigorous else 1,
        bootstrap_samples=64 if rigorous else 8,
        rebalance_freq="2QE",
        reoptimization_freq="YE",
        benchmark_group="US Market",
        benchmark_mandate_type="Relative vs benchmark",
        benchmark_auto_select=True,
        use_side_boom_portfolio=False,
        investor_horizon_years=args.horizon_years,
        investor_initial_capital=args.initial_capital,
        investor_monthly_contribution=args.monthly_contribution,
        investor_liquidity_need=args.liquidity_need,
        investor_max_drawdown=args.max_drawdown,
        investor_risk_aversion_score=args.risk_aversion,
        investor_objective=args.investor_objective,
        investor_base_currency=args.base_currency,
    )


def write_latest_local(results: dict, run_id: str | None = None) -> Path:
    out_dir = Path(__file__).resolve().with_name(".quant_cache") / "cloud"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dashboard_payload": results.get("dashboard_payload", {}),
        "data_freshness_report": results.get("data_freshness_report"),
        "promotion_gate": results.get("promotion_gate", {}),
        "suitability_gate": results.get("suitability_gate", {}),
    }
    path = out_dir / "latest_dashboard_payload.json"
    path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Daily cloud refresh for Quant Portfolio-Kaizen. Computes once, persists artifacts, and lets the UI render preloaded state."
    )
    parser.add_argument("--mode", choices=["fast", "rigorous"], default=os.getenv("QPK_CLOUD_REFRESH_MODE", "fast"))
    parser.add_argument("--period", default=os.getenv("QPK_CLOUD_REFRESH_PERIOD", "2y"))
    parser.add_argument("--benchmark", default=os.getenv("QPK_CLOUD_REFRESH_BENCHMARK", "SPY"))
    parser.add_argument("--objective", default=os.getenv("QPK_CLOUD_REFRESH_OBJECTIVE", "sortino"))
    parser.add_argument("--country", default=os.getenv("QPK_CLOUD_REFRESH_COUNTRY", "United States"))
    parser.add_argument("--tickers", default=os.getenv("QPK_CLOUD_REFRESH_EXTRA_TICKERS", ""))
    parser.add_argument("--max-tickers", type=int, default=int(os.getenv("QPK_CLOUD_REFRESH_MAX_TICKERS", "32")))
    parser.add_argument("--top-n", type=int, default=int(os.getenv("QPK_CLOUD_REFRESH_TOP_N", "5")))
    parser.add_argument("--preselect-n", type=int, default=int(os.getenv("QPK_CLOUD_REFRESH_PRESELECT_N", "10")))
    parser.add_argument("--workers", type=int, default=int(os.getenv("QPK_CLOUD_REFRESH_WORKERS", "1")))
    parser.add_argument("--ttl-hours", type=int, default=int(os.getenv("QPK_CLOUD_REFRESH_TTL_HOURS", "24")))
    parser.add_argument("--include-geopolitical", action="store_true", default=os.getenv("QPK_CLOUD_REFRESH_GEO", "0") == "1")
    parser.add_argument("--use-sec-edgar", action="store_true", default=os.getenv("QPK_CLOUD_REFRESH_SEC_EDGAR", "0") == "1")
    parser.add_argument("--use-options-snapshot", action="store_true", default=os.getenv("QPK_CLOUD_REFRESH_OPTIONS", "0") == "1")
    parser.add_argument("--use-forex-factory", action="store_true", default=os.getenv("QPK_CLOUD_REFRESH_FOREX_FACTORY", "0") == "1")
    parser.add_argument("--save-supabase", action="store_true", default=os.getenv("QPK_CLOUD_REFRESH_SAVE_SUPABASE", "1") == "1")
    parser.add_argument(
        "--require-supabase",
        action="store_true",
        default=os.getenv("QPK_CLOUD_REFRESH_REQUIRE_SUPABASE", "0") == "1",
        help="Fail the refresh if Supabase persistence fails. Use this in cloud jobs that feed the online app.",
    )
    parser.add_argument("--sec-user-agent", default=os.getenv("SEC_USER_AGENT", "QuantPortfolioKaizen/1.0 contact@example.com"))
    parser.add_argument("--horizon-years", type=float, default=float(os.getenv("QPK_CLOUD_REFRESH_HORIZON_YEARS", "3")))
    parser.add_argument("--initial-capital", type=float, default=float(os.getenv("QPK_CLOUD_REFRESH_INITIAL_CAPITAL", "100000")))
    parser.add_argument("--monthly-contribution", type=float, default=float(os.getenv("QPK_CLOUD_REFRESH_MONTHLY_CONTRIBUTION", "0")))
    parser.add_argument("--liquidity-need", default=os.getenv("QPK_CLOUD_REFRESH_LIQUIDITY_NEED", "Media"))
    parser.add_argument("--max-drawdown", type=float, default=float(os.getenv("QPK_CLOUD_REFRESH_MAX_DRAWDOWN", "0.20")))
    parser.add_argument("--risk-aversion", type=float, default=float(os.getenv("QPK_CLOUD_REFRESH_RISK_AVERSION", "5")))
    parser.add_argument("--investor-objective", default=os.getenv("QPK_CLOUD_REFRESH_INVESTOR_OBJECTIVE", "Balanced growth"))
    parser.add_argument("--base-currency", default=os.getenv("QPK_CLOUD_REFRESH_BASE_CURRENCY", "USD"))
    args = parser.parse_args()

    started = datetime.now(timezone.utc)
    config = build_cloud_config(args)
    print(f"[{started.isoformat(timespec='seconds')}] cloud refresh started")
    print(f"mode={args.mode} tickers={len(config.tickers)} benchmark={config.benchmark_ticker} objective={config.weight_objective}")
    results = run_pipeline(config)
    run_id = None
    if args.save_supabase:
        try:
            run_id = save_run_to_supabase(results, config, status="completed")
            print(f"saved_supabase_run_id={run_id}")
        except Exception as exc:
            print(f"supabase_save_error={type(exc).__name__}: {str(exc)[:300]}")
            if args.require_supabase:
                raise
    local_path = write_latest_local(results, run_id=run_id)
    elapsed = datetime.now(timezone.utc) - started
    print(f"local_latest_artifact={local_path}")
    print(f"done elapsed={elapsed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
