from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from quant_core.dashboard_payload import build_dashboard_payload
from quant_stockpicker_core import RunConfig, download_prices, run_pipeline
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


def _annualized_metrics(returns: pd.Series) -> dict[str, float]:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if r.empty:
        return {}
    annual_return = float((1.0 + r).prod() ** (252.0 / len(r)) - 1.0)
    annual_vol = float(r.std(ddof=1) * np.sqrt(252.0)) if len(r) > 1 else np.nan
    downside = float(np.sqrt(np.mean(np.minimum(r, 0.0) ** 2)) * np.sqrt(252.0))
    nav = (1.0 + r).cumprod()
    max_dd = float((nav / nav.cummax() - 1.0).min())
    q = float(r.quantile(0.05))
    tail = r[r <= q]
    cvar = float(tail.mean()) if not tail.empty else q
    return {
        "Annualized_Return": annual_return,
        "Annualized_Vol": annual_vol,
        "Downside_Deviation": downside,
        "Sortino": annual_return / downside if downside > 1e-12 else np.nan,
        "Max_Drawdown": max_dd,
        "CVaR_95": cvar,
    }


def _snapshot_weights(train_returns: pd.DataFrame, top_n: int) -> pd.Series:
    min_obs = min(63, max(20, len(train_returns) // 2))
    valid = train_returns.count()
    cols = valid[valid >= min_obs].index
    train = train_returns[cols].dropna(axis=1, how="all")
    if train.empty:
        return pd.Series(dtype=float)
    compounded = (1.0 + train.fillna(0.0)).prod() - 1.0
    downside = np.sqrt(np.minimum(train.fillna(0.0), 0.0).pow(2).mean()).replace(0.0, np.nan)
    score = (compounded / downside).replace([np.inf, -np.inf], np.nan).dropna()
    selected = score.sort_values(ascending=False).head(max(1, top_n)).index
    if len(selected) == 0:
        return pd.Series(dtype=float)
    risk = downside.reindex(selected).replace(0.0, np.nan)
    raw = (score.reindex(selected).clip(lower=0.05) / risk).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if raw.sum() <= 0:
        raw = 1.0 / train[selected].std(ddof=1).replace(0.0, np.nan)
    raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return raw / raw.sum() if raw.sum() > 0 else pd.Series(1.0 / len(selected), index=selected)


def _price_snapshot_context(
    prices: pd.DataFrame,
    *,
    benchmark: str,
    investable: list[str],
) -> pd.DataFrame:
    """Causal market-state proxies derived only from prices available as of the snapshot."""
    benchmark_prices = pd.to_numeric(prices[benchmark], errors="coerce").dropna()
    if benchmark_prices.empty:
        return pd.DataFrame()

    benchmark_returns = benchmark_prices.pct_change(fill_method=None).dropna()
    ma_window = min(200, max(63, len(benchmark_prices) // 2))
    moving_average = benchmark_prices.rolling(ma_window, min_periods=max(20, ma_window // 2)).mean()
    latest_price = float(benchmark_prices.iloc[-1])
    latest_ma = float(moving_average.iloc[-1]) if pd.notna(moving_average.iloc[-1]) else np.nan
    trend_return = (
        float(benchmark_prices.iloc[-1] / benchmark_prices.iloc[-64] - 1.0)
        if len(benchmark_prices) >= 64
        else float(benchmark_prices.iloc[-1] / benchmark_prices.iloc[0] - 1.0)
    )
    trend_regime = "Bullish" if trend_return > 0 and (not np.isfinite(latest_ma) or latest_price >= latest_ma) else "Bearish"

    short_vol = (
        float(benchmark_returns.tail(21).std(ddof=1) * np.sqrt(252.0))
        if len(benchmark_returns) >= 10
        else np.nan
    )
    long_vol = (
        float(benchmark_returns.tail(126).std(ddof=1) * np.sqrt(252.0))
        if len(benchmark_returns) >= 42
        else short_vol
    )
    vol_ratio = short_vol / long_vol if np.isfinite(short_vol) and np.isfinite(long_vol) and long_vol > 0 else np.nan
    if not np.isfinite(vol_ratio):
        volatility_regime = "Unavailable"
    elif vol_ratio >= 1.25:
        volatility_regime = "Elevated"
    elif vol_ratio <= 0.80:
        volatility_regime = "Compressed"
    else:
        volatility_regime = "Normal"

    running_max = benchmark_prices.cummax()
    current_drawdown = float(benchmark_prices.iloc[-1] / running_max.iloc[-1] - 1.0)

    breadth_flags: list[float] = []
    for ticker in investable:
        series = pd.to_numeric(prices[ticker], errors="coerce").dropna()
        if len(series) < 42:
            continue
        window = min(126, len(series))
        average = float(series.tail(window).mean())
        if np.isfinite(average) and average > 0:
            breadth_flags.append(float(series.iloc[-1] > average))
    breadth = float(np.mean(breadth_flags)) if breadth_flags else np.nan

    return pd.DataFrame(
        [
            {
                "As_Of": pd.Timestamp(benchmark_prices.index[-1]),
                "Benchmark": benchmark,
                "Trend_Regime": trend_regime,
                "Trend_Return_3M": trend_return,
                "Realized_Vol_21D": short_vol,
                "Realized_Vol_126D": long_vol,
                "Volatility_Regime": volatility_regime,
                "Current_Drawdown": current_drawdown,
                "Breadth_Above_126D_MA": breadth,
                "Method": "causal_price_proxy",
            }
        ]
    )


def build_fast_dashboard_snapshot(config: RunConfig) -> dict:
    """Build a causal, price-only daily artifact without running the full research pipeline."""
    tickers = tuple(dict.fromkeys(list(config.tickers) + [config.benchmark_ticker]))
    prices = download_prices(
        tickers,
        config.price_period,
        use_cache=config.use_persistent_cache,
        cache_ttl_hours=config.cache_ttl_hours,
    )
    if prices.empty or config.benchmark_ticker not in prices:
        raise RuntimeError("Fast dashboard snapshot could not obtain benchmark-aligned prices.")
    prices = prices.sort_index().ffill().dropna(axis=1, how="all")
    investable = [c for c in prices.columns if c != config.benchmark_ticker and prices[c].notna().sum() >= 160]
    if not investable:
        raise RuntimeError("Fast dashboard snapshot has no investable assets with sufficient history.")

    returns = prices.pct_change(fill_method=None)
    train_days = min(126, max(63, len(returns) // 3))
    rebalance_days = 21
    portfolio_returns = pd.Series(index=returns.index, dtype=float)
    holdings_rows: list[dict] = []
    latest_weights = pd.Series(dtype=float)
    for start in range(train_days, len(returns), rebalance_days):
        train = returns[investable].iloc[max(0, start - train_days) : start]
        weights = _snapshot_weights(train, config.top_n)
        if weights.empty:
            continue
        end = min(start + rebalance_days, len(returns))
        test = returns.loc[returns.index[start:end], weights.index].fillna(0.0)
        portfolio_returns.loc[test.index] = test @ weights
        signal_date = pd.Timestamp(returns.index[start - 1])
        execution_date = pd.Timestamp(returns.index[start])
        for ticker, weight in weights.items():
            holdings_rows.append(
                {
                    "Signal_Date": signal_date,
                    "Rebalance_Date": execution_date,
                    "Ticker": ticker,
                    "Weight": float(weight),
                    "Effective_Weight": float(weight),
                }
            )
        latest_weights = weights

    portfolio_returns = portfolio_returns.dropna()
    benchmark_returns = returns[config.benchmark_ticker].reindex(portfolio_returns.index).fillna(0.0)
    if portfolio_returns.empty or benchmark_returns.empty:
        raise RuntimeError("Fast dashboard snapshot could not construct a causal OOS path.")

    portfolio_nav = (1.0 + portfolio_returns).cumprod()
    benchmark_nav = (1.0 + benchmark_returns).cumprod()
    anchor = float(prices[config.benchmark_ticker].reindex(portfolio_nav.index).ffill().iloc[0])
    price_paths = pd.DataFrame(
        {
            "Date": portfolio_nav.index,
            f"{config.benchmark_ticker} observed price": prices[config.benchmark_ticker].reindex(portfolio_nav.index).ffill().values,
            "Sortino optimized synthetic NAV price": anchor * portfolio_nav.values / float(portfolio_nav.iloc[0]),
        }
    )
    drawdowns = pd.DataFrame({"Date": price_paths["Date"]})
    max_dd_rows = []
    for col in price_paths.columns[1:]:
        series = pd.to_numeric(price_paths[col], errors="coerce")
        dd = series / series.cummax() - 1.0
        drawdowns[col] = dd
        idx = dd.idxmin()
        max_dd_rows.append(
            {"Series": col, "Max_Drawdown": float(dd.loc[idx]), "Max_Drawdown_Date": price_paths.loc[idx, "Date"]}
        )
    max_drawdown_table = pd.DataFrame(max_dd_rows)
    path_bundle = {
        "price_paths": price_paths,
        "drawdowns": drawdowns,
        "max_drawdown_table": max_drawdown_table,
        "path_metadata": {
            "benchmark": config.benchmark_ticker,
            "source": "causal_monthly_price_snapshot",
            "drawdown_formula": "P_t / running_max(P_t) - 1",
        },
    }

    p_metrics = _annualized_metrics(portfolio_returns)
    b_metrics = _annualized_metrics(benchmark_returns)
    performance_rows = [{"Metric": key, "Value": value} for key, value in p_metrics.items()]
    performance_rows.extend({"Metric": f"Benchmark_{key}", "Value": value} for key, value in b_metrics.items())
    performance_summary = pd.DataFrame(performance_rows)
    last_train = returns[investable].tail(train_days)
    if latest_weights.empty:
        latest_weights = _snapshot_weights(last_train, config.top_n)
    portfolio = pd.DataFrame(
        {
            "Ticker": latest_weights.index,
            "Weight": latest_weights.values,
            "Sector": "Price-only snapshot",
            "Country": "Mixed",
            "Composite_Score": np.nan,
            "Optimization_Sortino": p_metrics.get("Sortino"),
        }
    )
    holdings = pd.DataFrame(holdings_rows)
    equity_curve = pd.DataFrame(
        {
            "Period_End": portfolio_returns.index,
            "Net_Return": portfolio_returns.values,
            "Benchmark_Return": benchmark_returns.values,
            "Portfolio_Equity": portfolio_nav.values,
            "Benchmark_Equity": benchmark_nav.values,
            "Active_Equity": (portfolio_nav / benchmark_nav).values,
        }
    )
    freshness = pd.DataFrame(
        [
            {
                "Namespace": "prices_daily",
                "Status": "fresh",
                "As_Of": pd.Timestamp(prices.index.max()),
                "Rows": int(len(prices)),
                "Fallback_Used": False,
            }
        ]
    )
    snapshot_meta = pd.DataFrame(
        [
            {
                "Snapshot_Mode": "daily_price_snapshot",
                "Is_User_Specific": False,
                "Analytics_Scope": "Prices, causal OOS path, allocation weights, and price-derived market context",
                "Benchmark": config.benchmark_ticker,
                "As_Of": pd.Timestamp(prices.index.max()),
                "Method": "causal_monthly_price_snapshot_v2",
            }
        ]
    )
    market_context = _price_snapshot_context(
        prices,
        benchmark=config.benchmark_ticker,
        investable=investable,
    )
    suitability_summary = pd.DataFrame(
        [{"Metric": "Snapshot_Status", "Value": "Precomputed dashboard; user suitability is evaluated on explicit runs."}]
    )
    promotion_summary = pd.DataFrame(
        [{"Metric": "Promotion_Status", "Value": "RESEARCH_SNAPSHOT_NOT_PROMOTED"}]
    )
    suitability_gate = {
        "status": "snapshot",
        "summary": suitability_summary,
        "breaches": pd.DataFrame(),
        "user_safe_summary": "Precomputed market snapshot. Run the allocation engine for a user-specific recommendation.",
    }
    promotion_gate = {
        "promotion_status": "RESEARCH_SNAPSHOT_NOT_PROMOTED",
        "summary": promotion_summary,
        "tests": pd.DataFrame(),
    }
    run_hash = hashlib.sha256(
        json.dumps(
            {
                "tickers": sorted(prices.columns.astype(str).tolist()),
                "asof": str(prices.index.max()),
                "benchmark": config.benchmark_ticker,
                "method": "causal_monthly_price_snapshot_v2",
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    results = {
        "prices": prices,
        "portfolio": portfolio,
        "performance_summary": performance_summary,
        "equity_curve": equity_curve,
        "backtest_perf": equity_curve,
        "backtest_holdings": holdings,
        "backtest_path_bundle": path_bundle,
        "return_diagnostics": {},
        "validation_diagnostics": {},
        "rejection_diagnostics": pd.DataFrame(),
        "global_yield_curves": pd.DataFrame(),
        "portfolio_vol_surface_matrix": pd.DataFrame(),
        "snapshot_meta": snapshot_meta,
        "market_context": market_context,
        "suitability_gate": suitability_gate,
        "promotion_gate": promotion_gate,
        "data_freshness_report": freshness,
        "model_registry": pd.DataFrame(
            [
                {
                    "run_hash": run_hash,
                    "code_version": "cloud-snapshot-v2",
                    "objective": "causal_price_snapshot",
                    "warnings": ["Price-only prewarm; full fundamentals and validation run on explicit optimization."],
                }
            ]
        ),
    }
    results["dashboard_payload"] = build_dashboard_payload(
        results,
        path_bundle,
        suitability_gate,
        promotion_gate,
        freshness_report=freshness,
    )
    return results


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
        "--full-pipeline",
        action="store_true",
        default=os.getenv("QPK_CLOUD_REFRESH_FULL_PIPELINE", "0") == "1",
        help="Run the full research pipeline instead of the causal price-only prewarm snapshot.",
    )
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
    results = run_pipeline(config) if args.full_pipeline else build_fast_dashboard_snapshot(config)
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
