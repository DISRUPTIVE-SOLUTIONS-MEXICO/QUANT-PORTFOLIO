from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from quant_core.dashboard_payload import build_dashboard_payload
from quant_stockpicker_core import (
    RunConfig,
    carry_trade_suggestions,
    download_prices,
    download_volume,
    fetch_options_snapshot,
    fetch_forex_factory_calendar,
    fetch_interbank_reference_rates,
    forex_factory_event_risk,
    geopolitical_thermometer,
    global_yield_curve_discrete_history,
    global_yield_curve_snapshot,
    market_regime,
    market_sentiment_sem,
    portfolio_implied_vol_surface,
    run_pipeline,
    summarize_options_snapshot,
    validate_carry_trade_strategies,
    xcdr_v3_sample_score,
)
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
        min_chunk=args.min_chunk,
        max_chunk=args.max_chunk,
        max_combos=args.max_combos,
        max_names_per_sector=3 if rigorous else 2,
        max_weight=args.max_weight,
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
    trend_regime = (
        "Bullish" if trend_return > 0 and (not np.isfinite(latest_ma) or latest_price >= latest_ma) else "Bearish"
    )

    short_vol = (
        float(benchmark_returns.tail(21).std(ddof=1) * np.sqrt(252.0)) if len(benchmark_returns) >= 10 else np.nan
    )
    long_vol = (
        float(benchmark_returns.tail(126).std(ddof=1) * np.sqrt(252.0)) if len(benchmark_returns) >= 42 else short_vol
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


def build_daily_market_intelligence(prices: pd.DataFrame, config: RunConfig) -> dict:
    """Refresh market intelligence without mutating the saved allocation policy."""

    def safe_call(label: str, default, fn):
        print(f"market_intelligence_stage={label}:start", flush=True)
        try:
            value = fn()
            if isinstance(value, pd.DataFrame):
                size = len(value)
            elif isinstance(value, tuple) and value and isinstance(value[0], pd.DataFrame):
                size = len(value[0])
            elif isinstance(value, dict):
                size = sum(len(frame) for frame in value.values() if isinstance(frame, pd.DataFrame))
            else:
                size = 0
            print(f"market_intelligence_stage={label}:done rows={size}", flush=True)
            return value
        except Exception as exc:
            print(
                f"market_intelligence_warning={label}:{type(exc).__name__}:{str(exc)[:180]}",
                flush=True,
            )
            return default

    macro_pair = safe_call(
        "macro",
        (pd.DataFrame(), pd.Series(dtype=object)),
        lambda: market_regime(
            prices,
            country=config.rate_country,
            use_cache=config.use_persistent_cache,
            cache_ttl_hours=config.cache_ttl_hours,
            use_latent_macro_regime=config.use_latent_macro_regime,
        ),
    )
    macro, latest_macro = macro_pair
    global_rates = safe_call(
        "global_yield_curves",
        pd.DataFrame(),
        lambda: global_yield_curve_snapshot(
            prices,
            use_cache=config.use_persistent_cache,
            cache_ttl_hours=config.cache_ttl_hours,
        ),
    )
    global_rate_history = safe_call(
        "global_rate_history",
        pd.DataFrame(),
        lambda: global_yield_curve_discrete_history(
            prices,
            use_cache=config.use_persistent_cache,
            cache_ttl_hours=config.cache_ttl_hours,
        ),
    )
    interbank_reference_rates = safe_call(
        "interbank_reference_rates",
        pd.DataFrame(),
        lambda: fetch_interbank_reference_rates(
            prices.index.min() - pd.Timedelta(days=30),
            prices.index.max() + pd.Timedelta(days=5),
            use_cache=config.use_persistent_cache,
            cache_ttl_hours=config.cache_ttl_hours,
        ),
    )
    forex_calendar = (
        safe_call(
            "forex_factory_calendar",
            pd.DataFrame(),
            lambda: fetch_forex_factory_calendar(
                use_cache=config.use_persistent_cache,
                cache_ttl_hours=config.cache_ttl_hours,
            ),
        )
        if config.use_forex_factory_calendar
        else pd.DataFrame()
    )
    forex_event_risk = forex_factory_event_risk(forex_calendar)
    geopolitical = (
        safe_call(
            "geopolitical",
            {"summary": pd.DataFrame(), "articles": pd.DataFrame(), "timeline": pd.DataFrame()},
            lambda: geopolitical_thermometer(
                use_cache=config.use_persistent_cache,
                cache_ttl_hours=config.cache_ttl_hours,
            ),
        )
        if config.use_gdelt
        else {"summary": pd.DataFrame(), "articles": pd.DataFrame(), "timeline": pd.DataFrame()}
    )
    sentiment = safe_call(
        "market_sentiment_sem",
        {},
        lambda: market_sentiment_sem(
            prices,
            macro=macro,
            forex_event_risk=forex_event_risk,
            geopolitical_summary=geopolitical.get("summary", pd.DataFrame()),
            benchmark=config.benchmark_ticker,
            lookback=756,
        ),
    )
    carry = carry_trade_suggestions(global_rates, forex_event_risk)
    carry_validation = safe_call(
        "carry_trade_validation",
        pd.DataFrame(),
        lambda: validate_carry_trade_strategies(
            carry,
            global_rates=global_rates,
            use_cache=config.use_persistent_cache,
            cache_ttl_hours=config.cache_ttl_hours,
        ),
    )
    return {
        "macro": macro,
        "latest_macro": latest_macro,
        "global_yield_curves": global_rates,
        "global_rate_history": global_rate_history,
        "interbank_reference_rates": interbank_reference_rates,
        "carry_trade_suggestions": carry,
        "carry_trade_validation": carry_validation,
        "market_sentiment_sem": sentiment,
        "alternative_data": {
            "forex_factory_calendar": forex_calendar,
            "forex_factory_event_risk": forex_event_risk,
            "summary": geopolitical.get("summary", pd.DataFrame()),
            "gdelt_timeline": geopolitical.get("timeline", pd.DataFrame()),
            "gdelt_articles": geopolitical.get("articles", pd.DataFrame()),
            "country_heatmap": geopolitical.get("country_heatmap", pd.DataFrame()),
        },
    }


def build_fast_dashboard_snapshot(config: RunConfig) -> dict:
    """Build a causal daily market artifact without reoptimizing saved portfolios."""
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
    volumes = download_volume(
        prices.columns,
        config.price_period,
        use_cache=config.use_persistent_cache,
        cache_ttl_hours=config.cache_ttl_hours,
    )
    if not volumes.empty:
        volumes = volumes.sort_index().reindex(prices.index).ffill().reindex(columns=prices.columns)
    minimum_observations = 720
    minimum_calendar_days = 365 * 3 - 14
    investable = []
    for column in prices.columns:
        if column == config.benchmark_ticker:
            continue
        history = pd.to_numeric(prices[column], errors="coerce").dropna()
        if len(history) < minimum_observations:
            continue
        if (pd.Timestamp(history.index[-1]) - pd.Timestamp(history.index[0])).days < minimum_calendar_days:
            continue
        investable.append(column)
    if not investable:
        raise RuntimeError("Fast dashboard snapshot has no investable assets with at least three years of history.")

    returns = prices.pct_change(fill_method=None)
    train_days = min(756, max(504, len(returns) // 3))
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
            f"{config.benchmark_ticker} observed price": prices[config.benchmark_ticker]
            .reindex(portfolio_nav.index)
            .ffill()
            .values,
            "Daily causal allocation proxy price": anchor * portfolio_nav.values / float(portfolio_nav.iloc[0]),
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
    p_metrics["XCDR_v3"] = xcdr_v3_sample_score(portfolio_returns, benchmark_returns)
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
            "Optimization_XCDR_v3": p_metrics.get("XCDR_v3"),
        }
    )
    max_options_tickers = max(0, int(os.getenv("QPK_CLOUD_REFRESH_MAX_OPTIONS_TICKERS", "12")))
    option_tickers = (
        portfolio.sort_values("Weight", ascending=False)["Ticker"].astype(str).head(max_options_tickers).tolist()
        if max_options_tickers > 0 and not portfolio.empty
        else []
    )
    if config.use_options_snapshot and option_tickers:
        print(f"market_intelligence_stage=options_snapshot:start tickers={len(option_tickers)}", flush=True)
        try:
            options_chain = fetch_options_snapshot(
                option_tickers,
                prices,
                max_expiries=max(1, int(config.option_expiries or 1)),
                max_workers=max(1, min(config.max_workers, len(option_tickers))),
                use_cache=config.use_persistent_cache,
                cache_ttl_hours=config.options_cache_ttl_hours,
            )
            options_summary = summarize_options_snapshot(options_chain)
            portfolio_vol_surface = portfolio_implied_vol_surface(options_chain, portfolio)
            print(
                f"market_intelligence_stage=options_snapshot:done "
                f"chain_rows={len(options_chain)} summary_rows={len(options_summary)}",
                flush=True,
            )
        except Exception as exc:
            print(
                f"market_intelligence_warning=options_snapshot:{type(exc).__name__}:{str(exc)[:180]}",
                flush=True,
            )
            options_chain = pd.DataFrame()
            options_summary = pd.DataFrame()
            portfolio_vol_surface = {
                "portfolio_vol_surface": pd.DataFrame(),
                "portfolio_vol_surface_matrix": pd.DataFrame(),
                "portfolio_vol_surface_diagnostics": pd.DataFrame(),
            }
    else:
        options_chain = pd.DataFrame()
        options_summary = pd.DataFrame()
        portfolio_vol_surface = {
            "portfolio_vol_surface": pd.DataFrame(),
            "portfolio_vol_surface_matrix": pd.DataFrame(),
            "portfolio_vol_surface_diagnostics": pd.DataFrame(),
        }
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
    market_intelligence = build_daily_market_intelligence(prices, config)
    freshness_rows = [
        {
            "Namespace": "prices_daily",
            "Status": "fresh",
            "As_Of": pd.Timestamp(prices.index.max()),
            "Rows": int(len(prices)),
            "Fallback_Used": False,
        }
    ]
    freshness_rows.append(
        {
            "Namespace": "volume_daily",
            "Status": "fresh" if not volumes.empty else "unavailable",
            "As_Of": pd.Timestamp(volumes.index.max()) if not volumes.empty else pd.Timestamp(prices.index.max()),
            "Rows": int(len(volumes)),
            "Fallback_Used": False,
        }
    )
    for namespace, key in (
        ("macro_daily", "macro"),
        ("global_yield_curves", "global_yield_curves"),
        ("global_rate_history", "global_rate_history"),
        ("interbank_reference_rates", "interbank_reference_rates"),
    ):
        frame = market_intelligence.get(key, pd.DataFrame())
        freshness_rows.append(
            {
                "Namespace": namespace,
                "Status": "fresh" if isinstance(frame, pd.DataFrame) and not frame.empty else "unavailable",
                "As_Of": pd.Timestamp(prices.index.max()),
                "Rows": int(len(frame)) if isinstance(frame, pd.DataFrame) else 0,
                "Fallback_Used": False,
            }
        )
    freshness_rows.append(
        {
            "Namespace": "options_yahoo_snapshot",
            "Status": "fresh" if not options_summary.empty or not options_chain.empty else "unavailable",
            "As_Of": pd.Timestamp(prices.index.max()),
            "Rows": int(len(options_chain)),
            "Fallback_Used": False,
        }
    )
    freshness = pd.DataFrame(freshness_rows)
    snapshot_meta = pd.DataFrame(
        [
            {
                "Snapshot_Mode": "daily_price_snapshot",
                "Is_User_Specific": False,
                "Analytics_Scope": "Prices, volume, security intelligence, causal OOS path, macro, rates, latent sentiment, events, and geopolitical context",
                "Benchmark": config.benchmark_ticker,
                "As_Of": pd.Timestamp(prices.index.max()),
                "Method": "causal_monthly_price_snapshot_v3",
            }
        ]
    )
    market_context = _price_snapshot_context(
        prices,
        benchmark=config.benchmark_ticker,
        investable=investable,
    )
    suitability_summary = pd.DataFrame(
        [
            {
                "Metric": "Snapshot_Status",
                "Value": "Precomputed dashboard; user suitability is evaluated on explicit runs.",
            }
        ]
    )
    promotion_summary = pd.DataFrame([{"Metric": "Promotion_Status", "Value": "RESEARCH_SNAPSHOT_NOT_PROMOTED"}])
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
                "method": "causal_monthly_price_snapshot_v3",
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    results = {
        "prices": prices,
        "volumes": volumes,
        # A daily price prewarm is market evidence, not an allocation mandate.
        # Keep official allocation/fundamental surfaces empty so it cannot
        # displace the latest full point-in-time research artifact.
        "portfolio": pd.DataFrame(),
        "price_snapshot_selection": portfolio,
        "options_chain": options_chain,
        "options_summary": options_summary,
        "portfolio_vol_surface": portfolio_vol_surface.get("portfolio_vol_surface", pd.DataFrame()),
        "portfolio_vol_surface_matrix": portfolio_vol_surface.get("portfolio_vol_surface_matrix", pd.DataFrame()),
        "portfolio_vol_surface_diagnostics": portfolio_vol_surface.get(
            "portfolio_vol_surface_diagnostics", pd.DataFrame()
        ),
        "performance_summary": performance_summary,
        "equity_curve": equity_curve,
        "backtest_perf": equity_curve,
        "backtest_holdings": holdings,
        "backtest_path_bundle": path_bundle,
        "return_diagnostics": {},
        "validation_diagnostics": {},
        "rejection_diagnostics": pd.DataFrame(),
        "global_yield_curves": pd.DataFrame(),
        "snapshot_meta": snapshot_meta,
        "market_context": market_context,
        "suitability_gate": suitability_gate,
        "promotion_gate": promotion_gate,
        "data_freshness_report": freshness,
        "model_registry": pd.DataFrame(
            [
                {
                    "run_hash": run_hash,
                    "code_version": "cloud-snapshot-v3",
                    "objective": "causal_market_and_security_snapshot",
                    "warnings": [
                        "Price-only prewarm; observed names are not persisted as recommended allocation.",
                        "Full fundamentals, sectors, suitability and validation require a full analysis run.",
                    ],
                }
            ]
        ),
    }
    results.update(market_intelligence)
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
    dashboard_payload = results.get("dashboard_payload", {})
    payload = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dashboard_payload": dashboard_payload,
        "data_freshness_report": results.get("data_freshness_report"),
        "promotion_gate": results.get("promotion_gate", {}),
        "suitability_gate": results.get("suitability_gate", {}),
    }
    scope = "full_analysis"
    status = dashboard_payload.get("status", {}) if isinstance(dashboard_payload, dict) else {}
    snapshot_meta = _json_safe(status.get("snapshot_meta", [])) if isinstance(status, dict) else []
    if isinstance(snapshot_meta, list) and snapshot_meta:
        mode = str(snapshot_meta[0].get("Snapshot_Mode", "")).lower()
        if mode == "daily_price_snapshot":
            scope = "daily_snapshot"
    elif isinstance(snapshot_meta, dict):
        mode = str(snapshot_meta.get("Snapshot_Mode", "")).lower()
        if mode == "daily_price_snapshot":
            scope = "daily_snapshot"

    scoped_filename = "latest_daily_snapshot_payload.json" if scope == "daily_snapshot" else "latest_full_analysis_payload.json"
    scoped_path = out_dir / scoped_filename
    serialized = json.dumps(_json_safe(payload), indent=2, ensure_ascii=False, default=str)
    scoped_path.write_text(serialized, encoding="utf-8")
    legacy_path = out_dir / "latest_dashboard_payload.json"
    legacy_path.write_text(serialized, encoding="utf-8")
    return scoped_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Daily cloud refresh for Quant Portfolio-Kaizen. Computes once, persists artifacts, and lets the UI render preloaded state."
    )
    parser.add_argument("--mode", choices=["fast", "rigorous"], default=os.getenv("QPK_CLOUD_REFRESH_MODE", "fast"))
    parser.add_argument("--period", default=os.getenv("QPK_CLOUD_REFRESH_PERIOD", "10y"))
    parser.add_argument("--benchmark", default=os.getenv("QPK_CLOUD_REFRESH_BENCHMARK", "SPY"))
    parser.add_argument("--objective", default=os.getenv("QPK_CLOUD_REFRESH_OBJECTIVE", "xcdr_v3"))
    parser.add_argument("--country", default=os.getenv("QPK_CLOUD_REFRESH_COUNTRY", "United States"))
    parser.add_argument("--tickers", default=os.getenv("QPK_CLOUD_REFRESH_EXTRA_TICKERS", ""))
    parser.add_argument("--max-tickers", type=int, default=int(os.getenv("QPK_CLOUD_REFRESH_MAX_TICKERS", "32")))
    parser.add_argument("--top-n", type=int, default=int(os.getenv("QPK_CLOUD_REFRESH_TOP_N", "5")))
    parser.add_argument("--preselect-n", type=int, default=int(os.getenv("QPK_CLOUD_REFRESH_PRESELECT_N", "10")))
    parser.add_argument("--min-chunk", type=int, default=int(os.getenv("QPK_CLOUD_REFRESH_MIN_CHUNK", "5")))
    parser.add_argument("--max-chunk", type=int, default=int(os.getenv("QPK_CLOUD_REFRESH_MAX_CHUNK", "8")))
    parser.add_argument("--max-combos", type=int, default=int(os.getenv("QPK_CLOUD_REFRESH_MAX_COMBOS", "10000")))
    parser.add_argument("--max-weight", type=float, default=float(os.getenv("QPK_CLOUD_REFRESH_MAX_WEIGHT", "0.20")))
    parser.add_argument("--workers", type=int, default=int(os.getenv("QPK_CLOUD_REFRESH_WORKERS", "1")))
    parser.add_argument("--ttl-hours", type=int, default=int(os.getenv("QPK_CLOUD_REFRESH_TTL_HOURS", "24")))
    parser.add_argument(
        "--include-geopolitical", action="store_true", default=os.getenv("QPK_CLOUD_REFRESH_GEO", "0") == "1"
    )
    parser.add_argument(
        "--use-sec-edgar", action="store_true", default=os.getenv("QPK_CLOUD_REFRESH_SEC_EDGAR", "0") == "1"
    )
    parser.add_argument(
        "--use-options-snapshot", action="store_true", default=os.getenv("QPK_CLOUD_REFRESH_OPTIONS", "0") == "1"
    )
    parser.add_argument(
        "--use-forex-factory", action="store_true", default=os.getenv("QPK_CLOUD_REFRESH_FOREX_FACTORY", "0") == "1"
    )
    parser.add_argument(
        "--save-supabase", action="store_true", default=os.getenv("QPK_CLOUD_REFRESH_SAVE_SUPABASE", "1") == "1"
    )
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
    parser.add_argument(
        "--sec-user-agent", default=os.getenv("SEC_USER_AGENT", "QuantPortfolioKaizen/1.0 contact@example.com")
    )
    parser.add_argument("--horizon-years", type=float, default=float(os.getenv("QPK_CLOUD_REFRESH_HORIZON_YEARS", "3")))
    parser.add_argument(
        "--initial-capital", type=float, default=float(os.getenv("QPK_CLOUD_REFRESH_INITIAL_CAPITAL", "100000"))
    )
    parser.add_argument(
        "--monthly-contribution", type=float, default=float(os.getenv("QPK_CLOUD_REFRESH_MONTHLY_CONTRIBUTION", "0"))
    )
    parser.add_argument("--liquidity-need", default=os.getenv("QPK_CLOUD_REFRESH_LIQUIDITY_NEED", "Media"))
    parser.add_argument(
        "--max-drawdown", type=float, default=float(os.getenv("QPK_CLOUD_REFRESH_MAX_DRAWDOWN", "0.20"))
    )
    parser.add_argument("--risk-aversion", type=float, default=float(os.getenv("QPK_CLOUD_REFRESH_RISK_AVERSION", "5")))
    parser.add_argument(
        "--investor-objective", default=os.getenv("QPK_CLOUD_REFRESH_INVESTOR_OBJECTIVE", "Balanced growth")
    )
    parser.add_argument("--base-currency", default=os.getenv("QPK_CLOUD_REFRESH_BASE_CURRENCY", "USD"))
    args = parser.parse_args()

    started = datetime.now(UTC)
    config = build_cloud_config(args)
    print(f"[{started.isoformat(timespec='seconds')}] cloud refresh started", flush=True)
    print(
        f"mode={args.mode} tickers={len(config.tickers)} benchmark={config.benchmark_ticker} objective={config.weight_objective}",
        flush=True,
    )
    results = run_pipeline(config) if args.full_pipeline else build_fast_dashboard_snapshot(config)
    run_id = None
    if args.save_supabase:
        try:
            run_id = save_run_to_supabase(results, config, status="completed")
            print(f"saved_supabase_run_id={run_id}", flush=True)
        except Exception as exc:
            print(f"supabase_save_error={type(exc).__name__}: {str(exc)[:300]}", flush=True)
            if args.require_supabase:
                raise
    local_path = write_latest_local(results, run_id=run_id)
    elapsed = datetime.now(UTC) - started
    print(f"local_latest_artifact={local_path}", flush=True)
    print(f"done elapsed={elapsed}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
