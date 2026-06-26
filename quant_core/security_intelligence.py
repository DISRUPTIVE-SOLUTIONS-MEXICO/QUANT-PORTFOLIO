from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_core.contracts import SecurityIntelligenceV1


@dataclass(frozen=True)
class SecurityIntelligenceConfig:
    """Frozen conventions for causal, cross-sectional security diagnostics."""

    minimum_observations: int = 252
    beta_lookback: int = 252
    residual_momentum_lookback: int = 126
    risk_lookback: int = 252
    liquidity_lookback: int = 63
    history_days: int = 756
    tail_probability: float = 0.10


def _datetime_index(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    if "__index__" in out.columns:
        out = out.set_index("__index__")
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out.loc[out.index.notna()]
    return out.sort_index()


def _period_return(prices: pd.Series, observations: int) -> float:
    series = pd.to_numeric(prices, errors="coerce").dropna()
    if len(series) <= observations:
        return np.nan
    return float(series.iloc[-1] / series.iloc[-observations - 1] - 1.0)


def _annualized_volatility(returns: pd.Series, observations: int) -> float:
    sample = pd.to_numeric(returns, errors="coerce").dropna().tail(observations)
    if len(sample) < max(20, observations // 4):
        return np.nan
    return float(sample.std(ddof=1) * np.sqrt(252.0))


def _downside_deviation(returns: pd.Series, observations: int) -> float:
    sample = pd.to_numeric(returns, errors="coerce").dropna().tail(observations)
    if len(sample) < max(20, observations // 4):
        return np.nan
    return float(np.sqrt(np.mean(np.minimum(sample.to_numpy(dtype=float), 0.0) ** 2)) * np.sqrt(252.0))


def _conditional_beta(asset: pd.Series, benchmark: pd.Series, mask: pd.Series) -> float:
    aligned = pd.concat([asset.rename("asset"), benchmark.rename("benchmark"), mask.rename("mask")], axis=1).dropna()
    selected = aligned.loc[aligned["mask"].astype(bool), ["asset", "benchmark"]]
    if len(selected) < 20:
        return np.nan
    variance = float(selected["benchmark"].var(ddof=1))
    if not np.isfinite(variance) or variance <= 1e-14:
        return np.nan
    return float(selected["asset"].cov(selected["benchmark"]) / variance)


def _market_model(
    asset: pd.Series,
    benchmark: pd.Series,
    observations: int,
    residual_observations: int,
    tail_probability: float,
) -> dict[str, float]:
    aligned = pd.concat([asset.rename("asset"), benchmark.rename("benchmark")], axis=1).dropna().tail(observations)
    if len(aligned) < max(63, observations // 2):
        return {
            "Beta_to_Xi_252D": np.nan,
            "Correlation_to_Xi_252D": np.nan,
            "Upside_Beta_252D": np.nan,
            "Downside_Beta_252D": np.nan,
            "Tail_Beta_252D": np.nan,
            "Residual_Momentum_126D": np.nan,
            "Idiosyncratic_Vol_252D": np.nan,
        }
    benchmark_variance = float(aligned["benchmark"].var(ddof=1))
    beta = (
        float(aligned["asset"].cov(aligned["benchmark"]) / benchmark_variance) if benchmark_variance > 1e-14 else np.nan
    )
    correlation = float(aligned["asset"].corr(aligned["benchmark"]))
    upside_beta = _conditional_beta(aligned["asset"], aligned["benchmark"], aligned["benchmark"] > 0.0)
    downside_beta = _conditional_beta(aligned["asset"], aligned["benchmark"], aligned["benchmark"] < 0.0)
    tail_threshold = float(aligned["benchmark"].quantile(tail_probability))
    tail_beta = _conditional_beta(
        aligned["asset"],
        aligned["benchmark"],
        aligned["benchmark"] <= tail_threshold,
    )

    design = np.column_stack([np.ones(len(aligned)), aligned["benchmark"].to_numpy(dtype=float)])
    response = aligned["asset"].to_numpy(dtype=float)
    coefficients, *_ = np.linalg.lstsq(design, response, rcond=None)
    residuals = response - design @ coefficients
    residual_window = residuals[-min(residual_observations, len(residuals)) :]
    residual_momentum = float(np.prod(1.0 + residual_window) - 1.0)
    idiosyncratic_vol = float(np.std(residuals, ddof=1) * np.sqrt(252.0))
    return {
        "Beta_to_Xi_252D": beta,
        "Correlation_to_Xi_252D": correlation,
        "Upside_Beta_252D": upside_beta,
        "Downside_Beta_252D": downside_beta,
        "Tail_Beta_252D": tail_beta,
        "Residual_Momentum_126D": residual_momentum,
        "Idiosyncratic_Vol_252D": idiosyncratic_vol,
    }


def _drawdown_state(prices: pd.Series, observations: int) -> dict[str, float | bool]:
    sample = pd.to_numeric(prices, errors="coerce").dropna().tail(observations)
    if len(sample) < 20:
        return {
            "Current_Drawdown": np.nan,
            "Max_Drawdown_252D": np.nan,
            "Recovery_Days_From_Trough": np.nan,
            "Drawdown_Recovered": False,
        }
    running_max = sample.cummax()
    drawdown = sample / running_max - 1.0
    trough_date = drawdown.idxmin()
    peak_value = float(running_max.loc[trough_date])
    after_trough = sample.loc[trough_date:]
    recovered = after_trough[after_trough >= peak_value]
    if len(recovered):
        recovery_days = float((pd.Timestamp(recovered.index[0]) - pd.Timestamp(trough_date)).days)
        drawdown_recovered = True
    else:
        recovery_days = float((pd.Timestamp(sample.index[-1]) - pd.Timestamp(trough_date)).days)
        drawdown_recovered = False
    return {
        "Current_Drawdown": float(drawdown.iloc[-1]),
        "Max_Drawdown_252D": float(drawdown.min()),
        "Recovery_Days_From_Trough": recovery_days,
        "Drawdown_Recovered": drawdown_recovered,
    }


def _liquidity_state(
    ticker: str,
    prices: pd.Series,
    returns: pd.Series,
    volumes: pd.DataFrame,
    observations: int,
) -> dict[str, float]:
    if volumes.empty or ticker not in volumes.columns:
        return {
            "ADV_USD_20D": np.nan,
            "ADV_USD_63D": np.nan,
            "Amihud_ILLIQ_63D": np.nan,
            "Zero_Volume_Days_63D": np.nan,
        }
    aligned = pd.concat(
        [
            pd.to_numeric(prices, errors="coerce").rename("price"),
            pd.to_numeric(volumes[ticker], errors="coerce").rename("volume"),
            pd.to_numeric(returns, errors="coerce").rename("return"),
        ],
        axis=1,
    ).dropna(subset=["price", "volume"])
    if aligned.empty:
        return {
            "ADV_USD_20D": np.nan,
            "ADV_USD_63D": np.nan,
            "Amihud_ILLIQ_63D": np.nan,
            "Zero_Volume_Days_63D": np.nan,
        }
    dollar_volume = aligned["price"] * aligned["volume"]
    tail = aligned.tail(observations)
    tail_dollar = dollar_volume.reindex(tail.index)
    illiquidity = (tail["return"].abs() / tail_dollar.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    return {
        "ADV_USD_20D": float(dollar_volume.tail(20).mean()) if len(dollar_volume) >= 5 else np.nan,
        "ADV_USD_63D": float(tail_dollar.mean()) if len(tail_dollar) >= 20 else np.nan,
        "Amihud_ILLIQ_63D": float(illiquidity.mean()) if illiquidity.notna().sum() >= 10 else np.nan,
        "Zero_Volume_Days_63D": float((tail["volume"] <= 0.0).sum()),
    }


def _strategy_consensus(strategy_scores: pd.DataFrame | None, as_of: pd.Timestamp) -> pd.DataFrame:
    if strategy_scores is None or strategy_scores.empty or "Ticker" not in strategy_scores:
        return pd.DataFrame()
    scores = strategy_scores.copy()
    if "As_Of" in scores:
        scores["As_Of"] = pd.to_datetime(scores["As_Of"], errors="coerce")
        scores = scores.loc[scores["As_Of"].notna() & (scores["As_Of"] <= as_of)]
        if "Strategy_ID" in scores:
            latest_by_strategy = scores.groupby("Strategy_ID")["As_Of"].transform("max")
            scores = scores.loc[scores["As_Of"] == latest_by_strategy]
    if scores.empty:
        return pd.DataFrame()
    scores["Score"] = (
        pd.to_numeric(scores["Score"], errors="coerce") if "Score" in scores else pd.Series(np.nan, index=scores.index)
    )
    scores["Selected"] = (
        scores["Selected"].fillna(False).astype(bool) if "Selected" in scores else pd.Series(False, index=scores.index)
    )
    scores["Weight"] = (
        pd.to_numeric(scores["Weight"], errors="coerce").fillna(0.0)
        if "Weight" in scores
        else pd.Series(0.0, index=scores.index)
    )
    strategy_key = "Strategy_ID" if "Strategy_ID" in scores else "Strategy"
    scores["Within_Strategy_Percentile"] = scores.groupby(strategy_key)["Score"].rank(pct=True, method="average")
    grouped = (
        scores.groupby("Ticker", as_index=False)
        .agg(
            Strategies_Scored=(strategy_key, "nunique"),
            Strategies_Selected=("Selected", "sum"),
            Strategy_Selection_Breadth=("Selected", "mean"),
            Consensus_Rank_0_1=("Within_Strategy_Percentile", "mean"),
            Consensus_Weight=("Weight", "sum"),
        )
        .sort_values(["Strategies_Selected", "Consensus_Rank_0_1"], ascending=False)
    )
    return grouped


def build_security_intelligence(
    prices: pd.DataFrame,
    *,
    benchmark: str,
    volumes: pd.DataFrame | None = None,
    strategy_scores: pd.DataFrame | None = None,
    as_of: pd.Timestamp | str | None = None,
    config: SecurityIntelligenceConfig | None = None,
) -> dict:
    """Build a causal security workbench from information available at ``as_of``.

    All estimators are descriptive live-snapshot diagnostics. They do not
    inherit an OOS or holdout label from the strategy laboratory.
    """

    cfg = config or SecurityIntelligenceConfig()
    px = _datetime_index(prices)
    vol = _datetime_index(volumes)
    if px.empty:
        return {
            "benchmark_xi": benchmark.upper(),
            "as_of": None,
            "metrics": pd.DataFrame(),
            "price_history": pd.DataFrame(),
            "strategy_consensus": pd.DataFrame(),
            "methodology": pd.DataFrame(),
        }
    decision_date = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp(px.index.max())
    px = px.loc[:decision_date]
    vol = vol.loc[:decision_date] if not vol.empty else vol
    benchmark = benchmark.upper()
    if benchmark not in px.columns:
        return {
            "benchmark_xi": benchmark,
            "as_of": decision_date,
            "metrics": pd.DataFrame(),
            "price_history": pd.DataFrame(),
            "strategy_consensus": pd.DataFrame(),
            "methodology": pd.DataFrame(),
        }

    returns = px.pct_change(fill_method=None)
    benchmark_returns = pd.to_numeric(returns[benchmark], errors="coerce")
    consensus = _strategy_consensus(strategy_scores, decision_date)
    consensus_by_ticker = consensus.set_index("Ticker") if not consensus.empty else pd.DataFrame()
    metric_rows: list[dict] = []
    for ticker in px.columns:
        series = pd.to_numeric(px[ticker], errors="coerce").dropna()
        if len(series) < cfg.minimum_observations:
            continue
        asset_returns = pd.to_numeric(returns[ticker], errors="coerce")
        risk_sample = asset_returns.dropna().tail(cfg.risk_lookback)
        cvar = np.nan
        if len(risk_sample) >= 63:
            threshold = float(risk_sample.quantile(0.05))
            tail = risk_sample.loc[risk_sample <= threshold]
            cvar = float(tail.mean()) if not tail.empty else threshold
        ma_63 = float(series.tail(63).mean()) if len(series) >= 63 else np.nan
        ma_200 = float(series.tail(200).mean()) if len(series) >= 200 else np.nan
        spot = float(series.iloc[-1])
        return_63 = _period_return(series, 63)
        if np.isfinite(ma_200) and spot >= ma_200 and (not np.isfinite(return_63) or return_63 >= 0.0):
            trend_state = "Bullish"
        elif np.isfinite(ma_200) and spot < ma_200 and np.isfinite(return_63) and return_63 < 0.0:
            trend_state = "Bearish"
        else:
            trend_state = "Transitional"
        row: dict[str, object] = {
            "Ticker": str(ticker),
            "As_Of": pd.Timestamp(series.index[-1]),
            "Evidence_Scope": "live_snapshot",
            "Benchmark_Xi": benchmark,
            "Observed_Price": spot,
            "Return_21D": _period_return(series, 21),
            "Return_63D": return_63,
            "Return_126D": _period_return(series, 126),
            "Return_252D": _period_return(series, 252),
            "Annualized_Vol_21D": _annualized_volatility(asset_returns, 21),
            "Annualized_Vol_63D": _annualized_volatility(asset_returns, 63),
            "Annualized_Vol_252D": _annualized_volatility(asset_returns, 252),
            "Downside_Deviation_63D": _downside_deviation(asset_returns, 63),
            "CVaR_95_Daily": cvar,
            "MA_63D": ma_63,
            "MA_200D": ma_200,
            "Trend_State": trend_state,
            "Observations": int(series.notna().sum()),
            "First_Price_Date": pd.Timestamp(series.index[0]),
            "Last_Price_Date": pd.Timestamp(series.index[-1]),
            "Stale_Days": int((decision_date.normalize() - pd.Timestamp(series.index[-1]).normalize()).days),
        }
        row.update(
            _market_model(
                asset_returns,
                benchmark_returns,
                cfg.beta_lookback,
                cfg.residual_momentum_lookback,
                cfg.tail_probability,
            )
        )
        row.update(_drawdown_state(series, cfg.risk_lookback))
        row.update(_liquidity_state(ticker, series, asset_returns, vol, cfg.liquidity_lookback))
        if not consensus_by_ticker.empty and ticker in consensus_by_ticker.index:
            consensus_row = consensus_by_ticker.loc[ticker]
            for column in consensus_by_ticker.columns:
                row[column] = consensus_row[column]
        else:
            row.update(
                {
                    "Strategies_Scored": 0,
                    "Strategies_Selected": 0,
                    "Strategy_Selection_Breadth": 0.0,
                    "Consensus_Rank_0_1": np.nan,
                    "Consensus_Weight": 0.0,
                }
            )
        metric_rows.append(row)

    metrics = pd.DataFrame(metric_rows)
    if not metrics.empty:
        metrics = metrics.sort_values(
            ["Strategies_Selected", "Consensus_Rank_0_1", "Residual_Momentum_126D"],
            ascending=[False, False, False],
            na_position="last",
        ).reset_index(drop=True)
    price_columns = [ticker for ticker in metrics.get("Ticker", pd.Series(dtype=str)).astype(str) if ticker in px]
    if benchmark in px and benchmark not in price_columns:
        price_columns.insert(0, benchmark)
    price_history = px[price_columns].tail(cfg.history_days).reset_index()
    if not price_history.empty:
        price_history = price_history.rename(columns={price_history.columns[0]: "Date"})
    methodology = pd.DataFrame(
        [
            {
                "Evidence_Scope": "live_snapshot",
                "Decision_Date": decision_date,
                "Benchmark_Xi": benchmark,
                "Minimum_Observations": cfg.minimum_observations,
                "Beta_Lookback": cfg.beta_lookback,
                "Residual_Momentum_Lookback": cfg.residual_momentum_lookback,
                "Risk_Lookback": cfg.risk_lookback,
                "Liquidity_Lookback": cfg.liquidity_lookback,
                "Signal_Timing": "all inputs timestamp <= decision date",
                "Beta_Formula": "Cov(r_i,r_xi)/Var(r_xi)",
                "Tail_Beta_Formula": "conditional beta when r_xi <= q10(r_xi)",
                "Residual_Momentum_Formula": "compound OLS market-model residuals over 126 observations",
            }
        ]
    )
    contract = SecurityIntelligenceV1(
        as_of=decision_date.to_pydatetime(),
        benchmark_xi=benchmark,
        tickers=tuple(price_columns),
        minimum_observations=cfg.minimum_observations,
        price_history_days=min(cfg.history_days, len(price_history)),
        formulas={
            "beta": "Cov(r_i,r_xi)/Var(r_xi)",
            "tail_beta": f"conditional beta when r_xi <= q{cfg.tail_probability:.2f}(r_xi)",
            "residual_momentum": f"compound OLS market-model residuals over {cfg.residual_momentum_lookback} observations",
            "drawdown": "P_t/running_max(P)-1",
        },
    )
    return {
        "contract": contract.model_dump(mode="json"),
        "benchmark_xi": benchmark,
        "as_of": decision_date,
        "metrics": metrics,
        "price_history": price_history,
        "strategy_consensus": consensus,
        "methodology": methodology,
    }
