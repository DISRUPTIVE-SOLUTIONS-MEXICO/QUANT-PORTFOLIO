from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from quant_core.strategy_registry import (
    STRATEGY_DEFINITIONS,
    StrategySpecification,
    strategy_registry_frame,
)

RESEARCH_GENERATION = "strategy-lab-g3b-equivalence-controlled-registry"
HOLDOUT_STATUS = "CONSUMED_FOR_DIAGNOSIS"


def _clean_prices(prices: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    frame = prices.copy()
    frame.index = pd.to_datetime(frame.index, errors="coerce")
    frame = frame.loc[frame.index.notna()].sort_index()
    frame = frame.loc[~frame.index.duplicated(keep="last")]
    frame.columns = [str(column).upper() for column in frame.columns]
    benchmark = benchmark.upper()
    if benchmark not in frame.columns:
        return pd.DataFrame()
    frame = frame.apply(pd.to_numeric, errors="coerce")
    minimum_observations = min(504, max(252, int(len(frame) * 0.60)))
    valid = frame.count()
    columns = valid[valid >= minimum_observations].index.tolist()
    if benchmark not in columns:
        columns.append(benchmark)
    return frame.loc[:, list(dict.fromkeys(columns))].dropna(how="all")


def _beta(asset: pd.Series, benchmark: pd.Series) -> float:
    aligned = pd.concat([asset, benchmark], axis=1).dropna()
    if len(aligned) < 20:
        return np.nan
    variance = float(aligned.iloc[:, 1].var(ddof=1))
    if not np.isfinite(variance) or variance <= 1e-12:
        return np.nan
    return float(aligned.iloc[:, 0].cov(aligned.iloc[:, 1]) / variance)


def _capture_ratio(asset: pd.Series, benchmark: pd.Series, *, positive: bool) -> float:
    aligned = pd.concat([asset, benchmark], axis=1).dropna()
    selected = aligned.iloc[:, 1] > 0 if positive else aligned.iloc[:, 1] < 0
    sample = aligned.loc[selected]
    denominator = float(sample.iloc[:, 1].mean()) if not sample.empty else np.nan
    if not np.isfinite(denominator) or abs(denominator) <= 1e-12:
        return np.nan
    return float(sample.iloc[:, 0].mean() / denominator)


def _tail_beta(asset: pd.Series, benchmark: pd.Series, quantile: float = 0.05) -> float:
    aligned = pd.concat([asset, benchmark], axis=1).dropna()
    if len(aligned) < 40:
        return np.nan
    threshold = float(aligned.iloc[:, 1].quantile(quantile))
    sample = aligned.loc[aligned.iloc[:, 1] <= threshold]
    denominator = float(sample.iloc[:, 1].mean()) if not sample.empty else np.nan
    if not np.isfinite(denominator) or abs(denominator) <= 1e-12:
        return np.nan
    return float(sample.iloc[:, 0].mean() / denominator)


def _residual_returns(history: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    benchmark_returns = history[benchmark]
    output: dict[str, pd.Series] = {}
    for ticker in history.columns:
        if ticker == benchmark:
            continue
        beta = _beta(history[ticker], benchmark_returns)
        if np.isfinite(beta):
            output[ticker] = history[ticker] - beta * benchmark_returns
    return pd.DataFrame(output, index=history.index)


def _scores(
    definition: StrategySpecification,
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    signal_position: int,
    benchmark: str,
) -> pd.Series:
    assets = [ticker for ticker in prices.columns if ticker != benchmark]
    end = signal_position + 1
    start = max(0, end - definition.lookback_days)
    history = returns.iloc[start:end]
    price_history = prices.iloc[start:end]
    benchmark_returns = history[benchmark]
    scores: dict[str, float] = {}

    residuals = _residual_returns(history, benchmark)
    for ticker in assets:
        asset_returns = history[ticker].dropna()
        aligned = pd.concat([asset_returns, benchmark_returns], axis=1).dropna()
        if len(aligned) < max(42, definition.lookback_days // 2):
            continue

        downside = float(np.sqrt(np.mean(np.minimum(aligned.iloc[:, 0], 0.0) ** 2)))
        upside_beta = _capture_ratio(aligned.iloc[:, 0], aligned.iloc[:, 1], positive=True)
        downside_beta = _capture_ratio(aligned.iloc[:, 0], aligned.iloc[:, 1], positive=False)
        tail_beta = _tail_beta(aligned.iloc[:, 0], aligned.iloc[:, 1])

        if definition.strategy_id in {
            "cross_sectional_momentum_12_1",
            "downside_controlled_momentum",
            "volatility_adjusted_trend",
            "dual_momentum",
        }:
            series = price_history[ticker].dropna()
            if len(series) < 230:
                continue
            recent = float(series.iloc[-22])
            distant = float(series.iloc[0])
            momentum = recent / distant - 1.0 if distant > 0 else np.nan
            if definition.strategy_id == "volatility_adjusted_trend":
                realized_vol = float(aligned.iloc[:, 0].tail(63).std(ddof=1) * np.sqrt(252.0))
                score = momentum / realized_vol if momentum > 0 and realized_vol > 1e-12 else np.nan
            elif definition.strategy_id == "dual_momentum":
                benchmark_prices = price_history[benchmark].dropna()
                if len(benchmark_prices) < 230:
                    continue
                benchmark_momentum = float(benchmark_prices.iloc[-22] / benchmark_prices.iloc[0] - 1.0)
                relative_momentum = momentum - benchmark_momentum
                score = relative_momentum if momentum > 0 and relative_momentum > 0 else np.nan
            else:
                score = momentum
        elif definition.strategy_id == "residual_momentum_6m":
            residual = residuals.get(ticker, pd.Series(dtype=float)).dropna()
            score = float(residual.sum()) if len(residual) >= 42 else np.nan
        elif definition.strategy_id == "asymmetric_capture":
            if not all(np.isfinite(value) for value in (upside_beta, downside_beta, tail_beta)):
                continue
            score = upside_beta - downside_beta - 0.50 * max(tail_beta - 1.0, 0.0)
        elif definition.strategy_id == "defensive_convexity":
            series = price_history[ticker].dropna()
            momentum = float(series.iloc[-1] / series.iloc[0] - 1.0) if len(series) >= 42 else np.nan
            if not all(np.isfinite(value) for value in (momentum, downside, upside_beta, downside_beta)):
                continue
            score = momentum + 0.35 * (upside_beta - downside_beta) - 4.0 * downside
        else:
            residual = residuals.get(ticker, pd.Series(dtype=float)).dropna()
            if len(residual) < 21:
                continue
            residual_vol = float(residual.tail(21).std(ddof=1))
            score = -float(residual.tail(5).sum()) / residual_vol if residual_vol > 1e-12 else np.nan

        if np.isfinite(score):
            scores[ticker] = float(score)
    return pd.Series(scores, dtype=float).sort_values(ascending=False)


def _capped_normalize(raw: pd.Series, max_weight: float) -> pd.Series:
    weights = raw.clip(lower=0.0).astype(float)
    if weights.empty or float(weights.sum()) <= 0:
        return pd.Series(dtype=float)
    weights /= float(weights.sum())
    effective_cap = max(max_weight, 1.0 / len(weights))
    fixed = pd.Series(0.0, index=weights.index)
    remaining = weights.copy()
    remaining_mass = 1.0
    while len(remaining):
        candidate = remaining / float(remaining.sum()) * remaining_mass
        breached = candidate > effective_cap + 1e-12
        if not breached.any():
            fixed.loc[candidate.index] = candidate
            break
        capped_names = candidate.index[breached]
        fixed.loc[capped_names] = effective_cap
        remaining = remaining.drop(capped_names)
        remaining_mass = 1.0 - float(fixed.sum())
        if remaining_mass <= 1e-12:
            break
    return fixed / float(fixed.sum()) if float(fixed.sum()) > 0 else fixed


def _strategy_weights(
    definition: StrategySpecification,
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    signal_position: int,
    benchmark: str,
    top_n: int,
    max_weight: float,
) -> tuple[pd.Series, pd.Series]:
    scores = _scores(definition, prices, returns, signal_position, benchmark)
    selected = scores.head(max(2, min(top_n, len(scores))))
    if selected.empty:
        return pd.Series(dtype=float), scores
    history = returns.iloc[max(0, signal_position - 125) : signal_position + 1]
    selected_history = history.loc[:, selected.index].to_numpy(dtype=float)
    downside = pd.Series(
        np.sqrt(np.nanmean(np.square(np.minimum(selected_history, 0.0)), axis=0)),
        index=selected.index,
        dtype=float,
    ).replace(0.0, np.nan)
    score_rank = selected.rank(pct=True).clip(lower=0.25)
    raw = (score_rank / downside).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if float(raw.sum()) <= 0:
        raw = pd.Series(1.0, index=selected.index)
    return _capped_normalize(raw, max_weight), scores


def _market_regime(prices: pd.Series, returns: pd.Series, position: int) -> str:
    price_history = prices.iloc[: position + 1].dropna()
    return_history = returns.iloc[: position + 1].dropna()
    if len(price_history) < 126 or len(return_history) < 63:
        return "insufficient_history"
    trend = float(price_history.iloc[-1] / price_history.iloc[-126] - 1.0)
    short_vol = float(return_history.tail(21).std(ddof=1))
    long_vol = float(return_history.tail(126).std(ddof=1))
    high_vol = np.isfinite(short_vol) and np.isfinite(long_vol) and short_vol > 1.20 * long_vol
    if trend >= 0 and not high_vol:
        return "expansion"
    if trend >= 0 and high_vol:
        return "fragile"
    if trend < 0 and high_vol:
        return "stress"
    return "recovery"


def _causal_risk_exposure(
    definition: StrategySpecification,
    benchmark_prices: pd.Series,
    benchmark_returns: pd.Series,
    position: int,
) -> tuple[float, dict[str, float]]:
    if definition.strategy_id != "downside_controlled_momentum":
        return 1.0, {
            "Trend_126D": np.nan,
            "Drawdown_252D": np.nan,
            "Volatility_Ratio_21_126": np.nan,
            "Downside_Ratio_21_126": np.nan,
        }
    price_history = benchmark_prices.iloc[: position + 1].dropna()
    return_history = benchmark_returns.iloc[: position + 1].dropna()
    if len(price_history) < 252 or len(return_history) < 126:
        return 0.35, {
            "Trend_126D": np.nan,
            "Drawdown_252D": np.nan,
            "Volatility_Ratio_21_126": np.nan,
            "Downside_Ratio_21_126": np.nan,
        }

    trend = float(price_history.iloc[-1] / price_history.iloc[-126] - 1.0)
    trailing_prices = price_history.tail(252)
    drawdown = float(trailing_prices.iloc[-1] / trailing_prices.max() - 1.0)
    short_returns = return_history.tail(21)
    long_returns = return_history.tail(126)
    short_vol = float(short_returns.std(ddof=1))
    long_vol = float(long_returns.std(ddof=1))
    short_downside = float(np.sqrt(np.mean(np.minimum(short_returns.to_numpy(dtype=float), 0.0) ** 2)))
    long_downside = float(np.sqrt(np.mean(np.minimum(long_returns.to_numpy(dtype=float), 0.0) ** 2)))
    vol_ratio = short_vol / long_vol if long_vol > 1e-12 else 1.0
    downside_ratio = short_downside / long_downside if long_downside > 1e-12 else 1.0

    trend_state = float(np.clip((trend + 0.10) / 0.20, 0.0, 1.0))
    drawdown_state = float(np.clip(1.0 + 3.0 * drawdown, 0.25, 1.0))
    volatility_state = float(np.clip(1.50 - 0.50 * vol_ratio, 0.25, 1.0))
    downside_state = float(np.clip(1.50 - 0.40 * downside_ratio, 0.25, 1.0))
    confidence = 0.35 * trend_state + 0.25 * drawdown_state + 0.20 * volatility_state + 0.20 * downside_state
    exposure = float(np.clip(0.20 + 0.80 * confidence, 0.25, 1.0))
    return exposure, {
        "Trend_126D": trend,
        "Drawdown_252D": drawdown,
        "Volatility_Ratio_21_126": vol_ratio,
        "Downside_Ratio_21_126": downside_ratio,
    }


def _annualized_metrics(strategy: pd.Series, benchmark: pd.Series) -> dict[str, float]:
    aligned = pd.concat([strategy.rename("strategy"), benchmark.rename("benchmark")], axis=1).dropna()
    if len(aligned) < 20:
        return {}
    strategy_returns = aligned["strategy"]
    benchmark_returns = aligned["benchmark"]
    strategy_array = strategy_returns.to_numpy(dtype=float)
    benchmark_array = benchmark_returns.to_numpy(dtype=float)
    annual_return = float(np.prod(1.0 + strategy_array) ** (252.0 / len(strategy_array)) - 1.0)
    benchmark_return = float(np.prod(1.0 + benchmark_array) ** (252.0 / len(benchmark_array)) - 1.0)
    annual_vol = float(strategy_returns.std(ddof=1) * np.sqrt(252.0))
    downside = float(np.sqrt(np.mean(np.minimum(strategy_returns, 0.0) ** 2)) * np.sqrt(252.0))
    nav = (1.0 + strategy_returns).cumprod()
    drawdown = nav / nav.cummax() - 1.0
    threshold = float(strategy_returns.quantile(0.05))
    tail = strategy_returns[strategy_returns <= threshold]
    active = strategy_returns - benchmark_returns
    tracking_error = float(active.std(ddof=1) * np.sqrt(252.0))
    return {
        "Annualized_Return": annual_return,
        "Benchmark_Annualized_Return": benchmark_return,
        "Active_Return": annual_return - benchmark_return,
        "Annualized_Volatility": annual_vol,
        "Downside_Deviation": downside,
        "Max_Drawdown": float(drawdown.min()),
        "CVaR_95_Daily": float(tail.mean()) if not tail.empty else threshold,
        "Upside_Capture": _capture_ratio(strategy_returns, benchmark_returns, positive=True),
        "Downside_Capture": _capture_ratio(strategy_returns, benchmark_returns, positive=False),
        "Beta_to_Xi": _beta(strategy_returns, benchmark_returns),
        "Tracking_Error": tracking_error,
        "Information_Ratio": float(active.mean() * 252.0 / tracking_error) if tracking_error > 1e-12 else np.nan,
    }


def _deduplicate_candidate_paths(
    candidate_returns: pd.DataFrame,
    *,
    tolerance: float = 1e-12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    canonical: list[str] = []
    rows: list[dict[str, object]] = []
    for candidate in candidate_returns.columns:
        canonical_candidate = str(candidate)
        equivalent = False
        for existing in canonical:
            aligned = pd.concat(
                [
                    candidate_returns[candidate].rename("candidate"),
                    candidate_returns[existing].rename("canonical"),
                ],
                axis=1,
            ).dropna()
            if aligned.empty:
                continue
            maximum_difference = float((aligned["candidate"] - aligned["canonical"]).abs().max())
            if maximum_difference <= tolerance:
                canonical_candidate = existing
                equivalent = True
                break
        if not equivalent:
            canonical.append(str(candidate))
        rows.append(
            {
                "Strategy": str(candidate),
                "Canonical_Strategy": canonical_candidate,
                "Equivalent_Path": equivalent,
                "Included_In_Selection": not equivalent,
                "Tolerance": tolerance,
            }
        )
    return candidate_returns.loc[:, canonical], pd.DataFrame(rows)


def _stationary_bootstrap_indices(length: int, rng: np.random.Generator, restart_probability: float) -> np.ndarray:
    indices = np.empty(length, dtype=int)
    indices[0] = int(rng.integers(0, length))
    for position in range(1, length):
        if rng.random() < restart_probability:
            indices[position] = int(rng.integers(0, length))
        else:
            indices[position] = (indices[position - 1] + 1) % length
    return indices


def _multiple_testing_diagnostics(
    candidate_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    *,
    samples: int,
    seed: int,
) -> pd.DataFrame:
    active = candidate_returns.sub(benchmark_returns, axis=0).dropna(how="all")
    active = active.dropna(axis=1, how="any")
    if len(active) < 126 or active.shape[1] < 2:
        return pd.DataFrame(
            [{"Metric": "Validation_Status", "Value": "Insufficient sample", "Pass": False, "Threshold": None}]
        )
    centered = active - active.mean()
    observed_wrc = float(np.sqrt(len(active)) * active.mean().max())
    std = active.std(ddof=1).replace(0.0, np.nan)
    observed_spa = float((np.sqrt(len(active)) * active.mean() / std).max())
    rng = np.random.default_rng(seed)
    bootstrap_wrc: list[float] = []
    bootstrap_spa: list[float] = []
    for _ in range(max(50, samples)):
        indices = _stationary_bootstrap_indices(len(active), rng, restart_probability=0.10)
        sample = centered.iloc[indices].reset_index(drop=True)
        bootstrap_wrc.append(float(np.sqrt(len(sample)) * sample.mean().max()))
        bootstrap_spa.append(float((np.sqrt(len(sample)) * sample.mean() / std).max()))
    wrc_p = float(np.mean(np.asarray(bootstrap_wrc) >= observed_wrc))
    spa_p = float(np.mean(np.asarray(bootstrap_spa) >= observed_spa))
    pbo = _pbo(candidate_returns)
    return pd.DataFrame(
        [
            {"Metric": "White_Reality_Check_PValue", "Value": wrc_p, "Pass": wrc_p < 0.05, "Threshold": 0.05},
            {"Metric": "Hansen_SPA_PValue", "Value": spa_p, "Pass": spa_p < 0.05, "Threshold": 0.05},
            {"Metric": "PBO", "Value": pbo, "Pass": np.isfinite(pbo) and pbo < 0.10, "Threshold": 0.10},
            {
                "Metric": "Promotion_Status",
                "Value": "RESEARCH_ONLY",
                "Pass": False,
                "Threshold": "Nested validation and frozen holdout required",
            },
        ]
    )


def _pbo(candidate_returns: pd.DataFrame, slices: int = 8) -> float:
    data = candidate_returns.dropna(how="any")
    if len(data) < 126 or data.shape[1] < 2:
        return np.nan
    slices = min(slices, max(4, len(data) // 21))
    if slices % 2:
        slices -= 1
    if slices < 4:
        return np.nan
    chunk_ids = np.array_split(np.arange(len(data)), slices)
    logits: list[float] = []
    for train_chunks in combinations(range(slices), slices // 2):
        if 0 not in train_chunks:
            continue
        train_index = np.concatenate([chunk_ids[index] for index in train_chunks])
        test_index = np.concatenate([chunk_ids[index] for index in range(slices) if index not in train_chunks])
        train = data.iloc[train_index]
        test = data.iloc[test_index]
        train_downside = np.sqrt(np.minimum(train, 0.0).pow(2).mean()).replace(0.0, np.nan)
        train_score = train.mean() / train_downside
        winner = train_score.idxmax()
        test_downside = np.sqrt(np.minimum(test, 0.0).pow(2).mean()).replace(0.0, np.nan)
        test_score = (test.mean() / test_downside).rank(pct=True)
        rank = float(np.clip(test_score.get(winner, np.nan), 1e-6, 1.0 - 1e-6))
        if np.isfinite(rank):
            logits.append(float(np.log(rank / (1.0 - rank))))
    return float(np.mean(np.asarray(logits) < 0.0)) if logits else np.nan


def _selection_utility(strategy: pd.Series, benchmark: pd.Series) -> float:
    metrics = _annualized_metrics(strategy, benchmark)
    benchmark_metrics = _annualized_metrics(benchmark, benchmark)
    if not metrics or not benchmark_metrics:
        return -np.inf
    required = [
        metrics.get("Active_Return"),
        metrics.get("Upside_Capture"),
        metrics.get("Downside_Capture"),
        metrics.get("Downside_Deviation"),
        metrics.get("CVaR_95_Daily"),
        metrics.get("Max_Drawdown"),
        benchmark_metrics.get("Downside_Deviation"),
        benchmark_metrics.get("CVaR_95_Daily"),
        benchmark_metrics.get("Max_Drawdown"),
    ]
    if not all(value is not None and np.isfinite(float(value)) for value in required):
        return -np.inf
    downside_excess = max(
        float(metrics["Downside_Deviation"]) - float(benchmark_metrics["Downside_Deviation"]),
        0.0,
    )
    cvar_excess = max(
        abs(float(metrics["CVaR_95_Daily"])) - abs(float(benchmark_metrics["CVaR_95_Daily"])),
        0.0,
    )
    drawdown_excess = max(
        abs(float(metrics["Max_Drawdown"])) - abs(float(benchmark_metrics["Max_Drawdown"])),
        0.0,
    )
    downside_capture_excess = max(float(metrics["Downside_Capture"]) - 1.0, 0.0)
    return float(
        float(metrics["Active_Return"])
        + 0.10 * (float(metrics["Upside_Capture"]) - 1.0)
        - 0.10 * downside_capture_excess
        - downside_excess
        - 2.0 * np.sqrt(252.0) * cvar_excess
        - 0.50 * drawdown_excess
    )


def _path_frame(
    strategy: pd.Series,
    benchmark: pd.Series,
    *,
    strategy_label: str,
    benchmark_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    aligned = pd.concat([strategy.rename(strategy_label), benchmark.rename(benchmark_label)], axis=1).dropna()
    paths = (1.0 + aligned).cumprod() * 100.0
    drawdowns = paths / paths.cummax() - 1.0
    return paths.reset_index(names="Date"), drawdowns.reset_index(names="Date")


def _nested_walk_forward(
    candidate_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    *,
    train_days: int,
    validation_days: int,
    test_days: int,
    purge_days: int,
    embargo_days: int,
    holdout_days: int,
) -> dict[str, pd.DataFrame | str]:
    data = candidate_returns.dropna(how="any")
    benchmark = benchmark_returns.reindex(data.index).dropna()
    data = data.reindex(benchmark.index)
    minimum = train_days + validation_days + test_days + purge_days + embargo_days + holdout_days
    if len(data) < minimum:
        return {
            "status": "INSUFFICIENT_NESTED_SAMPLE",
            "windows": pd.DataFrame(),
            "candidate_oos_returns": pd.DataFrame(),
            "selected_oos_returns": pd.DataFrame(),
            "holdout_returns": pd.DataFrame(),
            "selection_stability": pd.DataFrame(),
            "frozen_candidate": "",
        }

    development_end = len(data) - holdout_days
    first_test_start = train_days + purge_days + validation_days + embargo_days
    window_rows: list[dict] = []
    selected_oos = pd.Series(np.nan, index=data.index[:development_end], dtype=float)
    candidate_oos = pd.DataFrame(np.nan, index=data.index[:development_end], columns=data.columns, dtype=float)
    selections: list[str] = []
    validation_scores: dict[str, list[float]] = {str(column): [] for column in data.columns}

    window_id = 0
    for test_start in range(first_test_start, development_end, test_days):
        test_end = min(test_start + test_days, development_end)
        validation_end = test_start - embargo_days
        validation_start = validation_end - validation_days
        train_end = validation_start - purge_days
        train_start = train_end - train_days
        if min(train_start, train_end, validation_start, validation_end, test_start) < 0:
            continue
        train = data.iloc[train_start:train_end]
        validation = data.iloc[validation_start:validation_end]
        test = data.iloc[test_start:test_end]
        train_benchmark = benchmark.iloc[train_start:train_end]
        validation_benchmark = benchmark.iloc[validation_start:validation_end]
        if len(train) < train_days or len(validation) < validation_days or test.empty:
            continue

        robust_scores: dict[str, float] = {}
        for candidate in data.columns:
            train_score = _selection_utility(train[candidate], train_benchmark)
            validation_score = _selection_utility(validation[candidate], validation_benchmark)
            robust_score = (
                validation_score - 0.25 * abs(validation_score - train_score)
                if np.isfinite(train_score) and np.isfinite(validation_score)
                else -np.inf
            )
            robust_scores[str(candidate)] = float(robust_score)
            if np.isfinite(validation_score):
                validation_scores[str(candidate)].append(float(validation_score))
        selected = max(robust_scores, key=lambda candidate: robust_scores[candidate])
        if not np.isfinite(robust_scores[selected]):
            continue

        selected_oos.loc[test.index] = test[selected]
        candidate_oos.loc[test.index, :] = test
        selections.append(selected)
        window_rows.append(
            {
                "Window": window_id,
                "Train_Start": train.index[0],
                "Train_End": train.index[-1],
                "Purge_Days": purge_days,
                "Validation_Start": validation.index[0],
                "Validation_End": validation.index[-1],
                "Embargo_Days": embargo_days,
                "Test_Start": test.index[0],
                "Test_End": test.index[-1],
                "Selected_Strategy": selected,
                "Train_Utility": _selection_utility(train[selected], train_benchmark),
                "Validation_Utility": _selection_utility(validation[selected], validation_benchmark),
                "Test_Active_Return": _annualized_metrics(
                    test[selected],
                    benchmark.iloc[test_start:test_end],
                ).get("Active_Return", np.nan),
            }
        )
        window_id += 1

    selected_oos = selected_oos.dropna()
    candidate_oos = candidate_oos.reindex(selected_oos.index).dropna(how="any")
    if not selections or selected_oos.empty:
        return {
            "status": "INSUFFICIENT_NESTED_WINDOWS",
            "windows": pd.DataFrame(window_rows),
            "candidate_oos_returns": candidate_oos,
            "selected_oos_returns": pd.DataFrame(),
            "holdout_returns": pd.DataFrame(),
            "selection_stability": pd.DataFrame(),
            "frozen_candidate": "",
        }

    selection_counts = pd.Series(selections).value_counts()
    stability_rows: list[dict] = []
    for candidate in data.columns:
        scores = validation_scores[str(candidate)]
        stability_rows.append(
            {
                "Strategy": str(candidate),
                "Windows_Selected": int(selection_counts.get(candidate, 0)),
                "Selection_Rate": float(selection_counts.get(candidate, 0) / len(selections)),
                "Mean_Validation_Utility": float(np.mean(scores)) if scores else np.nan,
                "Validation_Utility_Std": float(np.std(scores, ddof=1)) if len(scores) > 1 else np.nan,
            }
        )
    stability = pd.DataFrame(stability_rows).sort_values(
        ["Windows_Selected", "Mean_Validation_Utility"],
        ascending=[False, False],
    )
    frozen_candidate = str(stability.iloc[0]["Strategy"])
    holdout = data.iloc[development_end:]
    holdout_benchmark = benchmark.iloc[development_end:]
    holdout_returns = pd.concat(
        [
            holdout[frozen_candidate].rename("Frozen strategy holdout"),
            holdout_benchmark.rename("Benchmark xi holdout"),
        ],
        axis=1,
    )
    selected_oos_frame = pd.concat(
        [
            selected_oos.rename("Nested selected policy"),
            benchmark.reindex(selected_oos.index).rename("Benchmark xi OOS"),
        ],
        axis=1,
    )
    return {
        "status": "NESTED_OOS_READY",
        "windows": pd.DataFrame(window_rows),
        "candidate_oos_returns": candidate_oos,
        "selected_oos_returns": selected_oos_frame,
        "holdout_returns": holdout_returns,
        "selection_stability": stability,
        "frozen_candidate": frozen_candidate,
    }


def _strict_promotion_evidence(
    nested: dict[str, pd.DataFrame | str],
    *,
    bootstrap_samples: int,
    seed: int,
    holdout_independent: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    selected_oos = nested["selected_oos_returns"]
    holdout = nested["holdout_returns"]
    candidate_oos = nested["candidate_oos_returns"]
    if not isinstance(selected_oos, pd.DataFrame) or selected_oos.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), "RESEARCH_ONLY"
    if not isinstance(holdout, pd.DataFrame) or holdout.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), "RESEARCH_ONLY"
    if not isinstance(candidate_oos, pd.DataFrame) or candidate_oos.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), "RESEARCH_ONLY"

    oos_strategy = selected_oos.iloc[:, 0]
    oos_benchmark = selected_oos.iloc[:, 1]
    holdout_strategy = holdout.iloc[:, 0]
    holdout_benchmark = holdout.iloc[:, 1]
    oos_metrics = _annualized_metrics(oos_strategy, oos_benchmark)
    holdout_metrics = _annualized_metrics(holdout_strategy, holdout_benchmark)
    benchmark_oos_metrics = _annualized_metrics(oos_benchmark, oos_benchmark)
    benchmark_holdout_metrics = _annualized_metrics(holdout_benchmark, holdout_benchmark)
    multiple_testing = _multiple_testing_diagnostics(
        candidate_oos,
        oos_benchmark.reindex(candidate_oos.index),
        samples=bootstrap_samples,
        seed=seed,
    )

    def downside_pass(metrics: dict[str, float], benchmark_metrics: dict[str, float]) -> bool:
        required = ["Downside_Deviation", "CVaR_95_Daily", "Max_Drawdown"]
        if not all(np.isfinite(metrics.get(key, np.nan)) for key in required):
            return False
        return bool(
            metrics["Downside_Deviation"] <= benchmark_metrics["Downside_Deviation"]
            and metrics["CVaR_95_Daily"] >= benchmark_metrics["CVaR_95_Daily"]
            and metrics["Max_Drawdown"] >= benchmark_metrics["Max_Drawdown"]
        )

    wrc = pd.to_numeric(
        multiple_testing.loc[multiple_testing["Metric"] == "White_Reality_Check_PValue", "Value"],
        errors="coerce",
    )
    spa = pd.to_numeric(
        multiple_testing.loc[multiple_testing["Metric"] == "Hansen_SPA_PValue", "Value"],
        errors="coerce",
    )
    pbo = pd.to_numeric(
        multiple_testing.loc[multiple_testing["Metric"] == "PBO", "Value"],
        errors="coerce",
    )
    oos_downside = downside_pass(oos_metrics, benchmark_oos_metrics)
    holdout_downside = downside_pass(holdout_metrics, benchmark_holdout_metrics)
    gate_rows = [
        {
            "Metric": "White_Reality_Check_PValue",
            "Value": float(wrc.iloc[0]) if len(wrc) else np.nan,
            "Threshold": 0.05,
            "Pass": bool(len(wrc) and float(wrc.iloc[0]) < 0.05),
        },
        {
            "Metric": "Hansen_SPA_PValue",
            "Value": float(spa.iloc[0]) if len(spa) else np.nan,
            "Threshold": 0.05,
            "Pass": bool(len(spa) and float(spa.iloc[0]) < 0.05),
        },
        {
            "Metric": "PBO",
            "Value": float(pbo.iloc[0]) if len(pbo) else np.nan,
            "Threshold": 0.10,
            "Pass": bool(len(pbo) and float(pbo.iloc[0]) < 0.10),
        },
        {
            "Metric": "OOS_Active_Return",
            "Value": oos_metrics.get("Active_Return", np.nan),
            "Threshold": 0.0,
            "Pass": bool(oos_metrics.get("Active_Return", -np.inf) > 0.0),
        },
        {
            "Metric": "OOS_Upside_Capture",
            "Value": oos_metrics.get("Upside_Capture", np.nan),
            "Threshold": 1.0,
            "Pass": bool(oos_metrics.get("Upside_Capture", -np.inf) > 1.0),
        },
        {
            "Metric": "OOS_Downside_Capture",
            "Value": oos_metrics.get("Downside_Capture", np.nan),
            "Threshold": 1.0,
            "Pass": bool(oos_metrics.get("Downside_Capture", np.inf) < 1.0),
        },
        {
            "Metric": "OOS_Downside_Preservation",
            "Value": oos_downside,
            "Threshold": True,
            "Pass": oos_downside,
        },
        {
            "Metric": "Holdout_Active_Return",
            "Value": holdout_metrics.get("Active_Return", np.nan),
            "Threshold": 0.0,
            "Pass": bool(holdout_metrics.get("Active_Return", -np.inf) > 0.0),
        },
        {
            "Metric": "Holdout_Downside_Preservation",
            "Value": holdout_downside,
            "Threshold": True,
            "Pass": holdout_downside,
        },
        {
            "Metric": "Holdout_Independence",
            "Value": holdout_independent,
            "Threshold": True,
            "Pass": holdout_independent,
        },
    ]
    passed = all(bool(row["Pass"]) for row in gate_rows)
    status = "PROMOTED" if passed else "RESEARCH_ONLY"
    gate_rows.append(
        {
            "Metric": "Promotion_Status",
            "Value": status,
            "Threshold": "All nested OOS and frozen holdout gates",
            "Pass": passed,
        }
    )
    oos_summary = pd.DataFrame([{"Evidence_Scope": "NESTED_OOS", **oos_metrics}])
    holdout_summary = pd.DataFrame(
        [
            {
                "Evidence_Scope": "FROZEN_FINAL_HOLDOUT",
                "Frozen_Candidate": nested["frozen_candidate"],
                **holdout_metrics,
            }
        ]
    )
    return pd.DataFrame(gate_rows), oos_summary, holdout_summary, status


def build_strategy_lab_artifact(
    prices: pd.DataFrame,
    *,
    benchmark: str,
    rebalance_days: int = 21,
    top_n: int = 8,
    max_weight: float = 0.20,
    transaction_cost_bps: float = 10.0,
    bootstrap_samples: int = 200,
    seed: int = 20260615,
    train_days: int = 252,
    validation_days: int = 126,
    test_days: int = 63,
    purge_days: int = 21,
    embargo_days: int = 5,
    holdout_days: int = 126,
) -> dict[str, pd.DataFrame | str | int | float]:
    """Build a causal, pre-registered long-only strategy research artifact.

    Signals use data through the signal close. New weights become effective
    on the next observed return, and transaction costs are charged on that
    first execution day. Candidate-family selection remains research-only
    until an external nested walk-forward and frozen holdout promotion gate.
    """

    benchmark = benchmark.upper()
    clean = _clean_prices(prices, benchmark)
    if clean.empty or len(clean) < 504 or len(clean.columns) < 4:
        return {
            "generation": RESEARCH_GENERATION,
            "status": "INSUFFICIENT_DATA",
            "summary": pd.DataFrame(),
            "price_paths": pd.DataFrame(),
            "drawdowns": pd.DataFrame(),
            "oos_summary": pd.DataFrame(),
            "oos_price_paths": pd.DataFrame(),
            "oos_drawdowns": pd.DataFrame(),
            "holdout_summary": pd.DataFrame(),
            "holdout_price_paths": pd.DataFrame(),
            "holdout_drawdowns": pd.DataFrame(),
            "walk_forward_windows": pd.DataFrame(),
            "selection_stability": pd.DataFrame(),
            "frozen_candidate": "",
            "signal_ic": pd.DataFrame(),
            "regime_performance": pd.DataFrame(),
            "weights": pd.DataFrame(),
            "latest_scores": pd.DataFrame(),
            "exposure_diagnostics": pd.DataFrame(),
            "exposure_timeline": pd.DataFrame(),
            "validation": pd.DataFrame(),
            "constitution": pd.DataFrame(),
            "research_lineage": pd.DataFrame(),
            "strategy_registry": strategy_registry_frame(),
            "candidate_equivalence": pd.DataFrame(),
        }

    returns = clean.pct_change(fill_method=None)
    first_signal = max(definition.lookback_days for definition in STRATEGY_DEFINITIONS)
    signal_positions = list(range(first_signal, len(clean) - 1, max(5, rebalance_days)))
    strategy_returns: dict[str, pd.Series] = {}
    weight_rows: list[dict] = []
    ic_rows: list[dict] = []
    latest_score_rows: list[dict] = []
    regime_rows: list[dict] = []
    exposure_rows: list[dict] = []

    for definition in STRATEGY_DEFINITIONS:
        daily = pd.Series(0.0, index=clean.index, dtype=float)
        previous_weights = pd.Series(dtype=float)
        for signal_index, signal_position in enumerate(signal_positions):
            next_signal = (
                signal_positions[signal_index + 1] if signal_index + 1 < len(signal_positions) else len(clean) - 1
            )
            weights, scores = _strategy_weights(
                definition,
                clean,
                returns,
                signal_position,
                benchmark,
                top_n,
                max_weight,
            )
            if weights.empty:
                continue
            exposure, exposure_state = _causal_risk_exposure(
                definition,
                clean[benchmark],
                returns[benchmark],
                signal_position,
            )
            if exposure < 1.0:
                weights = weights * exposure
                weights.loc["CASH"] = 1.0 - exposure
            signal_date = clean.index[signal_position]
            execution_position = signal_position + 1
            execution_date = clean.index[execution_position]
            holding_positions = range(execution_position, min(next_signal + 1, len(clean)))
            turnover = float(
                weights.sub(previous_weights, fill_value=0.0).abs().sum()
                if not previous_weights.empty
                else weights.abs().sum()
            )
            for position in holding_positions:
                asset_return = returns.iloc[position].reindex(weights.index).fillna(0.0)
                daily.iloc[position] = float(np.dot(asset_return.to_numpy(dtype=float), weights.to_numpy(dtype=float)))
            daily.iloc[execution_position] -= transaction_cost_bps / 10_000.0 * turnover

            forward_slice = returns.iloc[execution_position : min(next_signal + 1, len(clean))]
            forward_return = (1.0 + forward_slice.reindex(columns=scores.index).fillna(0.0)).prod() - 1.0
            aligned_ic = pd.concat([scores.rename("score"), forward_return.rename("forward")], axis=1).dropna()
            information_coefficient = (
                float(
                    np.corrcoef(
                        aligned_ic["score"].rank().to_numpy(dtype=float),
                        aligned_ic["forward"].rank().to_numpy(dtype=float),
                    )[0, 1]
                )
                if len(aligned_ic) >= 4
                else np.nan
            )
            regime = _market_regime(clean[benchmark], returns[benchmark], signal_position)
            if definition.strategy_id == "downside_controlled_momentum":
                exposure_rows.append(
                    {
                        "Signal_Date": signal_date,
                        "Execution_Date": execution_date,
                        "Strategy": definition.label,
                        "Strategy_ID": definition.strategy_id,
                        "Exposure": exposure,
                        "Cash_Weight": 1.0 - exposure,
                        "Regime": regime,
                        **exposure_state,
                    }
                )
            ic_rows.append(
                {
                    "Signal_Date": signal_date,
                    "Execution_Date": execution_date,
                    "Strategy": definition.label,
                    "Strategy_ID": definition.strategy_id,
                    "Regime": regime,
                    "Information_Coefficient": information_coefficient,
                    "Names_Scored": int(len(scores)),
                    "Turnover": turnover,
                }
            )
            for ticker, weight in weights.items():
                weight_rows.append(
                    {
                        "Signal_Date": signal_date,
                        "Execution_Date": execution_date,
                        "Strategy": definition.label,
                        "Strategy_ID": definition.strategy_id,
                        "Ticker": ticker,
                        "Weight": float(weight),
                        "Score": float(scores.get(ticker, np.nan)),
                        "Exposure": exposure,
                        "Regime": regime,
                    }
                )
            if signal_position == signal_positions[-1]:
                for ticker, score in scores.head(20).items():
                    latest_score_rows.append(
                        {
                            "As_Of": signal_date,
                            "Strategy": definition.label,
                            "Strategy_ID": definition.strategy_id,
                            "Ticker": ticker,
                            "Score": float(score),
                            "Selected": bool(ticker in weights.index),
                            "Weight": float(weights.get(ticker, 0.0)),
                        }
                    )
            previous_weights = weights

        strategy_returns[definition.label] = daily
        signal_frame = pd.DataFrame([row for row in ic_rows if row["Strategy_ID"] == definition.strategy_id])
        if not signal_frame.empty:
            for regime_name, group in signal_frame.groupby("Regime"):
                dates = pd.to_datetime(group["Execution_Date"])
                regime_return = daily.reindex(dates).dropna()
                regime_rows.append(
                    {
                        "Strategy": definition.label,
                        "Strategy_ID": definition.strategy_id,
                        "Regime": str(regime_name),
                        "Signals": int(len(group)),
                        "Mean_IC": float(pd.to_numeric(group["Information_Coefficient"], errors="coerce").mean()),
                        "Mean_Execution_Day_Return": (
                            float(regime_return.mean()) if not regime_return.empty else np.nan
                        ),
                    }
                )

    candidate_returns = pd.DataFrame(strategy_returns).loc[clean.index[first_signal + 1 :]]
    benchmark_returns = returns[benchmark].reindex(candidate_returns.index).fillna(0.0)
    summary_rows: list[dict] = []
    for definition in STRATEGY_DEFINITIONS:
        metrics = _annualized_metrics(candidate_returns[definition.label], benchmark_returns)
        summary_rows.append(
            {
                "Strategy": definition.label,
                "Strategy_ID": definition.strategy_id,
                "Evidence_Scope": "causal_path_family_selection_validation_only",
                **metrics,
                "Description": definition.hypothesis,
            }
        )
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary["Research_Rank"] = (
            summary["Active_Return"].rank(ascending=False, method="min")
            + summary["Max_Drawdown"].rank(ascending=False, method="min")
            + summary["Downside_Capture"].rank(ascending=True, method="min")
        )
        summary = summary.sort_values(["Research_Rank", "Active_Return"], ascending=[True, False]).reset_index(
            drop=True
        )

    price_paths = (1.0 + candidate_returns).cumprod() * 100.0
    price_paths.insert(0, f"{benchmark} benchmark price index", (1.0 + benchmark_returns).cumprod() * 100.0)
    price_paths = price_paths.reset_index(names="Date")
    drawdowns = price_paths.copy()
    for column in drawdowns.columns:
        if column == "Date":
            continue
        values = pd.to_numeric(drawdowns[column], errors="coerce")
        drawdowns[column] = values / values.cummax() - 1.0

    selection_candidates, candidate_equivalence = _deduplicate_candidate_paths(candidate_returns)
    nested = _nested_walk_forward(
        selection_candidates,
        benchmark_returns,
        train_days=train_days,
        validation_days=validation_days,
        test_days=test_days,
        purge_days=purge_days,
        embargo_days=embargo_days,
        holdout_days=holdout_days,
    )
    validation, oos_summary, holdout_summary, promotion_status = _strict_promotion_evidence(
        nested,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        holdout_independent=False,
    )
    selected_oos_returns = nested["selected_oos_returns"]
    holdout_returns = nested["holdout_returns"]
    if isinstance(selected_oos_returns, pd.DataFrame) and not selected_oos_returns.empty:
        oos_price_paths = (1.0 + selected_oos_returns).cumprod() * 100.0
        oos_drawdowns = oos_price_paths / oos_price_paths.cummax() - 1.0
        oos_price_paths = oos_price_paths.reset_index(names="Date")
        oos_drawdowns = oos_drawdowns.reset_index(names="Date")
    else:
        oos_price_paths = pd.DataFrame()
        oos_drawdowns = pd.DataFrame()
    if isinstance(holdout_returns, pd.DataFrame) and not holdout_returns.empty:
        holdout_price_paths = (1.0 + holdout_returns).cumprod() * 100.0
        holdout_drawdowns = holdout_price_paths / holdout_price_paths.cummax() - 1.0
        holdout_price_paths = holdout_price_paths.reset_index(names="Date")
        holdout_drawdowns = holdout_drawdowns.reset_index(names="Date")
    else:
        holdout_price_paths = pd.DataFrame()
        holdout_drawdowns = pd.DataFrame()
    constitution = pd.DataFrame(
        [
            {
                "Family": "Price-derived long-only strategy laboratory",
                "Benchmark_Xi": benchmark,
                "Candidates_PreRegistered": len(STRATEGY_DEFINITIONS),
                "Effective_Unique_Candidates": int(selection_candidates.shape[1]),
                "Signal_Lag_Days": 1,
                "Rebalance_Days": rebalance_days,
                "Top_N": top_n,
                "Max_Weight": max_weight,
                "Transaction_Cost_bps": transaction_cost_bps,
                "Minimum_History_Days": 504,
                "Train_Days": train_days,
                "Validation_Days": validation_days,
                "Test_Days": test_days,
                "Purge_Days": purge_days,
                "Embargo_Days": embargo_days,
                "Holdout_Days": holdout_days,
                "Research_Generation": RESEARCH_GENERATION,
                "Holdout_Status": HOLDOUT_STATUS,
                "Selection_Status": (
                    "Nested walk-forward; historical holdout is diagnostic-only for this new generation"
                ),
            }
        ]
    )
    research_lineage = pd.DataFrame(
        [
            {
                "Research_Generation": RESEARCH_GENERATION,
                "Parent_Generation": "strategy-lab-g3-governed-strategy-registry",
                "Change": (
                    "Added exact path-equivalence control so duplicate candidate trajectories remain documented "
                    "but count only once in nested selection and multiple-testing inference"
                ),
                "Holdout_Status": HOLDOUT_STATUS,
                "Promotion_Eligible": False,
                "Prospective_Evidence_Start": clean.index[-1],
                "Reason": (
                    "The historical holdout informed prior generations. Candidate-set expansion creates a new "
                    "generation that requires a new untouched prospective sample."
                ),
            }
        ]
    )
    exposure_diagnostics = pd.DataFrame(exposure_rows)
    exposure_timeline = pd.DataFrame()
    if not exposure_diagnostics.empty:
        exposure_timeline = exposure_diagnostics[["Execution_Date", "Exposure", "Cash_Weight"]].rename(
            columns={"Execution_Date": "Date"}
        )
    return {
        "generation": RESEARCH_GENERATION,
        "status": promotion_status,
        "summary": summary,
        "price_paths": price_paths,
        "drawdowns": drawdowns,
        "oos_summary": oos_summary,
        "oos_price_paths": oos_price_paths,
        "oos_drawdowns": oos_drawdowns,
        "holdout_summary": holdout_summary,
        "holdout_price_paths": holdout_price_paths,
        "holdout_drawdowns": holdout_drawdowns,
        "walk_forward_windows": nested["windows"],
        "selection_stability": nested["selection_stability"],
        "frozen_candidate": nested["frozen_candidate"],
        "signal_ic": pd.DataFrame(ic_rows),
        "regime_performance": pd.DataFrame(regime_rows),
        "weights": pd.DataFrame(weight_rows),
        "latest_scores": pd.DataFrame(latest_score_rows),
        "exposure_diagnostics": exposure_diagnostics,
        "exposure_timeline": exposure_timeline,
        "validation": validation,
        "constitution": constitution,
        "research_lineage": research_lineage,
        "strategy_registry": strategy_registry_frame(),
        "candidate_equivalence": candidate_equivalence,
        "benchmark_xi": benchmark,
        "observation_days": int(len(candidate_returns)),
    }
