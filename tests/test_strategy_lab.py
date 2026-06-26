from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from quant_core.contracts import EvidenceScope, StrategyResearchV1
from quant_core.dashboard_payload import build_dashboard_payload
from quant_core.strategy_lab import (
    STRATEGY_DEFINITIONS,
    _deduplicate_candidate_paths,
    build_strategy_lab_artifact,
)
from quant_core.strategy_registry import STRATEGY_SPECIFICATIONS


def _synthetic_prices(days: int = 900) -> pd.DataFrame:
    rng = np.random.default_rng(20260615)
    dates = pd.bdate_range("2022-01-03", periods=days)
    market = rng.normal(0.00035, 0.010, days)
    cycle = 0.0015 * np.sin(np.arange(days) / 31.0)
    returns = {
        "SPY": market,
        "AAA": 0.00045 + 1.05 * market + cycle + rng.normal(0.0, 0.005, days),
        "BBB": 0.00025 + 0.70 * market - cycle + rng.normal(0.0, 0.004, days),
        "CCC": 0.00055 + 1.20 * market + rng.normal(0.0, 0.007, days),
        "DDD": 0.00020 + 0.45 * market + rng.normal(0.0, 0.003, days),
        "EEE": 0.00035 - 0.15 * market + rng.normal(0.0, 0.006, days),
        "FFF": 0.00030 + 0.85 * market + rng.normal(0.0, 0.005, days),
    }
    return pd.DataFrame({ticker: 100.0 * np.cumprod(1.0 + values) for ticker, values in returns.items()}, index=dates)


def test_strategy_lab_builds_causal_research_artifact() -> None:
    artifact = build_strategy_lab_artifact(
        _synthetic_prices(),
        benchmark="SPY",
        bootstrap_samples=50,
    )

    assert artifact["status"] == "RESEARCH_ONLY"
    summary = artifact["summary"]
    validation = artifact["validation"]
    weights = artifact["weights"]
    assert isinstance(summary, pd.DataFrame)
    assert len(summary) == len(STRATEGY_DEFINITIONS)
    assert set(summary["Strategy_ID"]) == {definition.strategy_id for definition in STRATEGY_DEFINITIONS}
    assert {"Annualized_Return", "Max_Drawdown", "Upside_Capture", "Downside_Capture"}.issubset(summary.columns)
    assert isinstance(validation, pd.DataFrame)
    assert validation.loc[validation["Metric"] == "Promotion_Status", "Value"].iloc[0] == "RESEARCH_ONLY"
    assert isinstance(weights, pd.DataFrame)
    assert (pd.to_datetime(weights["Execution_Date"]) > pd.to_datetime(weights["Signal_Date"])).all()
    weight_sums = weights.groupby(["Strategy_ID", "Signal_Date"])["Weight"].sum()
    assert np.allclose(weight_sums.to_numpy(), 1.0)
    equivalence = artifact["candidate_equivalence"]
    assert isinstance(equivalence, pd.DataFrame)
    assert len(equivalence) == len(STRATEGY_DEFINITIONS)
    assert int(equivalence["Included_In_Selection"].sum()) <= len(STRATEGY_DEFINITIONS)


def test_only_implemented_registry_entries_can_enter_candidate_selection() -> None:
    engine_ids = {definition.strategy_id for definition in STRATEGY_DEFINITIONS}
    implemented_ids = {
        specification.strategy_id
        for specification in STRATEGY_SPECIFICATIONS
        if specification.engine_candidate and specification.implementation_status == "implemented"
    }
    blocked_ids = {
        specification.strategy_id
        for specification in STRATEGY_SPECIFICATIONS
        if specification.implementation_status.startswith("blocked")
    }
    assert engine_ids == implemented_ids
    assert engine_ids.isdisjoint(blocked_ids)
    assert {"volatility_adjusted_trend", "dual_momentum"}.issubset(engine_ids)


def test_exactly_equivalent_candidate_paths_count_as_one_trial() -> None:
    candidate_returns = pd.DataFrame(
        {
            "A": [0.01, -0.02, 0.03],
            "B": [0.01, -0.02, 0.03],
            "C": [0.01, -0.01, 0.03],
        }
    )
    unique, diagnostics = _deduplicate_candidate_paths(candidate_returns)
    assert list(unique.columns) == ["A", "C"]
    duplicate = diagnostics.loc[diagnostics["Strategy"] == "B"].iloc[0]
    assert duplicate["Canonical_Strategy"] == "A"
    assert bool(duplicate["Equivalent_Path"]) is True
    assert bool(duplicate["Included_In_Selection"]) is False


def test_future_contamination_does_not_change_prior_weights() -> None:
    prices = _synthetic_prices()
    cutoff = prices.index[620]
    baseline = build_strategy_lab_artifact(prices, benchmark="SPY", bootstrap_samples=50)

    contaminated = prices.copy()
    future_mask = contaminated.index > cutoff
    shocks = np.linspace(1.0, 5.0, int(future_mask.sum()))
    contaminated.loc[future_mask, "AAA"] *= shocks
    contaminated.loc[future_mask, "EEE"] *= shocks[::-1]
    altered = build_strategy_lab_artifact(contaminated, benchmark="SPY", bootstrap_samples=50)

    baseline_weights = baseline["weights"]
    altered_weights = altered["weights"]
    assert isinstance(baseline_weights, pd.DataFrame)
    assert isinstance(altered_weights, pd.DataFrame)
    keys = ["Strategy_ID", "Signal_Date", "Execution_Date", "Ticker"]
    before_baseline = baseline_weights.loc[pd.to_datetime(baseline_weights["Signal_Date"]) <= cutoff, keys + ["Weight"]]
    before_altered = altered_weights.loc[pd.to_datetime(altered_weights["Signal_Date"]) <= cutoff, keys + ["Weight"]]
    pd.testing.assert_frame_equal(
        before_baseline.sort_values(keys).reset_index(drop=True),
        before_altered.sort_values(keys).reset_index(drop=True),
        check_exact=False,
        atol=1e-12,
        rtol=1e-12,
    )


def test_strategy_contract_rejects_non_oos_promotion() -> None:
    with np.testing.assert_raises(ValueError):
        StrategyResearchV1(
            strategy_family="pre_registered_price_signals",
            candidate_ids=("momentum", "reversion"),
            benchmark_xi="SPY",
            evidence_scope=EvidenceScope.VALIDATION,
            observation_start=datetime(2022, 1, 1, tzinfo=UTC),
            observation_end=datetime(2025, 1, 1, tzinfo=UTC),
            signal_lag_days=1,
            rebalance_days=21,
            transaction_cost_bps=10.0,
            selected_candidate="momentum",
            promotion_status="promoted",
        )


def test_strategy_contract_requires_every_strict_promotion_gate() -> None:
    base = {
        "strategy_family": "pre_registered_price_signals",
        "candidate_ids": ("momentum", "reversion"),
        "benchmark_xi": "SPY",
        "evidence_scope": EvidenceScope.HOLDOUT,
        "observation_start": datetime(2022, 1, 1, tzinfo=UTC),
        "observation_end": datetime(2025, 1, 1, tzinfo=UTC),
        "signal_lag_days": 1,
        "rebalance_days": 21,
        "transaction_cost_bps": 10.0,
        "selected_candidate": "momentum",
        "promotion_status": "promoted",
    }
    with np.testing.assert_raises(ValueError):
        StrategyResearchV1(**base, validation_metrics={"WRC_p": 0.01})

    contract = StrategyResearchV1(
        **base,
        validation_metrics={
            "WRC_p": 0.01,
            "SPA_p": 0.02,
            "PBO": 0.05,
            "OOS_Active_Return": 0.03,
            "OOS_Upside_Capture": 1.05,
            "OOS_Downside_Capture": 0.92,
            "OOS_Downside_Preservation": True,
            "Holdout_Active_Return": 0.01,
            "Holdout_Downside_Preservation": True,
            "Holdout_Independence": True,
        },
    )
    assert contract.promotion_status == "promoted"


def test_daily_dashboard_does_not_synthesize_strategy_research() -> None:
    prices = _synthetic_prices()
    payload = build_dashboard_payload(
        {"prices": prices, "portfolio": pd.DataFrame()},
        {"path_metadata": {"benchmark": "SPY"}},
        {},
        {},
    )
    lab = payload["strategy_lab"]
    assert lab["status"] == "UNAVAILABLE_IN_DAILY_OVERLAY"
    assert isinstance(lab["summary"], pd.DataFrame)
    assert lab["summary"].empty
    assert lab["observation_days"] == 0
    assert lab["generation"] == "daily-market-overlay"


def test_nested_walk_forward_respects_purge_embargo_and_frozen_holdout() -> None:
    artifact = build_strategy_lab_artifact(
        _synthetic_prices(1100),
        benchmark="SPY",
        bootstrap_samples=50,
    )
    windows = artifact["walk_forward_windows"]
    holdout = artifact["holdout_price_paths"]
    assert isinstance(windows, pd.DataFrame)
    assert len(windows) >= 4
    assert isinstance(holdout, pd.DataFrame)
    assert not holdout.empty
    assert artifact["frozen_candidate"]
    for row in windows.itertuples(index=False):
        assert pd.Timestamp(row.Train_End) < pd.Timestamp(row.Validation_Start)
        assert pd.Timestamp(row.Validation_End) < pd.Timestamp(row.Test_Start)
        assert row.Purge_Days == 21
        assert row.Embargo_Days == 5
    assert pd.Timestamp(windows["Test_End"].max()) < pd.Timestamp(holdout["Date"].min())


def test_holdout_contamination_cannot_change_pre_holdout_selection() -> None:
    prices = _synthetic_prices(1100)
    baseline = build_strategy_lab_artifact(prices, benchmark="SPY", bootstrap_samples=50)
    holdout_paths = baseline["holdout_price_paths"]
    assert isinstance(holdout_paths, pd.DataFrame)
    holdout_start = pd.Timestamp(holdout_paths["Date"].min())

    contaminated = prices.copy()
    mask = contaminated.index >= holdout_start
    contaminated.loc[mask, "AAA"] *= np.linspace(1.0, 8.0, int(mask.sum()))
    contaminated.loc[mask, "EEE"] *= np.linspace(8.0, 0.5, int(mask.sum()))
    altered = build_strategy_lab_artifact(contaminated, benchmark="SPY", bootstrap_samples=50)

    assert baseline["frozen_candidate"] == altered["frozen_candidate"]
    pd.testing.assert_frame_equal(
        baseline["walk_forward_windows"].reset_index(drop=True),
        altered["walk_forward_windows"].reset_index(drop=True),
        check_exact=False,
        atol=1e-12,
        rtol=1e-12,
    )
    pd.testing.assert_frame_equal(
        baseline["oos_price_paths"].reset_index(drop=True),
        altered["oos_price_paths"].reset_index(drop=True),
        check_exact=False,
        atol=1e-12,
        rtol=1e-12,
    )


def test_downside_governor_is_causal_bounded_and_cash_balanced() -> None:
    prices = _synthetic_prices(1100)
    cutoff = prices.index[740]
    baseline = build_strategy_lab_artifact(prices, benchmark="SPY", bootstrap_samples=50)

    contaminated = prices.copy()
    future_mask = contaminated.index > cutoff
    contaminated.loc[future_mask, "SPY"] *= np.linspace(1.0, 4.0, int(future_mask.sum()))
    altered = build_strategy_lab_artifact(contaminated, benchmark="SPY", bootstrap_samples=50)

    baseline_exposure = baseline["exposure_diagnostics"]
    altered_exposure = altered["exposure_diagnostics"]
    assert isinstance(baseline_exposure, pd.DataFrame)
    assert not baseline_exposure.empty
    assert baseline_exposure["Exposure"].between(0.25, 1.0).all()
    assert np.allclose(
        baseline_exposure["Exposure"] + baseline_exposure["Cash_Weight"],
        1.0,
    )
    before_baseline = baseline_exposure.loc[pd.to_datetime(baseline_exposure["Signal_Date"]) <= cutoff].reset_index(
        drop=True
    )
    before_altered = altered_exposure.loc[pd.to_datetime(altered_exposure["Signal_Date"]) <= cutoff].reset_index(
        drop=True
    )
    pd.testing.assert_frame_equal(
        before_baseline,
        before_altered,
        check_exact=False,
        atol=1e-12,
        rtol=1e-12,
    )


def test_consumed_holdout_generation_is_never_promoted() -> None:
    artifact = build_strategy_lab_artifact(
        _synthetic_prices(1100),
        benchmark="SPY",
        bootstrap_samples=50,
    )
    validation = artifact["validation"]
    lineage = artifact["research_lineage"]
    assert isinstance(validation, pd.DataFrame)
    independence = validation.loc[validation["Metric"] == "Holdout_Independence"].iloc[0]
    assert independence["Pass"] is False or bool(independence["Pass"]) is False
    assert artifact["status"] == "RESEARCH_ONLY"
    assert isinstance(lineage, pd.DataFrame)
    assert lineage.loc[0, "Holdout_Status"] == "CONSUMED_FOR_DIAGNOSIS"
    assert bool(lineage.loc[0, "Promotion_Eligible"]) is False
