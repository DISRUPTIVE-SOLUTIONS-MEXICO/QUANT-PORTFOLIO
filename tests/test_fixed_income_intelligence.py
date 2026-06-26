from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from quant_core.contracts import EvidenceScope, FixedIncomeIntelligenceV1
from quant_core.dashboard_payload import build_dashboard_payload
from quant_core.fixed_income_intelligence import build_fixed_income_intelligence


def _rates(days: int = 420) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2024-01-02", periods=days, tz="UTC")
    two_year = np.linspace(4.8, 3.9, days) + 0.08 * np.sin(np.arange(days) / 19.0)
    ten_year = np.linspace(4.3, 4.4, days) + 0.10 * np.sin(np.arange(days) / 23.0)
    policy_dates = dates[::21]
    policy = np.linspace(5.5, 4.0, len(policy_dates))
    history = pd.concat(
        [
            pd.DataFrame(
                {
                    "Country": "United States",
                    "Observation_Date": dates,
                    "Tenor_Code": "SOV_2Y",
                    "Tenor": "2Y sovereign",
                    "Rate": two_year,
                    "Observation_Frequency": "Daily/business-day discrete",
                    "Source": "Synthetic official-source fixture",
                }
            ),
            pd.DataFrame(
                {
                    "Country": "United States",
                    "Observation_Date": dates,
                    "Tenor_Code": "SOV_10Y",
                    "Tenor": "10Y sovereign",
                    "Rate": ten_year,
                    "Observation_Frequency": "Daily/business-day discrete",
                    "Source": "Synthetic official-source fixture",
                }
            ),
            pd.DataFrame(
                {
                    "Country": "United States",
                    "Observation_Date": policy_dates,
                    "Tenor_Code": "POLICY_RATE",
                    "Tenor": "Policy",
                    "Rate": policy,
                    "Observation_Frequency": "Meeting-date discrete",
                    "Source": "Synthetic official-source fixture",
                }
            ),
        ],
        ignore_index=True,
    )
    snapshot = pd.DataFrame(
        [
            {
                "Country": "United States",
                "Policy_Rate": policy[-1],
                "Yield_2Y": two_year[-1],
                "Yield_10Y": ten_year[-1],
                "Rate_Source": "Synthetic official-source fixture",
                "Latest_Date": dates[-1],
                "Policy_Observation_Date": policy_dates[-1],
            }
        ]
    )
    references = pd.DataFrame(
        {
            "Observation_Date": dates,
            "Code": "SOFR",
            "Benchmark": "SOFR",
            "Jurisdiction": "United States",
            "Currency": "USD",
            "Tenor": "Overnight",
            "Rate": np.linspace(5.3, 3.8, days),
            "Observation_Frequency": "Daily/business-day discrete",
            "Status": "Active overnight risk-free/reference rate",
            "Source": "Synthetic official-source fixture",
        }
    )
    return snapshot, history, references


def test_fixed_income_intelligence_builds_curve_factors_and_stress_scenarios() -> None:
    snapshot, history, references = _rates()
    artifact = build_fixed_income_intelligence(snapshot, history, reference_rates=references)
    metrics = artifact["country_metrics"].set_index("Country")
    assert artifact["contract"]["evidence_scope"] == "live_snapshot"
    assert artifact["contract"]["minimum_real_tenors"] == 2
    assert metrics.loc["United States", "Curve_Quality"] == "High"
    assert metrics.loc["United States", "Sovereign_Tenor_Count"] == 2
    assert len(artifact["factor_history"]) >= 400
    assert len(artifact["stress_scenarios"]) == 6
    parallel_up = artifact["stress_scenarios"].set_index("Scenario").loc["Parallel +100 bp"]
    assert parallel_up["Approx_Price_Impact_2Y_pct"] < 0.0
    assert parallel_up["Approx_Price_Impact_10Y_pct"] < parallel_up["Approx_Price_Impact_2Y_pct"]
    assert artifact["reference_rate_summary"].iloc[0]["Code"] == "SOFR"


def test_fixed_income_intelligence_future_contamination_cannot_change_prior_state() -> None:
    snapshot, history, references = _rates()
    cutoff = pd.Timestamp(history["Observation_Date"].sort_values().iloc[-80])
    baseline = build_fixed_income_intelligence(snapshot, history, reference_rates=references, as_of=cutoff)
    contaminated = history.copy()
    future = contaminated["Observation_Date"] > cutoff
    contaminated.loc[future, "Rate"] = contaminated.loc[future, "Rate"] + 20.0
    altered = build_fixed_income_intelligence(snapshot, contaminated, reference_rates=references, as_of=cutoff)
    pd.testing.assert_frame_equal(
        baseline["country_metrics"].reset_index(drop=True),
        altered["country_metrics"].reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        baseline["factor_history"].reset_index(drop=True),
        altered["factor_history"].reset_index(drop=True),
    )


def test_fixed_income_intelligence_does_not_create_curve_from_policy_only() -> None:
    snapshot = pd.DataFrame(
        [{"Country": "Brazil", "Policy_Rate": 10.5, "Latest_Date": "2026-06-12", "Rate_Source": "BCB"}]
    )
    history = pd.DataFrame(
        {
            "Country": "Brazil",
            "Observation_Date": pd.date_range("2025-01-01", periods=80, freq="MS", tz="UTC"),
            "Tenor_Code": "POLICY_RATE",
            "Rate": 10.5,
        }
    )
    artifact = build_fixed_income_intelligence(snapshot, history)
    metric = artifact["country_metrics"].iloc[0]
    assert metric["Sovereign_Tenor_Count"] == 0
    assert metric["Curve_Quality"] == "Insufficient for curve analytics"
    assert artifact["stress_scenarios"].empty


def test_fixed_income_contract_requires_native_calendar_and_formulas() -> None:
    with pytest.raises(ValidationError, match="native observation calendars"):
        FixedIncomeIntelligenceV1(
            as_of=datetime(2026, 6, 15, tzinfo=UTC),
            evidence_scope=EvidenceScope.LIVE_SNAPSHOT,
            countries=("United States",),
            factor_observation_mode="daily_interpolated",
            formulas={
                "level": "mean",
                "slope": "10Y-2Y",
                "curvature_proxy": "2*2Y-short-10Y",
                "duration_convexity": "duration plus convexity",
                "quality": "coverage",
            },
        )


def test_dashboard_payload_exposes_fixed_income_workbench() -> None:
    snapshot, history, references = _rates()
    results = {
        "prices": pd.DataFrame(),
        "portfolio": pd.DataFrame(),
        "benchmark_ticker": "SPY",
        "global_yield_curves": snapshot,
        "global_rate_history": history,
        "interbank_reference_rates": references,
    }
    payload = build_dashboard_payload(
        results,
        {"path_metadata": {"benchmark": "SPY"}},
        {"summary": pd.DataFrame(), "breaches": pd.DataFrame()},
        {"summary": pd.DataFrame(), "tests": pd.DataFrame()},
    )
    fixed_income = payload["fixed_income_intelligence"]
    assert len(fixed_income["country_metrics"]) == 1
    assert len(fixed_income["factor_history"]) >= 400
    assert len(fixed_income["stress_scenarios"]) == 6
    assert fixed_income["contract"]["method"] == "causal_fixed_income_intelligence_v1"
    assert payload["contract"]["schema_version"] == "2026.06.19-publication-isolation-v11"
