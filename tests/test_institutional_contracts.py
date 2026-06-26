from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pandas as pd
import pytest
from pydantic import ValidationError

from quant_core.capability_manifest import CAPABILITIES, validate_manifest
from quant_core.contracts import (
    EvidenceScope,
    PortfolioPositionV2,
    PortfolioRunV2,
    PublicationState,
    ResearchEvidenceV2,
    SecurityIntelligenceV1,
    StrategySpecificationV1,
)
from quant_core.execution import PreTradePolicy, build_order_intent, evaluate_pretrade
from quant_core.strategy_registry import STRATEGY_SPECIFICATIONS, validate_strategy_registry
from supabase_store import build_publication_manifest, validate_publication_bundle


def _portfolio(
    *,
    promotion_status: str = "promoted",
    suitability_status: str = "approved",
    evidence_scope: EvidenceScope = EvidenceScope.OUT_OF_SAMPLE,
) -> PortfolioRunV2:
    return PortfolioRunV2(
        as_of=datetime(2026, 6, 12, tzinfo=UTC),
        portfolio_name="Institutional test",
        benchmark_xi="SPY",
        stress_set_omega=("QQQ", "ACWI"),
        evidence_scope=evidence_scope,
        positions=(
            PortfolioPositionV2(
                ticker="MSFT",
                sector="Technology",
                target_weight=0.5,
                reference_price=500.0,
                adv_usd=8_000_000_000.0,
            ),
            PortfolioPositionV2(
                ticker="XOM",
                sector="Energy",
                target_weight=0.5,
                reference_price=120.0,
                adv_usd=2_000_000_000.0,
            ),
        ),
        config_hash="c" * 64,
        data_hash="d" * 64,
        model_version="xcdr-v3",
        suitability_status=suitability_status,
        promotion_status=promotion_status,
    )


def test_feature_preservation_manifest_is_unique_and_complete():
    valid, errors = validate_manifest()
    assert valid, errors
    assert len(CAPABILITIES) >= 20
    assert {"portfolio.xcdr", "strategy.research_lab", "rates.sovereign_curves", "execution.paper"}.issubset(
        {capability.capability_id for capability in CAPABILITIES}
    )


def test_strategy_registry_is_unique_and_contract_valid():
    valid, errors = validate_strategy_registry()
    assert valid, errors
    assert len(STRATEGY_SPECIFICATIONS) >= 12
    assert any(item.engine_candidate for item in STRATEGY_SPECIFICATIONS)
    assert any(item.implementation_status == "blocked_data" for item in STRATEGY_SPECIFICATIONS)
    for item in STRATEGY_SPECIFICATIONS:
        StrategySpecificationV1(**item.__dict__)


def test_security_intelligence_contract_is_live_snapshot_and_formula_complete():
    contract = SecurityIntelligenceV1(
        as_of=datetime(2026, 6, 15, tzinfo=UTC),
        benchmark_xi="SPY",
        tickers=("SPY", "MSFT"),
        minimum_observations=252,
        price_history_days=756,
        formulas={
            "beta": "Cov(r_i,r_xi)/Var(r_xi)",
            "tail_beta": "conditional beta below q10",
            "residual_momentum": "compound market-model residuals",
            "drawdown": "P_t/running_max(P)-1",
        },
    )
    assert contract.evidence_scope == EvidenceScope.LIVE_SNAPSHOT
    assert contract.benchmark_xi == "SPY"


def test_portfolio_contract_rejects_weights_that_do_not_sum_to_one():
    with pytest.raises(ValidationError, match="target weights must sum"):
        PortfolioRunV2(
            as_of=datetime(2026, 6, 12, tzinfo=UTC),
            portfolio_name="Invalid",
            benchmark_xi="SPY",
            evidence_scope=EvidenceScope.OUT_OF_SAMPLE,
            positions=(PortfolioPositionV2(ticker="MSFT", target_weight=0.4),),
            config_hash="c" * 64,
            data_hash="d" * 64,
            model_version="xcdr-v3",
            suitability_status="approved",
            promotion_status="promoted",
        )


def test_research_evidence_cannot_promote_in_sample_results():
    with pytest.raises(ValidationError, match="Only OOS or holdout evidence"):
        ResearchEvidenceV2(
            strategy_id="xcdr-v3",
            strategy_version="3",
            benchmark_xi="SPY",
            stress_set_omega=("QQQ",),
            evidence_scope=EvidenceScope.IN_SAMPLE,
            observation_start=datetime(2023, 1, 1, tzinfo=UTC),
            observation_end=datetime(2026, 1, 1, tzinfo=UTC),
            oos_windows=0,
            promoted=True,
        )


def test_pretrade_rejects_research_only_and_mnpi_workflows():
    portfolio = _portfolio(promotion_status="research_only")
    order = build_order_intent(
        portfolio,
        user_id=uuid4(),
        portfolio_value=1_000_000.0,
        current_weights={"MSFT": 0.4, "XOM": 0.6},
    )
    decision = evaluate_pretrade(
        order,
        portfolio,
        data_age_hours=1.0,
        mnpi_flag=True,
        policy=PreTradePolicy(max_single_name_weight=0.6),
    )
    assert not decision.approved
    assert "promotion" in decision.hard_breaches
    assert "mnpi_firewall" in decision.hard_breaches


def test_pretrade_approves_clean_oos_paper_intent():
    portfolio = _portfolio()
    order = build_order_intent(
        portfolio,
        user_id=uuid4(),
        portfolio_value=1_000_000.0,
        current_weights={"MSFT": 0.45, "XOM": 0.55},
    )
    decision = evaluate_pretrade(
        order,
        portfolio,
        data_age_hours=2.0,
        policy=PreTradePolicy(max_single_name_weight=0.6),
    )
    assert decision.approved
    assert not decision.hard_breaches


def test_publication_bundle_is_rejected_when_required_artifacts_are_missing():
    artifacts = {
        "dashboard_payload": {"status": "ready"},
        "promotion_gate": {"promotion_status": "promoted"},
    }
    manifest = build_publication_manifest(str(uuid4()), artifacts, channel="global")
    valid, checks, rejections = validate_publication_bundle(manifest, artifacts)
    assert not valid
    assert not checks["suitability_gate_present"]
    assert "suitability_gate_present" in rejections
    assert manifest.state == PublicationState.STAGING
    assert manifest.publication_kind == "daily_snapshot"


def _daily_v9_publication_artifacts(*, include_security: bool) -> dict:
    dashboard = {
        "contract": {
            "schema_version": "2026.06.15-market-intelligence-v9",
            "analytics_scope": "daily_snapshot",
        },
        "market_intelligence": {"market_regime": [{"Regime": "bullish"}]},
    }
    if include_security:
        dashboard["security_intelligence"] = {
            "contract": {
                "schema_version": "2026.06.15-institutional-v4",
                "evidence_scope": "live_snapshot",
                "benchmark_xi": "SPY",
                "method": "causal_security_intelligence_v1",
                "tickers": ["SPY", "AAPL"],
                "minimum_observations": 252,
                "price_history_days": 252,
                "formulas": {
                    "beta": "Cov(r_i,r_xi)/Var(r_xi)",
                    "tail_beta": "conditional beta",
                    "residual_momentum": "compound OLS residuals",
                    "drawdown": "P_t/running_max(P)-1",
                },
            },
            "benchmark_xi": "SPY",
            "metrics": [{"Ticker": "SPY"}, {"Ticker": "AAPL"}],
            "price_history": [
                {"Date": value.isoformat(), "SPY": 100.0, "AAPL": 100.0}
                for value in pd.date_range("2025-01-02", periods=252, freq="B")
            ],
            "methodology": [{"Signal_Timing": "all inputs timestamp <= decision date"}],
        }
    return {
        "dashboard_payload": dashboard,
        "backtest_path_bundle": {},
        "suitability_gate": {},
        "promotion_gate": {},
        "data_freshness_report": {},
    }


def test_daily_v9_publication_rejects_missing_security_intelligence_contract():
    artifacts = _daily_v9_publication_artifacts(include_security=False)
    manifest = build_publication_manifest(str(uuid4()), artifacts)
    valid, checks, rejections = validate_publication_bundle(manifest, artifacts)
    assert valid is False
    assert checks["security_intelligence_present"] is False
    assert "security_intelligence_present" in rejections


def test_daily_v9_publication_accepts_complete_security_intelligence_contract():
    artifacts = _daily_v9_publication_artifacts(include_security=True)
    manifest = build_publication_manifest(str(uuid4()), artifacts)
    valid, checks, rejections = validate_publication_bundle(manifest, artifacts)
    assert valid is True
    assert checks["security_intelligence_present"] is True
    assert rejections == ()


def _daily_v10_publication_artifacts(*, include_fixed_income: bool) -> dict:
    artifacts = _daily_v9_publication_artifacts(include_security=True)
    artifacts["dashboard_payload"]["contract"]["schema_version"] = "2026.06.15-market-intelligence-v10"
    if include_fixed_income:
        dates = pd.date_range("2025-01-02", periods=126, freq="B", tz="UTC")
        artifacts["dashboard_payload"]["fixed_income_intelligence"] = {
            "contract": {
                "schema_version": "2026.06.15-institutional-v5",
                "evidence_scope": "live_snapshot",
                "method": "causal_fixed_income_intelligence_v1",
                "countries": ["United States", "Canada"],
                "minimum_real_tenors": 2,
                "factor_observation_mode": "native_calendar_event_time_last_observation_carried",
                "formulas": {
                    "level": "mean 2Y and 10Y",
                    "slope": "10Y-2Y",
                    "curvature_proxy": "2*2Y-short-10Y",
                    "duration_convexity": "local price approximation",
                    "quality": "coverage score",
                },
            },
            "country_metrics": [
                {"Country": "United States", "Sovereign_Tenor_Count": 2},
                {"Country": "Canada", "Sovereign_Tenor_Count": 2},
            ],
            "factor_history": [
                {
                    "Date": value.isoformat(),
                    "Country": "United States",
                    "Level_Factor": 4.0,
                    "Slope_10Y_2Y": 0.25,
                }
                for value in dates
            ],
            "stress_scenarios": [{"Country": "United States", "Scenario": f"Scenario {index}"} for index in range(6)],
            "methodology": [{"Curve_Interpolation": "None"}],
        }
    return artifacts


def test_daily_v10_publication_rejects_missing_fixed_income_contract():
    artifacts = _daily_v10_publication_artifacts(include_fixed_income=False)
    manifest = build_publication_manifest(str(uuid4()), artifacts)
    valid, checks, rejections = validate_publication_bundle(manifest, artifacts)
    assert valid is False
    assert checks["fixed_income_intelligence_present"] is False
    assert "fixed_income_intelligence_present" in rejections


def test_daily_v10_publication_accepts_complete_fixed_income_contract():
    artifacts = _daily_v10_publication_artifacts(include_fixed_income=True)
    manifest = build_publication_manifest(str(uuid4()), artifacts)
    valid, checks, rejections = validate_publication_bundle(manifest, artifacts)
    assert valid is True
    assert checks["fixed_income_intelligence_present"] is True
    assert rejections == ()


def _full_publication_artifacts(dates: pd.DatetimeIndex) -> dict:
    capability_matrix = [
        {
            "Module": module,
            "Completeness": 1.0,
            "Status": "complete",
            "Freshness_Requirement": freshness,
            "Missing_Evidence": "",
        }
        for module, freshness in [
            ("Market Intelligence", "daily"),
            ("Rates & Fixed Income", "daily"),
            ("Equity Fundamentals", "per full research run"),
            ("Benchmark xi", "per mandate/run"),
            ("XCDR Research", "per full research run"),
            ("Portfolio Construction", "per optimization"),
            ("Risk Laboratory", "per full research run"),
            ("Validation & Governance", "per full research run"),
            ("Data Quality", "every publication"),
        ]
    ]
    price_paths = [
        {
            "Date": value.isoformat(),
            "SPY observed price": 100.0 + index * 0.01,
            "XCDR research portfolio price": 100.0 + index * 0.02,
        }
        for index, value in enumerate(dates)
    ]
    drawdowns = [
        {
            "Date": value.isoformat(),
            "SPY observed price": -0.01,
            "XCDR research portfolio price": -0.005,
        }
        for value in dates
    ]
    return {
        "dashboard_payload": {
            "contract": {"analytics_scope": "full_analysis"},
            "status": {
                "snapshot_meta": [{"As_Of": dates.max().isoformat()}],
                "capability_completeness": capability_matrix,
            },
            "charts": {"price_paths": price_paths, "drawdowns": drawdowns},
            "allocation": {
                "recommended_portfolio": [
                    {"Ticker": "AAPL", "Sector": "Technology", "Weight": 0.4},
                    {"Ticker": "MSFT", "Sector": "Technology", "Weight": 0.6},
                ]
            },
            "tables": {
                "fundamentals": [
                    {"Ticker": "AAPL", "Sector": "Technology", "Weight": 0.4, "PIT_Confidence": 0.8},
                    {"Ticker": "MSFT", "Sector": "Technology", "Weight": 0.6, "PIT_Confidence": 0.8},
                ],
                "risk": [
                    {"Metric": "Annualized_Return", "Value": 0.1},
                    {"Metric": "Annualized_Vol", "Value": 0.15},
                    {"Metric": "CVaR_95", "Value": -0.02},
                    {"Metric": "Max_Drawdown", "Value": -0.12},
                ],
                "validation": [{"Metric": "WRC_p", "Value": 0.2, "Pass": False}],
            },
            "market_intelligence": {
                "macro_history": [{"Date": dates.max().isoformat(), "Market_Regime": "Bull"}],
                "global_yield_curves": [{"Country": "United States", "Yield_2Y": 4.0, "Yield_10Y": 4.3}],
                "global_rate_history": [{"Date": dates.max().isoformat(), "United States": 4.3}],
                "sentiment_timeline": [{"Date": dates.max().isoformat(), "Latent_Sentiment": 0.1}],
                "geopolitical_summary": [{"Topic": "Trade / Tariffs", "Robust_Z_Score": 0.0}],
            },
            "strategy_lab": {
                "summary": [
                    {"Strategy_ID": "momentum"},
                    {"Strategy_ID": "residual_momentum"},
                    {"Strategy_ID": "asymmetric_capture"},
                ],
                "constitution": [{"Signal_Lag_Days": 1, "Promotion_Status": "research_only"}],
                "walk_forward_windows": [
                    {
                        "Train_End": "2023-01-31",
                        "Validation_Start": "2023-02-22",
                        "Validation_End": "2023-08-18",
                        "Test_Start": "2023-08-28",
                    }
                ],
                "holdout_summary": [{"Evidence_Scope": "FROZEN_FINAL_HOLDOUT"}],
                "validation": [
                    {"Metric": "Holdout_Independence", "Value": False, "Pass": False},
                    {"Metric": "Promotion_Status", "Value": "RESEARCH_ONLY", "Pass": False},
                ],
                "research_lineage": [
                    {
                        "Research_Generation": "strategy-lab-g2-downside-throttle",
                        "Holdout_Status": "CONSUMED_FOR_DIAGNOSIS",
                        "Promotion_Eligible": False,
                    }
                ],
                "strategy_registry": [
                    {"Strategy_ID": "momentum", "Implementation_Status": "implemented"},
                    {"Strategy_ID": "residual_momentum", "Implementation_Status": "implemented"},
                    {"Strategy_ID": "asymmetric_capture", "Implementation_Status": "implemented"},
                ],
                "candidate_equivalence": [
                    {"Strategy": "momentum", "Canonical_Strategy": "momentum"},
                    {"Strategy": "residual_momentum", "Canonical_Strategy": "residual_momentum"},
                    {"Strategy": "asymmetric_capture", "Canonical_Strategy": "asymmetric_capture"},
                ],
            },
            "research": {
                "benchmark_governance": [{"Selected_Xi": "SPY", "Fit": 0.9}],
                "model_registry": [{"benchmark_xi": "SPY"}],
                "variance_model_selection": [],
                "pelt_regime_segments": [],
                "pelt_change_points": [],
                "options_summary": [],
                "options_chain": [],
                "factor_attribution": [],
                "hedge_suggestions": [],
            },
        },
        "backtest_path_bundle": {},
        "suitability_gate": {},
        "promotion_gate": {},
        "data_freshness_report": [{"Namespace": "prices_daily", "Status": "fresh"}],
    }


def test_full_analysis_publication_requires_three_year_price_history():
    artifacts = _full_publication_artifacts(pd.date_range("2025-01-01", periods=300, freq="B"))
    manifest = build_publication_manifest(str(uuid4()), artifacts)
    valid, checks, rejections = validate_publication_bundle(manifest, artifacts)
    assert valid is False
    assert checks["minimum_three_year_price_history"] is False
    assert "minimum_three_year_price_history" in rejections


def test_full_analysis_publication_requires_strategy_laboratory_contract():
    artifacts = _full_publication_artifacts(pd.date_range("2021-01-01", "2025-01-10", freq="B"))
    artifacts["dashboard_payload"]["strategy_lab"] = {}
    manifest = build_publication_manifest(str(uuid4()), artifacts)
    valid, checks, rejections = validate_publication_bundle(manifest, artifacts)
    assert valid is False
    assert checks["strategy_lab_present"] is False
    assert "strategy_lab_present" in rejections


def test_full_analysis_publication_requires_holdout_lineage_and_independence_gate():
    artifacts = _full_publication_artifacts(pd.date_range("2021-01-01", "2025-01-10", freq="B"))
    strategy_lab = artifacts["dashboard_payload"]["strategy_lab"]
    strategy_lab["research_lineage"] = []
    strategy_lab["validation"] = [
        {"Metric": "Promotion_Status", "Value": "RESEARCH_ONLY", "Pass": False},
    ]
    manifest = build_publication_manifest(str(uuid4()), artifacts)
    valid, checks, rejections = validate_publication_bundle(manifest, artifacts)
    assert valid is False
    assert checks["strategy_lab_present"] is False
    assert "strategy_lab_present" in rejections


def test_full_analysis_publication_accepts_three_year_history_and_unit_weights():
    artifacts = _full_publication_artifacts(pd.date_range("2021-01-01", "2025-01-10", freq="B"))
    manifest = build_publication_manifest(str(uuid4()), artifacts)
    valid, checks, rejections = validate_publication_bundle(manifest, artifacts)
    assert valid is True
    assert checks["minimum_three_year_price_history"] is True
    assert checks["portfolio_weights_valid"] is True
    assert checks["institutional_full_payload_complete"] is True
    assert rejections == ()


def test_full_analysis_publication_rejects_lightweight_price_only_snapshot():
    artifacts = _full_publication_artifacts(pd.date_range("2021-01-01", "2025-01-10", freq="B"))
    artifacts["dashboard_payload"]["tables"]["fundamentals"] = [
        {"Ticker": "AAPL", "Sector": "Price-only snapshot", "Weight": 0.4},
        {"Ticker": "MSFT", "Sector": "Price-only snapshot", "Weight": 0.6},
    ]
    manifest = build_publication_manifest(str(uuid4()), artifacts)
    valid, checks, rejections = validate_publication_bundle(manifest, artifacts)
    assert valid is False
    assert checks["institutional_full_payload_complete"] is False
    assert "institutional_full_payload_complete" in rejections


def test_full_analysis_publication_rejects_missing_capability_matrix():
    artifacts = _full_publication_artifacts(pd.date_range("2021-01-01", "2025-01-10", freq="B"))
    artifacts["dashboard_payload"]["status"].pop("capability_completeness")
    manifest = build_publication_manifest(str(uuid4()), artifacts)
    valid, checks, rejections = validate_publication_bundle(manifest, artifacts)
    assert valid is False
    assert checks["full_capability_matrix_present"] is False
    assert "full_capability_matrix_present" in rejections


def test_full_analysis_publication_rejects_partial_capability_matrix():
    artifacts = _full_publication_artifacts(pd.date_range("2021-01-01", "2025-01-10", freq="B"))
    artifacts["dashboard_payload"]["status"]["capability_completeness"][2]["Status"] = "partial"
    artifacts["dashboard_payload"]["status"]["capability_completeness"][2][
        "Missing_Evidence"
    ] = "tables.fundamentals"
    manifest = build_publication_manifest(str(uuid4()), artifacts)
    valid, checks, rejections = validate_publication_bundle(manifest, artifacts)
    assert valid is False
    assert checks["full_capability_core_complete"] is False
    assert "full_capability_core_complete" in rejections


def test_publication_rejects_future_dated_module_asof():
    artifacts = _daily_v10_publication_artifacts(include_fixed_income=True)
    artifacts["dashboard_payload"]["status"] = {"snapshot_meta": [{"As_Of": "2026-06-15T00:00:00+00:00"}]}
    artifacts["dashboard_payload"]["fixed_income_intelligence"]["as_of"] = "2026-06-17T00:00:00+00:00"
    manifest = build_publication_manifest(str(uuid4()), artifacts)
    valid, checks, rejections = validate_publication_bundle(manifest, artifacts)
    assert valid is False
    assert checks["publication_temporal_coherence"] is False
    assert "publication_temporal_coherence" in rejections
    assert "future_module_asof:fixed_income_intelligence.as_of" in rejections
