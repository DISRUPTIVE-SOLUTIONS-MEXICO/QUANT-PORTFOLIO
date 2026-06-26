from pathlib import Path
from uuid import uuid4

import pandas as pd

from cloud_job_worker import build_job_run_config, build_portfolio_contract
from quant_core.contracts import EvidenceScope


def _results(*, promotion_status: str = "research_only", include_oos: bool = True) -> dict:
    dates = pd.bdate_range("2023-01-02", periods=756)
    return {
        "portfolio": pd.DataFrame(
            [
                {
                    "Ticker": "AAA",
                    "Sector": "Technology",
                    "Country": "United States",
                    "Weight": 0.60,
                    "Composite_Score": 1.2,
                    "PIT_Confidence": 0.8,
                    "Dollar_Volume_63": 25_000_000.0,
                },
                {
                    "Ticker": "BBB",
                    "Sector": "Health Care",
                    "Country": "United States",
                    "Weight": 0.30,
                    "Composite_Score": 0.7,
                    "PIT_Confidence": 0.6,
                    "Dollar_Volume_63": 10_000_000.0,
                },
            ]
        ),
        "prices": pd.DataFrame({"AAA": range(100, 856), "BBB": range(50, 806)}, index=dates),
        "equity_curve": pd.DataFrame({"OOS_Return": [0.01, -0.005]}) if include_oos else pd.DataFrame(),
        "suitability_gate": {"status": "approved"},
        "promotion_gate": {"promotion_status": promotion_status},
        "model_registry": pd.DataFrame(
            [{"config_hash": "cfg", "data_hash": "data", "model_version": "xcdr-v3-test"}]
        ),
    }


def test_job_config_preserves_fundamental_style_and_xcdr_objective():
    config = build_job_run_config(
        {
            "tickers": ["aapl", "msft", "AAPL"],
            "benchmark_ticker": "spy",
            "filter_style": "growth",
            "objective": "xcdr_v3",
            "base_period": "5y",
        }
    )
    assert config.tickers == ("AAPL", "MSFT")
    assert config.fundamental_style == "growth"
    assert config.weight_objective == "xcdr_v3"
    assert config.price_period == "5y"
    assert not config.use_side_boom_portfolio


def test_portfolio_contract_normalizes_weights_and_preserves_research_state():
    user_id = str(uuid4())
    run_id = str(uuid4())
    config = build_job_run_config(
        {
            "tickers": ["AAA", "BBB"],
            "benchmark_ticker": "SPY",
            "filter_style": "quality",
            "base_period": "3y",
        }
    )
    contract = build_portfolio_contract(
        _results(),
        config,
        run_id=run_id,
        user_id=user_id,
        portfolio_name="Quality XCDR",
    )
    assert abs(sum(position.target_weight for position in contract.positions) - 1.0) < 1e-12
    assert contract.evidence_scope == EvidenceScope.OUT_OF_SAMPLE
    assert contract.promotion_status == "research_only"
    assert contract.suitability_status == "approved"
    assert all(position.reference_price is not None for position in contract.positions)


def test_snapshot_cannot_persist_as_promoted_portfolio():
    config = build_job_run_config(
        {
            "tickers": ["AAA", "BBB"],
            "benchmark_ticker": "SPY",
            "filter_style": "factor",
            "base_period": "3y",
        }
    )
    contract = build_portfolio_contract(
        _results(promotion_status="promoted", include_oos=False),
        config,
        run_id=str(uuid4()),
        user_id=str(uuid4()),
        portfolio_name="Snapshot",
    )
    assert contract.evidence_scope == EvidenceScope.LIVE_SNAPSHOT
    assert contract.promotion_status == "research_only"


def test_cloud_worker_workflow_is_serialized_and_service_role_only():
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github" / "workflows" / "process-quant-jobs.yml").read_text(encoding="utf-8")
    assert "cancel-in-progress: false" in workflow
    assert "SUPABASE_SERVICE_ROLE_KEY" in workflow
    assert "cloud_job_worker.py" in workflow
    assert "timeout-minutes: 180" in workflow


def test_job_api_enforces_active_job_and_daily_quota():
    root = Path(__file__).resolve().parents[1]
    source = (root / "apps" / "web" / "app" / "api" / "jobs" / "route.ts").read_text(encoding="utf-8")
    assert "already queued or running" in source
    assert "Daily optimization research limit reached" in source
    assert '.eq("user_id", userId)' in source
