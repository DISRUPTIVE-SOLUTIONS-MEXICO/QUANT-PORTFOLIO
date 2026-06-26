from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from quant_core.contracts import EvidenceScope, SecurityIntelligenceV1
from quant_core.dashboard_payload import build_dashboard_payload
from quant_core.security_intelligence import build_security_intelligence


def _market_data(days: int = 820) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(20260615)
    dates = pd.bdate_range("2023-01-02", periods=days)
    benchmark = rng.normal(0.00035, 0.009, days)
    convex = np.where(benchmark >= 0.0, 1.25 * benchmark, 0.55 * benchmark) + rng.normal(0.00015, 0.004, days)
    defensive = 0.35 * benchmark + rng.normal(0.0002, 0.003, days)
    prices = pd.DataFrame(
        {
            "SPY": 100.0 * np.cumprod(1.0 + benchmark),
            "CVX": 80.0 * np.cumprod(1.0 + convex),
            "DEF": 60.0 * np.cumprod(1.0 + defensive),
        },
        index=dates,
    )
    volumes = pd.DataFrame(
        {
            "SPY": rng.integers(60_000_000, 90_000_000, days),
            "CVX": rng.integers(3_000_000, 8_000_000, days),
            "DEF": rng.integers(1_000_000, 3_000_000, days),
        },
        index=dates,
    )
    return prices, volumes


def _latest_scores(as_of: pd.Timestamp) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"As_Of": as_of, "Strategy_ID": "momentum", "Ticker": "CVX", "Score": 1.5, "Selected": True, "Weight": 0.2},
            {
                "As_Of": as_of,
                "Strategy_ID": "momentum",
                "Ticker": "DEF",
                "Score": -0.5,
                "Selected": False,
                "Weight": 0.0,
            },
            {
                "As_Of": as_of,
                "Strategy_ID": "asymmetric_capture",
                "Ticker": "CVX",
                "Score": 1.0,
                "Selected": True,
                "Weight": 0.25,
            },
            {
                "As_Of": as_of,
                "Strategy_ID": "asymmetric_capture",
                "Ticker": "DEF",
                "Score": 0.2,
                "Selected": False,
                "Weight": 0.0,
            },
        ]
    )


def test_security_intelligence_builds_causal_risk_and_liquidity_state() -> None:
    prices, volumes = _market_data()
    artifact = build_security_intelligence(
        prices,
        benchmark="SPY",
        volumes=volumes,
        strategy_scores=_latest_scores(prices.index[-1]),
    )
    metrics = artifact["metrics"].set_index("Ticker")
    assert {"SPY", "CVX", "DEF"}.issubset(metrics.index)
    assert artifact["benchmark_xi"] == "SPY"
    assert len(artifact["price_history"]) == 756
    assert metrics.loc["CVX", "Evidence_Scope"] == "live_snapshot"
    assert metrics.loc["CVX", "ADV_USD_20D"] > 0.0
    assert metrics.loc["CVX", "Strategies_Selected"] == 2
    assert metrics.loc["CVX", "Strategy_Selection_Breadth"] == pytest.approx(1.0)
    assert np.isfinite(metrics.loc["CVX", "Upside_Beta_252D"])
    assert np.isfinite(metrics.loc["CVX", "Downside_Beta_252D"])
    assert metrics.loc["CVX", "Upside_Beta_252D"] > metrics.loc["CVX", "Downside_Beta_252D"]


def test_security_intelligence_future_contamination_cannot_change_prior_snapshot() -> None:
    prices, volumes = _market_data()
    cutoff = prices.index[650]
    baseline = build_security_intelligence(prices, benchmark="SPY", volumes=volumes, as_of=cutoff)
    contaminated = prices.copy()
    contaminated.loc[contaminated.index > cutoff, "CVX"] *= np.linspace(
        1.0,
        8.0,
        int((contaminated.index > cutoff).sum()),
    )
    altered = build_security_intelligence(contaminated, benchmark="SPY", volumes=volumes, as_of=cutoff)
    pd.testing.assert_frame_equal(
        baseline["metrics"].sort_values("Ticker").reset_index(drop=True),
        altered["metrics"].sort_values("Ticker").reset_index(drop=True),
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
    pd.testing.assert_frame_equal(baseline["price_history"], altered["price_history"])


def test_security_intelligence_does_not_infer_missing_volume() -> None:
    prices, _ = _market_data()
    artifact = build_security_intelligence(prices, benchmark="SPY")
    metrics = artifact["metrics"].set_index("Ticker")
    assert metrics["ADV_USD_20D"].isna().all()
    assert metrics["Amihud_ILLIQ_63D"].isna().all()


def test_security_intelligence_contract_requires_benchmark_and_formulas() -> None:
    with pytest.raises(ValidationError, match="benchmark_xi must be included"):
        SecurityIntelligenceV1(
            as_of=datetime(2026, 6, 15, tzinfo=UTC),
            evidence_scope=EvidenceScope.LIVE_SNAPSHOT,
            benchmark_xi="SPY",
            tickers=("CVX",),
            minimum_observations=252,
            price_history_days=756,
            formulas={
                "beta": "cov/var",
                "tail_beta": "conditional cov/var",
                "residual_momentum": "compound residuals",
                "drawdown": "price/running max - 1",
            },
        )


def test_dashboard_payload_exposes_security_workbench_without_ui_recalculation() -> None:
    prices, volumes = _market_data()
    results = {
        "prices": prices,
        "volumes": volumes,
        "portfolio": pd.DataFrame(),
        "benchmark_ticker": "SPY",
    }
    payload = build_dashboard_payload(
        results,
        {"path_metadata": {"benchmark": "SPY"}},
        {"summary": pd.DataFrame(), "breaches": pd.DataFrame()},
        {"summary": pd.DataFrame(), "tests": pd.DataFrame()},
    )
    security = payload["security_intelligence"]
    assert len(security["metrics"]) == 3
    assert len(security["price_history"]) == 756
    assert security["benchmark_xi"] == "SPY"
    assert security["contract"]["evidence_scope"] == "live_snapshot"
    assert security["contract"]["benchmark_xi"] == "SPY"
    assert payload["contract"]["schema_version"] == "2026.06.19-publication-isolation-v11"
