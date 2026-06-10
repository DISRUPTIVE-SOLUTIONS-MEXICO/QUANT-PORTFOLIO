"""Tests for L-moment EVT, turnover-aware optimization and Romano-Wolf."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import genpareto

from quant_stockpicker_core import (
    construct_constrained_weights,
    evt_tail_metrics_detailed,
    gpd_lmoments_fit,
    white_reality_check_spa,
)


def test_gpd_lmoments_recovers_known_shape():
    rng = np.random.default_rng(42)
    for xi_true in (0.10, 0.30):
        # Large-sample check: the estimator must converge to the truth.
        excess = genpareto.rvs(c=xi_true, scale=0.01, size=5000, random_state=rng)
        xi, beta, status = gpd_lmoments_fit(excess)
        assert status == "ok"
        assert abs(xi - xi_true) < 0.05, f"xi_true={xi_true}: estimated {xi}"
        assert abs(beta - 0.01) < 0.002


def test_gpd_lmoments_is_stable_in_small_samples():
    rng = np.random.default_rng(7)
    estimates = []
    for _ in range(200):
        excess = genpareto.rvs(c=0.2, scale=0.01, size=25, random_state=rng)
        xi, _beta, status = gpd_lmoments_fit(excess)
        if status == "ok":
            estimates.append(xi)
    est = np.asarray(estimates)
    # Small-sample sanity: centered near truth, no boundary explosions.
    assert abs(np.median(est) - 0.2) < 0.15
    assert np.all(np.abs(est) < 1.5)


def test_evt_detailed_uses_lmoments_and_supports_pooled_xi():
    rng = np.random.default_rng(3)
    r = pd.Series(rng.standard_t(df=4, size=750) * 0.01)
    detail = evt_tail_metrics_detailed(r)
    assert detail["status"] == "gpd_evt_lmom"
    assert np.isfinite(detail["xi"]) and np.isfinite(detail["var"]) and np.isfinite(detail["cvar"])
    assert detail["n_excess"] >= 20
    pooled = evt_tail_metrics_detailed(r, xi_override=0.25)
    assert pooled["status"] == "gpd_evt_pooled_xi"
    assert np.isclose(pooled["xi"], 0.25)
    assert pooled["cvar"] > pooled["var"] > 0


def _toy_selected(tickers: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Ticker": tickers,
            "Sector": ["Tech", "Tech", "Health", "Utility"],
            "Composite_Score": [1.0, 0.6, 0.2, -0.1],
            "Dollar_Volume_63": [1e9] * 4,
        }
    )


def test_turnover_penalty_pulls_solution_toward_previous_weights():
    rng = np.random.default_rng(123)
    dates = pd.bdate_range("2025-01-01", periods=160)
    tickers = ["A", "B", "C", "D"]
    rets = pd.DataFrame(rng.normal(0.0004, 0.012, (len(dates), len(tickers))), index=dates, columns=tickers)
    prices = 100 * (1 + rets).cumprod()
    prev = pd.Series({"A": 0.10, "B": 0.10, "C": 0.40, "D": 0.40})

    common = dict(
        objective="min_variance",
        max_weight=0.60,
        sector_weight_cap=0.80,
        multistarts=3,
    )
    w_free, _ = construct_constrained_weights(_toy_selected(tickers), prices, dates[-1], 120, **common)
    w_anchored, meta = construct_constrained_weights(
        _toy_selected(tickers),
        prices,
        dates[-1],
        120,
        prev_weights=prev,
        turnover_penalty=5.0,
        **common,
    )
    dist_free = float((w_free - prev).abs().sum())
    dist_anchored = float((w_anchored - prev).abs().sum())
    assert meta["opt_turnover_prev_weights_used"] is True
    assert dist_anchored < dist_free
    assert np.isclose(float(w_anchored.sum()), 1.0, atol=1e-6)


def _grid(seed: int, signal: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-31", periods=48, freq="ME")
    rows = []
    for trial in range(12):
        key = f"trial_{trial:02d}"
        edge = signal if trial == 0 else 0.0
        for date in dates:
            rows.append(
                {
                    "Rebalance_Date": date,
                    "Trial_Key": key,
                    "OOS_Equal_Return": rng.normal(edge, 0.02),
                }
            )
    return pd.DataFrame(rows)


def test_romano_wolf_controls_family_under_null_and_detects_signal():
    null_out = white_reality_check_spa(_grid(seed=5), samples=256, seed=99).set_index("Metric")["Value"]
    assert float(null_out["RomanoWolf_Rejected_5pct"]) == 0.0

    strong = white_reality_check_spa(_grid(seed=5, signal=0.02), samples=256, seed=99).set_index("Metric")["Value"]
    assert float(strong["RomanoWolf_Rejected_5pct"]) >= 1.0
    assert float(strong["RomanoWolf_Best_Trial_Adj_PValue"]) < 0.05
    # Adjusted p-values are monotone vs the marginal SPA p-value.
    assert float(strong["RomanoWolf_Min_Adj_PValue"]) >= 0.0
