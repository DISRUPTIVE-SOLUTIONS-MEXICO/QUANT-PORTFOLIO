"""Tests for the Hurst estimator and the XCDR-v3 lambda sensitivity audit."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_core.uncertainty_state import (
    _hurst_from_log_vol,
    estimate_hurst_rv,
    fractional_volterra_variance,
    xcdr_lambda_sensitivity,
)


def _fbm_path(hurst: float, n: int, seed: int) -> np.ndarray:
    """Exact fractional Brownian motion via Cholesky of the covariance kernel."""
    t = np.arange(1, n + 1, dtype=float)
    s, u = np.meshgrid(t, t)
    cov = 0.5 * (np.abs(s) ** (2 * hurst) + np.abs(u) ** (2 * hurst) - np.abs(s - u) ** (2 * hurst))
    cov += 1e-10 * np.eye(n)
    chol = np.linalg.cholesky(cov)
    rng = np.random.default_rng(seed)
    return chol @ rng.standard_normal(n)


def test_hurst_from_log_vol_recovers_known_exponents():
    for h_true in (0.12, 0.42):
        path = _fbm_path(h_true, n=750, seed=20260610)
        out = _hurst_from_log_vol(pd.Series(path), max_lag=25)
        assert out["status"] == "ok"
        assert abs(float(out["hurst"]) - h_true) < 0.08, f"H={h_true}: estimated {out['hurst']}"
        assert float(out["r2"]) > 0.95


def test_estimate_hurst_rv_orders_rough_vs_smooth_volatility():
    n = 1500
    rough_lv = 0.35 * _fbm_path(0.08, n, seed=11)
    smooth_lv = 0.35 * _fbm_path(0.45, n, seed=11)
    rng = np.random.default_rng(99)
    eps = rng.standard_normal(n)
    r_rough = pd.Series(0.01 * np.exp(rough_lv - rough_lv.mean()) * eps)
    r_smooth = pd.Series(0.01 * np.exp(smooth_lv - smooth_lv.mean()) * eps)
    h_rough = estimate_hurst_rv(r_rough)
    h_smooth = estimate_hurst_rv(r_smooth)
    assert h_rough["status"] == "ok" and h_smooth["status"] == "ok"
    assert float(h_rough["hurst"]) < float(h_smooth["hurst"])


def test_fractional_volterra_accepts_estimated_hurst():
    rng = np.random.default_rng(5)
    r = pd.Series(rng.normal(0.0, 0.012, 400))
    fixed = fractional_volterra_variance(r, hurst=0.10)
    estimated = fractional_volterra_variance(r, hurst="estimated")
    assert not estimated.dropna().empty
    assert len(estimated) == len(fixed)
    assert (estimated.dropna() >= 0).all()


def test_fractional_volterra_rejects_unknown_hurst_spec():
    r = pd.Series(np.random.default_rng(1).normal(0.0, 0.01, 100))
    try:
        fractional_volterra_variance(r, hurst="banana")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown hurst spec")


def test_xcdr_lambda_sensitivity_reports_ranking_stability():
    rng = np.random.default_rng(13)
    benchmark = pd.Series(rng.normal(0.0004, 0.011, 504))
    policies = {
        "defensive": pd.Series(np.where(benchmark >= 0, 0.9 * benchmark, 0.6 * benchmark)),
        "aggressive": pd.Series(np.where(benchmark >= 0, 1.3 * benchmark, 1.2 * benchmark)),
        "balanced": pd.Series(np.where(benchmark >= 0, 1.1 * benchmark, 0.9 * benchmark)),
    }
    out = xcdr_lambda_sensitivity(policies, benchmark, perturbation=0.5)
    assert not out.empty
    # 10 lambdas x 2 directions.
    assert len(out) == 20
    assert {"Lambda", "Direction", "Top_Changed", "Rank_Spearman_vs_Base", "Ranking_Fragile"}.issubset(out.columns)
    corr = out["Rank_Spearman_vs_Base"].dropna()
    assert ((corr >= -1.0) & (corr <= 1.0)).all()
    assert out["Top_Changed"].dtype == bool


def test_xcdr_lambda_sensitivity_requires_two_policies():
    rng = np.random.default_rng(3)
    benchmark = pd.Series(rng.normal(0.0, 0.01, 300))
    out = xcdr_lambda_sensitivity({"only": benchmark * 1.1}, benchmark)
    assert out.empty
