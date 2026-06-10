import unittest

import numpy as np
import pandas as pd

from quant_core.promotion_gate import evaluate_promotion_gate
from quant_core.uncertainty_state import (
    StrategyConstitution,
    build_uncertainty_state,
    causal_kalman_mean_filter,
    fractional_volterra_kernel,
    fractional_volterra_variance,
    rmt_clean_covariance,
    robust_alpha_shrinkage,
    upside_downside_diagnostics,
    xcdr_v2_score,
    xcdr_v3_growth_control_score,
    xodr_v1_omega_dominance_score,
)
from quant_stockpicker_core import RunConfig, fit_variance_architecture


class UncertaintyStateTests(unittest.TestCase):
    def test_strategy_constitution_limits_flexibility(self):
        c = StrategyConstitution(
            strategy_id="XCDR-v2",
            allowed_features=("icir", "rmt", "volterra"),
            allowed_hyperparameters={"lambda_cvar": (0.5, 1.0), "lambda_dd": (0.5, 1.0)},
            benchmark_set=("SPY", "QQQ", "USMV"),
            complexity_budget=20,
        )
        self.assertTrue(c.is_within_budget())
        self.assertLessEqual(c.to_record()["complexity_score"], 20)

    def test_fractional_volterra_kernel_is_causal_and_normalized(self):
        k = fractional_volterra_kernel(hurst=0.10, length=32)
        self.assertEqual(list(k.index[:3]), [1, 2, 3])
        self.assertAlmostEqual(float(k.sum()), 1.0, places=10)
        self.assertTrue((k >= 0).all())

    def test_fractional_volterra_variance_does_not_use_future_returns(self):
        idx = pd.bdate_range("2025-01-01", periods=120)
        r = pd.Series(np.linspace(-0.01, 0.01, len(idx)), index=idx)
        baseline = fractional_volterra_variance(r, hurst=0.20, length=21)
        shocked = r.copy()
        shocked.iloc[-1] = 1.0
        altered = fractional_volterra_variance(shocked, hurst=0.20, length=21)
        pd.testing.assert_series_equal(baseline.iloc[:-1], altered.iloc[:-1])

    def test_kalman_filter_prefix_is_not_changed_by_future_observation(self):
        idx = pd.bdate_range("2025-01-01", periods=80)
        r = pd.Series(np.sin(np.arange(len(idx)) / 10) * 0.01, index=idx)
        base = causal_kalman_mean_filter(r)
        shocked = r.copy()
        shocked.iloc[-1] = 0.50
        alt = causal_kalman_mean_filter(shocked)
        pd.testing.assert_series_equal(base["Filtered_Mean"].iloc[:-1], alt["Filtered_Mean"].iloc[:-1])

    def test_rmt_clean_covariance_is_psd(self):
        rng = np.random.default_rng(20260528)
        idx = pd.bdate_range("2024-01-01", periods=220)
        rets = pd.DataFrame(rng.normal(0, 0.01, (len(idx), 12)), index=idx)
        cov, meta = rmt_clean_covariance(rets)
        eig = np.linalg.eigvalsh(cov.values)
        self.assertGreaterEqual(float(eig.min()), -1e-8)
        self.assertIn("RMT_Noise_Fraction", meta)

    def test_robust_alpha_and_xcdr_degrade_with_uncertainty(self):
        idx = pd.bdate_range("2025-01-01", periods=120)
        port = pd.Series(0.0006 + 0.006 * np.sin(np.arange(len(idx)) / 7), index=idx)
        bench = pd.Series(0.0003 + 0.006 * np.sin(np.arange(len(idx)) / 7), index=idx)
        weights = pd.Series([0.5, 0.5], index=["A", "B"])
        low = xcdr_v2_score(port, bench, weights=weights, crlb=pd.Series([1e-6, 1e-6]), turnover=0.05)
        high = xcdr_v2_score(port, bench, weights=weights, crlb=pd.Series([1e-2, 1e-2]), turnover=0.50)
        self.assertLess(float(high["XCDR_v2"]), float(low["XCDR_v2"]))
        alpha = pd.Series([0.10, 0.10], index=["A", "B"])
        shrunk = robust_alpha_shrinkage(alpha, pd.Series([1e-6, 1e-2], index=["A", "B"]))
        self.assertLess(float(shrunk["B"]), float(shrunk["A"]))

    def test_variance_architecture_includes_fractional_volterra_and_qlike(self):
        rng = np.random.default_rng(20260529)
        idx = pd.bdate_range("2024-01-01", periods=320)
        r = pd.Series(0.012 * rng.standard_t(df=6, size=len(idx)), index=idx)
        out = fit_variance_architecture(r)
        self.assertIn("FractionalVolterra", out["Model"].astype(str).tolist())
        self.assertIn("QLIKE", out.columns)
        self.assertTrue(pd.to_numeric(out["QLIKE"], errors="coerce").notna().any())

    def test_uncertainty_promotion_gate_requires_new_research_metrics(self):
        cfg = RunConfig(
            tickers=("AAA", "SPY"),
            investor_max_drawdown=0.20,
            investor_cvar_max_daily=0.03,
        )
        perf_summary = pd.DataFrame(
            [
                {"Metric": "Max_Drawdown", "Value": -0.10},
                {"Metric": "CVaR_95_Daily", "Value": -0.015},
            ]
        )
        validation = {
            "summary": pd.DataFrame(
                [
                    {"Metric": "Deflated_Sortino", "Value": 0.7},
                    {"Metric": "CPCV_PBO", "Value": 0.05},
                    {"Metric": "Hansen_SPA_PValue", "Value": 0.01},
                    {"Metric": "White_Reality_Check_PValue", "Value": 0.01},
                    {"Metric": "ICIR", "Value": 0.20},
                    {"Metric": "DXCDR", "Value": 0.30},
                    {"Metric": "OOS_QLIKE_Delta", "Value": -0.02},
                ]
            )
        }
        gate = evaluate_promotion_gate(perf_summary, validation, cfg)
        self.assertEqual(gate["promotion_status"], "promoted")

    def test_build_uncertainty_state_contract(self):
        rng = np.random.default_rng(20260530)
        idx = pd.bdate_range("2025-01-01", periods=140)
        cols = ["A", "B", "C"]
        rets = pd.DataFrame(rng.normal(0.0002, 0.01, (len(idx), len(cols))), index=idx, columns=cols)
        state = build_uncertainty_state(rets, pd.Series([0.4, 0.3, 0.3], index=cols), pelt_regime="transition")
        frame = state.to_frame()
        self.assertIn("rmt_noise_fraction", frame.columns)
        self.assertEqual(frame.loc[0, "pelt_regime"], "transition")

    def test_upside_volatility_does_not_increase_downside_deviation(self):
        idx = pd.bdate_range("2025-01-01", periods=8)
        bench = pd.Series([0.01, -0.01, 0.02, -0.02, 0.01, -0.01, 0.02, -0.02], index=idx)
        base = pd.Series([0.008, -0.006, 0.012, -0.010, 0.008, -0.006, 0.012, -0.010], index=idx)
        upside_boost = base.copy()
        upside_boost[bench > 0] *= 4.0
        d_base = upside_downside_diagnostics(base, bench)
        d_boost = upside_downside_diagnostics(upside_boost, bench)
        self.assertGreater(float(d_boost["Upside_Deviation"]), float(d_base["Upside_Deviation"]))
        self.assertAlmostEqual(float(d_boost["Downside_Deviation"]), float(d_base["Downside_Deviation"]), places=12)

    def test_downside_capture_worsens_when_portfolio_falls_more_than_xi(self):
        idx = pd.bdate_range("2025-01-01", periods=6)
        bench = pd.Series([0.01, -0.01, 0.01, -0.02, 0.01, -0.01], index=idx)
        mild = pd.Series([0.012, -0.005, 0.012, -0.010, 0.012, -0.004], index=idx)
        severe = pd.Series([0.012, -0.020, 0.012, -0.040, 0.012, -0.030], index=idx)
        d_mild = upside_downside_diagnostics(mild, bench)
        d_severe = upside_downside_diagnostics(severe, bench)
        self.assertLess(float(d_mild["Downside_Capture"]), 1.0)
        self.assertGreater(float(d_severe["Downside_Capture"]), 1.0)

    def test_downside_preservation_fails_when_return_improves_by_breaking_tail(self):
        idx = pd.bdate_range("2025-01-01", periods=40)
        bench = pd.Series(0.001, index=idx)
        baseline = pd.Series(0.0003, index=idx)
        candidate = baseline.copy()
        candidate.iloc[:38] = 0.005
        candidate.iloc[38:] = -0.03
        diag = upside_downside_diagnostics(candidate, bench, baseline_returns=baseline, tolerance=0.05)
        self.assertGreater(float(candidate.mean()), float(baseline.mean()))
        self.assertFalse(bool(diag["Downside_Preservation_Pass"]))

    def test_xcdr_v3_rewards_upside_capture_but_penalizes_downside_capture(self):
        idx = pd.bdate_range("2025-01-01", periods=80)
        bench_vals = np.tile([0.010, -0.008, 0.012, -0.006], 20)
        bench = pd.Series(bench_vals, index=idx)
        good = pd.Series(np.where(bench > 0, 1.25 * bench, 0.60 * bench), index=idx)
        bad = pd.Series(np.where(bench > 0, 0.80 * bench, 1.35 * bench), index=idx)
        weights = pd.Series([0.5, 0.5], index=["A", "B"])
        s_good = xcdr_v3_growth_control_score(good, bench, weights=weights)
        s_bad = xcdr_v3_growth_control_score(bad, bench, weights=weights)
        self.assertTrue(bool(s_good["XCDR_v3_Capture_Pass"]))
        self.assertFalse(bool(s_bad["XCDR_v3_Capture_Pass"]))
        self.assertGreater(float(s_good["XCDR_v3_GrowthControl"]), float(s_bad["XCDR_v3_GrowthControl"]))

    def test_xodr_v1_rewards_omega_upside_and_blocks_tail_breach(self):
        idx = pd.bdate_range("2025-01-01", periods=96)
        omega = pd.DataFrame(
            {
                "SPY": np.tile([0.010, -0.008, 0.012, -0.006], 24),
                "USMV": np.tile([0.006, -0.004, 0.007, -0.003], 24),
                "QQQ": np.tile([0.014, -0.011, 0.016, -0.009], 24),
            },
            index=idx,
        )
        xi = omega["SPY"]
        good = pd.Series(np.where(omega.mean(axis=1) > 0, 1.35 * xi, 0.45 * xi), index=idx)
        bad = pd.Series(np.where(omega.mean(axis=1) > 0, 1.35 * xi, 1.80 * xi), index=idx)
        weights = pd.Series([0.5, 0.5], index=["A", "B"])
        s_good = xodr_v1_omega_dominance_score(good, xi, omega, weights=weights, downside_quantile=0.50)
        s_bad = xodr_v1_omega_dominance_score(bad, xi, omega, weights=weights, downside_quantile=0.50)
        self.assertTrue(bool(s_good["XODR_v1_Pass"]))
        self.assertFalse(bool(s_bad["XODR_v1_Pass"]))
        self.assertGreater(float(s_good["XODR_v1"]), float(s_bad["XODR_v1"]))


if __name__ == "__main__":
    unittest.main()
