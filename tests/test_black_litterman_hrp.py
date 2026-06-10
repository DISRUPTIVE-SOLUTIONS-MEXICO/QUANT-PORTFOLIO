import unittest

import numpy as np
import pandas as pd

from quant_stockpicker_core import (
    black_litterman_posterior_alpha,
    construct_constrained_weights,
    hierarchical_risk_parity_weights,
)


class BlackLittermanHrpTests(unittest.TestCase):
    def test_black_litterman_outputs_posterior_for_all_assets(self):
        tickers = ["A", "B", "C"]
        sigma = np.array([[0.04, 0.01, 0.00], [0.01, 0.09, 0.02], [0.00, 0.02, 0.16]])
        selected = pd.DataFrame(
            {
                "Ticker": tickers,
                "Market_Cap_AsOf": [100.0, 50.0, 25.0],
                "Bayesian_Alpha_Mean": [0.8, 0.1, -0.2],
                "Bayesian_Alpha_Std": [0.2, 0.5, 0.8],
                "CRLB_Mu": [0.01, 0.02, 0.04],
                "Valid_Fundamental_Ratios": [12, 10, 8],
            }
        ).set_index("Ticker")
        posterior, diag = black_litterman_posterior_alpha(selected, tickers, sigma, tau=0.05)
        self.assertEqual(list(posterior.index), tickers)
        self.assertEqual(len(diag), 3)
        self.assertTrue(np.isfinite(posterior.values).all())
        self.assertIn("BLInspired_Posterior_Alpha", diag.columns)
        self.assertEqual(diag["BLInspired_Mode"].iloc[0], "rank_shrinkage")

    def test_canonical_mode_recovers_equilibrium_with_zero_ic_views(self):
        tickers = ["A", "B", "C"]
        sigma = np.array([[0.04, 0.01, 0.00], [0.01, 0.09, 0.02], [0.00, 0.02, 0.16]])
        caps = np.array([100.0, 50.0, 25.0])
        selected = pd.DataFrame(
            {
                "Ticker": tickers,
                "Market_Cap_AsOf": caps,
                "Bayesian_Alpha_Mean": [0.8, 0.1, -0.2],
                "Bayesian_Alpha_Std": [0.2, 0.5, 0.8],
                "CRLB_Mu": [0.01, 0.02, 0.04],
                "Valid_Fundamental_Ratios": [12, 10, 8],
            }
        ).set_index("Ticker")
        risk_aversion = 2.5
        posterior, diag = black_litterman_posterior_alpha(
            selected, tickers, sigma, risk_aversion=risk_aversion, tau=0.05, bl_canonical=True, view_ic=0.0
        )
        # With IC = 0 every view equals the equilibrium return, so the
        # posterior must collapse back to pi = delta * Sigma * w_mkt in
        # return units (no z-standardization).
        w_mkt = caps / caps.sum()
        pi = risk_aversion * sigma @ w_mkt
        np.testing.assert_allclose(posterior.values, pi, rtol=1e-6, atol=1e-10)
        self.assertEqual(diag["BLInspired_Mode"].iloc[0], "canonical")

    def test_hrp_weights_are_long_only_and_sum_to_one(self):
        cov = pd.DataFrame(
            [[0.04, 0.01, 0.00], [0.01, 0.09, 0.02], [0.00, 0.02, 0.16]],
            index=["A", "B", "C"],
            columns=["A", "B", "C"],
        )
        weights, meta = hierarchical_risk_parity_weights(cov)
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=10)
        self.assertTrue((weights >= 0).all())
        self.assertIn("hrp_order", meta)

    def test_optimizer_accepts_black_litterman_and_hrp_objectives(self):
        rng = np.random.default_rng(123)
        dates = pd.bdate_range("2025-01-01", periods=140)
        tickers = ["A", "B", "C", "D"]
        rets = pd.DataFrame(rng.normal(0.0004, 0.012, (len(dates), len(tickers))), index=dates, columns=tickers)
        prices = 100 * (1 + rets).cumprod()
        selected = pd.DataFrame(
            {
                "Ticker": tickers,
                "Sector": ["Tech", "Tech", "Health", "Utility"],
                "Composite_Score": [1.0, 0.6, 0.2, -0.1],
                "Bayesian_Alpha_Mean": [0.8, 0.4, 0.1, -0.2],
                "Bayesian_Alpha_Std": [0.2, 0.3, 0.4, 0.5],
                "Market_Cap_AsOf": [100, 90, 80, 70],
                "Dollar_Volume_63": [1e9] * 4,
            }
        )
        for objective in ["black_litterman", "hrp"]:
            weights, meta = construct_constrained_weights(
                selected,
                prices,
                dates[-1],
                120,
                objective=objective,
                max_weight=0.60,
                sector_weight_cap=0.80,
                use_black_litterman=objective == "black_litterman",
                multistarts=2,
            )
            self.assertAlmostEqual(float(weights.sum()), 1.0, places=8)
            self.assertTrue((weights >= -1e-10).all())
            self.assertIn("status", meta)


if __name__ == "__main__":
    unittest.main()
