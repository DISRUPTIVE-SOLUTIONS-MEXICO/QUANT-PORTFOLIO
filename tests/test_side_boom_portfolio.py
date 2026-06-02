import unittest

import numpy as np
import pandas as pd

from quant_stockpicker_core import (
    DEFAULT_SIDE_ALPHA_CEREBRAS_WEIGHT,
    DEFAULT_SIDE_ALPHA_FIXED_WEIGHTS,
    DEFAULT_SIDE_ALPHA_TICKERS,
    RunConfig,
    normalize_side_tickers,
    optimize_side_boom_portfolio,
    side_boom_pelt_diagnostics,
    side_boom_walk_forward,
)


class SideBoomPortfolioTests(unittest.TestCase):
    def test_aliases_normalize_company_names(self):
        tickers = normalize_side_tickers(["CEREBRAS", "MICROSOFT", "APPLE", "LENOVO", "TSMC", "CISCO"])
        self.assertEqual(tickers, ("CBRS", "MSFT", "AAPL", "LNVGY", "TSM", "CSCO"))

    def test_side_portfolio_keeps_fixed_cbrs_weight_and_sums_to_one(self):
        rng = np.random.default_rng(7)
        dates = pd.bdate_range("2025-01-01", periods=180)
        tickers = ["CBRS", "MSFT", "NVDA", "PFE", "SPY"]
        rets = pd.DataFrame(rng.normal(0.0004, 0.012, (len(dates), len(tickers))), index=dates, columns=tickers)
        prices = 100.0 * (1.0 + rets).cumprod()
        prices.loc[dates[:140], "CBRS"] = np.nan
        cfg = RunConfig(
            tickers=("MSFT", "NVDA", "PFE"),
            benchmark_ticker="SPY",
            side_boom_tickers=("CEREBRAS", "MICROSOFT", "NVDA", "PFIZER"),
            side_boom_fixed_ticker="CBRS",
            side_boom_fixed_weight=0.14,
            side_boom_fixed_weights=(("NVDA", 0.10),),
            sortino_multistarts=2,
        )
        portfolio, curve, diag = optimize_side_boom_portfolio(prices, cfg, lookback=126)
        weights = portfolio.set_index("Ticker")["Weight"]
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=8)
        self.assertAlmostEqual(float(weights.loc["CBRS"]), 0.14, places=8)
        self.assertAlmostEqual(float(weights.loc["NVDA"]), 0.10, places=8)
        self.assertFalse(curve.empty)
        self.assertIn("Side_Boom_Equity", curve.columns)
        self.assertFalse(diag.empty)

    def test_side_walk_forward_uses_cash_before_fixed_ticker_is_tradable(self):
        rng = np.random.default_rng(11)
        dates = pd.bdate_range("2025-01-01", periods=160)
        tickers = ["CBRS", "MSFT", "NVDA", "PFE", "SPY"]
        rets = pd.DataFrame(rng.normal(0.0005, 0.011, (len(dates), len(tickers))), index=dates, columns=tickers)
        prices = 100.0 * (1.0 + rets).cumprod()
        prices.loc[dates[:100], "CBRS"] = np.nan
        perf = pd.DataFrame(
            {
                "Signal_Date": [dates[50], dates[120]],
                "OOS_Start": [dates[55], dates[125]],
                "OOS_End": [dates[75], dates[145]],
            }
        )
        cfg = RunConfig(
            tickers=("MSFT", "NVDA", "PFE"),
            benchmark_ticker="SPY",
            side_boom_tickers=("CEREBRAS", "MICROSOFT", "NVDA", "PFIZER"),
            side_boom_fixed_ticker="CBRS",
            side_boom_fixed_weight=0.14,
            side_boom_fixed_weights=(("NVDA", 0.10),),
            side_boom_min_obs=20,
            sortino_multistarts=2,
        )
        side_perf, holdings, diag = side_boom_walk_forward(prices, cfg, perf=perf, lookback=60)
        self.assertEqual(len(side_perf), 2)
        first_cash = float(holdings[(holdings["OOS_End"].eq(dates[75])) & (holdings["Ticker"].eq("CASH"))]["Weight"].sum())
        second_cbrs = float(holdings[(holdings["OOS_End"].eq(dates[145])) & (holdings["Ticker"].eq("CBRS"))]["Weight"].sum())
        first_nvda = float(holdings[(holdings["OOS_End"].eq(dates[75])) & (holdings["Ticker"].eq("NVDA"))]["Weight"].sum())
        second_nvda = float(holdings[(holdings["OOS_End"].eq(dates[145])) & (holdings["Ticker"].eq("NVDA"))]["Weight"].sum())
        self.assertAlmostEqual(first_cash, 0.14, places=8)
        self.assertAlmostEqual(second_cbrs, 0.14, places=8)
        self.assertAlmostEqual(first_nvda, 0.10, places=8)
        self.assertAlmostEqual(second_nvda, 0.10, places=8)
        self.assertTrue((holdings.groupby("OOS_End")["Weight"].sum().round(8) == 1.0).all())
        self.assertFalse(diag.empty)

    def test_default_private_side_alpha_weights_sum_to_one_with_cbrs_residual(self):
        cfg = RunConfig(tickers=("LITE", "CIEN"), benchmark_ticker="SPY")
        self.assertEqual(cfg.side_boom_tickers, DEFAULT_SIDE_ALPHA_TICKERS)
        fixed = dict(cfg.side_boom_fixed_weights)
        self.assertEqual(fixed, dict(DEFAULT_SIDE_ALPHA_FIXED_WEIGHTS))
        total = cfg.side_boom_fixed_weight + sum(fixed.values())
        self.assertAlmostEqual(total, 1.0, places=8)
        self.assertAlmostEqual(cfg.side_boom_fixed_weight, DEFAULT_SIDE_ALPHA_CEREBRAS_WEIGHT, places=12)
        self.assertEqual(cfg.side_boom_mode, "private_side_alpha_firewall")

    def test_side_boom_pelt_diagnostics_from_curve(self):
        idx = pd.date_range("2024-01-01", periods=160, freq="B")
        returns = np.r_[np.tile([0.001, -0.001], 40), np.tile([0.030, -0.028], 40)]
        curve = pd.DataFrame({"Period_End": idx, "Side_Boom_Return": returns})

        out = side_boom_pelt_diagnostics(curve)

        self.assertFalse(out["side_boom_pelt_regime_segments"].empty)
        self.assertFalse(out["side_boom_pelt_timeline"].empty)
        self.assertEqual(set(out["side_boom_pelt_timeline"]["Series"]), {"SIDE_BOOM"})


if __name__ == "__main__":
    unittest.main()
