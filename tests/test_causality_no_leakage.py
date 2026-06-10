import unittest

import numpy as np
import pandas as pd

from quant_stockpicker_core import RunConfig, backtest, fundamentals_asof, optimize_chunks


def synthetic_prices(tickers=("AAA", "BBB", "CCC"), periods=180):
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2025-01-01", periods=periods)
    rets = pd.DataFrame(rng.normal(0.0004, 0.012, (periods, len(tickers))), index=dates, columns=tickers)
    prices = 100.0 * (1.0 + rets).cumprod()
    prices["SPY"] = 100.0 * (1.0 + rets.mean(axis=1)).cumprod()
    return prices


def synthetic_panel(tickers=("AAA", "BBB", "CCC"), availability="2025-01-15"):
    rows = []
    for i, ticker in enumerate(tickers):
        rows.append(
            {
                "Ticker": ticker,
                "Sector": "Technology" if i < 2 else "Healthcare",
                "Country": "United States",
                "Period_End": pd.Timestamp("2024-12-31"),
                "Availability_Date": pd.Timestamp(availability),
                "_shares": 100_000_000.0,
                "_debt": 100_000_000.0 + i,
                "_cash": 20_000_000.0,
                "_ebit": 40_000_000.0 + i * 1_000_000.0,
                "_revenue": 300_000_000.0 + i * 5_000_000.0,
                "_net_income": 25_000_000.0 + i * 1_000_000.0,
                "_equity": 200_000_000.0,
                "_assets": 350_000_000.0,
                "_liabilities": 150_000_000.0,
                "_cfo": 35_000_000.0,
                "_capex": -8_000_000.0,
                "_dividends": -2_000_000.0,
                "_depreciation_amortization": 5_000_000.0,
                "_interest": -3_000_000.0,
                "_retained": 80_000_000.0,
                "_tax": 8_000_000.0,
                "_pretax": 33_000_000.0,
                "_working_capital": 60_000_000.0,
                "_current_assets": 120_000_000.0,
                "_current_liabilities": 60_000_000.0,
                "_gross_profit": 140_000_000.0,
                "Piotroski": 7.0,
            }
        )
    return pd.DataFrame(rows)


class CausalityNoLeakageTests(unittest.TestCase):
    def test_availability_date_blocks_future_fundamental(self):
        prices = synthetic_prices(("AAA",), 80)
        panel = synthetic_panel(("AAA",), availability="2025-01-15")
        future = panel.iloc[0].copy()
        future["Period_End"] = pd.Timestamp("2025-12-31")
        future["Availability_Date"] = pd.Timestamp("2026-03-31")
        future["_net_income"] = 9_999_999_999.0
        contaminated = pd.concat([panel, pd.DataFrame([future])], ignore_index=True)
        asof = pd.Timestamp("2025-03-31")
        out = fundamentals_asof(contaminated, prices, asof)
        self.assertEqual(pd.Timestamp(out.iloc[0]["Availability_Date"]), pd.Timestamp("2025-01-15"))
        self.assertLess(out.iloc[0]["EPS"], 1.0)

    def test_optimization_ignores_prices_after_asof(self):
        prices = synthetic_prices(periods=180)
        asof = prices.index[120]
        cs = pd.DataFrame(
            {
                "Ticker": ["AAA", "BBB", "CCC"],
                "Sector": ["Technology", "Technology", "Healthcare"],
                "Country": ["United States"] * 3,
                "Composite_Score": [1.0, 0.5, 0.2],
                "Fundamental_Gate": [True, True, True],
                "Dollar_Volume_63": [1e9, 1e9, 1e9],
            }
        )
        altered = prices.copy()
        altered.loc[altered.index > asof, "AAA"] *= 1000.0
        _, opts_clean = optimize_chunks(
            cs, prices, asof, lookback=80, min_chunk=2, max_chunk=2, preselect_n=3, bootstrap_samples=0, max_combos=100
        )
        _, opts_altered = optimize_chunks(
            cs, altered, asof, lookback=80, min_chunk=2, max_chunk=2, preselect_n=3, bootstrap_samples=0, max_combos=100
        )
        self.assertEqual(opts_clean.iloc[0]["Tickers"], opts_altered.iloc[0]["Tickers"])
        self.assertAlmostEqual(float(opts_clean.iloc[0]["Sortino"]), float(opts_altered.iloc[0]["Sortino"]), places=10)

    def test_purging_creates_gap_between_train_and_validation(self):
        prices = synthetic_prices(periods=180)
        cs = pd.DataFrame(
            {
                "Ticker": ["AAA", "BBB", "CCC"],
                "Sector": ["Technology", "Technology", "Healthcare"],
                "Country": ["United States"] * 3,
                "Composite_Score": [1.0, 0.5, 0.2],
                "Fundamental_Gate": [True, True, True],
                "Dollar_Volume_63": [1e9, 1e9, 1e9],
            }
        )
        asof = prices.index[-1]
        purge_days = 7
        _, opts = optimize_chunks(
            cs,
            prices,
            asof,
            lookback=120,
            min_chunk=2,
            max_chunk=2,
            preselect_n=3,
            bootstrap_samples=0,
            purge_days=purge_days,
            nested_validation_fraction=0.25,
            max_combos=100,
        )
        row = opts.iloc[0]
        ret_index = (
            prices.loc[:asof, ["AAA", "BBB", "CCC"]].tail(121).pct_change(fill_method=None).dropna(how="all").index
        )
        train_pos = ret_index.get_loc(pd.Timestamp(row["Train_End"]))
        val_pos = ret_index.get_loc(pd.Timestamp(row["Validation_Start"]))
        self.assertGreaterEqual(val_pos - train_pos - 1, purge_days)
        self.assertLess(pd.Timestamp(row["Validation_Start"]), pd.Timestamp(row["Opt_End"]) + pd.Timedelta(days=1))

    def test_backtest_embargo_moves_signal_before_execution(self):
        prices = synthetic_prices(periods=180)
        panel = synthetic_panel()
        macro = pd.DataFrame({"hawkish_score": 0.0, "bullish_score": 0.0}, index=prices.index)
        volumes = pd.DataFrame(1_000_000, index=prices.index, columns=prices.columns)
        cfg = RunConfig(
            tickers=("AAA", "BBB", "CCC"),
            benchmark_ticker="SPY",
            price_period="1y",
            top_n=3,
            preselect_n=3,
            min_chunk=2,
            max_chunk=2,
            max_combos=20,
            rebalance_freq="ME",
            embargo_days=3,
            use_garch=False,
            use_options_snapshot=False,
            validation_bootstrap_samples=0,
            bootstrap_samples=0,
            max_oos_trials_per_rebalance=5,
        )
        perf, _, _ = backtest(prices, panel, volumes, macro, cfg)
        self.assertFalse(perf.empty)
        for _, row in perf.iterrows():
            signal_pos = prices.index.get_loc(pd.Timestamp(row["Signal_Date"]))
            exec_pos = prices.index.get_loc(pd.Timestamp(row["Rebalance_Date"]))
            self.assertEqual(exec_pos - signal_pos, 3)

    def test_backtest_rebalances_semiannually_but_reoptimizes_annually(self):
        prices = synthetic_prices(periods=520)
        panel = synthetic_panel()
        macro = pd.DataFrame({"hawkish_score": 0.0, "bullish_score": 0.0}, index=prices.index)
        volumes = pd.DataFrame(1_000_000, index=prices.index, columns=prices.columns)
        cfg = RunConfig(
            tickers=("AAA", "BBB", "CCC"),
            benchmark_ticker="SPY",
            price_period="2y",
            top_n=3,
            preselect_n=3,
            min_chunk=2,
            max_chunk=2,
            max_combos=20,
            rebalance_freq="2QE",
            reoptimization_freq="YE",
            embargo_days=3,
            use_garch=False,
            use_options_snapshot=False,
            validation_bootstrap_samples=0,
            bootstrap_samples=0,
            max_oos_trials_per_rebalance=5,
            lookback_grid=(63,),
            chunk_size_grid=(2,),
        )
        perf, holdings, opt_grid = backtest(prices, panel, volumes, macro, cfg)
        self.assertFalse(perf.empty)
        self.assertIn("Reoptimized", perf.columns)
        self.assertGreater(perf["Reoptimized"].astype(bool).sum(), 0)
        self.assertLess(perf["Reoptimized"].astype(bool).sum(), len(perf))
        self.assertTrue(holdings["Reoptimized"].isin([True, False]).all())
        self.assertTrue(opt_grid["Reoptimized"].eq(True).all())


if __name__ == "__main__":
    unittest.main()
