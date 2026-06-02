import unittest

import pandas as pd

import quant_stockpicker_core as core


class InterbankReferenceRatesTests(unittest.TestCase):
    def test_fetch_interbank_reference_rates_long_schema_and_sofr_differential(self):
        dates = pd.date_range("2026-01-01", periods=4, freq="B")
        original_reader = core.pdr.DataReader

        def fake_reader(code, source, start, end):
            self.assertEqual(source, "fred")
            values = {
                "SOFR": [3.50, 3.55, 3.60, 3.65],
                "IUDSOIA": [3.70, 3.75, 3.80, 3.85],
                "ECBESTRVOLWGTTRMDMNRT": [1.90, 1.91, 1.92, 1.93],
                "IRSTCI01JPM156N": [0.70, 0.71, 0.72, 0.73],
            }.get(code, [3.00, 3.05, 3.10, 3.15])
            return pd.DataFrame({code: values}, index=dates)

        try:
            core.pdr.DataReader = fake_reader
            out = core.fetch_interbank_reference_rates("2026-01-01", "2026-01-10", use_cache=False)
        finally:
            core.pdr.DataReader = original_reader

        self.assertFalse(out.empty)
        self.assertIn("SOFR", set(out["Benchmark"]))
        self.assertIn("SONIA", set(out["Benchmark"]))
        self.assertIn("ESTR", set(out["Benchmark"]))
        self.assertIn("TONAR", set(out["Benchmark"]))
        self.assertFalse(out["Benchmark"].astype(str).str.contains("LI" + "BOR", case=False, na=False).any())
        self.assertIn("Jurisdiction", out.columns)
        self.assertIn("Currency", out.columns)
        self.assertIn("Level_Diff_vs_SOFR_bps", out.columns)
        self.assertIn("Data_Staleness_Days", out.columns)
        self.assertIn("Comparable_To_Current_Funding", out.columns)
        latest_sonia = out[out["Benchmark"].eq("SONIA")].sort_values("Observation_Date").iloc[-1]
        self.assertAlmostEqual(latest_sonia["Level_Diff_vs_SOFR_bps"], 20.0)
        self.assertEqual(latest_sonia["Currency"], "GBP")
        self.assertTrue(bool(latest_sonia["Comparable_To_Current_Funding"]))


if __name__ == "__main__":
    unittest.main()
