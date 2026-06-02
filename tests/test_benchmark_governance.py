import unittest

import pandas as pd

from quant_stockpicker_core import benchmark_governance_diagnostics, suggest_benchmark


class BenchmarkGovernanceTests(unittest.TestCase):
    def test_country_mandate_suggests_country_etf(self):
        self.assertEqual(
            suggest_benchmark("Country", "Mexico", benchmark_group="Country", investor_objective="Balanced growth"),
            "EWW",
        )

    def test_sector_mandate_suggests_sector_etf(self):
        self.assertEqual(
            suggest_benchmark("Sector", "United States", dominant_sector="Technology", benchmark_group="US Sector"),
            "XLK",
        )

    def test_custom_benchmark_warns_for_relative_metric(self):
        diag = benchmark_governance_diagnostics(
            "FOO",
            "Custom",
            "Relative vs benchmark",
            "United States",
            "Balanced growth",
            "information_ratio",
            ("AAPL", "MSFT"),
            cross_section=pd.DataFrame(
                {
                    "Ticker": ["AAPL", "MSFT"],
                    "Country": ["United States", "United States"],
                    "Sector": ["Technology", "Technology"],
                }
            ),
        )
        row = diag.iloc[0]
        self.assertFalse(bool(row["Benchmark_Is_Coherent"]))
        self.assertIn("custom benchmark is not recognized", row["Warnings"])


if __name__ == "__main__":
    unittest.main()
