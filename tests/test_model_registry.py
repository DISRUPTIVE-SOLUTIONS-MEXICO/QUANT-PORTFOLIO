import unittest

import pandas as pd

from quant_stockpicker_core import (
    RunConfig,
    build_model_registry_record,
    canonical_hash,
)


class ModelRegistryTests(unittest.TestCase):
    def test_canonical_hash_is_order_invariant(self):
        self.assertEqual(canonical_hash({"a": 1, "b": 2}), canonical_hash({"b": 2, "a": 1}))

    def test_registry_record_contains_reproducibility_keys(self):
        cfg = RunConfig(tickers=("AAPL", "MSFT"), benchmark_ticker="SPY", weight_objective="sortino")
        dates = pd.bdate_range("2025-01-01", periods=5)
        prices = pd.DataFrame({"AAPL": [1, 2, 3, 4, 5], "MSFT": [2, 3, 4, 5, 6]}, index=dates)
        panel = pd.DataFrame({"Ticker": ["AAPL"], "Availability_Date": [pd.Timestamp("2025-01-03")]})
        macro = pd.DataFrame({"hawkish_score": [0.1]}, index=[dates[-1]])
        cs = pd.DataFrame({"Ticker": ["AAPL", "MSFT"]})
        portfolio = pd.DataFrame({"Ticker": ["AAPL"], "Weight": [1.0]})
        perf = pd.DataFrame({"Period_End": [dates[-1]], "Net_Return": [0.01]})
        perf_summary = pd.DataFrame({"Metric": ["Total_Return"], "Value": [0.01]})
        record = build_model_registry_record(
            cfg,
            prices,
            panel,
            macro,
            cs,
            portfolio,
            perf,
            perf_summary,
            pd.DataFrame(),
            pd.DataFrame(),
            {"summary": pd.DataFrame()},
            {"total_sec": 1.0},
        )
        for key in [
            "run_hash",
            "code_version",
            "app_version",
            "model_version",
            "schema_version",
            "config_hash",
            "universe_hash",
            "data_hash",
            "config",
            "data_quality",
        ]:
            self.assertIn(key, record)
        self.assertEqual(record["benchmark"], "SPY")
        self.assertEqual(record["objective"], "sortino")


if __name__ == "__main__":
    unittest.main()
