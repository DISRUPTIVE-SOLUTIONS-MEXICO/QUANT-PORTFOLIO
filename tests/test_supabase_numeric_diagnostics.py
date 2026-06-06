import math
import unittest
from pathlib import Path

import numpy as np

from supabase_store import _finite_float, _risk_diagnostic_rows


class SupabaseNumericDiagnosticTests(unittest.TestCase):
    def test_finite_float_rejects_text_metadata_and_non_finite_values(self):
        self.assertIsNone(_finite_float(Path("artifact.json")))
        self.assertIsNone(_finite_float("RESEARCH_SNAPSHOT_NOT_PROMOTED"))
        self.assertIsNone(_finite_float(float("nan")))
        self.assertIsNone(_finite_float(np.inf))

    def test_finite_float_accepts_numeric_scalars(self):
        self.assertEqual(_finite_float(True), 1.0)
        self.assertEqual(_finite_float("0.25"), 0.25)
        self.assertTrue(math.isclose(_finite_float(np.float64(-1.5)), -1.5))

    def test_risk_rows_exclude_artifact_paths_and_status_strings(self):
        rows = _risk_diagnostic_rows(
            "run-1",
            [
                {"Metric": "Annualized_Return", "Value": 0.12},
                {"Metric": "artifact_local_path", "Value": r"C:\cache\artifact.json"},
                {"Metric": "promotion_gate_status", "Value": "RESEARCH_ONLY"},
            ],
        )
        self.assertEqual(
            rows,
            [{"run_id": "run-1", "metric": "Annualized_Return", "value": 0.12}],
        )


if __name__ == "__main__":
    unittest.main()
