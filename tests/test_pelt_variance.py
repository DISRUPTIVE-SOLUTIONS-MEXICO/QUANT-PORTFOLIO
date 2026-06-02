import unittest

import numpy as np
import pandas as pd

import quant_stockpicker_core as core


class PeltVarianceTests(unittest.TestCase):
    def test_pelt_detects_variance_regime_shift(self):
        rng = np.random.default_rng(20260518)
        idx = pd.date_range("2024-01-01", periods=160, freq="B")
        r = pd.Series(
            np.r_[
                rng.normal(0.0010, 0.004, 80),
                rng.normal(-0.0010, 0.030, 80),
            ],
            index=idx,
        )
        out = core.pelt_change_point_analysis(r, min_size=20, penalty=10.0)
        segments = out["pelt_regime_segments"]
        changes = out["pelt_change_points"]
        timeline = out["pelt_timeline"]

        self.assertGreaterEqual(len(segments), 2)
        self.assertFalse(changes.empty)
        self.assertIn("Variance up", set(changes["Change_Type"]))
        self.assertTrue(timeline["Is_Change_Point"].any())
        self.assertGreater(segments["Ann_Vol"].max(), segments["Ann_Vol"].min() * 3.0)

    def test_pelt_requires_enough_history(self):
        r = pd.Series([0.001, -0.002, 0.003] * 10)
        out = core.pelt_change_point_analysis(r, min_size=20)
        self.assertTrue(out["pelt_regime_segments"].empty)
        self.assertTrue(out["pelt_change_points"].empty)
        self.assertTrue(out["pelt_timeline"].empty)

    def test_variance_architecture_includes_student_t_garch(self):
        rng = np.random.default_rng(20260519)
        idx = pd.date_range("2024-01-01", periods=260, freq="B")
        r = pd.Series(0.012 * rng.standard_t(df=5, size=len(idx)), index=idx)
        out = core.fit_variance_architecture(r)
        self.assertFalse(out.empty)
        self.assertIn("StudentT_GARCH11", out["Model"].astype(str).tolist())
        self.assertIn("AIC", out.columns)
        self.assertIn("BIC", out.columns)


if __name__ == "__main__":
    unittest.main()
