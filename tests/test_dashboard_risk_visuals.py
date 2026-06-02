import unittest

import pandas as pd

from quant_dashboard_utils import build_drawdown_frame


class DashboardRiskVisualTests(unittest.TestCase):
    def test_drawdown_falls_back_to_perf_when_equity_curve_is_flat(self):
        dates = pd.date_range("2026-01-31", periods=4, freq="ME")
        curve = pd.DataFrame({"Period_End": dates, "Portfolio_Equity": [1.0, 1.0, 1.0, 1.0]})
        perf = pd.DataFrame({"Period_End": dates, "Net_Return": [0.05, -0.10, 0.02, -0.03]})

        out = build_drawdown_frame(curve, perf)

        self.assertFalse(out.empty)
        self.assertLess(out["Drawdown"].min(), 0.0)
        self.assertAlmostEqual(out["Portfolio_Equity"].iloc[1], 0.945)


if __name__ == "__main__":
    unittest.main()
