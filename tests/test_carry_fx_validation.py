import unittest

import numpy as np
import pandas as pd

import quant_stockpicker_core as core


class CarryFxValidationTests(unittest.TestCase):
    def test_carry_validation_adds_fx_risk_adjusted_score(self):
        dates = pd.date_range("2025-01-01", periods=120, freq="B")
        fx = pd.DataFrame(
            {
                "MXN": np.linspace(0.050, 0.055, len(dates)),
                "USD": 1.0,
            },
            index=dates,
        )
        carry = pd.DataFrame(
            {
                "Long_Currency": ["MXN"],
                "Short_Currency": ["USD"],
                "Carry_10Y_Spread": [6.0],
                "Carry_Trade_Score": [5.0],
                "Long_Curve_10Y_2Y": [0.5],
                "Event_Risk_Penalty": [0.0],
            }
        )

        out = core.validate_carry_trade_strategies(carry, fx_usd_values=fx, use_cache=False)

        self.assertIn("FX_Risk_Adjusted_Carry_Score", out.columns)
        self.assertIn("FX_Ann_Vol_Pct", out.columns)
        self.assertEqual(out.loc[0, "FX_Data_Status"], "ok_yahoo_spot_proxy")
        self.assertGreater(out.loc[0, "FX_Risk_Adjusted_Carry_Score"], 0)
        self.assertIn("not arbitrage", out.loc[0, "Mathematical_Admissibility"])

    def test_carry_suggestion_note_points_to_validation(self):
        rates = pd.DataFrame(
            {
                "Country": ["Mexico", "United States"],
                "Yield_10Y": [9.0, 4.0],
                "Curve_10Y_2Y": [0.5, 0.1],
                "Term_Premium_Proxy": [1.0, 0.0],
                "Regime_Hawkish_Dovish": ["Hawkish", "Dovish"],
                "Regime_Bull_Bear": ["Bull", "Bull"],
            }
        )
        out = core.carry_trade_suggestions(rates)
        self.assertFalse(out.empty)
        self.assertTrue(out["Risk_Note"].str.contains("FX-risk-adjusted validation").any())


if __name__ == "__main__":
    unittest.main()
