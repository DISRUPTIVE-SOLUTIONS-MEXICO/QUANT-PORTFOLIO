import unittest

from quant_stockpicker_core import build_suitability_constraints


class SuitabilityEngineTests(unittest.TestCase):
    def test_conservative_profile_caps_risk(self):
        out = build_suitability_constraints(
            horizon_years=1.0,
            initial_capital=10_000.0,
            monthly_contribution=100.0,
            liquidity_need="Alta",
            max_drawdown=0.08,
            risk_aversion_score=9.0,
            investor_objective="Preservacion de capital",
        )
        self.assertEqual(out["Suitability_Profile"], "Conservador")
        self.assertLessEqual(out["Target_Vol"], 0.10)
        self.assertLessEqual(out["Max_Weight"], 0.12)
        self.assertLessEqual(out["Sector_Weight_Cap"], 0.25)
        self.assertLessEqual(out["Top_N_Max"], out["N_Capital_Max"])

    def test_incoherent_aggressive_short_horizon_blocks(self):
        out = build_suitability_constraints(
            horizon_years=0.5,
            initial_capital=50_000.0,
            monthly_contribution=0.0,
            liquidity_need="Media",
            max_drawdown=0.06,
            risk_aversion_score=9.0,
            investor_objective="Alta conviccion",
        )
        self.assertTrue(out["Hard_Block"])
        self.assertGreaterEqual(len(out["Warnings"]), 1)

    def test_long_horizon_low_aversion_allows_more_capacity(self):
        out = build_suitability_constraints(
            horizon_years=10.0,
            initial_capital=250_000.0,
            monthly_contribution=2_000.0,
            liquidity_need="Baja",
            max_drawdown=0.45,
            risk_aversion_score=1.0,
            investor_objective="Crecimiento agresivo",
        )
        self.assertIn(out["Suitability_Profile"], {"Agresivo", "Especulativo"})
        self.assertGreaterEqual(out["Top_N_Max"], 12)
        self.assertFalse(out["Hard_Block"])


if __name__ == "__main__":
    unittest.main()
