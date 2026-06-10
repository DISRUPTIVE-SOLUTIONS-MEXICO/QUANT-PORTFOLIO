import unittest

import pandas as pd

from quant_stockpicker_core import (
    RunConfig,
    kaizen_contextual_bandit_diagnostics,
    kaizen_promotion_gate,
    kaizen_reward_series,
    merge_sec_nlp_into_panel,
    sec_text_features,
)


class SecNlpKaizenTests(unittest.TestCase):
    def test_sec_text_features_detect_risk_and_deterioration(self):
        prev = "Management discusses automation and normal liquidity."
        text = "Risk factors include litigation, litigation, impairment, debt covenant default and going concern."
        features = sec_text_features(text, previous_text=prev)
        self.assertGreater(features["SEC_Text_litigation_Mentions"], 0)
        self.assertGreater(features["SEC_Text_impairment_Mentions"], 0)
        self.assertGreater(features["TextRisk_Score"], 0)
        self.assertGreaterEqual(features["SEC_Text_Risk_Deterioration"], 0)

    def test_sec_nlp_merge_is_point_in_time_by_availability(self):
        panel = pd.DataFrame(
            {
                "Ticker": ["AAA", "AAA"],
                "Availability_Date": [pd.Timestamp("2025-03-31"), pd.Timestamp("2025-09-30")],
                "ROE": [0.1, 0.2],
            }
        )
        nlp = pd.DataFrame(
            {
                "Ticker": ["AAA", "AAA"],
                "Availability_Date": [pd.Timestamp("2025-04-15"), pd.Timestamp("2025-08-15")],
                "TextRisk_Score": [10.0, 1.0],
            }
        )
        merged = merge_sec_nlp_into_panel(panel, nlp)
        self.assertTrue(pd.isna(merged.loc[0, "TextRisk_Score"]))
        self.assertEqual(float(merged.loc[1, "TextRisk_Score"]), 1.0)

    def test_kaizen_reward_penalizes_drawdown_turnover_and_cost(self):
        perf = pd.DataFrame(
            {
                "Signal_Date": pd.date_range("2025-01-31", periods=4, freq="ME"),
                "OOS_End": pd.date_range("2025-02-28", periods=4, freq="ME"),
                "Net_Return": [0.02, -0.08, 0.01, -0.05],
                "Turnover": [0.10, 0.80, 0.20, 0.60],
                "Transaction_Cost": [0.001, 0.004, 0.001, 0.003],
            }
        )
        cfg = RunConfig(tickers=("AAA", "BBB", "SPY"), investor_max_drawdown=0.05)
        rewards = kaizen_reward_series(perf, cfg, action_id="sortino_balanced")
        self.assertEqual(len(rewards), 4)
        self.assertLess(float(rewards.loc[1, "Reward"]), float(rewards.loc[0, "Reward"]))
        self.assertGreaterEqual(float(rewards["Suitability_Breach"].sum()), 1.0)

    def test_promotion_gate_and_bandit_return_candidate_actions(self):
        cfg = RunConfig(tickers=("AAA", "BBB", "SPY"), investor_max_drawdown=0.20)
        perf = pd.DataFrame(
            {
                "Signal_Date": pd.date_range("2025-01-31", periods=8, freq="ME"),
                "OOS_End": pd.date_range("2025-02-28", periods=8, freq="ME"),
                "Net_Return": [0.01, 0.02, -0.01, 0.015, 0.005, 0.01, -0.005, 0.012],
                "Turnover": [0.2] * 8,
                "Transaction_Cost": [0.001] * 8,
                "Model_Confidence": [0.5] * 8,
            }
        )
        validation = {
            "summary": pd.DataFrame(
                [
                    {"Metric": "Deflated_Sortino", "Value": 0.5},
                    {"Metric": "CPCV_PBO", "Value": 0.1},
                    {"Metric": "Hansen_SPA_PValue", "Value": 0.05},
                    {"Metric": "Mean_IC", "Value": 0.03},
                ]
            )
        }
        perf_summary = pd.DataFrame([{"Metric": "Max_Drawdown", "Value": -0.03}])
        gate = kaizen_promotion_gate(perf_summary, validation, cfg)
        self.assertTrue(bool(gate.iloc[0]["Promotion_Gate_Passed"]))
        diag = kaizen_contextual_bandit_diagnostics(
            cfg, pd.Series({"hawkish_score": 0.2, "bullish_score": 0.6}), perf, validation, perf_summary
        )
        self.assertFalse(diag["actions"].empty)
        self.assertIn("Recommended", diag["actions"].columns)
        self.assertEqual(int(diag["actions"]["Recommended"].sum()), 1)


if __name__ == "__main__":
    unittest.main()
