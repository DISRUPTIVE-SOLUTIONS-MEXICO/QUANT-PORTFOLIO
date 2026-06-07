import unittest

import numpy as np
import pandas as pd

from quant_core.backtest_paths import build_backtest_path_bundle, price_drawdown_frame
from quant_core.dashboard_payload import build_dashboard_payload
from quant_core.data_freshness import build_data_freshness_report
from quant_core.pit_confidence import add_pit_confidence
from quant_core.promotion_gate import evaluate_promotion_gate
from quant_core.suitability_gate import evaluate_suitability_gate
from quant_stockpicker_core import RunConfig


class CoreContractsTests(unittest.TestCase):
    def test_daily_nav_and_drawdown_contract(self):
        dates = pd.bdate_range("2025-01-01", periods=8)
        prices = pd.DataFrame(
            {
                "AAA": [100, 102, 101, 99, 103, 104, 102, 105],
                "SPY": [500, 505, 503, 501, 507, 509, 506, 512],
            },
            index=dates,
        )
        perf = pd.DataFrame(
            {
                "Rebalance_Date": [dates[0]],
                "OOS_Start": [dates[0]],
                "OOS_End": [dates[-1]],
                "Fixed_TC": [0.0],
                "Impact_TC": [0.0],
            }
        )
        holdings = pd.DataFrame({"Rebalance_Date": [dates[0]], "Ticker": ["AAA"], "Effective_Weight": [1.0]})
        bundle = build_backtest_path_bundle(perf, holdings, prices, "SPY")
        self.assertIn("price_paths", bundle)
        self.assertFalse(bundle["price_paths"].empty)
        dd = bundle["drawdowns"]
        self.assertFalse(dd.empty)
        self.assertLess(pd.to_numeric(dd["Sortino optimized synthetic NAV price"], errors="coerce").min(), 0.0)

    def test_suitability_gate_blocks_excess_risk(self):
        cfg = RunConfig(tickers=("AAA", "SPY"), target_vol=0.10, investor_max_drawdown=0.08, investor_cvar_max_daily=0.015)
        portfolio = pd.DataFrame({"Ticker": ["AAA"], "Weight": [0.20], "realized_weight_ann_vol": [0.25], "hist_cvar_95_daily": [0.02], "realized_weight_max_drawdown": [-0.15]})
        perf_summary = pd.DataFrame([{"Metric": "Annualized_Vol", "Value": 0.25}, {"Metric": "Max_Drawdown", "Value": -0.15}])
        gate = evaluate_suitability_gate(cfg, portfolio, perf_summary)
        self.assertEqual(gate["status"], "blocked")
        self.assertGreaterEqual(len(gate["breaches"]), 2)

    def test_promotion_gate_rejects_high_pbo(self):
        cfg = RunConfig(tickers=("AAA", "SPY"), weight_objective="sortino")
        validation = {
            "summary": pd.DataFrame(
                [
                    {"Metric": "Deflated_Sortino", "Value": 0.8},
                    {"Metric": "CPCV_PBO", "Value": 0.9},
                    {"Metric": "Hansen_SPA_PValue", "Value": 0.05},
                    {"Metric": "ICIR", "Value": 0.2},
                ]
            )
        }
        gate = evaluate_promotion_gate(pd.DataFrame(), validation, cfg)
        self.assertEqual(gate["promotion_status"], "rejected")
        self.assertIn("PBO", gate["failed_tests"]["Test"].tolist())

    def test_pit_confidence_penalizes_missing_ratio(self):
        panel = pd.DataFrame(
            {
                "Ticker": ["AAA"],
                "Availability_Date": [pd.Timestamp("2024-01-01")],
                "SEC_Facts_Coverage": [0],
                "Valid_Fundamental_Ratios": [2],
                "ROIC": [np.nan],
                "EV_EBITDA": [8.0],
            }
        )
        out = add_pit_confidence(panel, asof_date=pd.Timestamp("2025-01-01"))
        self.assertLess(float(out.loc[0, "PIT_Confidence"]), 0.5)
        self.assertEqual(float(out.loc[0, "ROIC_PIT_Confidence"]), 0.0)
        self.assertGreater(float(out.loc[0, "EV_EBITDA_PIT_Confidence"]), 0.0)

    def test_dashboard_payload_schema(self):
        results = {"portfolio": pd.DataFrame({"Ticker": ["AAA"], "Weight": [1.0]}), "validation_diagnostics": {"summary": pd.DataFrame()}}
        path_bundle = {"price_paths": pd.DataFrame({"Date": pd.date_range("2025-01-01", periods=2), "P": [1.0, 1.1]}), "drawdowns": pd.DataFrame(), "max_drawdown_table": pd.DataFrame()}
        suitability = {"summary": pd.DataFrame([{"Gate_Status": "approved"}]), "breaches": pd.DataFrame(), "user_safe_summary": "ok"}
        promotion = {"summary": pd.DataFrame([{"Promotion_Status": "watchlist"}]), "tests": pd.DataFrame()}
        payload = build_dashboard_payload(results, path_bundle, suitability, promotion)
        self.assertIn("status", payload)
        self.assertIn("allocation", payload)
        self.assertIn("charts", payload)
        self.assertIn("tables", payload)
        self.assertIn("research", payload)
        self.assertIn("diagnostics", payload)
        self.assertEqual(payload["contract"]["schema_version"], "2026.06.07-full-research-v4")

    def test_data_freshness_report_flags_stale_sources(self):
        now = pd.Timestamp.utcnow()
        inv = pd.DataFrame(
            {
                "Namespace": ["prices_daily", "fundamentals_sec_companyfacts"],
                "Rows": [100, 10],
                "Created_At": [now - pd.Timedelta(hours=2), now - pd.Timedelta(hours=300)],
                "Age_Hours": [2.0, 300.0],
                "Key": ["a", "b"],
            }
        )
        out = build_data_freshness_report(inv)
        self.assertIn("Last_Update_Central", out.columns)
        self.assertIn("TTL_Hours", out.columns)
        sec = out[out["Namespace"].eq("fundamentals_sec_companyfacts")].iloc[0]
        self.assertIn(sec["Status"], {"stale", "expired"})


if __name__ == "__main__":
    unittest.main()
