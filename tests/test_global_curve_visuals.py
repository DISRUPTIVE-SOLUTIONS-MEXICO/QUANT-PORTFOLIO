import unittest

import pandas as pd

from quant_dashboard_utils import (
    balanced_rate_history_sample,
    country_flag,
    latest_rate_observations,
    prepare_discrete_rate_plot_data,
    prepare_global_curve_matrix,
    spread_label_positions,
)


class GlobalCurveVisualTests(unittest.TestCase):
    def test_prepare_global_curve_matrix_sorts_and_renames(self):
        curves = pd.DataFrame(
            {
                "Country": ["A", "B"],
                "Policy_Rate": [1.0, 2.0],
                "Yield_2Y": [1.5, 2.5],
                "Yield_10Y": [3.0, 2.0],
                "Curve_10Y_2Y": [1.5, -0.5],
                "Term_Premium_Proxy": [2.0, 0.0],
            }
        )
        matrix = prepare_global_curve_matrix(curves)
        self.assertEqual(matrix.index.tolist(), ["A", "B"])
        self.assertEqual(matrix.columns.tolist(), ["Policy", "2Y", "10Y", "10Y-2Y", "10Y-Policy"])
        self.assertAlmostEqual(matrix.loc["B", "10Y-2Y"], -0.5)

    def test_discrete_rate_plot_uses_calendar_window_not_tail_count(self):
        daily_dates = pd.bdate_range("2023-01-01", "2026-01-01")
        monthly_dates = pd.date_range("2023-01-01", "2026-01-01", freq="MS")
        history = pd.concat(
            [
                pd.DataFrame(
                    {
                        "Country": "DailyLand",
                        "Observation_Date": daily_dates,
                        "Tenor_Code": "SOV_10Y",
                        "Rate": 4.0,
                    }
                ),
                pd.DataFrame(
                    {
                        "Country": "MonthlyLand",
                        "Observation_Date": monthly_dates,
                        "Tenor_Code": "SOV_10Y",
                        "Rate": 8.0,
                    }
                ),
            ],
            ignore_index=True,
        )
        out = prepare_discrete_rate_plot_data(history, "SOV_10Y", max_countries=2, lookback_days=365 * 3, normalize_frequency="native")
        daily = out[out["Country"].eq("DailyLand")]
        monthly = out[out["Country"].eq("MonthlyLand")]

        self.assertLessEqual(daily["Observation_Date"].min(), pd.Timestamp("2023-02-01"))
        self.assertLessEqual(monthly["Observation_Date"].min(), pd.Timestamp("2023-02-01"))
        self.assertGreater(len(daily), 80)

    def test_discrete_rate_plot_can_resample_to_monthly_comparable_view(self):
        daily_dates = pd.bdate_range("2025-01-01", "2025-06-30")
        history = pd.DataFrame(
            {
                "Country": "DailyLand",
                "Observation_Date": daily_dates,
                "Tenor_Code": "SOV_10Y",
                "Rate": range(len(daily_dates)),
                "Observation_Frequency": "Daily/business-day discrete",
                "Source": "test",
            }
        )
        out = prepare_discrete_rate_plot_data(history, "SOV_10Y", max_countries=1, lookback_days=365, normalize_frequency="month_end")
        self.assertLessEqual(len(out), 6)
        self.assertTrue(out["Observation_Frequency"].eq("Monthly comparable view").all())
        self.assertTrue(out["Native_Observation_Frequency"].eq("Daily/business-day discrete").all())

    def test_slope_labels_use_flags_and_are_spread_apart(self):
        self.assertEqual(country_flag("United States"), "🇺🇸")
        positions = spread_label_positions([4.4, 4.42, 4.43], min_gap=0.20)
        self.assertGreaterEqual(positions[1] - positions[0], 0.19)
        self.assertGreaterEqual(positions[2] - positions[1], 0.19)

    def test_rate_tables_are_balanced_not_plain_tail(self):
        rows = []
        for country in ["A", "United States"]:
            for tenor in ["SOV_2Y", "SOV_10Y"]:
                for i in range(4):
                    rows.append(
                        {
                            "Country": country,
                            "Observation_Date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
                            "Tenor_Code": tenor,
                            "Rate": i,
                        }
                    )
        hist = pd.DataFrame(rows)
        latest = latest_rate_observations(hist)
        sample = balanced_rate_history_sample(hist, rows_per_group=2)

        self.assertEqual(len(latest), 4)
        self.assertEqual(set(latest["Country"]), {"A", "United States"})
        self.assertEqual(sample.groupby(["Country", "Tenor_Code"]).size().max(), 2)


if __name__ == "__main__":
    unittest.main()
