import unittest
from unittest.mock import patch

import pandas as pd

import quant_stockpicker_core as core


class GlobalRateHistoryTests(unittest.TestCase):
    def test_daily_country_rates_are_not_truncated_to_one_year(self):
        idx = pd.bdate_range("2023-01-01", "2026-05-01")
        prices = pd.DataFrame({"SPY": range(len(idx))}, index=idx)

        def fake_discrete(start, end, country, use_cache=True, cache_ttl_hours=24):
            dates = pd.bdate_range(start, end)
            return pd.DataFrame(
                {
                    "SOV_10Y": 4.0,
                    "SOV_2Y": 3.5,
                    "POLICY_RATE": 3.0,
                    "Country_Rate_Source": "test",
                },
                index=dates,
            )

        with patch.object(core, "fetch_discrete_country_rate_frame", side_effect=fake_discrete):
            hist = core.global_yield_curve_discrete_history(
                prices,
                countries=["United States"],
                lookback_days=365 * 3,
                use_cache=False,
            )

        ten_y = hist[hist["Tenor_Code"].eq("SOV_10Y")]
        self.assertGreater(len(ten_y), 700)
        self.assertLessEqual(ten_y["Observation_Date"].min(), pd.Timestamp("2023-06-01"))


if __name__ == "__main__":
    unittest.main()
