import unittest
from unittest.mock import patch

import pandas as pd

import quant_stockpicker_core as core

TREASURY_CSV = """Date,"1 Mo","3 Mo","6 Mo","1 Yr","2 Yr","3 Yr","5 Yr","7 Yr","10 Yr","20 Yr","30 Yr"
06/05/2026,3.71,3.78,3.81,3.88,4.17,4.22,4.29,4.41,4.55,5.03,5.01
06/04/2026,3.71,3.78,3.78,3.82,4.05,4.10,4.18,4.32,4.47,4.98,4.97
"""


class USTreasuryCurveTests(unittest.TestCase):
    def test_official_curve_parses_all_required_tenors(self):
        with patch.object(core, "http_read_text", return_value=TREASURY_CSV) as read:
            out = core.fetch_us_treasury_yield_curve(
                "2026-06-04",
                "2026-06-05",
                use_cache=False,
                timeout=9,
            )

        self.assertEqual(read.call_count, 1)
        self.assertIn("daily-treasury-rates.csv/2026/all", read.call_args.args[0])
        self.assertEqual(read.call_args.kwargs["timeout"], 9)
        self.assertEqual(len(out), 2)
        self.assertAlmostEqual(float(out.loc[pd.Timestamp("2026-06-05"), "US2Y"]), 4.17)
        self.assertAlmostEqual(float(out.loc[pd.Timestamp("2026-06-05"), "US10Y"]), 4.55)
        self.assertAlmostEqual(float(out.loc[pd.Timestamp("2026-06-05"), "SOV_2Y"]), 4.17)
        self.assertAlmostEqual(float(out.loc[pd.Timestamp("2026-06-05"), "SOV_10Y"]), 4.55)
        self.assertEqual(
            out.loc[pd.Timestamp("2026-06-05"), "Country_Rate_Source"],
            "U.S. Treasury daily par yield curve",
        )

    def test_official_curve_filters_dates_after_combining_years(self):
        with patch.object(core, "http_read_text", return_value=TREASURY_CSV):
            out = core.fetch_us_treasury_yield_curve(
                "2026-06-05",
                "2026-06-05",
                use_cache=False,
            )
        self.assertEqual(list(out.index), [pd.Timestamp("2026-06-05")])

    def test_direct_us_rates_use_official_treasury_source(self):
        expected = pd.DataFrame(
            {
                "US2Y": [4.17],
                "US10Y": [4.55],
                "SOV_2Y": [4.17],
                "SOV_10Y": [4.55],
                "Country_Rate_Source": ["U.S. Treasury daily par yield curve"],
            },
            index=pd.DatetimeIndex(["2026-06-05"], name="Date"),
        )
        with patch.object(core, "fetch_us_treasury_yield_curve", return_value=expected):
            out = core.fetch_direct_country_rates(
                "2026-06-04",
                "2026-06-05",
                "United States",
                use_cache=False,
            )
        self.assertAlmostEqual(float(out.iloc[-1]["SOV_10Y"]), 4.55)
        self.assertEqual(out.iloc[-1]["Country_Rate_Source"], "U.S. Treasury daily par yield curve")


if __name__ == "__main__":
    unittest.main()
