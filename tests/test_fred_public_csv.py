import unittest

import pandas as pd

import quant_stockpicker_core as core


class FredPublicCsvTests(unittest.TestCase):
    def test_fetch_fred_series_frame_parses_public_csv(self):
        original_read = core.http_read_text

        def fake_read(url, user_agent="QuantStockPicker/1.0", timeout=20):
            self.assertIn("id=DGS10", url)
            self.assertIn("cosd=2026-01-01", url)
            self.assertIn("coed=2026-01-05", url)
            self.assertEqual(timeout, 7)
            return "observation_date,DGS10\n2026-01-01,4.10\n2026-01-02,.\n2026-01-05,4.20\n"

        try:
            core.http_read_text = fake_read
            out = core.fetch_fred_series_frame("DGS10", "2026-01-01", "2026-01-05", timeout=7)
        finally:
            core.http_read_text = original_read

        self.assertEqual(list(out.columns), ["DGS10"])
        self.assertEqual(out.index.name, "Date")
        self.assertAlmostEqual(float(out.loc[pd.Timestamp("2026-01-01"), "DGS10"]), 4.10)
        self.assertTrue(pd.isna(out.loc[pd.Timestamp("2026-01-02"), "DGS10"]))

    def test_fred_panel_keeps_successful_series_when_one_fails(self):
        original_fetch = core.fetch_fred_series_frame
        dates = pd.date_range("2026-01-01", periods=2, freq="D")

        def fake_fetch(code, start, end, timeout=12):
            if code == "BAD":
                raise TimeoutError("public endpoint timeout")
            return pd.DataFrame({code: [1.0, 2.0]}, index=dates)

        try:
            core.fetch_fred_series_frame = fake_fetch
            out = core._fetch_fred_panel(
                {"GOOD": "Good_Series", "BAD": "Bad_Series"},
                "2026-01-01",
                "2026-01-02",
                max_workers=2,
            )
        finally:
            core.fetch_fred_series_frame = original_fetch

        self.assertEqual(list(out.columns), ["Good_Series"])
        self.assertEqual(len(out), 2)


if __name__ == "__main__":
    unittest.main()
