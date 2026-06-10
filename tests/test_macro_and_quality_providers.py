"""Network-free tests for macro providers, governed scraping and data quality."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd

import quant_core.data.macro_global as macro_global
import quant_core.data.macro_mx as macro_mx
from quant_core.data.ocr import validate_ocr_frame
from quant_core.data.quality import detect_price_anomalies
from quant_core.data.scraping import GovernedScraper, parse_banxico_policy_announcements

BANXICO_JSON = {
    "bmx": {
        "series": [
            {
                "idSerie": "SF61745",
                "datos": [
                    {"fecha": "02/01/2026", "dato": "11.25"},
                    {"fecha": "03/01/2026", "dato": "11.25"},
                    {"fecha": "06/01/2026", "dato": "11.00"},
                ],
            }
        ]
    }
}

INEGI_JSON = {
    "Series": [
        {
            "INDICADOR": "737121",
            "OBSERVATIONS": [
                {"TIME_PERIOD": "2026/01", "OBS_VALUE": "102.5"},
                {"TIME_PERIOD": "2026/02", "OBS_VALUE": "103.1"},
            ],
        }
    ]
}

WORLDBANK_JSON = [
    {"page": 1},
    [
        {"date": "2024", "value": 1.789},
        {"date": "2023", "value": 1.654},
        {"date": "2022", "value": None},
    ],
]

BOE_CSV = "DATE,IUDBEDR\n02 Jan 2026,5.25\n03 Jan 2026,5.25\n06 Jan 2026,5.00\n"

BANXICO_POLICY_HTML = """
<html><body><table>
<tr><th>Fecha</th><th>Decisión de política monetaria</th><th>Tasa objetivo</th></tr>
<tr><td>06/02/2026</td><td>Recorte de 25 pb</td><td>11.00%</td></tr>
<tr><td>19/03/2026</td><td>Sin cambio</td><td>11.00%</td></tr>
</table></body></html>
"""


def test_banxico_series_parses_catalog_fixture():
    with patch.object(macro_mx, "http_read_json", return_value=BANXICO_JSON):
        out = macro_mx.fetch_banxico_series(
            {"SF61745": "POLICY_RATE"}, "2026-01-01", "2026-01-31", token="fixture", use_cache=False
        )
    assert list(out.columns) == ["POLICY_RATE"]
    assert len(out) == 3
    assert float(out["POLICY_RATE"].iloc[-1]) == 11.00


def test_banxico_degrades_honestly_without_token():
    out = macro_mx.fetch_banxico_series({"SF61745": "POLICY_RATE"}, "2026-01-01", "2026-01-31", token="")
    assert out.empty


def test_inegi_series_parses_bie_fixture():
    with patch.object(macro_mx, "http_read_json", return_value=INEGI_JSON):
        out = macro_mx.fetch_inegi_series({"737121": "IGAE"}, token="fixture", use_cache=False)
    assert "IGAE" in out.columns
    assert len(out) == 2
    assert float(out["IGAE"].iloc[-1]) == 103.1


def test_worldbank_indicator_parses_fixture():
    with patch.object(macro_global, "http_read_json", return_value=WORLDBANK_JSON):
        out = macro_global.fetch_worldbank_indicator("MEX", "FP.CPI.TOTL.ZG", 2022, 2024)
    assert "FP.CPI.TOTL.ZG" in out.columns
    assert len(out) == 3
    assert out.index.is_monotonic_increasing


def test_boe_iadb_parses_csv_fixture():
    with patch.object(macro_global, "http_read_text", return_value=BOE_CSV):
        out = macro_global.fetch_boe_iadb_series({"IUDBEDR": "BANK_RATE"}, "2026-01-01", "2026-01-31")
    assert list(out.columns) == ["BANK_RATE"]
    assert len(out) == 3
    assert float(out["BANK_RATE"].iloc[-1]) == 5.00


def test_governed_scraper_respects_robots_disallow():
    scraper = GovernedScraper(min_interval_seconds=0.0, max_retries=1)
    robots = "User-agent: *\nDisallow: /private/\n"
    with patch("quant_core.data.scraping.http_read_text", return_value=robots):
        allowed = scraper._robots_allows("https://example.org/public/page")
    # Cached parser is reused; the disallowed path must be rejected.
    assert allowed is True
    assert scraper._robots_allows("https://example.org/private/secret") is False


def test_governed_scraper_snapshots_content():
    scraper = GovernedScraper(min_interval_seconds=0.0, max_retries=1, respect_robots=False)
    html = "<html><body>ok</body></html>"
    with (
        patch("quant_core.data.scraping.http_read_text", return_value=html),
        patch("quant_core.data.scraping.PERSISTENT_CACHE") as cache,
    ):
        content, provenance = scraper.fetch("https://example.org/data")
    assert content == html
    assert provenance.content_hash
    assert cache.set_df.called


def test_parse_banxico_policy_announcements_fixture():
    out = parse_banxico_policy_announcements(BANXICO_POLICY_HTML)
    assert len(out) == 2
    assert "Decision" in out.columns
    assert float(out["Rate"].iloc[0]) == 11.00
    assert out["Date"].is_monotonic_increasing


def test_validate_ocr_frame_blocks_implausible_data():
    good = pd.DataFrame({"Date": ["2026-01-01", "2026-02-01"], "Rate": [11.0, 10.75]})
    passed, violations = validate_ocr_frame(good, numeric_columns={"Rate": (0.0, 30.0)}, date_column="Date")
    assert passed and not violations

    bad = pd.DataFrame({"Date": ["2026-02-01", "2026-01-01"], "Rate": [110.0, -5.0]})
    passed, violations = validate_ocr_frame(bad, numeric_columns={"Rate": (0.0, 30.0)}, date_column="Date")
    assert not passed
    assert any(v.startswith("out_of_range") for v in violations)
    assert any(v.startswith("non_monotonic_dates") for v in violations)

    assert validate_ocr_frame(pd.DataFrame()) == (False, ["empty_frame"])


def test_detect_price_anomalies_flags_jumps_stale_and_gaps():
    idx = pd.bdate_range("2024-01-02", periods=120)
    rng = np.random.default_rng(11)
    normal = 100 * (1 + pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)).cumprod()
    jumpy = normal.copy()
    jumpy.iloc[60] = jumpy.iloc[59] * 1.80  # 80% single-day jump
    stale = pd.Series(100.0, index=idx)
    prices = pd.DataFrame({"NORMAL": normal, "JUMPY": jumpy, "STALE": stale})
    out = detect_price_anomalies(prices).set_index("Ticker")
    assert bool(out.loc["JUMPY", "Jump_Flag"])
    assert bool(out.loc["STALE", "Stale_Flag"])
    assert not bool(out.loc["NORMAL", "Anomaly_Flag"])
