"""Network-free tests for the zero-cost multi-source data layer."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd

import quant_core.data.prices as prices_mod
import quant_core.data.universe as universe_mod
import quant_stockpicker_core as core
from quant_core.data.prices import parse_stooq_csv, stooq_symbol
from quant_core.data.reconcile import reconcile_price_frames
from quant_core.data.universe import build_delisted_registry, parse_sp500_constituents_html

STOOQ_CSV = """Date,Open,High,Low,Close,Volume
2024-01-02,100.0,101.0,99.0,100.5,1000000
2024-01-03,100.5,102.0,100.0,101.5,1100000
2024-01-04,101.5,103.0,101.0,102.5,1200000
2024-01-05,102.5,104.0,102.0,103.5,1300000
2024-01-08,103.5,105.0,103.0,104.5,1400000
"""

WIKI_SNAPSHOT_HTML = """
<html><body><table>
<tr><th>Symbol</th><th>Security</th><th>GICS Sector</th></tr>
<tr><td>AAA</td><td>Alpha Corp</td><td>Technology</td></tr>
<tr><td>BBB</td><td>Beta Inc</td><td>Healthcare</td></tr>
<tr><td>BRK.B</td><td>Berkshire</td><td>Financials</td></tr>
</table></body></html>
"""


def test_stooq_symbol_mapping_handles_class_shares():
    assert stooq_symbol("AAPL") == "aapl.us"
    assert stooq_symbol("BRK-B") == "brk-b.us"
    assert stooq_symbol("brk.b") == "brk-b.us"


def test_parse_stooq_csv_returns_close_frame():
    frame = parse_stooq_csv(STOOQ_CSV, "TEST")
    assert list(frame.columns) == ["TEST"]
    assert len(frame) == 5
    assert float(frame.iloc[-1, 0]) == 104.5
    assert frame.index.is_monotonic_increasing


def test_parse_stooq_csv_rejects_garbage():
    assert parse_stooq_csv("No data", "X").empty
    assert parse_stooq_csv("", "X").empty


def test_fetch_stooq_prices_uses_http_reader_and_provenance():
    with patch.object(prices_mod, "http_read_text", return_value=STOOQ_CSV) as reader:
        frame, provenance = prices_mod.fetch_stooq_prices(["AAA", "BBB"], use_cache=False)
    assert reader.call_count == 2
    assert sorted(frame.columns) == ["AAA", "BBB"]
    assert len(provenance) == 2
    assert all(p.source == "stooq" and p.status == "ok" for p in provenance)
    assert all(p.content_hash for p in provenance)


def test_reconcile_flags_divergent_ticker_and_fills_gaps():
    idx = pd.bdate_range("2024-01-02", periods=60)
    rng = np.random.default_rng(3)
    base = 100 * (1 + pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)).cumprod()
    primary = pd.DataFrame({"GOOD": base, "BAD": base})
    primary.loc[idx[-10] :, "GOOD"] = np.nan  # gap to be filled
    secondary = pd.DataFrame({"GOOD": base, "BAD": base * 1.02})  # 200 bps off
    consensus, report = reconcile_price_frames(primary, secondary, tolerance_bps=50.0)
    rep = report.set_index("Ticker")
    assert bool(rep.loc["BAD", "Price_Quality_Warning"])
    assert not bool(rep.loc["GOOD", "Price_Quality_Warning"])
    # The gap in GOOD was filled from the secondary source.
    assert consensus["GOOD"].notna().all()
    # BAD keeps the primary values (auditable choice), not the divergent ones.
    assert np.isclose(float(consensus["BAD"].iloc[0]), float(base.iloc[0]))


def test_parse_sp500_constituents_html_normalizes_tickers():
    out = parse_sp500_constituents_html(WIKI_SNAPSHOT_HTML)
    assert list(out["Ticker"]) == ["AAA", "BBB", "BRK-B"]
    assert "Sector" in out.columns


def test_wayback_snapshot_flow_parses_archived_page():
    availability = {
        "archived_snapshots": {
            "closest": {
                "available": True,
                "url": "http://web.archive.org/web/20150101000000/https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            }
        }
    }
    with (
        patch.object(universe_mod, "http_read_json", return_value=availability),
        patch.object(universe_mod, "http_read_text", return_value=WIKI_SNAPSHOT_HTML),
    ):
        out = universe_mod.fetch_sp500_constituents_wayback("2015-01-01")
    assert not out.empty
    assert set(out["Ticker"]) == {"AAA", "BBB", "BRK-B"}
    assert (out["Source_Status"] == "wayback_snapshot").all()
    assert out["Snapshot_URL"].iloc[0].startswith("https://")


def test_wayback_returns_empty_when_no_snapshot():
    with patch.object(universe_mod, "http_read_json", return_value={"archived_snapshots": {}}):
        out = universe_mod.fetch_sp500_constituents_wayback("1999-01-01")
    assert out.empty


def test_build_delisted_registry_combines_evidence():
    removals = pd.DataFrame(
        {
            "Ticker": ["DEAD", "ALIVE", "GONE"],
            "Removed_Date": pd.to_datetime(["2020-05-01", "2021-02-01", "2019-08-01"]),
        }
    )
    listed = pd.DataFrame({"Ticker": ["ALIVE", "OTHER"]})
    stooq_known = pd.DataFrame({"Ticker": ["DEAD"]})
    registry = build_delisted_registry(removals, listed, extra_symbols=stooq_known)
    assert set(registry["Ticker"]) == {"DEAD", "GONE"}
    reg = registry.set_index("Ticker")
    assert reg.loc["DEAD", "Recovery_Status"] == "price_history_available"
    assert reg.loc["GONE", "Recovery_Status"] == "no_known_free_history"


def test_oos_returns_apply_delisting_assumption_on_joint_evidence():
    idx = pd.bdate_range("2024-01-02", periods=40)
    live = pd.Series(np.linspace(100, 110, len(idx)), index=idx)
    dead = pd.Series(100.0, index=idx)  # stale price all window
    px = pd.DataFrame({"LIVE": live, "DEAD": dead, "STALE_BUT_TRADED": dead})
    volumes = pd.DataFrame(
        {
            "LIVE": pd.Series(1e6, index=idx),
            "DEAD": pd.Series(0.0, index=idx),
            "STALE_BUT_TRADED": pd.Series(5e5, index=idx),
        }
    )
    out = core.oos_returns_with_delisting(
        px,
        ["LIVE", "DEAD", "STALE_BUT_TRADED"],
        idx[0],
        idx[-1],
        delisting_return=-0.30,
        volumes=volumes,
    )
    assert out.loc["LIVE"] > 0.0
    assert np.isclose(out.loc["DEAD"], -0.30)
    # Stale price alone is not enough evidence: it still traded.
    assert np.isclose(out.loc["STALE_BUT_TRADED"], 0.0)


def test_download_prices_falls_back_to_stooq_when_yahoo_is_empty():
    stooq_frame = parse_stooq_csv(STOOQ_CSV, "AAA")
    with (
        patch.object(core.yf, "download", return_value=pd.DataFrame()),
        patch.object(core, "fetch_stooq_prices", return_value=(stooq_frame, [])) as stooq,
    ):
        out = core.download_prices(["AAA"], period="3y", use_cache=False)
    assert stooq.called
    assert "AAA" in out.columns
    assert len(out) == 5


def test_download_prices_period_to_start_parses_common_specs():
    assert core._period_to_start("max") is None
    three_years = core._period_to_start("3y")
    assert three_years is not None
    assert (pd.Timestamp.today() - three_years).days >= 3 * 360
    six_months = core._period_to_start("6mo")
    assert six_months is not None and six_months > three_years


TIINGO_JSON = [
    {"date": "2024-01-02T00:00:00.000Z", "adjClose": 100.5},
    {"date": "2024-01-03T00:00:00.000Z", "adjClose": 101.5},
    {"date": "2024-01-04T00:00:00.000Z", "adjClose": 102.5},
]


def test_tiingo_provider_parses_fixture_and_degrades_without_token():
    with patch.object(prices_mod, "http_read_json", return_value=TIINGO_JSON):
        frame, provenance = prices_mod.fetch_tiingo_prices(["AAA"], token="fixture", use_cache=False)
    assert list(frame.columns) == ["AAA"]
    assert len(frame) == 3
    assert all("token=***" in p.url or not p.url for p in provenance)

    empty, prov = prices_mod.fetch_tiingo_prices(["AAA"], token="")
    assert empty.empty and prov == []
