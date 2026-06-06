from __future__ import annotations

import numpy as np
import pandas as pd

import cloud_daily_refresh as cloud
from quant_stockpicker_core import RunConfig


def test_fast_dashboard_snapshot_builds_causal_render_contract(monkeypatch):
    index = pd.bdate_range("2024-01-02", periods=280)
    rng = np.random.default_rng(17)
    returns = pd.DataFrame(
        {
            "AAA": rng.normal(0.0007, 0.012, len(index)),
            "BBB": rng.normal(0.0004, 0.009, len(index)),
            "CCC": rng.normal(0.0002, 0.007, len(index)),
            "SPY": rng.normal(0.00035, 0.010, len(index)),
        },
        index=index,
    )
    prices = 100.0 * (1.0 + returns).cumprod()
    monkeypatch.setattr(cloud, "download_prices", lambda *args, **kwargs: prices)

    config = RunConfig(
        tickers=("AAA", "BBB", "CCC"),
        benchmark_ticker="SPY",
        price_period="2y",
        top_n=2,
        compute_mode="fast",
        use_sec_edgar=False,
        use_options_snapshot=False,
        use_forex_factory_calendar=False,
        use_gdelt=False,
        use_garch=False,
        use_latent_macro_regime=False,
        use_kaizen_bandit=False,
    )
    results = cloud.build_fast_dashboard_snapshot(config)

    assert not results["portfolio"].empty
    assert np.isclose(results["portfolio"]["Weight"].sum(), 1.0)
    assert not results["backtest_path_bundle"]["price_paths"].empty
    assert not results["backtest_path_bundle"]["drawdowns"].empty
    assert results["promotion_gate"]["promotion_status"] == "RESEARCH_SNAPSHOT_NOT_PROMOTED"
    assert results["dashboard_payload"]["allocation"]["recommended_portfolio"].shape[0] <= 2

