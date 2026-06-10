from __future__ import annotations

import numpy as np
import pandas as pd

import cloud_daily_refresh as cloud
from quant_stockpicker_core import RunConfig


def test_fast_dashboard_snapshot_builds_causal_render_contract(monkeypatch):
    index = pd.bdate_range("2022-01-03", periods=900)
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
    monkeypatch.setattr(
        cloud,
        "build_daily_market_intelligence",
        lambda *args, **kwargs: {
            "macro": pd.DataFrame({"POLICY_RATE": [4.0]}, index=[index[-1]]),
            "latest_macro": pd.Series({"Regime_Hawkish_Dovish": "Hawkish"}),
            "global_yield_curves": pd.DataFrame([{"Country": "United States", "Yield_2Y": 4.0, "Yield_10Y": 4.4}]),
            "global_rate_history": pd.DataFrame(
                [{"Country": "United States", "Observation_Date": index[-1], "Tenor_Code": "SOV_10Y", "Rate": 4.4}]
            ),
            "interbank_reference_rates": pd.DataFrame(
                [{"Observation_Date": index[-1], "Benchmark": "SOFR", "Rate": 4.3}]
            ),
            "carry_trade_suggestions": pd.DataFrame(),
            "carry_trade_validation": pd.DataFrame(),
            "market_sentiment_sem": {"timeline": pd.DataFrame({"Date": [index[-1]], "Latent_Sentiment": [0.2]})},
            "alternative_data": {
                "forex_factory_calendar": pd.DataFrame([{"Date": index[-1], "Event": "CPI"}]),
                "forex_factory_event_risk": pd.DataFrame(),
                "summary": pd.DataFrame(),
                "gdelt_timeline": pd.DataFrame(),
                "gdelt_articles": pd.DataFrame(),
            },
        },
    )

    config = RunConfig(
        tickers=("AAA", "BBB", "CCC"),
        benchmark_ticker="SPY",
        price_period="10y",
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

    assert results["portfolio"].empty
    assert np.isclose(results["price_snapshot_selection"]["Weight"].sum(), 1.0)
    assert not results["backtest_path_bundle"]["price_paths"].empty
    path_columns = set(results["backtest_path_bundle"]["price_paths"].columns)
    assert "Daily causal allocation proxy price" in path_columns
    assert all("Sortino optimized" not in column for column in path_columns)
    assert not results["backtest_path_bundle"]["drawdowns"].empty
    assert results["promotion_gate"]["promotion_status"] == "RESEARCH_SNAPSHOT_NOT_PROMOTED"
    assert results["dashboard_payload"]["allocation"]["recommended_portfolio"].empty
    assert not results["dashboard_payload"]["market_snapshot"]["observed_selection"].empty
    assert results["snapshot_meta"].iloc[0]["Snapshot_Mode"] == "daily_price_snapshot"
    assert not bool(results["snapshot_meta"].iloc[0]["Is_User_Specific"])
    assert not results["market_context"].empty
    assert results["market_context"].iloc[0]["Method"] == "causal_price_proxy"
    assert not results["global_yield_curves"].empty
    assert not results["market_sentiment_sem"]["timeline"].empty
    assert not results["dashboard_payload"]["market_intelligence"]["macro_history"].empty
    assert not results["dashboard_payload"]["market_intelligence"]["global_yield_curves"].empty
    assert set(results["performance_summary"]["Metric"]) >= {
        "Annualized_Return",
        "Annualized_Vol",
        "Sortino",
        "Max_Drawdown",
        "CVaR_95",
        "Benchmark_Annualized_Return",
        "XCDR_v3",
    }
    assert "snapshot_meta" in results["dashboard_payload"]["status"]
    assert "market_context" in results["dashboard_payload"]["status"]
