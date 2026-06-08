from __future__ import annotations

import pandas as pd


def _frame(value) -> pd.DataFrame:
    return value if isinstance(value, pd.DataFrame) else pd.DataFrame()


def _nested_frame(container, key: str) -> pd.DataFrame:
    return _frame(container.get(key)) if isinstance(container, dict) else pd.DataFrame()


def _frame_map(container) -> dict[str, pd.DataFrame]:
    if not isinstance(container, dict):
        return {}
    return {
        str(key): value
        for key, value in container.items()
        if isinstance(value, pd.DataFrame)
    }


def _tail_frame(value, rows: int = 756) -> pd.DataFrame:
    frame = _frame(value)
    return frame.tail(rows).reset_index(drop=True) if not frame.empty else frame


def build_dashboard_payload(
    results: dict,
    path_bundle: dict,
    suitability_gate: dict,
    promotion_gate: dict,
    freshness_report: pd.DataFrame | None = None,
) -> dict:
    """Stable frontend contract. UI code should render this object, not recompute analytics."""
    validation = results.get("validation_diagnostics", {})
    risk = results.get("return_diagnostics", {})
    latent = results.get("latent_regime_diagnostics", {})
    alternative = results.get("alternative_data", {})
    kaizen = results.get("kaizen_diagnostics", {})
    return {
        "contract": {
            "schema_version": "2026.06.08-market-intelligence-v5",
            "analytics_scope": "full_analysis" if _frame(results.get("portfolio")).shape[0] else "market_snapshot",
            "render_policy": "frontend_read_only",
        },
        "status": {
            "suitability": suitability_gate.get("summary", suitability_gate.get("diagnostics", pd.DataFrame())),
            "suitability_breaches": suitability_gate.get("breaches", pd.DataFrame()),
            "promotion": promotion_gate.get("summary", pd.DataFrame()),
            "promotion_tests": promotion_gate.get("tests", pd.DataFrame()),
            "data_freshness": freshness_report if freshness_report is not None else pd.DataFrame(),
            "snapshot_meta": results.get("snapshot_meta", pd.DataFrame()),
            "market_context": results.get("market_context", pd.DataFrame()),
        },
        "allocation": {
            "recommended_portfolio": _frame(results.get("portfolio")),
            "side_sleeve": _frame(results.get("side_boom_portfolio")),
            "weights": _frame(results.get("portfolio")),
        },
        "market_snapshot": {
            "observed_selection": _frame(results.get("price_snapshot_selection")),
            "price_only": bool(not _frame(results.get("price_snapshot_selection")).empty),
            "context": _frame(results.get("market_context")),
        },
        "market_intelligence": {
            "latest_macro": pd.DataFrame([results.get("latest_macro", pd.Series(dtype=object))]),
            "macro_history": _tail_frame(results.get("macro"), 1260),
            "global_yield_curves": _frame(results.get("global_yield_curves")),
            "global_rate_history": _tail_frame(results.get("global_rate_history"), 2500),
            "interbank_reference_rates": _tail_frame(results.get("interbank_reference_rates"), 2500),
            "carry_trade_suggestions": _frame(results.get("carry_trade_suggestions")),
            "carry_trade_validation": _frame(results.get("carry_trade_validation")),
            "sentiment_timeline": _nested_frame(results.get("market_sentiment_sem", {}), "timeline").tail(756),
            "sentiment_latest": _nested_frame(results.get("market_sentiment_sem", {}), "latest"),
            "sentiment_loadings": _nested_frame(results.get("market_sentiment_sem", {}), "loadings"),
            "sentiment_structural_links": _nested_frame(results.get("market_sentiment_sem", {}), "structural_links"),
            "sentiment_diagnostics": _nested_frame(results.get("market_sentiment_sem", {}), "diagnostics"),
            "forex_factory_calendar": _nested_frame(alternative, "forex_factory_calendar").head(120),
            "forex_factory_event_risk": _nested_frame(alternative, "forex_factory_event_risk"),
            "geopolitical_summary": _nested_frame(alternative, "summary"),
            "geopolitical_timeline": _nested_frame(alternative, "gdelt_timeline").tail(756),
        },
        "charts": {
            "price_paths": path_bundle.get("price_paths", pd.DataFrame()),
            "drawdowns": path_bundle.get("drawdowns", pd.DataFrame()),
            "forecast_cone": risk.get("gbm_forecast_paths", pd.DataFrame()) if isinstance(risk, dict) else pd.DataFrame(),
            "conditional_vol": risk.get("variance_conditional_paths", pd.DataFrame()) if isinstance(risk, dict) else pd.DataFrame(),
            "rate_curves": results.get("global_yield_curves", pd.DataFrame()),
            "options_surface": results.get("portfolio_vol_surface_matrix", pd.DataFrame()),
        },
        "tables": {
            "fundamentals": results.get("portfolio", pd.DataFrame()),
            "risk": results.get("performance_summary", pd.DataFrame()),
            "validation": validation.get("summary", pd.DataFrame()) if isinstance(validation, dict) else pd.DataFrame(),
            "rejections": results.get("rejection_diagnostics", pd.DataFrame()),
            "max_drawdown": path_bundle.get("max_drawdown_table", pd.DataFrame()),
        },
        "research": {
            "optimization_grid": _frame(results.get("optimization_grid")),
            "sector_diagnostics": _frame(results.get("sector_diagnostics")),
            "overfit_diagnostics": _frame(results.get("overfit_diagnostics")),
            "factor_attribution": _frame(results.get("factor_attribution")),
            "oos_factor_attribution": _frame(results.get("oos_factor_attribution")),
            "monitoring_diagnostics": _frame(results.get("monitoring_diagnostics")),
            "benchmark_governance": _frame(results.get("benchmark_governance")),
            "model_registry": _frame(results.get("model_registry")),
            "regime_performance": _frame(results.get("regime_performance")),
            "stress_tests": _frame(results.get("stress_tests")),
            "hedge_suggestions": _frame(results.get("hedge_suggestions")),
            "decision_attribution": _frame(results.get("decision_attribution")),
            "capital_ledger": _frame(results.get("capital_ledger")),
            "variance_model_selection": _nested_frame(risk, "variance_model_selection"),
            "variance_conditional_paths": _nested_frame(risk, "variance_conditional_paths"),
            "gbm_forecast_paths": _nested_frame(risk, "gbm_forecast_paths"),
            "pelt_regime_segments": _nested_frame(risk, "pelt_regime_segments"),
            "pelt_change_points": _nested_frame(risk, "pelt_change_points"),
            "pelt_timeline": _nested_frame(risk, "pelt_timeline"),
            "side_pelt_regime_segments": _frame(results.get("side_boom_pelt_regime_segments")),
            "side_pelt_change_points": _frame(results.get("side_boom_pelt_change_points")),
            "side_pelt_timeline": _frame(results.get("side_boom_pelt_timeline")),
            "options_chain": _frame(results.get("options_chain")),
            "options_summary": _frame(results.get("options_summary")),
            "vol_surface": _frame(results.get("portfolio_vol_surface")),
            "vol_surface_diagnostics": _frame(results.get("portfolio_vol_surface_diagnostics")),
            "global_rate_history": _frame(results.get("global_rate_history")),
            "interbank_reference_rates": _frame(results.get("interbank_reference_rates")),
            "carry_trade_suggestions": _frame(results.get("carry_trade_suggestions")),
            "carry_trade_validation": _frame(results.get("carry_trade_validation")),
            "latent_regime_summary": _nested_frame(latent, "summary"),
            "latent_regime_history": _nested_frame(latent, "history"),
            "latent_markov_forecast": _nested_frame(latent, "markov_forecast"),
            "alternative_data_summary": _nested_frame(alternative, "summary"),
            "kaizen_summary": _nested_frame(kaizen, "summary"),
            "cache_inventory": _frame(results.get("cache_inventory")),
            "timings": _frame(results.get("timings")),
        },
        "diagnostics": {
            "return": _frame_map(risk),
            "validation": _frame_map(validation),
            "latent_regime": _frame_map(latent),
            "alternative_data": _frame_map(alternative),
            "kaizen": _frame_map(kaizen),
        },
        "explanations": {
            "user_safe_summary": suitability_gate.get("user_safe_summary", ""),
            "technical_audit": "Core analytics were produced by the backend contract; frontend is render-only.",
            "warnings": list(results.get("model_registry", pd.DataFrame()).get("warnings", [])) if isinstance(results.get("model_registry"), pd.DataFrame) else [],
        },
    }
