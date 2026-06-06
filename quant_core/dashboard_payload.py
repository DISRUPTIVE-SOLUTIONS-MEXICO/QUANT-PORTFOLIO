from __future__ import annotations

import pandas as pd


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
    return {
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
            "recommended_portfolio": results.get("portfolio", pd.DataFrame()),
            "side_sleeve": results.get("side_boom_portfolio", pd.DataFrame()),
            "weights": results.get("portfolio", pd.DataFrame()),
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
        "explanations": {
            "user_safe_summary": suitability_gate.get("user_safe_summary", ""),
            "technical_audit": "Core analytics were produced by the backend contract; frontend is render-only.",
            "warnings": list(results.get("model_registry", pd.DataFrame()).get("warnings", [])) if isinstance(results.get("model_registry"), pd.DataFrame) else [],
        },
    }
