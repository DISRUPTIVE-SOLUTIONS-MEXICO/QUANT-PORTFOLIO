from __future__ import annotations

import pandas as pd

from quant_core.fixed_income_intelligence import build_fixed_income_intelligence
from quant_core.security_intelligence import build_security_intelligence
from quant_core.strategy_lab import build_strategy_lab_artifact


def _frame(value) -> pd.DataFrame:
    return value if isinstance(value, pd.DataFrame) else pd.DataFrame()


def _nested_frame(container, key: str) -> pd.DataFrame:
    return _frame(container.get(key)) if isinstance(container, dict) else pd.DataFrame()


def _frame_map(container) -> dict[str, pd.DataFrame]:
    if not isinstance(container, dict):
        return {}
    return {str(key): value for key, value in container.items() if isinstance(value, pd.DataFrame)}


def _tail_frame(value, rows: int = 756) -> pd.DataFrame:
    frame = _frame(value)
    return frame.tail(rows).reset_index(drop=True) if not frame.empty else frame



def _payload_value_at(payload: dict, path: str):
    value = payload
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _value_count(value) -> int:
    if isinstance(value, pd.DataFrame):
        return int(len(value.index))
    if isinstance(value, pd.Series):
        return int(value.notna().sum())
    if isinstance(value, dict):
        return int(sum(1 for item in value.values() if item is not None and str(item).strip() != ""))
    if isinstance(value, list):
        return int(len(value))
    if value is None:
        return 0
    return int(str(value).strip() != "")


def _build_capability_completeness(payload: dict) -> pd.DataFrame:
    """Module-level evidence map for the UI and publication audit trail.

    The score is descriptive rather than a promotion gate. It prevents the
    frontend from making a thin daily snapshot look equivalent to the full
    research package.
    """
    checks = [
        (
            "Market Intelligence",
            "Macro, public benchmarks, SEM/geopolitical attention and event risk.",
            "daily",
            [
                ("market_intelligence.latest_macro", 1),
                ("market_intelligence.macro_history", 30),
                ("market_intelligence.geopolitical_summary", 1),
                ("market_intelligence.forex_factory_calendar", 1),
            ],
        ),
        (
            "Rates & Fixed Income",
            "Source-aware sovereign curves, reference rates and carry validation.",
            "daily",
            [
                ("market_intelligence.global_yield_curves", 2),
                ("market_intelligence.global_rate_history", 30),
                ("fixed_income_intelligence.country_metrics", 2),
                ("fixed_income_intelligence.reference_rate_summary", 1),
            ],
        ),
        (
            "Equity Fundamentals",
            "Sector-relative fundamentals, PIT confidence, options and reject diagnostics.",
            "per full research run",
            [
                ("tables.fundamentals", 2),
                ("research.sector_diagnostics", 1),
                ("research.options_summary", 1),
                ("tables.rejections", 1),
            ],
        ),
        (
            "Benchmark xi",
            "Mandate-compatible benchmark chosen before optimization.",
            "per mandate/run",
            [
                ("research.benchmark_governance", 1),
                ("research.model_registry", 1),
                ("strategy_lab.benchmark_xi", 1),
            ],
        ),
        (
            "XCDR Research",
            "Strategy lab, candidate comparison, OOS paths and promotion evidence.",
            "per full research run",
            [
                ("strategy_lab.summary", 3),
                ("strategy_lab.oos_price_paths", 252),
                ("strategy_lab.validation", 1),
                ("strategy_lab.constitution", 1),
            ],
        ),
        (
            "Portfolio Construction",
            "Immutable research weights with constraints and allocation evidence.",
            "per optimization",
            [
                ("allocation.recommended_portfolio", 2),
                ("charts.price_paths", 252),
                ("charts.drawdowns", 252),
                ("tables.risk", 4),
            ],
        ),
        (
            "Risk Laboratory",
            "Conditional variance, PELT, forecast cones, tail and hedge diagnostics.",
            "per full research run",
            [
                ("research.variance_model_selection", 1),
                ("research.pelt_change_points", 1),
                ("research.gbm_forecast_paths", 1),
                ("research.hedge_suggestions", 1),
            ],
        ),
        (
            "Validation & Governance",
            "Nested walk-forward evidence, WRC/SPA/PBO and model registry.",
            "per full research run",
            [
                ("status.promotion_tests", 1),
                ("tables.validation", 1),
                ("research.model_registry", 1),
                ("status.data_freshness", 1),
            ],
        ),
        (
            "Data Quality",
            "Freshness, source provenance and fallback-state observability.",
            "every publication",
            [
                ("status.data_freshness", 1),
                ("status.snapshot_meta", 1),
                ("status.market_context", 1),
                ("research.cache_inventory", 1),
            ],
        ),
    ]
    output = []
    for module, description, freshness, requirements in checks:
        observed = 0
        missing = []
        detail = []
        for requirement_path, minimum in requirements:
            count = _value_count(_payload_value_at(payload, requirement_path))
            passed = count >= minimum
            observed += int(passed)
            if not passed:
                missing.append(requirement_path)
            detail.append(f"{requirement_path}: {count}/{minimum}")
        score = observed / max(1, len(requirements))
        output.append(
            {
                "Module": module,
                "Completeness": round(score, 4),
                "Status": "complete" if score >= 1.0 else "partial" if score > 0 else "missing",
                "Freshness_Requirement": freshness,
                "Evidence_Count": observed,
                "Required_Count": len(requirements),
                "Missing_Evidence": ", ".join(missing),
                "Description": description,
                "Diagnostics": " | ".join(detail),
            }
        )
    return pd.DataFrame(output)


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
    path_metadata = path_bundle.get("path_metadata", {}) if isinstance(path_bundle, dict) else {}
    benchmark = str(
        path_metadata.get("benchmark") or results.get("benchmark_ticker") or results.get("benchmark") or "SPY"
    )
    has_portfolio = not _frame(results.get("portfolio")).empty
    strategy_lab = results.get("strategy_lab")
    if not isinstance(strategy_lab, dict) and has_portfolio:
        strategy_lab = build_strategy_lab_artifact(
            _frame(results.get("prices")),
            benchmark=benchmark,
            bootstrap_samples=80,
        )
    elif not isinstance(strategy_lab, dict):
        strategy_lab = {
            "generation": "daily-market-overlay",
            "status": "UNAVAILABLE_IN_DAILY_OVERLAY",
            "benchmark_xi": benchmark,
            "observation_days": 0,
            "summary": pd.DataFrame(),
        }
    security_intelligence = results.get("security_intelligence")
    if not isinstance(security_intelligence, dict):
        security_intelligence = build_security_intelligence(
            _frame(results.get("prices")),
            benchmark=benchmark,
            volumes=_frame(results.get("volumes")),
            strategy_scores=_frame(strategy_lab.get("latest_scores")),
        )
    fixed_income_intelligence = results.get("fixed_income_intelligence")
    if not isinstance(fixed_income_intelligence, dict):
        prices = _frame(results.get("prices"))
        market_as_of = prices.index.max() if not prices.empty else None
        fixed_income_intelligence = build_fixed_income_intelligence(
            _frame(results.get("global_yield_curves")),
            _frame(results.get("global_rate_history")),
            reference_rates=_frame(results.get("interbank_reference_rates")),
            carry_validation=_frame(results.get("carry_trade_validation")),
            as_of=market_as_of,
        )
    payload = {
        "contract": {
            "schema_version": "2026.06.19-publication-isolation-v11",
            "analytics_scope": "full_analysis" if has_portfolio else "market_snapshot",
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
            "geopolitical_articles": _nested_frame(alternative, "gdelt_articles").head(500),
            "geopolitical_country_heatmap": _nested_frame(alternative, "country_heatmap"),
        },
        "security_intelligence": {
            "contract": security_intelligence.get("contract", {}),
            "benchmark_xi": str(security_intelligence.get("benchmark_xi", benchmark)),
            "as_of": security_intelligence.get("as_of"),
            "metrics": _frame(security_intelligence.get("metrics")),
            "price_history": _tail_frame(security_intelligence.get("price_history"), 756),
            "strategy_consensus": _frame(security_intelligence.get("strategy_consensus")),
            "methodology": _frame(security_intelligence.get("methodology")),
        },
        "fixed_income_intelligence": {
            "contract": fixed_income_intelligence.get("contract", {}),
            "as_of": fixed_income_intelligence.get("as_of"),
            "country_metrics": _frame(fixed_income_intelligence.get("country_metrics")),
            "factor_history": _frame(fixed_income_intelligence.get("factor_history")),
            "stress_scenarios": _frame(fixed_income_intelligence.get("stress_scenarios")),
            "reference_rate_summary": _frame(fixed_income_intelligence.get("reference_rate_summary")),
            "carry_candidates": _frame(fixed_income_intelligence.get("carry_candidates")),
            "methodology": _frame(fixed_income_intelligence.get("methodology")),
        },
        "charts": {
            "price_paths": path_bundle.get("price_paths", pd.DataFrame()),
            "drawdowns": path_bundle.get("drawdowns", pd.DataFrame()),
            "forecast_cone": risk.get("gbm_forecast_paths", pd.DataFrame())
            if isinstance(risk, dict)
            else pd.DataFrame(),
            "conditional_vol": risk.get("variance_conditional_paths", pd.DataFrame())
            if isinstance(risk, dict)
            else pd.DataFrame(),
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
        "strategy_lab": {
            "generation": str(strategy_lab.get("generation", "")),
            "status": str(strategy_lab.get("status", "UNAVAILABLE")),
            "benchmark_xi": str(strategy_lab.get("benchmark_xi", "")),
            "observation_days": int(strategy_lab.get("observation_days", 0) or 0),
            "summary": _frame(strategy_lab.get("summary")),
            "price_paths": _tail_frame(strategy_lab.get("price_paths"), 1260),
            "drawdowns": _tail_frame(strategy_lab.get("drawdowns"), 1260),
            "oos_summary": _frame(strategy_lab.get("oos_summary")),
            "oos_price_paths": _tail_frame(strategy_lab.get("oos_price_paths"), 1260),
            "oos_drawdowns": _tail_frame(strategy_lab.get("oos_drawdowns"), 1260),
            "holdout_summary": _frame(strategy_lab.get("holdout_summary")),
            "holdout_price_paths": _tail_frame(strategy_lab.get("holdout_price_paths"), 756),
            "holdout_drawdowns": _tail_frame(strategy_lab.get("holdout_drawdowns"), 756),
            "walk_forward_windows": _frame(strategy_lab.get("walk_forward_windows")),
            "selection_stability": _frame(strategy_lab.get("selection_stability")),
            "frozen_candidate": str(strategy_lab.get("frozen_candidate", "")),
            "signal_ic": _tail_frame(strategy_lab.get("signal_ic"), 2500),
            "regime_performance": _frame(strategy_lab.get("regime_performance")),
            "weights": _tail_frame(strategy_lab.get("weights"), 4000),
            "latest_scores": _frame(strategy_lab.get("latest_scores")),
            "exposure_diagnostics": _tail_frame(strategy_lab.get("exposure_diagnostics"), 2500),
            "exposure_timeline": _tail_frame(strategy_lab.get("exposure_timeline"), 2500),
            "validation": _frame(strategy_lab.get("validation")),
            "constitution": _frame(strategy_lab.get("constitution")),
            "research_lineage": _frame(strategy_lab.get("research_lineage")),
            "strategy_registry": _frame(strategy_lab.get("strategy_registry")),
            "candidate_equivalence": _frame(strategy_lab.get("candidate_equivalence")),
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
            "warnings": list(results.get("model_registry", pd.DataFrame()).get("warnings", []))
            if isinstance(results.get("model_registry"), pd.DataFrame)
            else [],
        },
    }
    payload["status"]["capability_completeness"] = _build_capability_completeness(payload)
    return payload
