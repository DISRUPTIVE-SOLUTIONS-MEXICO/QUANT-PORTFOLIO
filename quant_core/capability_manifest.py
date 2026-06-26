from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Capability:
    capability_id: str
    owner_layer: str
    canonical_calculation: str
    canonical_artifact: str
    primary_view: str
    secondary_views: tuple[str, ...]
    freshness_requirement: str
    validation_status: str


CAPABILITIES: tuple[Capability, ...] = (
    Capability(
        "market.prices",
        "data",
        "quant_core.data.prices.download_prices",
        "market_snapshot.price_history",
        "Command Center",
        ("Equity Research", "Portfolio Construction", "Risk Laboratory"),
        "daily",
        "source-reconciled",
    ),
    Capability(
        "market.universe_pit",
        "pit_quality",
        "quant_core.data.universe.load_sp500_wikipedia_asof",
        "research_evidence.universe",
        "Data Quality",
        ("Equity Research", "Validation & Governance"),
        "per rebalance",
        "public PIT approximation",
    ),
    Capability(
        "security.workbench",
        "features",
        "quant_core.security_intelligence.build_security_intelligence",
        "security_intelligence.metrics",
        "Equity Research",
        ("Strategy Laboratory", "Portfolio Construction", "Risk Laboratory"),
        "daily",
        "causal live-snapshot diagnostics",
    ),
    Capability(
        "fundamentals.sector_relative",
        "features",
        "AlphaResearchEngine sector-relative robust scoring",
        "portfolio_run.fundamentals",
        "Equity Research",
        ("Portfolio Construction", "Data Quality"),
        "filing availability date",
        "SEC-gated where available",
    ),
    Capability(
        "fundamentals.mahalanobis",
        "signals",
        "sectoral robust Mahalanobis distance",
        "research_evidence.sector_diagnostics",
        "Equity Research",
        ("Validation & Governance",),
        "per research run",
        "research-grade",
    ),
    Capability(
        "benchmark.xi",
        "benchmark",
        "benchmark governance fit function",
        "research_evidence.benchmark_governance",
        "Command Center",
        ("XCDR Research", "Validation & Governance"),
        "per mandate/run",
        "causal selection required",
    ),
    Capability(
        "benchmark.omega",
        "benchmark",
        "mandate-specific stress benchmark set",
        "research_evidence.omega_stress",
        "XCDR Research",
        ("Risk Laboratory", "Validation & Governance"),
        "per strategy constitution",
        "pre-registered",
    ),
    Capability(
        "portfolio.xcdr",
        "portfolio",
        "quant_core.uncertainty_state.xcdr_v3_growth_control_score",
        "portfolio_run.weights",
        "Portfolio Construction",
        ("Command Center", "XCDR Research", "My Portfolios"),
        "annual reoptimization",
        "research-only until promotion",
    ),
    Capability(
        "portfolio.tail_aware_growth",
        "signals",
        "tail-aware convex opportunity universe and growth sleeve",
        "research_evidence.growth_sleeve",
        "XCDR Research",
        ("Equity Research", "Risk Laboratory"),
        "per research run",
        "research-grade",
    ),
    Capability(
        "strategy.research_lab",
        "signals",
        "quant_core.strategy_lab.build_strategy_lab_artifact",
        "research_evidence.strategy_lab",
        "Strategy Laboratory",
        ("Market Intelligence", "Risk Laboratory", "Validation & Governance"),
        "per daily snapshot and full research run",
        "nested OOS selection with purge, embargo and frozen final holdout",
    ),
    Capability(
        "strategy.registry",
        "signals",
        "quant_core.strategy_registry.STRATEGY_SPECIFICATIONS",
        "research_evidence.strategy_lab.strategy_registry",
        "Strategy Laboratory",
        ("Data Quality", "Validation & Governance"),
        "per immutable research generation",
        "implemented, planned and blocked families explicitly separated",
    ),
    Capability(
        "strategy.equivalence_control",
        "backtest_validation",
        "quant_core.strategy_lab._deduplicate_candidate_paths",
        "research_evidence.strategy_lab.candidate_equivalence",
        "Strategy Laboratory",
        ("Validation & Governance",),
        "per immutable candidate matrix",
        "identical paths count once in nested selection and WRC/SPA/PBO",
    ),
    Capability(
        "strategy.signal_diagnostics",
        "backtest_validation",
        "lagged signal IC, regime attribution, capture and turnover diagnostics",
        "research_evidence.strategy_signal_diagnostics",
        "Strategy Laboratory",
        ("Validation & Governance", "Data Quality"),
        "per strategy research run",
        "one-period lag plus test-block WRC/SPA/PBO and holdout gate",
    ),
    Capability(
        "strategy.downside_governor",
        "risk",
        "quant_core.strategy_lab._causal_risk_exposure",
        "research_evidence.strategy_lab.exposure_diagnostics",
        "Strategy Laboratory",
        ("Risk Laboratory", "Validation & Governance"),
        "per causal strategy decision",
        "fixed generation; prospective evidence required after holdout consumption",
    ),
    Capability(
        "risk.covariance",
        "risk",
        "Ledoit-Wolf and RMT PSD covariance",
        "risk_report.covariance",
        "Risk Laboratory",
        ("Portfolio Construction",),
        "per rebalance",
        "PSD tested",
    ),
    Capability(
        "risk.variance_models",
        "risk",
        "ARCH/GARCH/EGARCH/Fractional Volterra architecture selection",
        "risk_report.variance_models",
        "Risk Laboratory",
        ("Validation & Governance",),
        "daily snapshot / rebalance",
        "AIC/BIC/LogLik plus OOS QLIKE",
    ),
    Capability(
        "risk.pelt",
        "risk",
        "PELT change-point segmentation",
        "risk_report.pelt_regimes",
        "Risk Laboratory",
        ("Market Intelligence",),
        "daily",
        "causal-prefix tested",
    ),
    Capability(
        "risk.tail",
        "risk",
        "historical CVaR and EVT residual tails",
        "risk_report.tail",
        "Risk Laboratory",
        ("Portfolio Construction", "Paper Execution"),
        "per rebalance / daily monitor",
        "promotion-gated",
    ),
    Capability(
        "risk.uncertainty",
        "risk",
        "Kalman/RMT/Fisher/CRLB/entropy uncertainty state",
        "risk_report.uncertainty_state",
        "XCDR Research",
        ("Risk Laboratory", "Validation & Governance"),
        "per decision date",
        "causal-prefix tested",
    ),
    Capability(
        "validation.walk_forward",
        "backtest_validation",
        "nested walk-forward with purging and embargo",
        "research_evidence.walk_forward",
        "Validation & Governance",
        ("XCDR Research",),
        "per research run",
        "future-contamination tested",
    ),
    Capability(
        "validation.multiple_testing",
        "backtest_validation",
        "WRC/SPA/PBO/Deflated Sortino/Romano-Wolf",
        "research_evidence.promotion_tests",
        "Validation & Governance",
        ("Command Center",),
        "per research run",
        "null calibration tested",
    ),
    Capability(
        "rates.sovereign_curves",
        "market_intelligence",
        "official/public sovereign term-structure providers",
        "market_snapshot.yield_curves",
        "Rates & Fixed Income",
        ("Command Center", "Market Intelligence"),
        "daily",
        "minimum two real tenors",
    ),
    Capability(
        "rates.term_structure_workbench",
        "market_intelligence",
        "causal curve factors, source quality and duration/convexity stress scenarios",
        "fixed_income_intelligence",
        "Rates & Fixed Income",
        ("Risk Laboratory", "Command Center"),
        "daily",
        "native-calendar causal diagnostics",
    ),
    Capability(
        "rates.carry",
        "market_intelligence",
        "FX-adjusted carry validation",
        "market_snapshot.carry",
        "Rates & Fixed Income",
        ("Risk Laboratory",),
        "daily",
        "research-only",
    ),
    Capability(
        "macro.regime",
        "market_intelligence",
        "hawkish/dovish and bull/bear regime engine",
        "market_snapshot.macro_state",
        "Market Intelligence",
        ("Command Center", "XCDR Research"),
        "daily",
        "public-data proxy",
    ),
    Capability(
        "sentiment.sem",
        "market_intelligence",
        "causal latent SEM market sentiment",
        "market_snapshot.sentiment",
        "Market Intelligence",
        ("Command Center",),
        "daily",
        "experimental",
    ),
    Capability(
        "geopolitical.attention",
        "market_intelligence",
        "within-topic robust GDELT/RSS abnormal attention",
        "market_snapshot.geopolitical",
        "Market Intelligence",
        ("Risk Laboratory",),
        "daily",
        "attention proxy, not causal probability",
    ),
    Capability(
        "options.snapshot",
        "market_intelligence",
        "Yahoo option-chain snapshot analytics",
        "market_snapshot.options",
        "Equity Research",
        ("Risk Laboratory",),
        "daily",
        "snapshot-only",
    ),
    Capability(
        "portfolio.my_portfolios",
        "artifact_registry",
        "versioned user-scoped portfolio runs",
        "portfolio_run",
        "My Portfolios",
        ("Portfolio Construction", "Paper Execution"),
        "on optimization/rebalance",
        "RLS required",
    ),
    Capability(
        "execution.paper",
        "execution",
        "pre-trade checks and immutable paper blotter",
        "order_intent",
        "Paper Execution",
        ("My Portfolios", "Administration"),
        "on demand",
        "human approval required",
    ),
    Capability(
        "data.provenance",
        "artifact_registry",
        "hash-keyed source provenance and freshness",
        "data_provenance",
        "Data Quality",
        ("Administration", "Validation & Governance"),
        "every fetch",
        "mandatory for promotion",
    ),
    Capability(
        "publication.atomic",
        "artifact_registry",
        "staging validation and atomic active pointer",
        "publication_manifest",
        "Administration",
        ("Data Quality",),
        "every publication",
        "rollback-capable",
    ),
)


def manifest_records() -> list[dict]:
    records = []
    for capability in CAPABILITIES:
        record = asdict(capability)
        record["secondary_views"] = list(record["secondary_views"])
        records.append(record)
    return records


def write_manifest(path: str | Path) -> Path:
    target = Path(path)
    target.write_text(json.dumps(manifest_records(), indent=2, sort_keys=True), encoding="utf-8")
    return target


def validate_manifest() -> tuple[bool, tuple[str, ...]]:
    errors: list[str] = []
    ids = [capability.capability_id for capability in CAPABILITIES]
    if len(ids) != len(set(ids)):
        errors.append("capability_id values must be unique")
    for capability in CAPABILITIES:
        if not capability.owner_layer or not capability.canonical_artifact or not capability.primary_view:
            errors.append(f"{capability.capability_id}: required manifest fields are missing")
    return not errors, tuple(errors)
