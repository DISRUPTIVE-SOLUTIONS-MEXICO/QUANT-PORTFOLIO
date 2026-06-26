"""Core quantitative services for Quant Portfolio-Kaizen.

The Streamlit layer should render outputs from this package instead of
recomputing portfolio mathematics in the UI.
"""

from quant_core.contracts import (
    DataProvenanceRecord,
    FixedIncomeIntelligenceV1,
    MarketSnapshotV2,
    OrderIntentV1,
    PortfolioRunV2,
    PreTradeDecisionV1,
    PublicationManifest,
    ResearchEvidenceV2,
    RiskReportV2,
    SecurityIntelligenceV1,
    StrategyResearchV1,
)
from quant_core.fixed_income_intelligence import FixedIncomeIntelligenceConfig, build_fixed_income_intelligence
from quant_core.security_intelligence import SecurityIntelligenceConfig, build_security_intelligence
from quant_core.strategy_lab import STRATEGY_DEFINITIONS, build_strategy_lab_artifact
from quant_core.uncertainty_state import (
    StrategyConstitution,
    UncertaintyState,
    VarianceModelResult,
    upside_downside_diagnostics,
    xcdr_v3_growth_control_score,
    xodr_v1_omega_dominance_score,
)

__all__ = [
    "StrategyConstitution",
    "UncertaintyState",
    "VarianceModelResult",
    "upside_downside_diagnostics",
    "xcdr_v3_growth_control_score",
    "xodr_v1_omega_dominance_score",
    "DataProvenanceRecord",
    "FixedIncomeIntelligenceV1",
    "MarketSnapshotV2",
    "OrderIntentV1",
    "PortfolioRunV2",
    "PreTradeDecisionV1",
    "PublicationManifest",
    "ResearchEvidenceV2",
    "RiskReportV2",
    "SecurityIntelligenceV1",
    "StrategyResearchV1",
    "SecurityIntelligenceConfig",
    "build_security_intelligence",
    "FixedIncomeIntelligenceConfig",
    "build_fixed_income_intelligence",
    "STRATEGY_DEFINITIONS",
    "build_strategy_lab_artifact",
]
