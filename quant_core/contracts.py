from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CONTRACT_SCHEMA_VERSION = "2026.06.15-institutional-v5"


class FrozenContract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


class EvidenceScope(StrEnum):
    IN_SAMPLE = "in_sample"
    VALIDATION = "validation"
    OUT_OF_SAMPLE = "out_of_sample"
    HOLDOUT = "holdout"
    LIVE_SNAPSHOT = "live_snapshot"


class PublicationState(StrEnum):
    STAGING = "staging"
    VALIDATED = "validated"
    ACTIVE = "active"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


class OrderStatus(StrEnum):
    DRAFT = "draft"
    PRETRADE_REJECTED = "pretrade_rejected"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    PAPER_SUBMITTED = "paper_submitted"
    PAPER_FILLED = "paper_filled"
    CANCELLED = "cancelled"


class DataProvenanceRecord(FrozenContract):
    schema_version: str = CONTRACT_SCHEMA_VERSION
    dataset: str
    source: str
    source_url: str | None = None
    retrieved_at: datetime
    availability_date: datetime | None = None
    content_sha256: str
    license_note: str
    rows: int = Field(ge=0)
    stale: bool = False
    fallback_used: bool = False
    quality_warnings: tuple[str, ...] = ()


class MarketSnapshotV2(FrozenContract):
    schema_version: str = CONTRACT_SCHEMA_VERSION
    snapshot_id: UUID = Field(default_factory=uuid4)
    as_of: datetime
    evidence_scope: EvidenceScope = EvidenceScope.LIVE_SNAPSHOT
    benchmark_xi: str
    stress_set_omega: tuple[str, ...] = ()
    market_regime: str = "unknown"
    rates_regime: str = "unknown"
    macro_state: dict[str, Any] = Field(default_factory=dict)
    yield_curves: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    sentiment: dict[str, Any] = Field(default_factory=dict)
    geopolitical: dict[str, Any] = Field(default_factory=dict)
    freshness: tuple[DataProvenanceRecord, ...] = ()

    @field_validator("benchmark_xi")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        normalized = value.upper()
        if not normalized:
            raise ValueError("benchmark_xi is required")
        return normalized


class ResearchEvidenceV2(FrozenContract):
    schema_version: str = CONTRACT_SCHEMA_VERSION
    strategy_id: str
    strategy_version: str
    benchmark_xi: str
    stress_set_omega: tuple[str, ...]
    evidence_scope: EvidenceScope
    observation_start: datetime
    observation_end: datetime
    oos_windows: int = Field(ge=0)
    metrics: dict[str, float | int | str | bool | None] = Field(default_factory=dict)
    promotion_tests: tuple[dict[str, Any], ...] = ()
    promoted: bool = False
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_dates_and_promotion(self) -> ResearchEvidenceV2:
        if self.observation_end < self.observation_start:
            raise ValueError("observation_end must be on or after observation_start")
        if self.promoted and self.evidence_scope not in {EvidenceScope.OUT_OF_SAMPLE, EvidenceScope.HOLDOUT}:
            raise ValueError("Only OOS or holdout evidence can be promoted")
        return self


class StrategyResearchV1(FrozenContract):
    schema_version: str = CONTRACT_SCHEMA_VERSION
    strategy_family: str
    candidate_ids: tuple[str, ...]
    benchmark_xi: str
    evidence_scope: EvidenceScope
    observation_start: datetime
    observation_end: datetime
    signal_lag_days: int = Field(ge=1)
    rebalance_days: int = Field(ge=1)
    transaction_cost_bps: float = Field(ge=0.0)
    selected_candidate: str | None = None
    promotion_status: Literal["research_only", "promoted", "rejected"] = "research_only"
    validation_metrics: dict[str, float | int | str | bool | None] = Field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_strategy_research(self) -> StrategyResearchV1:
        if self.observation_end < self.observation_start:
            raise ValueError("observation_end must be on or after observation_start")
        if self.selected_candidate and self.selected_candidate not in self.candidate_ids:
            raise ValueError("selected_candidate must be included in candidate_ids")
        if self.promotion_status == "promoted" and self.evidence_scope not in {
            EvidenceScope.OUT_OF_SAMPLE,
            EvidenceScope.HOLDOUT,
        }:
            raise ValueError("Promoted strategy research requires OOS or holdout evidence")
        if self.promotion_status == "promoted":
            required = {
                "WRC_p": lambda value: float(value) < 0.05,
                "SPA_p": lambda value: float(value) < 0.05,
                "PBO": lambda value: float(value) < 0.10,
                "OOS_Active_Return": lambda value: float(value) > 0.0,
                "OOS_Upside_Capture": lambda value: float(value) > 1.0,
                "OOS_Downside_Capture": lambda value: float(value) < 1.0,
                "OOS_Downside_Preservation": lambda value: value is True,
                "Holdout_Active_Return": lambda value: float(value) > 0.0,
                "Holdout_Downside_Preservation": lambda value: value is True,
                "Holdout_Independence": lambda value: value is True,
            }
            failures = [
                metric
                for metric, predicate in required.items()
                if metric not in self.validation_metrics or not predicate(self.validation_metrics[metric])
            ]
            if failures:
                raise ValueError(f"Promoted strategy research failed strict gates: {', '.join(failures)}")
        return self


class StrategySpecificationV1(FrozenContract):
    schema_version: str = CONTRACT_SCHEMA_VERSION
    strategy_id: str
    label: str
    family: str
    asset_class: str
    directionality: str
    lookback_days: int = Field(ge=0)
    holding_horizon: str
    hypothesis: str
    signal_formula: str
    benchmark_policy: str
    required_inputs: tuple[str, ...]
    availability_rule: str
    suitable_regimes: tuple[str, ...]
    failure_modes: tuple[str, ...]
    cost_sensitivity: Literal["low", "low-medium", "medium", "high", "very high"]
    liquidity_requirement: str
    implementation_status: Literal["implemented", "planned_pit", "planned_data", "blocked_data", "blocked_execution"]
    evidence_status: str
    engine_candidate: bool = False

    @model_validator(mode="after")
    def validate_specification(self) -> StrategySpecificationV1:
        if self.engine_candidate and self.implementation_status != "implemented":
            raise ValueError("Engine candidates must be implemented")
        if self.engine_candidate and self.lookback_days < 21:
            raise ValueError("Engine candidates require at least 21 observations")
        if not self.required_inputs:
            raise ValueError("required_inputs cannot be empty")
        if not self.failure_modes:
            raise ValueError("failure_modes cannot be empty")
        return self


class SecurityIntelligenceV1(FrozenContract):
    schema_version: str = CONTRACT_SCHEMA_VERSION
    as_of: datetime
    evidence_scope: EvidenceScope = EvidenceScope.LIVE_SNAPSHOT
    benchmark_xi: str
    method: str = "causal_security_intelligence_v1"
    tickers: tuple[str, ...]
    minimum_observations: int = Field(ge=63)
    price_history_days: int = Field(ge=63)
    formulas: dict[str, str]

    @field_validator("benchmark_xi")
    @classmethod
    def normalize_security_benchmark(cls, value: str) -> str:
        normalized = value.upper()
        if not normalized:
            raise ValueError("benchmark_xi is required")
        return normalized

    @model_validator(mode="after")
    def validate_security_intelligence(self) -> SecurityIntelligenceV1:
        if not self.tickers:
            raise ValueError("at least one security is required")
        if self.benchmark_xi not in self.tickers:
            raise ValueError("benchmark_xi must be included in tickers")
        required = {"beta", "tail_beta", "residual_momentum", "drawdown"}
        missing = required.difference(self.formulas)
        if missing:
            raise ValueError(f"missing security intelligence formulas: {', '.join(sorted(missing))}")
        return self


class FixedIncomeIntelligenceV1(FrozenContract):
    schema_version: str = CONTRACT_SCHEMA_VERSION
    as_of: datetime
    evidence_scope: EvidenceScope = EvidenceScope.LIVE_SNAPSHOT
    method: str = "causal_fixed_income_intelligence_v1"
    countries: tuple[str, ...]
    minimum_real_tenors: int = Field(default=2, ge=2)
    factor_observation_mode: str
    formulas: dict[str, str]

    @model_validator(mode="after")
    def validate_fixed_income_intelligence(self) -> FixedIncomeIntelligenceV1:
        if not self.countries:
            raise ValueError("at least one country is required")
        required = {"level", "slope", "curvature_proxy", "duration_convexity", "quality"}
        missing = required.difference(self.formulas)
        if missing:
            raise ValueError(f"missing fixed-income formulas: {', '.join(sorted(missing))}")
        if "native_calendar" not in self.factor_observation_mode:
            raise ValueError("fixed-income factors must preserve native observation calendars")
        return self


class PortfolioPositionV2(FrozenContract):
    ticker: str
    sector: str | None = None
    country: str | None = None
    target_weight: float = Field(ge=0.0, le=1.0)
    current_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    reference_price: float | None = Field(default=None, gt=0.0)
    adv_usd: float | None = Field(default=None, ge=0.0)
    composite_score: float | None = None
    pit_confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("ticker")
    @classmethod
    def normalize_position_ticker(cls, value: str) -> str:
        normalized = value.upper()
        if not normalized:
            raise ValueError("ticker is required")
        return normalized


class PortfolioRunV2(FrozenContract):
    schema_version: str = CONTRACT_SCHEMA_VERSION
    run_id: UUID = Field(default_factory=uuid4)
    user_id: UUID | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    as_of: datetime
    portfolio_name: str
    base_currency: str = "USD"
    objective: str = "xcdr_v3"
    benchmark_xi: str
    stress_set_omega: tuple[str, ...] = ()
    evidence_scope: EvidenceScope
    positions: tuple[PortfolioPositionV2, ...]
    config_hash: str
    data_hash: str
    model_version: str
    suitability_status: Literal["approved", "blocked", "watchlist"]
    promotion_status: Literal["promoted", "research_only", "rejected", "watchlist"]

    @model_validator(mode="after")
    def validate_weights(self) -> PortfolioRunV2:
        total = sum(position.target_weight for position in self.positions)
        if self.positions and abs(total - 1.0) > 1e-6:
            raise ValueError(f"target weights must sum to 1.0, observed {total:.10f}")
        if self.promotion_status == "promoted" and self.evidence_scope not in {
            EvidenceScope.OUT_OF_SAMPLE,
            EvidenceScope.HOLDOUT,
        }:
            raise ValueError("Promoted portfolios require OOS or holdout evidence")
        return self


class RiskReportV2(FrozenContract):
    schema_version: str = CONTRACT_SCHEMA_VERSION
    run_id: UUID
    as_of: datetime
    evidence_scope: EvidenceScope
    annualized_return: float | None = None
    annualized_volatility: float | None = Field(default=None, ge=0.0)
    downside_deviation: float | None = Field(default=None, ge=0.0)
    cvar_95_daily: float | None = None
    max_drawdown: float | None = Field(default=None, ge=-1.0, le=0.0)
    upside_capture: float | None = None
    downside_capture: float | None = None
    beta_to_xi: float | None = None
    tracking_error: float | None = Field(default=None, ge=0.0)
    xcdr_v3: float | None = None
    variance_model: str | None = None
    pelt_regime: str | None = None
    risk_contributions: dict[str, float] = Field(default_factory=dict)
    limit_breaches: tuple[str, ...] = ()


class ArtifactDescriptor(FrozenContract):
    name: str
    content_sha256: str
    bytes: int = Field(ge=0)
    required: bool = True
    evidence_scope: EvidenceScope | None = None


class PublicationManifest(FrozenContract):
    schema_version: str = CONTRACT_SCHEMA_VERSION
    publication_id: UUID = Field(default_factory=uuid4)
    run_id: UUID
    channel: Literal["global", "user", "research"]
    publication_kind: Literal["daily_snapshot", "full_analysis", "user_portfolio", "research_evidence"]
    state: PublicationState = PublicationState.STAGING
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    validated_at: datetime | None = None
    activated_at: datetime | None = None
    supersedes_publication_id: UUID | None = None
    artifacts: tuple[ArtifactDescriptor, ...]
    quality_checks: dict[str, bool] = Field(default_factory=dict)
    rejection_reasons: tuple[str, ...] = ()

    def is_publishable(self) -> bool:
        required = [artifact for artifact in self.artifacts if artifact.required]
        return bool(required) and all(self.quality_checks.values()) and not self.rejection_reasons


class OrderLegV1(FrozenContract):
    ticker: str
    side: Literal["BUY", "SELL"]
    target_weight: float = Field(ge=0.0, le=1.0)
    current_weight: float = Field(ge=0.0, le=1.0)
    delta_weight: float
    reference_price: float = Field(gt=0.0)
    estimated_quantity: float = Field(ge=0.0)
    estimated_notional: float = Field(ge=0.0)
    adv_usd: float | None = Field(default=None, ge=0.0)
    estimated_spread_bps: float = Field(default=0.0, ge=0.0)
    estimated_impact_bps: float = Field(default=0.0, ge=0.0)


class OrderIntentV1(FrozenContract):
    schema_version: str = CONTRACT_SCHEMA_VERSION
    order_intent_id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    run_id: UUID
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    portfolio_value: float = Field(gt=0.0)
    base_currency: str = "USD"
    benchmark_xi: str
    evidence_scope: EvidenceScope
    legs: tuple[OrderLegV1, ...]
    status: OrderStatus = OrderStatus.DRAFT
    human_approval_required: bool = True
    approved_by: UUID | None = None
    approved_at: datetime | None = None
    config_hash: str
    artifact_sha256: str

    @model_validator(mode="after")
    def approval_integrity(self) -> OrderIntentV1:
        if self.status in {OrderStatus.APPROVED, OrderStatus.PAPER_SUBMITTED, OrderStatus.PAPER_FILLED}:
            if not self.approved_by or not self.approved_at:
                raise ValueError("approved orders require approved_by and approved_at")
        return self


class PreTradeCheckV1(FrozenContract):
    check: str
    passed: bool
    observed: float | str | bool | None = None
    limit: float | str | bool | None = None
    severity: Literal["hard", "warning"] = "hard"
    explanation: str


class PreTradeDecisionV1(FrozenContract):
    schema_version: str = CONTRACT_SCHEMA_VERSION
    decision_id: UUID = Field(default_factory=uuid4)
    order_intent_id: UUID
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    approved: bool
    checks: tuple[PreTradeCheckV1, ...]
    hard_breaches: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def decision_consistency(self) -> PreTradeDecisionV1:
        failed_hard = [check.check for check in self.checks if not check.passed and check.severity == "hard"]
        if self.approved and (failed_hard or self.hard_breaches):
            raise ValueError("approved decision cannot contain hard breaches")
        return self
