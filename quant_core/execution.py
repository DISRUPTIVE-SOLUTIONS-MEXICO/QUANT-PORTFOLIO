from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import UUID

import numpy as np

from quant_core.contracts import (
    EvidenceScope,
    OrderIntentV1,
    OrderLegV1,
    OrderStatus,
    PortfolioRunV2,
    PreTradeCheckV1,
    PreTradeDecisionV1,
)


@dataclass(frozen=True)
class PreTradePolicy:
    max_single_name_weight: float = 0.15
    max_adv_participation: float = 0.02
    max_estimated_cost_bps: float = 50.0
    max_data_age_hours: float = 30.0
    require_suitability_approval: bool = True
    require_strategy_promotion: bool = True
    require_human_approval: bool = True
    minimum_evidence_scope: EvidenceScope = EvidenceScope.OUT_OF_SAMPLE


def build_order_intent(
    portfolio: PortfolioRunV2,
    *,
    user_id: UUID,
    portfolio_value: float,
    current_weights: dict[str, float] | None = None,
    reference_prices: dict[str, float] | None = None,
    adv_usd: dict[str, float] | None = None,
    spread_bps: dict[str, float] | None = None,
    impact_bps: dict[str, float] | None = None,
) -> OrderIntentV1:
    current = {str(key).upper(): float(value) for key, value in (current_weights or {}).items()}
    prices = {str(key).upper(): float(value) for key, value in (reference_prices or {}).items()}
    adv = {str(key).upper(): float(value) for key, value in (adv_usd or {}).items()}
    spreads = {str(key).upper(): float(value) for key, value in (spread_bps or {}).items()}
    impacts = {str(key).upper(): float(value) for key, value in (impact_bps or {}).items()}
    legs: list[OrderLegV1] = []
    for position in portfolio.positions:
        ticker = position.ticker
        current_weight = current.get(ticker, position.current_weight)
        delta = position.target_weight - current_weight
        if abs(delta) <= 1e-8:
            continue
        reference_price = prices.get(ticker, position.reference_price)
        if reference_price is None or not np.isfinite(reference_price) or reference_price <= 0:
            raise ValueError(f"Missing valid reference price for {ticker}")
        notional = abs(delta) * float(portfolio_value)
        legs.append(
            OrderLegV1(
                ticker=ticker,
                side="BUY" if delta > 0 else "SELL",
                target_weight=position.target_weight,
                current_weight=current_weight,
                delta_weight=delta,
                reference_price=reference_price,
                estimated_quantity=notional / reference_price,
                estimated_notional=notional,
                adv_usd=adv.get(ticker, position.adv_usd),
                estimated_spread_bps=max(spreads.get(ticker, 0.0), 0.0),
                estimated_impact_bps=max(impacts.get(ticker, 0.0), 0.0),
            )
        )
    return OrderIntentV1(
        user_id=user_id,
        run_id=portfolio.run_id,
        portfolio_value=portfolio_value,
        base_currency=portfolio.base_currency,
        benchmark_xi=portfolio.benchmark_xi,
        evidence_scope=portfolio.evidence_scope,
        legs=tuple(legs),
        status=OrderStatus.DRAFT,
        config_hash=portfolio.config_hash,
        artifact_sha256=portfolio.sha256(),
    )


def evaluate_pretrade(
    order: OrderIntentV1,
    portfolio: PortfolioRunV2,
    *,
    policy: PreTradePolicy | None = None,
    data_age_hours: float,
    suitability_status: str | None = None,
    promotion_status: str | None = None,
    mnpi_flag: bool = False,
    artifact_hash_matches: bool = True,
) -> PreTradeDecisionV1:
    policy = policy or PreTradePolicy()
    suitability = suitability_status or portfolio.suitability_status
    promotion = promotion_status or portfolio.promotion_status
    checks: list[PreTradeCheckV1] = []

    def add(
        name: str,
        passed: bool,
        observed: Any,
        limit: Any,
        explanation: str,
        severity: Literal["hard", "warning"] = "hard",
    ) -> None:
        checks.append(
            PreTradeCheckV1(
                check=name,
                passed=bool(passed),
                observed=observed,
                limit=limit,
                explanation=explanation,
                severity=severity,
            )
        )

    add(
        "suitability",
        not policy.require_suitability_approval or suitability == "approved",
        suitability,
        "approved",
        "The portfolio must pass the investor suitability gate.",
    )
    add(
        "promotion",
        not policy.require_strategy_promotion or promotion == "promoted",
        promotion,
        "promoted",
        "Research-only strategies cannot advance to the paper blotter.",
    )
    add(
        "evidence_scope",
        order.evidence_scope in {EvidenceScope.OUT_OF_SAMPLE, EvidenceScope.HOLDOUT},
        order.evidence_scope.value,
        policy.minimum_evidence_scope.value,
        "Execution requires out-of-sample or holdout evidence.",
    )
    add(
        "data_freshness",
        np.isfinite(data_age_hours) and data_age_hours <= policy.max_data_age_hours,
        float(data_age_hours),
        policy.max_data_age_hours,
        "Reference prices and risk diagnostics must be fresh.",
    )
    add(
        "artifact_integrity",
        artifact_hash_matches,
        artifact_hash_matches,
        True,
        "Order inputs must match the immutable portfolio artifact.",
    )
    add(
        "mnpi_firewall",
        not mnpi_flag,
        mnpi_flag,
        False,
        "Material non-public information must never enter a shared or executable workflow.",
    )

    for leg in order.legs:
        add(
            f"single_name_weight:{leg.ticker}",
            leg.target_weight <= policy.max_single_name_weight + 1e-12,
            leg.target_weight,
            policy.max_single_name_weight,
            "Target weight must remain inside the single-name concentration cap.",
        )
        participation = (
            leg.estimated_notional / leg.adv_usd
            if leg.adv_usd is not None and np.isfinite(leg.adv_usd) and leg.adv_usd > 0
            else np.inf
        )
        add(
            f"adv_participation:{leg.ticker}",
            np.isfinite(participation) and participation <= policy.max_adv_participation,
            float(participation) if np.isfinite(participation) else "missing",
            policy.max_adv_participation,
            "Estimated notional must remain within the configured ADV participation cap.",
        )
        estimated_cost = leg.estimated_spread_bps + leg.estimated_impact_bps
        add(
            f"estimated_cost:{leg.ticker}",
            estimated_cost <= policy.max_estimated_cost_bps,
            estimated_cost,
            policy.max_estimated_cost_bps,
            "Spread and impact estimate must remain inside the execution-cost budget.",
        )

    hard_breaches = tuple(check.check for check in checks if not check.passed and check.severity == "hard")
    warnings = tuple(check.check for check in checks if not check.passed and check.severity == "warning")
    return PreTradeDecisionV1(
        order_intent_id=order.order_intent_id,
        evaluated_at=datetime.now(UTC),
        approved=not hard_breaches,
        checks=tuple(checks),
        hard_breaches=hard_breaches,
        warnings=warnings,
    )


def approval_status(decision: PreTradeDecisionV1, *, human_approved: bool) -> OrderStatus:
    if not decision.approved:
        return OrderStatus.PRETRADE_REJECTED
    return OrderStatus.APPROVED if human_approved else OrderStatus.AWAITING_APPROVAL
