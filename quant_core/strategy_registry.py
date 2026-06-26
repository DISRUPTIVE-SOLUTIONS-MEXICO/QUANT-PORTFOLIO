from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class StrategySpecification:
    strategy_id: str
    label: str
    family: str
    asset_class: str
    directionality: str
    lookback_days: int
    holding_horizon: str
    hypothesis: str
    signal_formula: str
    benchmark_policy: str
    required_inputs: tuple[str, ...]
    availability_rule: str
    suitable_regimes: tuple[str, ...]
    failure_modes: tuple[str, ...]
    cost_sensitivity: str
    liquidity_requirement: str
    implementation_status: str
    evidence_status: str
    engine_candidate: bool = False


STRATEGY_SPECIFICATIONS: tuple[StrategySpecification, ...] = (
    StrategySpecification(
        "cross_sectional_momentum_12_1",
        "Cross-sectional momentum 12-1",
        "Equity factor",
        "Listed equities",
        "Long-only cross-sectional",
        252,
        "1-3 months",
        "Persistent relative strength may survive after excluding the most recent reversal-prone month.",
        "M_i=P_{i,t-21}/P_{i,t-252}-1",
        "Mandate-specific equity benchmark xi",
        ("Adjusted close", "Trading calendar"),
        "Signal close t; execution starts t+1",
        ("expansion", "recovery"),
        ("momentum crash", "crowding", "high turnover"),
        "medium",
        "daily dollar volume required",
        "implemented",
        "nested OOS; generation holdout consumed",
        True,
    ),
    StrategySpecification(
        "volatility_adjusted_trend",
        "Volatility-adjusted trend",
        "Trend following",
        "Listed equities",
        "Long-only time-series filtered",
        252,
        "1-3 months",
        "Positive persistent trends scaled by recent realized risk may improve signal comparability across names.",
        "T_i=[P_{i,t-21}/P_{i,t-252}-1]_+/(sigma_{i,63}sqrt(252))",
        "Mandate-specific equity benchmark xi",
        ("Adjusted close", "Trading calendar"),
        "Signal close t; execution starts t+1",
        ("expansion", "recovery"),
        ("whipsaw", "gap risk", "volatility estimator lag"),
        "medium",
        "daily dollar volume required",
        "implemented",
        "new G3 candidate; prospective evidence required",
        True,
    ),
    StrategySpecification(
        "dual_momentum",
        "Dual momentum",
        "Trend and relative strength",
        "Listed equities",
        "Long-only with absolute and relative filters",
        252,
        "1-3 months",
        "Assets should receive capital only when both absolute and benchmark-relative momentum are positive.",
        "D_i=1{M_i>0}1{M_i-M_xi>0}(M_i-M_xi)",
        "Mandate-specific equity benchmark xi",
        ("Adjusted close", "Benchmark adjusted close"),
        "Signal close t; execution starts t+1",
        ("expansion", "recovery"),
        ("late exits", "benchmark regime reversal", "concentrated leadership"),
        "low-medium",
        "daily dollar volume required",
        "implemented",
        "new G3 candidate; prospective evidence required",
        True,
    ),
    StrategySpecification(
        "residual_momentum_6m",
        "Residual momentum versus xi",
        "Benchmark-relative alpha",
        "Listed equities",
        "Long-only residual ranking",
        126,
        "1-2 months",
        "Idiosyncratic trend orthogonalized to the mandate benchmark may be more portable than raw beta.",
        "RM_i=sum(r_i-beta_i,xi r_xi)",
        "Causally selected benchmark xi",
        ("Adjusted close", "Benchmark adjusted close"),
        "Rolling beta and signal use observations through t",
        ("expansion", "recovery"),
        ("beta instability", "omitted factors", "residual crowding"),
        "medium",
        "daily dollar volume required",
        "implemented",
        "nested OOS; generation holdout consumed",
        True,
    ),
    StrategySpecification(
        "asymmetric_capture",
        "Asymmetric capture",
        "Conditional beta",
        "Listed equities",
        "Long-only convexity ranking",
        126,
        "1-2 months",
        "Favorable upside versus downside participation can improve benchmark-relative payoff asymmetry.",
        "A_i=beta_i^+-beta_i^- -0.5[TailBeta_i-1]_+",
        "Causally selected benchmark xi",
        ("Adjusted close", "Benchmark adjusted close"),
        "Conditional samples end at signal close t",
        ("expansion", "fragile"),
        ("small conditional samples", "tail-state instability", "hidden factor exposure"),
        "medium",
        "daily dollar volume required",
        "implemented",
        "nested OOS; generation holdout consumed",
        True,
    ),
    StrategySpecification(
        "defensive_convexity",
        "Defensive convexity",
        "Defensive equity",
        "Listed equities",
        "Long-only downside-aware",
        126,
        "1-2 months",
        "Positive trend with controlled downside deviation and conditional beta asymmetry may preserve capital.",
        "C_i=M_i+0.35(beta_i^+-beta_i^-)-4D_i^-",
        "Causally selected benchmark xi",
        ("Adjusted close", "Benchmark adjusted close"),
        "All moments use observations through t",
        ("fragile", "stress", "recovery"),
        ("defensive crowding", "upside sacrifice", "rate sensitivity"),
        "low-medium",
        "daily dollar volume required",
        "implemented",
        "nested OOS; generation holdout consumed",
        True,
    ),
    StrategySpecification(
        "short_term_residual_reversion",
        "Short-term residual reversion",
        "Mean reversion",
        "Listed equities",
        "Long-only residual reversal",
        63,
        "5-21 trading days",
        "Large short-horizon benchmark-residual moves may partially mean revert after liquidity shocks.",
        "R_i=-sum_{j=0}^4 epsilon_{i,t-j}/sigma(epsilon_i,21)",
        "Causally selected benchmark xi",
        ("Adjusted close", "Benchmark adjusted close"),
        "Residual and scale end at signal close t",
        ("fragile", "recovery"),
        ("catching structural breaks", "transaction costs", "earnings gaps"),
        "high",
        "high liquidity required",
        "implemented",
        "nested OOS; generation holdout consumed",
        True,
    ),
    StrategySpecification(
        "downside_controlled_momentum",
        "Downside-controlled momentum",
        "Dynamic risk allocation",
        "Listed equities and cash",
        "Long-only momentum with causal cash throttle",
        252,
        "1-3 months",
        "Momentum exposure should contract when benchmark trend, drawdown and volatility state deteriorate.",
        "w_t^risk=e_t w_t, e_t=clip(g(T_t,DD_t,V_t,S_t),0.25,1)",
        "Causally selected benchmark xi",
        ("Adjusted close", "Benchmark adjusted close", "Cash convention"),
        "Governor state uses benchmark observations through t",
        ("expansion", "recovery", "fragile", "stress"),
        ("cash drag", "fast gap risk", "state-estimator lag"),
        "low-medium",
        "daily dollar volume required",
        "implemented",
        "G2 diagnostic; prospective evidence required",
        True,
    ),
    StrategySpecification(
        "sector_relative_quality_growth",
        "Sector-relative quality and growth",
        "Fundamental equity",
        "Listed equities",
        "Long-only sector-neutral",
        0,
        "3-12 months",
        "Sector-relative ROIC, FCF, revenue growth and balance-sheet quality may identify durable compounders.",
        "Q_i=z_sector(ROIC,FCF,RevenueGrowth,Leverage,Piotroski)",
        "Country and sector matched benchmark xi",
        ("SEC filing facts", "Availability_Date", "Sector taxonomy", "Adjusted close"),
        "Only filings accepted on or before signal date",
        ("expansion", "recovery"),
        ("stale filings", "accounting comparability", "valuation compression"),
        "low",
        "monthly dollar volume required",
        "planned_pit",
        "requires a dedicated PIT panel before causal execution",
    ),
    StrategySpecification(
        "cross_asset_time_series_momentum",
        "Cross-asset time-series momentum",
        "Macro trend",
        "Futures and liquid ETFs",
        "Long-only V1; long-short research extension",
        0,
        "1-6 months",
        "Persistent trends across equity, rates, commodities and FX can diversify equity-specific risk.",
        "TSMOM_a=sign(R_a,12m)/sigma_a",
        "Risk-balanced cross-asset composite",
        ("Continuous futures or liquid ETF prices", "Roll-adjustment metadata"),
        "Signals use only completed sessions and causal rolls available through t",
        ("expansion", "stress", "recovery"),
        ("whipsaw", "roll bias", "proxy ETF tracking error"),
        "medium",
        "liquid ETF or futures proxy required",
        "planned_data",
        "ETF implementation feasible; futures-quality histories remain incomplete",
    ),
    StrategySpecification(
        "sovereign_curve_carry_roll",
        "Sovereign carry and roll-down",
        "Fixed income relative value",
        "Sovereign rates",
        "Long-only duration buckets",
        0,
        "1-6 months",
        "Carry plus roll-down can be harvested when curve shape compensates for duration and policy risk.",
        "CR_m=Carry_m+RollDown_m-lambda_DV01 DV01_m-lambda_tail Tail_m",
        "Currency and duration matched sovereign benchmark",
        ("Official yield curves", "Tenor mapping", "FX hedge proxy"),
        "Curve observations must be source-dated and available before allocation",
        ("dovish", "stable_inflation"),
        ("policy shock", "curve interpolation", "unhedged FX"),
        "low",
        "sovereign ETF or paper instrument proxy",
        "planned_data",
        "curve analytics available; executable instrument mapping pending",
    ),
    StrategySpecification(
        "fx_carry_value_momentum",
        "FX carry, value and momentum",
        "Currency allocation",
        "FX",
        "Long-only funded basket",
        0,
        "1-3 months",
        "Rate differential should be combined with valuation, momentum and crash-risk controls.",
        "FX_i=Carry_i+Value_i+Momentum_i-lambda CrashBeta_i",
        "Base-currency cash or broad USD index",
        ("Policy rates", "Spot FX", "Forward or CIP proxy", "Event calendar"),
        "All rates and FX observations must be available before the decision timestamp",
        ("stable_policy", "risk_on"),
        ("carry crash", "funding squeeze", "basis risk"),
        "medium",
        "liquid currency ETF or institutional FX execution",
        "planned_data",
        "carry suggestions exist; causal hedge and execution histories pending",
    ),
    StrategySpecification(
        "variance_risk_premium",
        "Variance risk premium",
        "Options volatility",
        "Index options",
        "Defined-risk option structures",
        0,
        "1-2 months",
        "Implied variance can exceed subsequent realized variance, but left-tail exposure must be explicitly bounded.",
        "VRP=IV^2-E_t[RV^2]",
        "Underlying index plus cash collateral",
        ("Historical option surfaces", "Bid/ask", "Open interest", "Realized variance"),
        "Historical chain must be timestamped before trade construction",
        ("range_bound", "stable_volatility"),
        ("short-vol crash", "surface sparsity", "liquidity and assignment"),
        "high",
        "institutional options liquidity required",
        "blocked_data",
        "Yahoo snapshot is insufficient for causal historical validation",
    ),
    StrategySpecification(
        "statistical_arbitrage_pairs",
        "Statistical arbitrage pairs",
        "Relative value",
        "Listed equities",
        "Market-neutral long-short",
        0,
        "1-20 trading days",
        "Stable cointegration and residual mean reversion may support hedged relative-value trades.",
        "z_t=(spread_t-mu_t)/sigma_t",
        "Cash plus factor-neutral residual benchmark",
        ("Survivorship-aware universe", "Borrow availability", "Bid/ask", "Intraday execution"),
        "Pair selection and hedge ratio must be frozen before test",
        ("range_bound", "high_dispersion"),
        ("cointegration break", "borrow recall", "crowded unwind"),
        "very high",
        "high liquidity and short availability required",
        "blocked_execution",
        "not admissible for retail-safe long-only production",
    ),
    StrategySpecification(
        "event_driven_earnings",
        "Earnings event drift",
        "Event driven",
        "Listed equities",
        "Long-only event basket",
        0,
        "2-60 trading days",
        "Unexpected earnings information may diffuse gradually when surprise and guidance are measured point-in-time.",
        "PEAD_i=Surprise_i x Revision_i x Liquidity_i",
        "Sector and country matched event benchmark",
        ("Historical earnings timestamps", "Consensus history", "Guidance revisions", "Prices"),
        "Publication timestamp must be available before the first executable session",
        ("expansion", "high_dispersion"),
        ("timestamp leakage", "consensus revisions", "gap execution"),
        "high",
        "high liquidity required",
        "blocked_data",
        "free consensus histories are not sufficiently point-in-time",
    ),
)


STRATEGY_DEFINITIONS: tuple[StrategySpecification, ...] = tuple(
    specification for specification in STRATEGY_SPECIFICATIONS if specification.engine_candidate
)


def strategy_registry_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for specification in STRATEGY_SPECIFICATIONS:
        row = asdict(specification)
        for key in ("required_inputs", "suitable_regimes", "failure_modes"):
            row[key] = " | ".join(row[key])
        rows.append(
            {
                "Strategy_ID": row["strategy_id"],
                "Strategy": row["label"],
                "Family": row["family"],
                "Asset_Class": row["asset_class"],
                "Directionality": row["directionality"],
                "Holding_Horizon": row["holding_horizon"],
                "Hypothesis": row["hypothesis"],
                "Signal_Formula": row["signal_formula"],
                "Benchmark_Policy": row["benchmark_policy"],
                "Required_Inputs": row["required_inputs"],
                "Availability_Rule": row["availability_rule"],
                "Suitable_Regimes": row["suitable_regimes"],
                "Failure_Modes": row["failure_modes"],
                "Cost_Sensitivity": row["cost_sensitivity"],
                "Liquidity_Requirement": row["liquidity_requirement"],
                "Implementation_Status": row["implementation_status"],
                "Evidence_Status": row["evidence_status"],
                "Engine_Candidate": row["engine_candidate"],
            }
        )
    return pd.DataFrame(rows)


def validate_strategy_registry() -> tuple[bool, tuple[str, ...]]:
    errors: list[str] = []
    identifiers = [specification.strategy_id for specification in STRATEGY_SPECIFICATIONS]
    if len(identifiers) != len(set(identifiers)):
        errors.append("strategy_id values must be unique")
    for specification in STRATEGY_SPECIFICATIONS:
        if specification.engine_candidate and specification.implementation_status != "implemented":
            errors.append(f"{specification.strategy_id}: engine candidates must be implemented")
        if specification.engine_candidate and specification.lookback_days < 21:
            errors.append(f"{specification.strategy_id}: engine candidates require a causal lookback")
        if not specification.required_inputs:
            errors.append(f"{specification.strategy_id}: required inputs are missing")
        if not specification.failure_modes:
            errors.append(f"{specification.strategy_id}: failure modes are missing")
        availability = specification.availability_rule.lower()
        if not any(marker in availability for marker in ("t+1", "before", "through t", "end at signal close")):
            errors.append(f"{specification.strategy_id}: availability rule is not explicit")
    return not errors, tuple(errors)
