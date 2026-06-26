# Quant Portfolio-Kaizen Institutional API Contract

The Python engine is the analytical authority. Next.js renders immutable
artifacts, authenticates users, and queues work. It never recomputes portfolio
metrics or bypasses a promotion gate.

## Publication Channels

Atomic pointers are independent:

```text
global:daily_snapshot
global:full_analysis
user:<user_id>:user_portfolio
research:research_evidence
```

A daily price refresh cannot replace the active full research publication.
Readers continue serving the prior active pointer while a new publication is
in `staging` or `validated`.

## Versioned Contracts

- `MarketSnapshotV2`
- `ResearchEvidenceV2`
- `StrategyResearchV1`
- `SecurityIntelligenceV1`
- `FixedIncomeIntelligenceV1`
- `PortfolioRunV2`
- `RiskReportV2`
- `DataProvenanceRecord`
- `PublicationManifest`
- `OrderIntentV1`
- `PreTradeDecisionV1`

The Pydantic definitions in `quant_core/contracts.py` are canonical. JSON is
serialized deterministically and hashed with SHA-256.

`SecurityIntelligenceV1` governs the daily per-instrument workbench. Its
render-ready artifact contains observed price history, asymmetric benchmark
betas, tail beta, residual momentum, drawdown state, CVaR, dollar-volume
liquidity and cross-strategy consensus. Every estimator is truncated at the
artifact decision date and remains labelled `live_snapshot`; it cannot be
misrepresented as OOS or holdout evidence.

`FixedIncomeIntelligenceV1` governs the rates workbench. The dashboard schema
`2026.06.15-market-intelligence-v10` requires:

- source-quality metrics with observed policy, 2Y and 10Y state;
- native-calendar event-time level, slope, policy-gap and curvature factors;
- local duration/convexity stress scenarios;
- SOFR, SONIA, ESTR and TONAR changes and staleness;
- explicit methodology and limitations.

A v11 daily publication is rejected when this contract, at least 126 factor
observations, stress scenarios or methodology are missing. A partial rates
refresh therefore cannot replace the previous active snapshot.

`StrategyResearchV1` is accompanied by immutable dashboard artifacts:

- `walk_forward_windows`: explicit train, purge, validation, embargo and test boundaries;
- `oos_summary`, `oos_price_paths`, `oos_drawdowns`: concatenated untouched test evidence;
- `selection_stability`: validation-only family selection frequency;
- `holdout_summary`, `holdout_price_paths`, `holdout_drawdowns`: frozen final holdout;
- `validation`: strict WRC, SPA, PBO, active-return, capture and downside gates.
- `exposure_diagnostics`, `exposure_timeline`: causal risk governor state and
  the render-ready exposure/cash series;
- `research_lineage`: immutable generation ancestry, holdout status and
  prospective-evidence start date.
- `strategy_registry`: versioned hypotheses, inputs, benchmark policies,
  failure modes and implementation admissibility;
- `candidate_equivalence`: canonical path mapping used to prevent numerically
  duplicate strategies from inflating the effective multiple-testing family.

Full-history family curves are diagnostic only and cannot satisfy promotion.
`StrategyResearchV1` additionally requires `Holdout_Independence=true`.
A holdout marked `CONSUMED_FOR_DIAGNOSIS` is valid diagnostic evidence but is
ineligible to promote the strategy generation that was designed from it.

## HTTP Routes

### `GET /api/dashboard`

Returns a merged rendering bundle:

- last active `full_analysis`;
- current `daily_snapshot` market overlay;
- notices identifying missing evidence.

No placeholder metric is inferred.

### `GET /api/capabilities`

Returns the Feature Preservation Manifest. Each capability identifies its
canonical owner, calculation, artifact, view, freshness and validation state.

### `POST /api/jobs`

Queues an authenticated XCDR optimization job.

```json
{
  "tickers": ["AAPL", "MSFT", "NVDA"],
  "benchmark_ticker": "QQQ",
  "filter_style": "growth",
  "objective": "xcdr_v3",
  "base_period": "5y",
  "portfolio_name": "Growth XCDR",
  "initial_capital": 100000,
  "monthly_contribution": 1000,
  "risk_aversion": 5,
  "max_drawdown": 0.2,
  "liquidity_need": "Medium",
  "base_currency": "USD"
}
```

`filter_style` is a backend input, not display metadata. It changes the
sector-relative fundamental constitution used by `score_cross_section` while
retaining PIT confidence, liquidity, Mahalanobis and downside penalties.

The route normalizes and deduplicates tickers, rejects concurrent jobs and
limits each user to three optimization requests per rolling 24 hours. The
worker, not the route, performs PIT filtering, XCDR optimization, validation,
run publication and immutable portfolio-version persistence.

### `GET /api/jobs?job_id=<uuid>`

Returns user-scoped job state. RLS prevents cross-user reads.

### `GET /api/portfolios`

Returns the authenticated user's immutable portfolio catalog.

### `GET /api/orders`

Returns user-scoped paper order intents and pre-trade decisions.

### `POST /api/orders`

Queues a Python pre-trade evaluation for an immutable portfolio version.

```json
{
  "portfolio_version_id": "uuid",
  "portfolio_value": 1000000,
  "current_weights": {"AAPL": 0.1},
  "acknowledgement": "paper_execution_only"
}
```

The API does not approve or route an order. The Python worker enforces
suitability, promotion, OOS scope, freshness, artifact integrity, MNPI
segregation, concentration, ADV participation and estimated cost.

## Promotion Rule

```text
recommended allocation =
  suitability approved
  AND research promoted
  AND OOS or holdout evidence
  AND no hard risk breach
```

Anything else must be labelled `research-only`, `watchlist`, or `blocked`.
