# Quant Portfolio-Kaizen Institutional Architecture

## System Invariant

```text
Public data -> PIT/quality -> features -> signals -> benchmark xi
-> strategy research -> XCDR portfolio -> risk -> causal validation -> immutable artifacts
-> API -> Next.js/PWA + Electron
```

The UI is a renderer. Python is the source of truth. Supabase is the
authentication, persistence and audit boundary.

## Product Surfaces

- **Next.js/PWA**: production terminal for desktop and mobile.
- **Electron**: hardened desktop shell loading the same web interface.
- **Streamlit**: internal research laboratory and diagnostic fallback.
- **Python engine**: stateless quantitative computation and governed workers.
- **Supabase**: Auth, Postgres, RLS, immutable run artifacts and pointers.
- **GitHub Actions/local compute**: zero-cost scheduled refresh and research.

## Analytical Layers

```text
DataLayer
  -> PITQualityLayer
  -> FeatureLayer
  -> SignalLayer
  -> StrategyResearchLayer
  -> BenchmarkGovernanceLayer
  -> PortfolioLayer
  -> RiskLayer
  -> BacktestValidationLayer
  -> ArtifactRegistry
```

The Feature Preservation Manifest maps every capability to one canonical
calculation and artifact. Secondary screens reuse that artifact; they do not
duplicate calculations.

## Security Intelligence Workbench

Equity research includes a canonical per-instrument state built from observed
prices, volume and the latest causal strategy scores:

\[
\beta_i^{\pm}
=
\frac{\operatorname{Cov}(r_i,r_\xi\mid r_\xi\gtrless0)}
{\operatorname{Var}(r_\xi\mid r_\xi\gtrless0)},
\qquad
\beta_i^{tail}
=
\frac{\operatorname{Cov}(r_i,r_\xi\mid r_\xi\le q_{0.10})}
{\operatorname{Var}(r_\xi\mid r_\xi\le q_{0.10})}.
\]

Residual momentum compounds market-model residuals over the frozen lookback.
The same artifact carries drawdown, CVaR, realized volatility, ADV, Amihud
illiquidity and strategy-selection breadth. It is a causal live snapshot, not
backtest evidence, and the frontend performs no quantitative recomputation.

## Fixed-Income Intelligence Workbench

Rates research has one canonical public-data artifact. It preserves each
source's native observation calendar and constructs event-time factors only
from observations available at the decision date:

\[
L_t=\frac{y_t(2Y)+y_t(10Y)}{2},\qquad
S_t=y_t(10Y)-y_t(2Y),\qquad
C_t=2y_t(2Y)-y_t(short)-y_t(10Y).
\]

No daily interpolation is used to make policy or monthly series look
continuous. A country needs at least two observed sovereign maturities before
curve stress analytics are admissible. Local duration/convexity scenarios use

\[
\frac{\Delta P}{P}
\approx
-D_{\mathrm{mod}}\Delta y+\frac{1}{2}\mathcal{C}(\Delta y)^2
\]

and are labelled sensitivity proxies rather than executable bond valuations.
The artifact also records source quality, staleness, reference-rate changes
and FX-adjusted carry diagnostics.

## Quantitative Constitution

`StrategyConstitution` freezes:

- permitted features and availability-date requirements;
- benchmark xi and Omega stress set;
- complexity and trial budgets;
- transaction cost, ADV, concentration and sector limits;
- WRC, SPA, PBO, ICIR, CVaR and drawdown promotion thresholds;
- prohibition on test/holdout model selection;
- absolute segregation of MNPI/private information.

Every result declares one evidence scope:

```text
in_sample | validation | out_of_sample | holdout | live_snapshot
```

## Strategy Laboratory

General quantitative strategy research is separated from portfolio
optimization. The canonical laboratory pre-registers price-derived,
long-only candidate families and enforces:

```text
signal close t -> execution t+1 -> explicit turnover cost
-> train -> purge -> validation-only selection -> embargo -> untouched test
-> concatenated nested OOS -> frozen candidate -> final holdout
-> WRC/SPA/PBO and downside promotion gate
```

Full-history candidate paths remain diagnostics. Family selection is promoted
only when concatenated untouched test blocks and the frozen final holdout pass
every statistical and downside gate. Current families include cross-sectional
momentum, residual momentum versus xi, asymmetric capture, defensive convexity
and residual mean reversion.

Material strategy changes create a new immutable research generation. If a
historical holdout informed the diagnosis that motivated a change, that
holdout is marked `CONSUMED_FOR_DIAGNOSIS` and cannot promote the new
generation. Promotion then requires prospective evidence beginning after the
generation timestamp.

The candidate matrix applies exact path-equivalence control before nested
selection and WRC/SPA/PBO. Distinct hypotheses remain in the registry, but
identical realized return paths count as one effective trial:

\[
\mathcal{K}_{eff}
=
\mathcal{K}/\sim,\qquad
k_i\sim k_j
\Longleftrightarrow
\max_t |r_t^{(i)}-r_t^{(j)}|\le\varepsilon.
\]

The governed strategy registry is broader than the executable candidate set.
Every family declares its hypothesis, horizon, benchmark policy, required
inputs, availability rule, failure modes, liquidity assumptions and evidence
status. Families blocked by missing PIT histories or execution constraints
remain visible but cannot enter portfolio selection:

```text
strategy_registry
  -> implemented + engine_candidate
  -> planned_pit / planned_data
  -> blocked_data / blocked_execution
```

The first governed downside generation applies a fixed causal exposure state:

\[
e_t=\operatorname{clip}\left(
0.20+0.80(0.35T_t+0.25D_t+0.20V_t+0.20S_t),0.25,1
\right),
\]

where trend, drawdown, volatility clustering and downside semivariance use
benchmark observations available at the signal close. Risk weights are scaled
by \(e_t\); \(1-e_t\) remains in zero-return cash. Thresholds are frozen before
prospective evaluation and are never tuned on test or consumed holdout data.

## Atomic Publications

Daily market data and full research have separate pointers:

```text
global:daily_snapshot
global:full_analysis
```

Publication lifecycle:

```text
staging -> artifact validation -> validated -> atomic promotion -> active
```

Failure preserves the previous active pointer. Hashes, quality checks and
rejection reasons are append-only.

## Portfolio And Paper Execution

Saved user portfolios are immutable versions protected by Auth and RLS.
Paper execution follows:

```text
PortfolioRunV2
-> OrderIntentV1
-> Python pre-trade evaluation
-> PreTradeDecisionV1
-> named human approval
-> simulated fill
```

There is no broker connectivity or real order routing.

## Mathematical Core

The active objective is XCDR/XODR research, not Sortino:

\[
\max_{w\in\Delta_N}
\frac{
\operatorname{ActiveReturn}(w,\xi)+
\lambda_U\operatorname{UpsideCapture}(w,\xi)
}{
\epsilon+D_-(w)+\lambda_C\operatorname{CVaR}(w)
+\lambda_D|DD(w)|+\lambda_U^{risk}U(w)
}
-\lambda_T TO(w).
\]

Fundamentals are sector-relative; covariance is shrinkage/RMT-cleaned;
uncertainty layers include Kalman, Fisher/CRLB, entropy, PELT,
ARCH/GARCH/EGARCH, EVT and governed Volterra diagnostics.

## Trust Boundaries

1. Public sources are untrusted input and always carry provenance/freshness.
2. The service-role key exists only in workers and server routes.
3. Browser clients use anon keys plus user JWT and RLS.
4. User portfolio and order data are tenant-scoped.
5. Private/MNPI information never enters shared signals, RAG or publications.
6. The Electron renderer has no Node.js access and cannot open arbitrary URLs.
