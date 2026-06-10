# Quant Portfolio-Kaizen Architecture

## Principle

```text
Core = source of truth
UI   = renderer
DB   = append-only audit layer
```

## Local Research Mode

```text
yfinance + Stooq / SEC / FRED / Banxico / INEGI / central banks / scraping
  -> quant_core.data (providers, reconciliation, provenance, quality)
  -> quant_core + quant_stockpicker_core.py
  -> dashboard_payload
  -> Streamlit renderer
  -> Supabase artifacts + local .quant_cache
```

## Zero-Cost Data Layer (`quant_core/data`)

- `base.py`: hash-keyed `PersistentCache` + validated HTTP readers (single
  source of truth; the monolith re-exports them).
- `prices.py` / `reconcile.py`: yfinance primary, Stooq redundancy/backfill,
  cross-source close reconciliation with `Price_Quality_Warning` flags.
- `universe.py`: PIT S&P 500 via Wikipedia changes + Wayback snapshots;
  delisting registry; backtests apply `delisting_return_assumption` on stale
  price + zero volume evidence.
- `macro_mx.py` / `macro_global.py`: Banxico SIE catalog, INEGI BIE, FRED,
  ECB, BoC, BCB, BoE, SNB, World Bank, IMF.
- `scraping.py`: governed scraping (robots.txt, throttle, backoff, hash-keyed
  snapshots). `ocr.py`: pdfplumber -> pytesseract last resort behind a
  mandatory plausibility gate. `quality.py`: ingest anomaly scan.
- Full catalog with licensing notes: `DATA_SOURCES.md`.

## Cloud Mode

```text
Vercel Next.js UI
  -> API routes / worker trigger
  -> Supabase Auth + Postgres + Storage
  -> stateless quant job
  -> run_artifacts.dashboard_payload
```

## Versioning

- UI-only changes update `APP_VERSION`.
- Mathematical/model changes update `MODEL_VERSION`.
- Database changes update `SCHEMA_VERSION` and add a migration file.
- Every run stores `run_hash`, `code_version`, `config_hash`, `universe_hash`
  and `data_hash`.

## Post-PhD Research Governance

Advanced models enter the stack only as uncertainty reducers:

```text
PIT data -> signal reliability -> Kalman/state-space -> RMT denoising
-> GARCH/EGARCH/Fractional Volterra -> Fisher/CRLB + entropy
-> benchmark-relative control -> promotion gate
```

- `StrategyConstitution` freezes allowed features, hyperparameters, benchmark
  sets, complexity budget, trial budget and promotion gates.
- `UncertaintyState` is the single-row state vector for RMT noise, effective
  rank, Kalman confidence, CRLB/Fisher, entropy, Volterra H and regime labels.
- `VarianceModelResult` records AIC, BIC, log-likelihood, OOS QLIKE and bands
  (with an optional `arch`-package GARCH cross-check row compared on QLIKE).
- Production promotion requires strict evidence: DXCDR, PBO, WRC, SPA, ICIR,
  OOS QLIKE, drawdown and CVaR gates; Romano-Wolf adjusted p-values are
  reported as family-wise diagnostics.

## Statistical Conventions and Honesty Rules

- XCDR/XODR risk denominators mix only annualized loss units: daily CVaR is
  clipped at zero and scaled by sqrt(252); max drawdown stays a sample-path
  fraction (see `quant_core/uncertainty_state.py`).
- The Deflated Sortino uses a null-bootstrap estimator scale (the Mertens
  Sharpe approximation over-rejects for Sortino) and an `effective_trial_count`
  that includes the hyperparameter grid, bandit arms and PSO evaluations.
- "Black-Litterman" diagnostics are prefixed `BLInspired_`: the default mode
  is a rank-shrinkage device; `black_litterman_canonical=True` enables the
  return-unit canonical posterior with Grinold-style views.
- `CRLB_Mu` is the classical standard error of the mean (sigma^2/T) — the
  CRLB framing documents why it is the right shrinkage scale, nothing more.
- The power-law variance kernel is rough-volatility-*inspired*; H can be
  estimated from data (`estimate_hurst_rv`, Gatheral-Jaisson-Rosenbaum).
- GPD tails are estimated by L-moments with sector-pooled tail indices.
- Null-calibration tests (`tests/test_validation_null_calibration.py`) verify
  that WRC/SPA/PBO/DSR do not over-reject when there is no signal.

## Research Batches at Zero Cost

`xcdr-windows-batch.yml` builds a 10y+ market cache, shards 12+ walk-forward
windows across a GitHub Actions matrix, merges partials, recomputes WRC/SPA/
PBO/Romano-Wolf on the full date x candidate matrix, and publishes refreshed
`research_artifacts/` via automated PR. `--pit-universe` filters each window
to point-in-time S&P 500 membership; `scripts/survivorship_sensitivity.py`
bounds survivorship inflation against a current-constituents leg.

## Multiuser Boundary

Shared market data is global. User portfolios, configs, jobs, runs and chat
sessions are user-scoped with RLS.
