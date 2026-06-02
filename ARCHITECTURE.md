# Quant Portfolio-Kaizen Architecture

## Principle

```text
Core = source of truth
UI   = renderer
DB   = append-only audit layer
```

## Local Research Mode

```text
yfinance / SEC / FRED / public APIs
  -> quant_core + quant_stockpicker_core.py
  -> dashboard_payload
  -> Streamlit renderer
  -> Supabase artifacts + local .quant_cache
```

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
- `VarianceModelResult` records AIC, BIC, log-likelihood, OOS QLIKE and bands.
- Production promotion requires strict evidence: DXCDR, PBO, WRC, SPA, ICIR,
  OOS QLIKE, drawdown and CVaR gates.

## Multiuser Boundary

Shared market data is global. User portfolios, configs, jobs, runs and chat
sessions are user-scoped with RLS.
