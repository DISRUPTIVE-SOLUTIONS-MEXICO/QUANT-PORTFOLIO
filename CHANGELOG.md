# Changelog

## 0.3.0 - 2026-06-10

Statistical rigor and zero-cost data robustness release.

- XCDR/XODR risk denominators unified to annualized loss units (daily CVaR
  clipped and sqrt(252)-scaled); lambda sensitivity audit for the XCDR-v3
  scalarization.
- Deflated Sortino recalibrated with a null-bootstrap estimator scale and an
  honest `effective_trial_count` (grid x objectives x bandit arms + PSO);
  null-calibration tests for WRC/SPA/PBO/DSR added to the suite.
- Black-Litterman diagnostics renamed `BLInspired_*` with an optional
  canonical return-unit mode; Hurst exponent estimation
  (Gatheral-Jaisson-Rosenbaum) for the power-law variance kernel.
- New `quant_core/data` package: Stooq price redundancy with cross-source
  reconciliation, Wayback-backed PIT S&P 500 universe, delisting registry and
  an explicit backtest delisting-return assumption; Banxico SIE catalog,
  INEGI BIE, BoE/SNB/World Bank/IMF providers; governed scraping engine and
  an OCR last-resort chain behind a plausibility gate; ingest anomaly scan
  surfaced in a new Data Provenance app subsection (`DATA_SOURCES.md`).
- Walk-forward research batches run on free GitHub Actions compute:
  schedule sharding, partial merge with full-matrix WRC/SPA/PBO, PIT-universe
  mode and a survivorship sensitivity report (`xcdr-windows-batch.yml`).
- EVT tails estimated by Hosking-Wallis L-moments with sector-pooled tail
  indices; optional arch-package GARCH cross-check on OOS QLIKE; in-optimizer
  L1 turnover penalty against previous weights; Romano-Wolf stepdown FWER
  diagnostics in the reality-check battery and promotion gate.
- CI hardened: ruff and format checks are blocking, mypy gate on
  `quant_core/`, universal dependency lockfile (`requirements.lock.txt`).

## 0.2.0 - 2026-05-20

- Added backend `dashboard_payload` contract for render-only frontend work.
- Added core daily synthetic NAV, price-path and drawdown bundle.
- Added blocking suitability gate and statistical promotion gate.
- Added PIT confidence scoring for public-data fundamentals.
- Added data freshness report by source with Central Time timestamps.
- Added local/cloud run artifact persistence for dashboard payload, gates and backtest paths.
- Added Student-t GARCH(1,1) candidate in variance model selection.
- Added Supabase migration files for versioned runs, artifacts, multiuser jobs and RAG assistant tables.

