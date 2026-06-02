# Quant Portfolio-Kaizen Cloud API Contract

This contract is for the future Vercel/Next.js frontend. The frontend must not
recompute portfolio analytics. It reads Supabase artifacts produced by the
Python engine.

## Core Principle

```text
Frontend = renderer
Supabase = persistence and auth boundary
Quant engine = source of truth
```

## Endpoints

These can be implemented as Vercel API routes or Supabase Edge Functions.

### GET /api/runs

Returns the current user's runs, newest first.

Response:

```json
{
  "runs": [
    {
      "run_id": "uuid",
      "created_at": "timestamp",
      "benchmark_ticker": "SPY",
      "objective": "sortino",
      "app_version": "0.2.0",
      "model_version": "qp-kaizen-core-v0.2.0",
      "schema_version": "20260520_001",
      "status": "completed"
    }
  ]
}
```

### GET /api/latest-dashboard

Returns the latest `dashboard_payload` artifact for the current user.

Response:

```json
{
  "run_id": "uuid",
  "dashboard_payload": {}
}
```

### GET /api/run-status/:job_id

Returns job state.

```json
{
  "job_id": "uuid",
  "status": "queued | running | completed | failed | blocked",
  "result_run_id": "uuid | null",
  "error": null
}
```

### POST /api/run-optimization

Creates a queued optimization job. The actual heavy compute should be done by a
worker, not synchronously by the frontend route.

Request:

```json
{
  "universe_id": "uuid",
  "preset_id": "uuid",
  "risk_profile_id": "uuid",
  "config": {
    "tickers": ["AAPL", "MSFT"],
    "benchmark_ticker": "SPY",
    "filter_style": "growth | value | quality | factor | custom",
    "weight_objective": "sortino"
  }
}
```

Response:

```json
{
  "job_id": "uuid",
  "status": "queued"
}
```

## Artifact Names

The `run_artifacts` table stores:

```text
dashboard_payload
backtest_path_bundle
suitability_gate
promotion_gate
data_freshness_report
```

## Gate Rules

The frontend must enforce:

```text
if suitability_gate.status != "approved":
  show Allocation Blocked

if promotion_gate.promotion_status != "promoted":
  show Research Only
```

No UI component may label a portfolio as recommended unless both gates pass.

