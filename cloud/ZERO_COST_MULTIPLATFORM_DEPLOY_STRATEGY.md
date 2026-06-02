# Quant Portfolio-Kaizen — Zero-Cost Multiplatform Deployment Strategy

## Objective

Deploy Quant Portfolio-Kaizen online for desktop and mobile users while keeping
cloud cost at zero or near-zero. The production experience must be render-first:
users should see the latest portfolio dashboard immediately, while heavy
quantitative computation runs on a schedule or explicit recompute.

## Core Constraint

Streamlit is a persistent Python/WebSocket app. Vercel is optimized for
stateless serverless functions and Next.js. Therefore, the zero-cost strategy
must separate:

```text
Heavy quant compute != web render
```

The app should never recompute the full research pipeline on first page load.

## Target Architecture

```text
Daily scheduled worker
  -> Quant engine
  -> Supabase runs + run_artifacts
  -> Render-only app
  -> Desktop browser / Mobile browser / PWA shell
```

Concrete repository implementation:

```text
.github/workflows/daily-cloud-refresh.yml
  -> cloud_daily_refresh.py
  -> run_pipeline(config)
  -> save_run_to_supabase(...) with required persistence in cloud
  -> run_artifacts.dashboard_payload
  -> stockpicker_app.py loads latest artifact on open
```

## Deployment Phases

### Phase 1 — Immediate Zero-Cost Deploy

Use:

```text
Streamlit Community Cloud
Supabase Free
GitHub Actions scheduled worker
```

Purpose:

- Fastest production pilot.
- Works on desktop and mobile browser.
- Uses existing Streamlit UI and auth.
- Heavy compute is precomputed daily.

Flow:

```text
08:40 Central Time
  GitHub Action runs cloud_daily_refresh.py
  Dashboard payload is saved to Supabase

User opens app
  Streamlit wakes if hibernated
  App loads latest dashboard_payload
  User sees dashboard without running pipeline
```

Operational runbook:

```text
cloud/ZERO_COST_DEPLOY_RUNBOOK.md
```

Limitations:

- Streamlit free apps can hibernate.
- First wake may still have platform cold-start.
- UI is responsive, but not a true native mobile app.
- This is the best zero-cost pilot, not the final SaaS architecture.

### Phase 2 — Multiplatform Production UI

Use:

```text
Next.js on Vercel
Supabase Auth/Postgres/Storage
GitHub Actions or external Python worker
```

Purpose:

- Desktop and mobile responsive product.
- Installable PWA.
- Better UX than Streamlit.
- Vercel renders artifacts, not heavy quant jobs.

Flow:

```text
Next.js UI
  -> Supabase Auth
  -> GET latest dashboard_payload
  -> Render portfolio/risk/market/evidence

Python worker
  -> Scheduled by GitHub Actions
  -> Writes artifacts to Supabase
```

Do not run the full quant engine inside Vercel Functions unless the workload is
tiny. Use Vercel API routes only for lightweight reads, job creation and status.

### Phase 3 — True Multiuser Product

Use:

```text
Next.js PWA
Supabase Auth + RLS
Shared market cache
User-scoped runs/artifacts
RAG assistant
```

User-scoped data:

```text
profiles
user_risk_profiles
user_universes
user_run_configs
jobs
runs
run_artifacts
chat_sessions
chat_messages
```

Shared data:

```text
market_prices_daily
fundamentals_cache
macro_cache
rates_cache
options_snapshot_cache
news_cache
data_freshness_report
```

## Recommended Free-Tier Stack

### Frontend

Immediate:

```text
Streamlit Community Cloud
```

Final:

```text
Vercel + Next.js PWA
```

### Database / Auth / Storage

```text
Supabase Free
```

Use Supabase for:

- Auth
- Postgres
- run artifacts
- user profiles
- app knowledge base
- RAG chunks

### Scheduled Compute

```text
GitHub Actions
```

Use GitHub Actions for:

- daily refresh
- public-data cache warmup
- dashboard artifact generation
- optional research batch

Reason: heavy Python jobs are more natural in GitHub Actions than in Vercel
serverless functions.

## Daily Refresh Policy

Current workflow:

```text
.github/workflows/daily-cloud-refresh.yml
```

Schedules:

```text
13:40 UTC  -> 08:40 Central during daylight saving time
14:40 UTC  -> 08:40 Central during standard time
```

The workflow guards internally using `America/Chicago` so duplicate UTC
schedules do not produce duplicate refreshes.

Target user-facing status:

```text
Last refresh: today 08:40 CT
Data status: Fresh / Stale / Error
Next refresh: next market morning
```

## App Loading Contract

On app open:

1. Try Supabase latest `dashboard_payload`.
2. If unavailable, try local `.quant_cache/cloud/latest_dashboard_payload.json`.
3. If unavailable, show setup state and ask user to run allocation manually.

Never block the first screen on:

- yfinance download
- SEC EDGAR
- GDELT
- options chains
- GARCH/PELT
- full optimizer

## Secrets

GitHub Actions secrets:

```text
SUPABASE_URL
SUPABASE_SERVICE_ROLE_KEY
BANXICO_TOKEN
SEC_USER_AGENT
```

Streamlit secrets:

```text
[auth]
cookie_key = "..."

[auth.users.chris]
name = "Chris"
email = "..."
password_hash = "$2b$..."
role = "admin"

SUPABASE_URL = "..."
SUPABASE_SERVICE_ROLE_KEY = "..."
BANXICO_TOKEN = "..."
SEC_USER_AGENT = "..."
```

Vercel final frontend:

```text
NEXT_PUBLIC_SUPABASE_URL
NEXT_PUBLIC_SUPABASE_ANON_KEY
```

Vercel server-side only:

```text
SUPABASE_SERVICE_ROLE_KEY
CRON_SECRET
```

Never expose the Supabase service role in browser code.

## Multiplatform UX Strategy

Desktop navigation:

```text
Dashboard
Portfolio
Risk
Market
Evidence
Assistant
Advanced
```

Mobile navigation:

```text
Home
Portfolio
Risk
Market
Chat
```

Mobile rules:

- single-column layout
- no large raw tables by default
- technical tables inside expanders
- charts with fixed height
- touch targets >= 44px
- user-safe explanations before raw diagnostics

## Production Gates

Frontend must enforce:

```text
if suitability_gate.status != approved:
  show Blocked

if promotion_gate.status != promoted:
  show Research-only

if data_freshness.status == stale:
  show Stale data warning
```

No UI component should label a portfolio as recommended unless suitability and
promotion gates pass.

## Cost-Control Rules

For 10 users:

```text
Daily market refresh: 1/day
Options snapshot: 1/day
News/geopolitical: 1/day
Heavy optimization: max 1/user/week
Rebalance: semiannual
Reoptimization: annual
User universe: <= 50 tickers
Backtests: <= 3/user/day
Research batches: admin-only
```

## What Not To Do

Do not:

- deploy Streamlit as a heavy compute app on Vercel serverless;
- run full optimization on page load;
- expose service_role to browser;
- call yfinance/GDELT/options per user on every visit;
- treat Yahoo options snapshot as historical options backtest;
- let chatbot bypass suitability or promotion gates;
- show research-only output as a recommendation.

## Final Recommendation

Immediate deploy:

```text
Streamlit Community Cloud + Supabase + GitHub Actions
```

Final multiplatform product:

```text
Next.js PWA on Vercel + Supabase + external Python scheduled worker
```

This is the correct zero-cost path because it preserves mathematical rigor,
avoids slow first loads, supports desktop/mobile, and keeps heavy quant compute
outside the user-facing request path.
