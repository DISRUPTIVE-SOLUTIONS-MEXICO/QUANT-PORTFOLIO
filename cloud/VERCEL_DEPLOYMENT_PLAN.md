# Vercel Deployment Plan

## Recommended Runtime Split

```text
Next.js UI on Vercel
  -> Supabase Auth
  -> Supabase Postgres / Storage
  -> Python quant worker for scheduled or queued compute
```

Do not deploy Streamlit as the production app on Vercel. Keep Streamlit as the
local research/admin lab.

## Environment Variables

Server-side only:

```text
SUPABASE_URL
SUPABASE_SERVICE_ROLE_KEY
BANXICO_TOKEN
```

Client-side:

```text
NEXT_PUBLIC_SUPABASE_URL
NEXT_PUBLIC_SUPABASE_ANON_KEY
```

Never expose `SUPABASE_SERVICE_ROLE_KEY` to the browser.

## Cron

Daily before 09:00 Central Time:

```text
/api/cron/prewarm-market-data
```

The cron should update shared market cache and data freshness, not run every
user's optimization.

## Implemented Zero-Cost Preload Path

The current repository uses a zero-cost scheduled worker before the final
Next.js migration:

```text
GitHub Actions daily 08:40 America/Chicago
  -> cloud_daily_refresh.py
  -> run_pipeline(config)
  -> Supabase run_artifacts.dashboard_payload with required persistence
  -> Streamlit / future Vercel UI renders latest artifact on first load
```

Workflow:

```text
.github/workflows/daily-cloud-refresh.yml
```

Runtime script:

```text
cloud_daily_refresh.py
```

Operational runbook:

```text
cloud/ZERO_COST_DEPLOY_RUNBOOK.md
```

The app reads the latest artifact automatically when:

```text
QPK_LOAD_LATEST_DASHBOARD_ON_START=1
```

This keeps the public web app render-first. Users should not pay the cold
compute cost unless they explicitly press the allocation engine.

## Cost-Control Rules

```text
max users: 10
max user universe: 50 tickers
max heavy optimizations: 1 per user per week
max backtests: 3 per user per day
rebalance: semiannual
reoptimize: annual
options snapshot: daily
news/geopolitical: daily
```
