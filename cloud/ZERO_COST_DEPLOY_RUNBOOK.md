# Quant Portfolio-Kaizen — Zero-Cost Deploy Runbook

This runbook is the operational path for putting the app online without paid
infrastructure while keeping the first screen preloaded.

## Production Shape

```text
GitHub Actions daily worker
  -> cloud_daily_refresh.py
  -> Supabase run + dashboard artifact
  -> Streamlit app renders latest artifact on open
```

This is the zero-cost pilot architecture. Streamlit remains the online app for
now. The later SaaS architecture can move the UI to Vercel/Next.js while keeping
Supabase as the source of truth.

## Required Secrets

Create these as GitHub repository secrets:

```text
SUPABASE_URL
SUPABASE_SERVICE_ROLE_KEY
BANXICO_TOKEN
SEC_USER_AGENT
```

Create these in Streamlit Community Cloud secrets:

```toml
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_SERVICE_ROLE_KEY = "server-side-service-role-key"
BANXICO_TOKEN = "optional-banxico-token"
SEC_USER_AGENT = "YourCompany/1.0 contact@email.com"
```

The Streamlit app also accepts the Supabase aliases:

```toml
[supabase]
url = "https://your-project.supabase.co"
service_key = "server-side-service-role-key"
```

Never expose `SUPABASE_SERVICE_ROLE_KEY` in browser-side JavaScript or public
configuration.

## One-Time Setup

1. Push the repository to GitHub.
2. Add the GitHub secrets above.
3. Deploy `stockpicker_app.py` on Streamlit Community Cloud.
4. Paste Streamlit secrets using `.streamlit/secrets.toml.example` as template.
5. Confirm Supabase migrations are applied.
6. Run the GitHub workflow `Daily cloud refresh` manually once.
7. Open the app and verify that it says the latest precomputed dashboard artifact
   was loaded.

## Daily Refresh

The existing workflow runs at both UTC times needed to cover Central Time
daylight saving and standard time:

```text
.github/workflows/daily-cloud-refresh.yml
```

By default the workflow evaluates Central time as `America/Mexico_City`.
Set the GitHub repository variable below if the app should follow another
Central timezone:

```text
QPK_CLOUD_REFRESH_TIMEZONE=America/Chicago
```

It gates execution to 08:40 in the configured timezone and then runs:

```bash
python cloud_daily_refresh.py --save-supabase --require-supabase
```

`--require-supabase` is intentional: if the job cannot persist the artifact, the
workflow must fail instead of giving a false green status.

## App Cold-Start Behavior

On first load, the app tries:

```text
Supabase latest dashboard artifact
  -> local .quant_cache/cloud/latest_dashboard_payload.json
  -> normal empty state
```

The user should see cached market/regime/portfolio state before any heavy
optimization is triggered.

## Recommended Free-Tier Limits

```text
max users: 10
max user universe: 50 tickers
max heavy optimizations: 1 per user per week
max backtests: 3 per user per day
scheduled refresh: once per market day before 09:00 Central
options/geopolitical refresh: daily
rebalance: semiannual
reoptimization: annual
```

## Smoke Validation

Run locally before pushing:

```bash
python -m py_compile stockpicker_app.py cloud_daily_refresh.py supabase_store.py
python -m ruff check stockpicker_app.py cloud_daily_refresh.py supabase_store.py --select F,E9
python -m pytest tests -q
```

Run a fast cloud refresh locally:

```bash
python cloud_daily_refresh.py --mode fast --max-tickers 20 --top-n 5 --preselect-n 10 --save-supabase --require-supabase
```

If Supabase is unavailable, remove `--require-supabase` only for local testing.
Do not remove it from the production GitHub Action.

## Production Readiness Gate

The app may be deployed for pilot use when:

```text
all tests pass
daily refresh persists to Supabase
the app loads the latest artifact on open
auth works for admin/analyst/viewer users
service role key is not present in frontend code or public logs
research strategy remains labeled research-only unless strict gates pass
```
