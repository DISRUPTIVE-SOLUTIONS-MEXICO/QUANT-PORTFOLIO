# Security Deployment Hardening Report

## Executive Summary

The project is now hardened for a zero-cost public pilot in three areas:

1. Public files are scanned for high-confidence secret leaks and vendor-specific
   tooling references.
2. Local secrets and cache artifacts are excluded from Git and Docker contexts.
3. The daily cloud refresh requires Supabase persistence, preventing false
   success when the online dashboard artifact is not saved.

No live secret values are documented in this report.

## Threat Model

Primary assets:

- Supabase service-role credential.
- Streamlit authentication cookie secret.
- User password hashes.
- Supabase run artifacts and user-specific portfolio configurations.
- Local `.env`, `.streamlit/secrets.toml`, audit logs, cache files and research
  artifacts.

Primary adversarial paths:

- Accidental commit of `.env` or Streamlit secrets.
- Docker build context containing local secrets.
- Public documentation containing a live JWT or long token.
- Server-side service-role key exposed to browser-side code.
- Daily refresh job appearing successful despite failing to persist artifacts.
- User-specific portfolio data being queried without ownership checks.

## Controls Implemented

### 1. Public Secret Hygiene Scan

Implemented:

```text
scripts/security_hygiene_scan.py
tests/test_security_hygiene.py
```

The scanner detects:

- Supabase-like JWT credentials in public files.
- Long credential-like assignments to names such as token, secret, password,
  API key or service-role key.
- Vendor-specific AI tooling references in public project files.

Excluded from public scans by design:

- `.env`
- `.streamlit/secrets.toml`
- `.streamlit/credentials.toml`
- local caches
- audit logs
- build outputs

Those files are not scanned as public files because they must not be committed or
copied into deployment artifacts.

### 2. Git And Docker Secret Boundary

Hardened:

```text
.gitignore
.dockerignore
```

Protected patterns include:

```text
.env
.env.*
.streamlit/secrets.toml
.streamlit/credentials.toml
.quant_cache/
.quant_cache_vendor_yf/
audit.jsonl
*.jsonl
*.log
```

This reduces accidental leakage through source control and container build
contexts.

### 3. Streamlit / Supabase Secret Loading

The app reads Supabase credentials from environment variables or Streamlit
secrets. Supported forms:

```toml
SUPABASE_URL = "..."
SUPABASE_SERVICE_ROLE_KEY = "..."
```

or:

```toml
[supabase]
url = "..."
service_key = "..."
```

The service-role key must remain server-side only.

### 4. Required Supabase Persistence For Cloud Prewarm

The cloud refresh script supports:

```bash
python cloud_daily_refresh.py --save-supabase --require-supabase
```

The GitHub Actions workflow uses this mode. If Supabase persistence fails, the
workflow fails instead of producing a misleading green deployment signal.

## Residual Risks

### P0: Local Live Secrets Still Need Manual Rotation

If live credentials were ever pasted locally, rotate them before public deploy:

- Supabase service-role key.
- Streamlit cookie signing key.
- Any API token used in `.env` or Streamlit secrets.

This cannot be completed by code alone because rotation must occur in the
provider dashboards.

### P1: Service-Role Key Is Powerful

The service-role credential bypasses Supabase row-level security. It is required
only for server-side jobs and persistence. It must never be placed in:

- frontend JavaScript
- public Vercel client environment variables
- public logs
- screenshots
- issue comments

### P1: Multiuser Access Requires Continuous RLS Testing

User-specific tables should keep row-level security policies based on
`auth.uid() = user_id`. Any new table that stores user-specific portfolios,
chat messages or runs must include explicit ownership policies.

### P2: Public Data Sources Can Fail Open

Yahoo, GDELT, RSS, SEC and public rate APIs may return partial data. The app
should continue to display stale-data diagnostics and avoid silently treating
missing data as a valid alpha signal.

## Test Plan

Run before deploy:

```bash
python scripts/security_hygiene_scan.py
python -m py_compile stockpicker_app.py cloud_daily_refresh.py supabase_store.py scripts/security_hygiene_scan.py
python -m ruff check stockpicker_app.py cloud_daily_refresh.py supabase_store.py scripts/security_hygiene_scan.py --select F,E9
python -m pytest tests -q
```

Expected:

```text
security hygiene scan passed
py_compile passed
ruff F/E9 passed
pytest passed
```

## Deployment Checklist

Before making the repository public:

1. Remove or rotate any credential that has appeared in local files.
2. Confirm `.env` and `.streamlit/secrets.toml` are not committed.
3. Add GitHub repository secrets, not plaintext files:
   - `SUPABASE_URL`
   - `SUPABASE_SERVICE_ROLE_KEY`
   - `BANXICO_TOKEN`
   - `SEC_USER_AGENT`
4. Add Streamlit Cloud secrets through the UI.
5. Run `Daily cloud refresh` manually once.
6. Confirm the online app loads the latest Supabase dashboard artifact.
7. Confirm research strategies remain labeled research-only unless strict gates
   pass.

## Final Security Verdict

The codebase is materially safer for public deployment after this hardening
pass, but it still requires manual credential rotation before publishing if any
live key has ever been present in local project files or chat history.

