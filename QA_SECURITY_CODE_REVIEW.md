# Quant Portfolio-Kaizen QA, Code Review And Security Audit

Date: 2026-06-01  
Scope: `C:\Users\chris\FINANZAS`

## Executive Summary

The project has a strong research/testing foundation: the full pytest suite passes with `83 passed`, and the core quant safeguards around causality, suitability, benchmark governance, PELT/GARCH, uncertainty state, RMT covariance, private side sleeve controls, and promotion gates are covered.

The main release blockers are not mathematical unit-test failures; they are deployment/security governance issues:

1. Real secrets exist in local project files.
2. CI does not execute the full pytest suite.
3. Ruff is currently soft-failed and reports actionable security/QA issues.
4. Supabase multiuser persistence is incomplete because `runs` and `run_artifacts` are not fully RLS-governed and `save_run_to_supabase()` does not attach `user_id`.
5. Heavy research computation is still synchronous in the Streamlit app path.

## Verification Performed

Commands run:

```powershell
python -m pytest tests -q
python -m unittest discover -v tests
python -m ruff check .
python -m bandit -r . -c pyproject.toml -q
python -m pip_audit -r requirements.txt
```

Results:

```text
pytest: 83 passed, 4 warnings
unittest discover: 68 tests passed
ruff: 87 findings
bandit: not installed in local environment
pip-audit: not installed in local environment
```

## Remediation Update

Applied on 2026-06-01:

```text
pytest: 83 passed, 4 warnings
py_compile: stockpicker_app.py, quant_stockpicker_core.py, supabase_store.py, cloud_jobs.py OK
targeted ruff checks: URL/XML/loop-closure and simple app errors OK
```

Changes implemented:

- CI now runs `python -m pytest tests -q`, not `unittest discover`.
- Added `defusedxml` as a runtime dependency.
- Public-data URL readers now reject unsupported URL schemes and missing hosts.
- External XML parsing now uses `defusedxml.ElementTree`.
- Manual ticker parsing now uses the central ticker sanitizer.
- Supabase run/artifact persistence accepts `user_id`.
- `load_run_bundle()` verifies run ownership before loading service-role child tables.
- `jobs` status helper can be scoped by `user_id`.
- Supabase migration now enables RLS on `runs` and `run_artifacts` and adds owner policies.
- Cleaned selected app QA findings: unused imports, boolean comparisons, and name shadowing.

Still external/manual:

- Rotate the Supabase service-role key in Supabase.
- Rotate the Streamlit auth cookie key.
- Replace any deployed secrets with fresh environment variables.
- Apply the Supabase migration in the hosted database.
- Install dev security tools locally if desired: `python -m pip install -r requirements-dev.txt`.

## Critical Findings

### P0-1: Real Secrets Are Present In Local Project Files

Evidence:
- `C:\Users\chris\FINANZAS\.env`
- `C:\Users\chris\FINANZAS\.streamlit\secrets.toml`
- `.gitignore` excludes them, but local files still contain live credentials.

Impact:
If this folder is zipped, backed up, synced with external tooling, or accidentally copied into Docker context, Supabase service-role access and auth hashes/cookie secrets can be exposed.

Recommendation:
Rotate the Supabase service-role key and Streamlit cookie key before deployment. Keep only `.env.example` and `secrets.toml.example` in source, and inject real secrets through Vercel/Supabase/Streamlit secret stores.

Status:
Code-side guardrails improved; key rotation still requires manual action in Supabase/Streamlit/Vercel secret stores.

### P0-2: Supabase `service_role` Is Used As The Main Persistence Client

Evidence:
- `supabase_store.py:66-72` loads `SUPABASE_SERVICE_ROLE_KEY`.
- `cloud_jobs.py:48-57` allows latest dashboard lookup without `user_id`.
- `supabase_store.py:142-169` inserts runs without a `user_id`.

Impact:
`service_role` bypasses RLS. If any API route or frontend-adjacent runtime exposes this path incorrectly, users can cross-read or mutate other users' artifacts.

Recommendation:
Use `service_role` only in trusted backend workers. API routes should authenticate the user, pass `user_id`, and never expose service-role credentials to the browser. `save_run_to_supabase()` should require or accept `user_id` and persist it into `runs` and `run_artifacts`.

Status:
Partially fixed. Persistence now accepts `user_id`, job/status helpers support user scoping, and `load_run_bundle()` verifies ownership before service-role child-table loads. Production API routes must pass authenticated `user_id`.

## High Findings

### P1-1: CI Does Not Run The Full Test Suite

Evidence:
- `.github/workflows/ci.yml:45-46` runs `python -m unittest discover -v tests`.
- Local `pytest` runs 83 tests; `unittest discover` runs only 68.
- Pytest-style functions in `tests/test_pso_research_controls.py` are not covered by the CI command.

Impact:
Important research controls can regress while CI remains green.

Recommendation:
Change CI to:

```yaml
- name: Run unit tests
  run: python -m pytest tests -q
```

Status:
Fixed in `.github/workflows/ci.yml`.

### P1-2: Ruff Is Soft-Failed And Reports Real QA/Security Issues

Evidence:
- `.github/workflows/ci.yml:62-64` uses `ruff check --exit-zero`.
- Local `ruff check .` reports 87 findings.

Notable findings:
- `quant_stockpicker_core.py:4794` parses untrusted XML with stdlib `xml.etree`.
- `quant_stockpicker_core.py:839-848` URL readers do not enforce URL scheme allowlists.
- multiple `try/except/continue` blocks swallow data-source failures silently.
- unused imports and name redefinitions in `stockpicker_app.py`.

Impact:
The app can mask failed data pulls, produce overly clean dashboards, and keep unsafe patterns invisible.

Recommendation:
Fix the high-signal Ruff findings first, then remove `--exit-zero`.

Status:
Partially fixed. The targeted security/QA findings around external XML, public URL scheme validation, loop closure, unused app imports, boolean comparisons, and name shadowing were corrected. The broader monolith still has lint debt.

### P1-3: Multiuser RLS Is Incomplete For Run Artifacts

Evidence:
- `supabase/migrations/20260520_001_run_artifacts_and_versions.sql:22-29` creates `run_artifacts`.
- `supabase/migrations/20260520_002_multiuser_rag_jobs.sql:123-128` adds `user_id` to `runs` and `run_artifacts`.
- The migration enables RLS for new user tables, but not explicitly for `runs` or `run_artifacts` in the shown section.

Impact:
Portfolio artifacts can become cross-tenant readable unless all reads are service-side filtered and RLS policies are added.

Recommendation:
Enable RLS on `runs` and `run_artifacts`, add owner policies, and ensure all inserts include `user_id`.

Status:
Fixed in migration and persistence interfaces. Apply the migration in Supabase before production.

### P1-4: Heavy Pipeline Runs Synchronously In Streamlit UI

Evidence:
- `stockpicker_app.py:2794-2796` caches and calls `run_pipeline(config)`.
- `stockpicker_app.py:2840-2866` executes the full causal pipeline inside the UI interaction.

Impact:
Long research or data-source stalls can block UX, cause duplicate work, and make Vercel/serverless deployment impractical.

Recommendation:
Move production execution to jobs. The UI should submit `jobs`, poll status, and render the latest completed `run_artifacts`.

Status:
Not fully implemented in Streamlit. Existing cloud job helpers are safer, but production UI/API still needs the async job flow.

## Medium Findings

### P2-1: Manual Ticker Input Bypasses Existing Sanitizer

Evidence:
- `security.py:58-77` defines ticker sanitization.
- `stockpicker_app.py:660-662` implements `parse_tickers()` without using it.
- `stockpicker_app.py:2628-2642` uses `parse_tickers()` for manual universe and private side tickers.

Impact:
Malformed tickers can pass into yfinance/cache keys. This is lower risk than SQL injection because no string SQL is used, but it is avoidable input-surface risk.

Recommendation:
Rewrite `parse_tickers()` to call `sanitize_ticker_list()`.

Status:
Fixed.

### P2-2: ForexFactory XML Parsing Should Use `defusedxml`

Evidence:
- `quant_stockpicker_core.py:4792-4794`.

Impact:
External XML should not be parsed with stdlib XML in a production app.

Recommendation:
Add `defusedxml` and use `defusedxml.ElementTree.fromstring`.

Status:
Fixed.

### P2-3: URL Fetch Helpers Need Scheme And Host Guardrails

Evidence:
- `quant_stockpicker_core.py:839-848`.

Impact:
If future code passes user-controlled URLs into these helpers, `file:` or unexpected schemes can be opened.

Recommendation:
Validate `urllib.parse.urlparse(url).scheme in {"https"}` and optionally allowlist known hosts.

Status:
Partially fixed. URL helpers now reject unsupported schemes and missing hosts. Host allowlisting remains optional future hardening.

### P2-4: Dependency Governance Is Too Loose

Evidence:
- `requirements.txt` has mostly unpinned dependencies.
- Local `bandit` and `pip-audit` are not installed even though listed in `requirements-dev.txt`.

Impact:
Reproducibility and CVE triage are weak. A future dependency release can change yfinance schemas, Streamlit behavior, or auth library behavior.

Recommendation:
Add a locked constraints file for production and make `pip-audit` a non-soft-fail gate.

## QA Strengths

- Strong causal tests exist for availability dates, embargo, purging, OOS separation, and future-contamination style controls.
- Suitability and benchmark governance have direct tests.
- Private Side Alpha fixed-weight behavior and trading-history fallback are tested.
- Uncertainty state tests cover RMT PSD, Volterra causality, Kalman prefix stability, XCDR/XODR degradation with uncertainty, and downside/upside decomposition.
- Streamlit auth is loaded before data widgets and backend triggers.
- Suitability hard blocks disable the run button.
- Pipeline rate limiting exists.

## Recommended Fix Order

1. Rotate all exposed Supabase and app secrets.
2. Change CI tests from `unittest discover` to `pytest`.
3. Add `user_id` to run/artifact persistence and enable RLS on `runs`/`run_artifacts`.
4. Replace Streamlit synchronous production runs with Supabase `jobs` + artifact polling.
5. Fix high-signal Ruff findings: XML, URL scheme validation, swallowed exceptions.
6. Use `sanitize_ticker_list()` in `parse_tickers()`.
7. Pin production dependencies through a constraints/lock workflow.
8. Make Ruff/Bandit/pip-audit hard gates after fixes.

## Final QA Verdict

Current status: strong research prototype / local lab, not production-ready for multiuser cloud deployment yet.

Production readiness requires security hardening, full CI coverage, tenant isolation, job-based execution, and stricter dependency/audit gates. The mathematical research layer is much more mature than the deployment-security layer.
