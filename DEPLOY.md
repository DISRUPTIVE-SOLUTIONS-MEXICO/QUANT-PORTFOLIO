# Deployment Runbook

## Production Topology

```text
Vercel Next.js/PWA
  -> Supabase Auth + RLS + atomic publication pointers
  -> queued jobs
  -> GitHub Actions or trusted local Python worker
  -> immutable artifacts
```

Streamlit remains the internal research lab. It is not the primary production
surface.

## 1. Apply Supabase Migrations

Apply in order:

1. `20260520_001_run_artifacts_and_versions.sql`
2. `20260520_002_multiuser_rag_jobs.sql`
3. `20260612_003_institutional_publication_and_paper_execution.sql`

Verify that `promote_publication(uuid)` is executable only by `service_role`.

## 2. Configure Vercel

Project root: repository root.

Build command:

```text
npm run web:build
```

Install command:

```text
npm ci
```

Environment variables:

```text
SUPABASE_URL                  server only
SUPABASE_SERVICE_ROLE_KEY     server only
NEXT_PUBLIC_SUPABASE_URL      browser safe
NEXT_PUBLIC_SUPABASE_ANON_KEY browser safe, protected by RLS
```

Never create a `NEXT_PUBLIC_` service-role variable.

## 3. Daily Atomic Refresh

The workflow runs at 07:00 Central using two UTC candidates and a timezone
guard. It writes `daily_snapshot` staging artifacts, validates them, then
atomically promotes `global:daily_snapshot`.

The prior daily snapshot and the full-analysis pointer remain readable until
promotion succeeds.

## 4. Full Research Refresh

The governed full pipeline runs semiannually or manually. It promotes only
`global:full_analysis`; daily updates cannot overwrite it.

## 5. Tenant Job Worker

`POST /api/jobs` only enqueues tenant-scoped work. The heavy Python process is
run by `.github/workflows/process-quant-jobs.yml` at 09:00 Central on weekdays,
or manually with:

```powershell
python cloud_job_worker.py --max-jobs 1
```

Required GitHub secrets:

```text
SUPABASE_URL
SUPABASE_SERVICE_ROLE_KEY
SEC_USER_AGENT
BANXICO_TOKEN             optional outside Mexico-rate research
```

The workflow uses `requirements.lock.txt`, processes jobs serially and never
cancels an in-flight optimization. A job moves through:

```text
queued -> running -> completed | failed
```

An optimization is marked complete only after its user-scoped run artifacts
and immutable `portfolio_versions` record are durable. Pre-trade jobs create a
paper order only after ownership, artifact hash, OOS scope, promotion,
suitability, freshness, concentration, ADV and cost checks.

## 6. Desktop

Development:

```powershell
npm run desktop:dev
```

Production:

```powershell
$env:QPK_WEB_URL="https://your-vercel-domain"
npm run start --workspace @qpk/desktop
```

The Electron shell enables sandboxing and context isolation and denies device
permissions.

### Offline artifact verification

The production web runtime reads only active Supabase publications by default.
For Electron or an isolated production-like verification that must read the
immutable local registry, set:

```text
QPK_ALLOW_LOCAL_ARTIFACTS=1
```

Do not set this variable in Vercel. It is an explicit offline/self-hosted
capability, not a cloud fallback.

The development Content Security Policy permits `unsafe-eval` only because the
Next.js development runtime uses it for debugging call-stack reconstruction.
The production policy removes `unsafe-eval`.

## 6. Quality Gates

```powershell
npm ci
npm run quality
python -m pytest tests -q
python -m ruff check .
python -m mypy quant_core --ignore-missing-imports
python scripts/security_hygiene_scan.py
```

CI fails for high/critical npm advisories. A current moderate PostCSS advisory
is an accepted temporary build-time risk while Next.js pins the affected
version; no untrusted CSS is processed and the risk must be re-evaluated when
a patched stable Next.js release is available.

## 7. Rollback

Rollback changes only the applicable atomic pointer to a previously validated
publication. Never delete historical manifests, run artifacts, portfolio
versions, pre-trade decisions or audit events.

## 8. Acceptance

- Full analysis and daily market overlay both render.
- No page recomputes analytics.
- Missing data show source, cause and last valid evidence.
- User portfolios are isolated by RLS.
- Paper orders cannot bypass Python pre-trade checks or human approval.
- Desktop widths 1440/1920 and mobile widths 375/768 pass visual regression.
- WRC/SPA/PBO/ICIR/downside failures remain visible and block promotion.
