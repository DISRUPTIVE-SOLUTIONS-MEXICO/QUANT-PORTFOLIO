# Supabase Cloud Schema

Apply migrations in order from `supabase/migrations` using the Supabase SQL
Editor. The local Python client uses the service-role key server-side only; do
not expose it in Vercel client bundles.

## Migration Order

1. `20260520_001_run_artifacts_and_versions.sql`
2. `20260520_002_multiuser_rag_jobs.sql`
3. `20260612_003_institutional_publication_and_paper_execution.sql`

## Current Version Contract

- `app_version`: UI/API release.
- `model_version`: quantitative engine release.
- `schema_version`: expected Supabase schema release.
- `code_version`: SHA-256 digest over key local source files.

Historical runs are append-only. Improvements create new runs with new version
metadata instead of overwriting old results.

The third migration adds an atomic publication pointer. Workers insert a
`staging` manifest, validate every required artifact, set the manifest to
`validated`, then call the server-only `promote_publication` RPC. Readers keep
serving the prior active snapshot until that transaction succeeds.
