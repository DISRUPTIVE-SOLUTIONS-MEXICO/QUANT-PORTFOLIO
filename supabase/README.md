# Supabase Cloud Schema

Apply migrations in order from `supabase/migrations` using the Supabase SQL
Editor. The local Python client uses the service-role key server-side only; do
not expose it in Vercel client bundles.

## Migration Order

1. `20260520_001_run_artifacts_and_versions.sql`
2. `20260520_002_multiuser_rag_jobs.sql`

## Current Version Contract

- `app_version`: UI/API release.
- `model_version`: quantitative engine release.
- `schema_version`: expected Supabase schema release.
- `code_version`: SHA-256 digest over key local source files.

Historical runs are append-only. Improvements create new runs with new version
metadata instead of overwriting old results.

