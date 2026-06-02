-- Quant Portfolio-Kaizen cloud audit baseline.
-- Apply in Supabase SQL Editor. Safe to run more than once.

create table if not exists public.schema_migrations (
  version text primary key,
  description text not null,
  applied_at timestamptz not null default now()
);

alter table if exists public.runs
  add column if not exists app_version text,
  add column if not exists model_version text,
  add column if not exists schema_version text,
  add column if not exists run_hash text,
  add column if not exists code_version text,
  add column if not exists config_hash text,
  add column if not exists universe_hash text,
  add column if not exists data_hash text,
  add column if not exists objective text,
  add column if not exists warnings jsonb;

create table if not exists public.run_artifacts (
  artifact_id uuid primary key default gen_random_uuid(),
  run_id uuid not null references public.runs(run_id) on delete cascade,
  artifact_name text not null,
  artifact_json jsonb not null,
  artifact_sha256 text,
  created_at timestamptz not null default now()
);

create index if not exists idx_runs_run_hash
  on public.runs(run_hash);

create index if not exists idx_runs_versions
  on public.runs(app_version, model_version, schema_version);

create index if not exists idx_run_artifacts_run_id
  on public.run_artifacts(run_id);

create index if not exists idx_run_artifacts_name
  on public.run_artifacts(artifact_name);

insert into public.schema_migrations(version, description)
values ('20260520_001', 'Run artifacts and app/model/schema version audit fields')
on conflict (version) do nothing;

