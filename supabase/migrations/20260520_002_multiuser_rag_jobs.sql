-- Quant Portfolio-Kaizen multiuser/RAG/job schema.
-- Apply after 20260520_001. Safe to run more than once.

create table if not exists public.profiles (
  user_id uuid primary key references auth.users(id) on delete cascade,
  display_name text,
  base_currency text default 'USD',
  role text not null default 'user',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.user_risk_profiles (
  profile_id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  horizon_years numeric,
  initial_capital numeric,
  monthly_contribution numeric,
  liquidity_need text,
  max_drawdown numeric,
  risk_aversion_score numeric,
  investor_objective text,
  suitability_profile text,
  constraints jsonb,
  created_at timestamptz not null default now()
);

create table if not exists public.user_universes (
  universe_id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  name text not null,
  tickers text[] not null default '{}',
  source text default 'custom',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.user_filter_presets (
  preset_id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  name text not null,
  filter_style text not null check (filter_style in ('growth','value','quality','factor','custom')),
  parameters jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create table if not exists public.user_run_configs (
  config_id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  universe_id uuid references public.user_universes(universe_id) on delete set null,
  preset_id uuid references public.user_filter_presets(preset_id) on delete set null,
  config jsonb not null,
  created_at timestamptz not null default now()
);

create table if not exists public.jobs (
  job_id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  job_type text not null,
  status text not null default 'queued' check (status in ('queued','running','completed','failed','blocked')),
  config jsonb not null default '{}'::jsonb,
  result_run_id uuid references public.runs(run_id) on delete set null,
  error text,
  created_at timestamptz not null default now(),
  started_at timestamptz,
  finished_at timestamptz
);

create table if not exists public.chat_sessions (
  session_id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  title text,
  active_run_id uuid references public.runs(run_id) on delete set null,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.chat_messages (
  message_id uuid primary key default gen_random_uuid(),
  session_id uuid not null references public.chat_sessions(session_id) on delete cascade,
  user_id uuid not null references auth.users(id) on delete cascade,
  role text not null check (role in ('user','assistant','system','tool')),
  content text not null,
  citations jsonb not null default '[]'::jsonb,
  created_at timestamptz not null default now()
);

create table if not exists public.app_knowledge_base (
  document_id uuid primary key default gen_random_uuid(),
  title text not null,
  category text not null,
  source_path text,
  content text not null,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.app_knowledge_chunks (
  chunk_id uuid primary key default gen_random_uuid(),
  document_id uuid not null references public.app_knowledge_base(document_id) on delete cascade,
  chunk_index integer not null,
  content text not null,
  search_vector tsvector generated always as (to_tsvector('english', content)) stored,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create index if not exists idx_app_knowledge_chunks_fts
  on public.app_knowledge_chunks using gin(search_vector);

create table if not exists public.assistant_tool_audit (
  audit_id uuid primary key default gen_random_uuid(),
  user_id uuid references auth.users(id) on delete set null,
  session_id uuid references public.chat_sessions(session_id) on delete set null,
  tool_name text not null,
  input_json jsonb,
  output_summary text,
  blocked boolean not null default false,
  created_at timestamptz not null default now()
);

alter table if exists public.runs add column if not exists user_id uuid references auth.users(id) on delete set null;
alter table if exists public.run_artifacts add column if not exists user_id uuid references auth.users(id) on delete set null;

create index if not exists idx_jobs_user_status on public.jobs(user_id, status, created_at desc);
create index if not exists idx_runs_user_created on public.runs(user_id, created_at desc);
create index if not exists idx_run_artifacts_user_run on public.run_artifacts(user_id, run_id);

alter table public.runs enable row level security;
alter table public.run_artifacts enable row level security;
alter table public.profiles enable row level security;
alter table public.user_risk_profiles enable row level security;
alter table public.user_universes enable row level security;
alter table public.user_filter_presets enable row level security;
alter table public.user_run_configs enable row level security;
alter table public.jobs enable row level security;
alter table public.chat_sessions enable row level security;
alter table public.chat_messages enable row level security;
alter table public.assistant_tool_audit enable row level security;

do $$
declare
  t text;
begin
  foreach t in array array[
    'profiles',
    'user_risk_profiles',
    'user_universes',
    'user_filter_presets',
    'user_run_configs',
    'runs',
    'run_artifacts',
    'jobs',
    'chat_sessions',
    'chat_messages',
    'assistant_tool_audit'
  ]
  loop
    execute format('drop policy if exists "%s owner read" on public.%I', t, t);
    execute format('drop policy if exists "%s owner write" on public.%I', t, t);
  end loop;
end $$;

create policy "profiles owner read" on public.profiles
  for select using (auth.uid() = user_id);
create policy "profiles owner write" on public.profiles
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

create policy "user_risk_profiles owner read" on public.user_risk_profiles
  for select using (auth.uid() = user_id);
create policy "user_risk_profiles owner write" on public.user_risk_profiles
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

create policy "user_universes owner read" on public.user_universes
  for select using (auth.uid() = user_id);
create policy "user_universes owner write" on public.user_universes
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

create policy "user_filter_presets owner read" on public.user_filter_presets
  for select using (auth.uid() = user_id);
create policy "user_filter_presets owner write" on public.user_filter_presets
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

create policy "user_run_configs owner read" on public.user_run_configs
  for select using (auth.uid() = user_id);
create policy "user_run_configs owner write" on public.user_run_configs
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

create policy "runs owner read" on public.runs
  for select using (auth.uid() = user_id);
create policy "runs owner write" on public.runs
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

create policy "run_artifacts owner read" on public.run_artifacts
  for select using (auth.uid() = user_id);
create policy "run_artifacts owner write" on public.run_artifacts
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

create policy "jobs owner read" on public.jobs
  for select using (auth.uid() = user_id);
create policy "jobs owner write" on public.jobs
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

create policy "chat_sessions owner read" on public.chat_sessions
  for select using (auth.uid() = user_id);
create policy "chat_sessions owner write" on public.chat_sessions
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

create policy "chat_messages owner read" on public.chat_messages
  for select using (auth.uid() = user_id);
create policy "chat_messages owner write" on public.chat_messages
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

create policy "assistant_tool_audit owner read" on public.assistant_tool_audit
  for select using (auth.uid() = user_id);
create policy "assistant_tool_audit owner write" on public.assistant_tool_audit
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

insert into public.schema_migrations(version, description)
values ('20260520_002', 'Multiuser portfolios, jobs and zero-cost RAG assistant schema')
on conflict (version) do nothing;
