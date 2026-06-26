-- Quant Portfolio-Kaizen institutional publication and paper-execution layer.
-- Apply after 20260520_002. Safe to run more than once.

create table if not exists public.publication_manifests (
  publication_id uuid primary key default gen_random_uuid(),
  run_id uuid not null references public.runs(run_id) on delete cascade,
  user_id uuid references auth.users(id) on delete cascade,
  channel text not null check (channel in ('global', 'user', 'research')),
  publication_kind text not null check (
    publication_kind in ('daily_snapshot', 'full_analysis', 'user_portfolio', 'research_evidence')
  ),
  state text not null default 'staging'
    check (state in ('staging', 'validated', 'active', 'rejected', 'superseded')),
  schema_version text not null,
  manifest_json jsonb not null,
  manifest_sha256 text not null,
  supersedes_publication_id uuid references public.publication_manifests(publication_id) on delete set null,
  created_at timestamptz not null default now(),
  validated_at timestamptz,
  activated_at timestamptz,
  rejection_reason text
);

create unique index if not exists idx_publication_manifest_hash
  on public.publication_manifests(manifest_sha256);
create index if not exists idx_publication_channel_state
  on public.publication_manifests(channel, publication_kind, state, activated_at desc);
create index if not exists idx_publication_user_state
  on public.publication_manifests(user_id, state, activated_at desc);

create table if not exists public.publication_pointers (
  pointer_key text primary key,
  publication_id uuid not null references public.publication_manifests(publication_id) on delete restrict,
  updated_at timestamptz not null default now()
);

create table if not exists public.user_portfolios (
  portfolio_id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  name text not null,
  base_currency text not null default 'USD',
  active_version_id uuid,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.portfolio_versions (
  version_id uuid primary key default gen_random_uuid(),
  portfolio_id uuid not null references public.user_portfolios(portfolio_id) on delete cascade,
  user_id uuid not null references auth.users(id) on delete cascade,
  run_id uuid not null references public.runs(run_id) on delete restrict,
  contract_json jsonb not null,
  contract_sha256 text not null,
  created_at timestamptz not null default now()
);

alter table public.user_portfolios
  drop constraint if exists fk_user_portfolios_active_version;
alter table public.user_portfolios
  add constraint fk_user_portfolios_active_version
  foreign key (active_version_id) references public.portfolio_versions(version_id) on delete set null;

create unique index if not exists idx_portfolio_version_hash
  on public.portfolio_versions(portfolio_id, contract_sha256);
create index if not exists idx_user_portfolios_owner
  on public.user_portfolios(user_id, updated_at desc);
create unique index if not exists idx_user_portfolios_owner_name
  on public.user_portfolios(user_id, name);

create table if not exists public.order_intents (
  order_intent_id uuid primary key,
  user_id uuid not null references auth.users(id) on delete cascade,
  run_id uuid not null references public.runs(run_id) on delete restrict,
  portfolio_version_id uuid references public.portfolio_versions(version_id) on delete restrict,
  status text not null check (
    status in (
      'draft',
      'pretrade_rejected',
      'awaiting_approval',
      'approved',
      'paper_submitted',
      'paper_filled',
      'cancelled'
    )
  ),
  contract_json jsonb not null,
  contract_sha256 text not null,
  created_at timestamptz not null default now(),
  approved_by uuid references auth.users(id) on delete set null,
  approved_at timestamptz
);

create table if not exists public.pretrade_decisions (
  decision_id uuid primary key,
  order_intent_id uuid not null references public.order_intents(order_intent_id) on delete cascade,
  user_id uuid not null references auth.users(id) on delete cascade,
  approved boolean not null,
  contract_json jsonb not null,
  contract_sha256 text not null,
  evaluated_at timestamptz not null
);

create table if not exists public.paper_fills (
  fill_id uuid primary key default gen_random_uuid(),
  order_intent_id uuid not null references public.order_intents(order_intent_id) on delete cascade,
  user_id uuid not null references auth.users(id) on delete cascade,
  ticker text not null,
  side text not null check (side in ('BUY', 'SELL')),
  quantity numeric not null check (quantity >= 0),
  fill_price numeric not null check (fill_price > 0),
  simulated_cost numeric not null default 0,
  filled_at timestamptz not null default now()
);

create table if not exists public.institutional_audit_events (
  event_id uuid primary key default gen_random_uuid(),
  user_id uuid references auth.users(id) on delete set null,
  event_type text not null,
  resource_type text not null,
  resource_id text,
  event_json jsonb not null default '{}'::jsonb,
  event_sha256 text not null,
  created_at timestamptz not null default now()
);

alter table public.publication_manifests enable row level security;
alter table public.publication_pointers enable row level security;
alter table public.user_portfolios enable row level security;
alter table public.portfolio_versions enable row level security;
alter table public.order_intents enable row level security;
alter table public.pretrade_decisions enable row level security;
alter table public.paper_fills enable row level security;
alter table public.institutional_audit_events enable row level security;

drop policy if exists "publication owner read" on public.publication_manifests;
create policy "publication owner read" on public.publication_manifests
  for select using (
    channel in ('global', 'research')
    or auth.uid() = user_id
  );

drop policy if exists "publication pointers read" on public.publication_pointers;
create policy "publication pointers read" on public.publication_pointers
  for select using (
    pointer_key like 'global:%'
    or pointer_key like 'research:%'
    or pointer_key like 'user:' || auth.uid()::text || ':%'
  );

drop policy if exists "user portfolios owner access" on public.user_portfolios;
drop policy if exists "user portfolios owner read" on public.user_portfolios;
create policy "user portfolios owner read" on public.user_portfolios
  for select using (auth.uid() = user_id);

drop policy if exists "portfolio versions owner access" on public.portfolio_versions;
drop policy if exists "portfolio versions owner read" on public.portfolio_versions;
create policy "portfolio versions owner read" on public.portfolio_versions
  for select using (auth.uid() = user_id);

drop policy if exists "order intents owner access" on public.order_intents;
drop policy if exists "order intents owner read" on public.order_intents;
create policy "order intents owner read" on public.order_intents
  for select using (auth.uid() = user_id);

drop policy if exists "pretrade decisions owner access" on public.pretrade_decisions;
drop policy if exists "pretrade decisions owner read" on public.pretrade_decisions;
create policy "pretrade decisions owner read" on public.pretrade_decisions
  for select using (auth.uid() = user_id);

drop policy if exists "paper fills owner access" on public.paper_fills;
drop policy if exists "paper fills owner read" on public.paper_fills;
create policy "paper fills owner read" on public.paper_fills
  for select using (auth.uid() = user_id);

drop policy if exists "audit owner read" on public.institutional_audit_events;
create policy "audit owner read" on public.institutional_audit_events
  for select using (auth.uid() = user_id);

create or replace function public.promote_publication(p_publication_id uuid)
returns uuid
language plpgsql
security definer
set search_path = public
as $$
declare
  candidate public.publication_manifests%rowtype;
  pointer text;
  previous_id uuid;
begin
  select * into candidate
  from public.publication_manifests
  where publication_id = p_publication_id
  for update;

  if candidate.publication_id is null then
    raise exception 'publication not found';
  end if;
  if candidate.state <> 'validated' then
    raise exception 'only validated publications can become active';
  end if;

  pointer := case
    when candidate.channel = 'user'
      then 'user:' || candidate.user_id::text || ':' || candidate.publication_kind
    else candidate.channel || ':' || candidate.publication_kind
  end;

  select publication_id into previous_id
  from public.publication_pointers
  where pointer_key = pointer
  for update;

  if previous_id is not null and previous_id <> candidate.publication_id then
    update public.publication_manifests
    set state = 'superseded'
    where publication_id = previous_id and state = 'active';
  end if;

  update public.publication_manifests
  set state = 'active', activated_at = now(), supersedes_publication_id = previous_id
  where publication_id = candidate.publication_id;

  insert into public.publication_pointers(pointer_key, publication_id, updated_at)
  values (pointer, candidate.publication_id, now())
  on conflict (pointer_key) do update
    set publication_id = excluded.publication_id,
        updated_at = excluded.updated_at;

  return candidate.publication_id;
end;
$$;

revoke all on function public.promote_publication(uuid) from public, anon, authenticated;
grant execute on function public.promote_publication(uuid) to service_role;

insert into public.schema_migrations(version, description)
values (
  '20260612_003',
  'Atomic publication manifests, versioned user portfolios and paper execution'
)
on conflict (version) do nothing;
