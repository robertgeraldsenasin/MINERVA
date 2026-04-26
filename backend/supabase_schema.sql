-- MINERVA Supabase / Postgres schema
-- Purpose:
-- 1) store Unity-ready story cards
-- 2) return only unused cards per authenticated player
-- 3) record player decisions for learning analytics and replay prevention

create extension if not exists pgcrypto;

create table if not exists public.cards (
    id text primary key,
    deck_version text not null default 'v1',
    day integer,
    verdict text check (verdict in ('real', 'fake', 'neutral')),
    truth_type text check (truth_type in ('blue', 'red', 'neutral')),
    is_misinformation boolean,
    targets text[] not null default '{}',
    linked_blue_truth_id text,
    difficulty_bin text,
    text text not null,
    card_json jsonb not null,
    active boolean not null default true,
    created_at timestamptz not null default now()
);

create index if not exists idx_cards_active_day on public.cards(active, day);
create index if not exists idx_cards_targets_gin on public.cards using gin(targets);
create index if not exists idx_cards_verdict on public.cards(verdict);

create table if not exists public.user_card_state (
    user_id uuid not null references auth.users(id) on delete cascade,
    card_id text not null references public.cards(id) on delete cascade,
    status text not null check (status in ('seen', 'answered', 'discarded', 'used')),
    player_verdict text check (player_verdict in ('real', 'fake', 'neutral')),
    shared boolean not null default false,
    confidence numeric(5,4),
    day integer,
    first_seen_at timestamptz not null default now(),
    last_seen_at timestamptz not null default now(),
    used_at timestamptz,
    metadata jsonb not null default '{}'::jsonb,
    primary key (user_id, card_id)
);

create index if not exists idx_user_card_state_user_status on public.user_card_state(user_id, status);
create index if not exists idx_user_card_state_user_day on public.user_card_state(user_id, day);

create table if not exists public.deck_versions (
    deck_version text primary key,
    description text,
    imported_at timestamptz not null default now(),
    imported_by uuid,
    source_note text
);

alter table public.cards enable row level security;
alter table public.user_card_state enable row level security;
alter table public.deck_versions enable row level security;

-- Cards are read-only to authenticated players.
drop policy if exists cards_select_authenticated on public.cards;
create policy cards_select_authenticated
on public.cards
for select
to authenticated
using (active = true);

-- Only service role should write cards/deck versions.
drop policy if exists cards_no_client_write on public.cards;
create policy cards_no_client_write
on public.cards
for all
to authenticated
using (false)
with check (false);

drop policy if exists deck_versions_select_authenticated on public.deck_versions;
create policy deck_versions_select_authenticated
on public.deck_versions
for select
to authenticated
using (true);

drop policy if exists deck_versions_no_client_write on public.deck_versions;
create policy deck_versions_no_client_write
on public.deck_versions
for all
to authenticated
using (false)
with check (false);

-- Players can only see and modify their own state rows.
drop policy if exists user_card_state_select_own on public.user_card_state;
create policy user_card_state_select_own
on public.user_card_state
for select
to authenticated
using (auth.uid() = user_id);

drop policy if exists user_card_state_insert_own on public.user_card_state;
create policy user_card_state_insert_own
on public.user_card_state
for insert
to authenticated
with check (auth.uid() = user_id);

drop policy if exists user_card_state_update_own on public.user_card_state;
create policy user_card_state_update_own
on public.user_card_state
for update
to authenticated
using (auth.uid() = user_id)
with check (auth.uid() = user_id);

-- RPC: get next unseen cards for the authenticated player.
create or replace function public.get_next_cards(
    p_day integer default null,
    p_limit integer default 10,
    p_candidate_ids text[] default null
)
returns setof jsonb
language sql
security invoker
set search_path = public
as $$
    select c.card_json || jsonb_build_object(
        'id', c.id,
        'day', c.day,
        'verdict', c.verdict,
        'truth_type', c.truth_type,
        'targets', c.targets,
        'linked_blue_truth_id', c.linked_blue_truth_id
    )
    from public.cards c
    left join public.user_card_state s
        on s.card_id = c.id
       and s.user_id = auth.uid()
       and s.status in ('discarded', 'used')
    where c.active = true
      and (p_day is null or c.day = p_day)
      and (p_candidate_ids is null or c.targets && p_candidate_ids)
      and s.card_id is null
    order by c.day nulls first, c.id asc
    limit greatest(coalesce(p_limit, 10), 1);
$$;

grant execute on function public.get_next_cards(integer, integer, text[]) to authenticated;

-- RPC: save a player's decision and mark the card as used/discarded.
create or replace function public.submit_card_result(
    p_card_id text,
    p_player_verdict text,
    p_shared boolean default false,
    p_confidence numeric default null,
    p_day integer default null,
    p_metadata jsonb default '{}'::jsonb
)
returns jsonb
language plpgsql
security invoker
set search_path = public
as $$
declare
    v_user uuid;
    v_status text;
begin
    v_user := auth.uid();
    if v_user is null then
        raise exception 'Not authenticated';
    end if;

    v_status := case
        when coalesce(p_shared, false) then 'used'
        else 'answered'
    end;

    insert into public.user_card_state (
        user_id, card_id, status, player_verdict, shared, confidence, day, metadata, first_seen_at, last_seen_at, used_at
    )
    values (
        v_user,
        p_card_id,
        v_status,
        p_player_verdict,
        coalesce(p_shared, false),
        p_confidence,
        p_day,
        coalesce(p_metadata, '{}'::jsonb),
        now(),
        now(),
        case when coalesce(p_shared, false) then now() else null end
    )
    on conflict (user_id, card_id)
    do update set
        status = excluded.status,
        player_verdict = excluded.player_verdict,
        shared = excluded.shared,
        confidence = excluded.confidence,
        day = excluded.day,
        metadata = coalesce(public.user_card_state.metadata, '{}'::jsonb) || excluded.metadata,
        last_seen_at = now(),
        used_at = case
            when excluded.shared then now()
            else public.user_card_state.used_at
        end;

    return jsonb_build_object(
        'status', 'ok',
        'card_id', p_card_id,
        'shared', coalesce(p_shared, false),
        'saved_at', now()
    );
end;
$$;

grant execute on function public.submit_card_result(text, text, boolean, numeric, integer, jsonb) to authenticated;

-- Optional admin import helper.
create or replace function public.import_cards(p_cards jsonb, p_deck_version text default 'v1')
returns jsonb
language plpgsql
security definer
set search_path = public
as $$
declare
    rec jsonb;
    n int := 0;
begin
    if jsonb_typeof(p_cards) <> 'array' then
        raise exception 'p_cards must be a JSON array';
    end if;

    insert into public.deck_versions(deck_version, description)
    values (p_deck_version, 'Imported via import_cards RPC')
    on conflict (deck_version) do nothing;

    for rec in select * from jsonb_array_elements(p_cards)
    loop
        insert into public.cards (
            id, deck_version, day, verdict, truth_type, is_misinformation, targets,
            linked_blue_truth_id, difficulty_bin, text, card_json, active
        )
        values (
            rec->>'id',
            p_deck_version,
            nullif(rec->>'day', '')::integer,
            lower(rec->>'verdict'),
            lower(rec->'classification'->>'truth_type'),
            coalesce((rec->'classification'->>'is_misinformation')::boolean, false),
            coalesce(array(select jsonb_array_elements_text(coalesce(rec->'targets', '[]'::jsonb))), '{}'::text[]),
            rec->'classification'->>'linked_blue_truth_id',
            rec->>'difficulty_bin',
            rec->>'text',
            rec,
            true
        )
        on conflict (id) do update set
            deck_version = excluded.deck_version,
            day = excluded.day,
            verdict = excluded.verdict,
            truth_type = excluded.truth_type,
            is_misinformation = excluded.is_misinformation,
            targets = excluded.targets,
            linked_blue_truth_id = excluded.linked_blue_truth_id,
            difficulty_bin = excluded.difficulty_bin,
            text = excluded.text,
            card_json = excluded.card_json,
            active = true;
        n := n + 1;
    end loop;

    return jsonb_build_object('status', 'ok', 'imported', n, 'deck_version', p_deck_version);
end;
$$;

-- IMPORTANT:
-- Revoke this from regular clients and call with service role only.
revoke all on function public.import_cards(jsonb, text) from public, anon, authenticated;
