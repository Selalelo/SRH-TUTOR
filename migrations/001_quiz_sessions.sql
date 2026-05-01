-- Quiz sessions for the SPLA tutor
--
-- Each row represents one 20-question quiz attempt by a user. The 20
-- questions (with their grounded answer keys, source, and page) are
-- generated server-side at the start of the quiz from real document
-- chunks, then served one at a time. This prevents the LLM from
-- hallucinating questions during the quiz turns.
--
-- Run this in your Supabase SQL editor.

create table if not exists public.quiz_sessions (
    id            uuid primary key default gen_random_uuid(),
    user_id       uuid not null references auth.users(id) on delete cascade,
    questions     jsonb not null,                       -- array of 20 question objects
    answers       jsonb not null default '[]'::jsonb,   -- log of user answers (one per question answered)
    current_index integer not null default 0,           -- 0..20; equals 20 when finished
    score         integer not null default 0,
    topic         text,                                 -- optional topic the user asked to be quizzed on
    started_at    timestamptz not null default now(),
    completed_at  timestamptz,                          -- null while active
    cancelled_at  timestamptz                           -- null unless cancelled
);

-- Look up a user's active (not completed, not cancelled) session quickly
create index if not exists quiz_sessions_active_idx
    on public.quiz_sessions (user_id, started_at desc)
    where completed_at is null and cancelled_at is null;

create index if not exists quiz_sessions_user_idx
    on public.quiz_sessions (user_id, started_at desc);
