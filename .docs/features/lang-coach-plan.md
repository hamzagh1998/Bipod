# Language Coach MVP: Decision Record and Reviewable Slice Plan

## 1. Locked Decisions

- Surface: add a new in-app route/page at `/coach`; reuse current Bipod auth.
- Frontend architecture: hybrid. Keep existing legacy frontend unchanged; build coach UI separately in React.
- Data model: add coach-specific tables with explicit coaching fields (`target_language`, `cefr_level`, per-turn scores, mistakes).
- Voice UX: push-to-talk streaming, with partial transcript updates while user is speaking.
- Session language: one target language per session.
- Correction mode: strict correction (report all detected issues).
- Progress tracking: include all three in MVP: per-turn scores/mistakes, session review, and trend/dashboard data.
- Model policy: default to strongest available model, allow per-session model selection, and auto-downgrade when latency guard is exceeded.
- Privacy: transcript-only by default; raw audio retention is explicit opt-in.
- Done criteria: ship only when end-to-end voice turn, review page, and dashboard are all working.

## 2. Enforced Product/Tech Defaults

- Feedback contract per turn: always return `reply` and `score`; include `correction` and `explanation` only when needed.
- Explanation language: bilingual (target language + learner language).
- Score system: `0-100` overall plus optional sub-scores (`grammar`, `vocabulary`, `fluency`, `pronunciation_proxy`).
- Streaming protocol: NDJSON event stream (consistent with existing `/chat/stream` style) for MVP simplicity.
- Migration strategy: additive schema only in MVP (no destructive migrations), because current app uses `Base.metadata.create_all`.

## 3. MVP Interface Contract (Docs-Level)

- New page route: `GET /coach` serves the coach app shell.
- Session APIs:
- `POST /api/v1/coach/sessions` create session (`target_language`, `native_language`, `cefr_level`, optional preferred model).
- `GET /api/v1/coach/sessions` list sessions.
- `GET /api/v1/coach/sessions/{id}` session detail + aggregate stats.
- Turn APIs:
- `POST /api/v1/coach/turns/stream` push-to-talk upload + streamed events.
- Event types: `stt_partial`, `stt_final`, `coach_reply`, `feedback`, `score`, `done`, `error`, `model_fallback`.
- Review APIs:
- `GET /api/v1/coach/sessions/{id}/turns` timeline.
- `GET /api/v1/coach/sessions/{id}/mistakes` strict mistake inventory.
- `GET /api/v1/coach/progress` trend metrics for dashboard.

## 4. Reviewable Implementation Slices

### Slice 0: Spec and Contracts
- Deliverable: API/event schema doc and DB schema draft.
- Review gate: names and payloads are stable and accepted before coding.

### Slice 1: Data Layer (Additive)
- Deliverable: coach tables + ORM models + read/write service methods.
- Review gate: creates cleanly on existing DB; existing chat tables untouched.

### Slice 2: Coach Session API
- Deliverable: create/list/get session endpoints with auth reuse.
- Review gate: session ownership enforced; schema validation complete.

### Slice 3: Streaming STT Pipeline Skeleton
- Deliverable: push-to-talk endpoint emits `stt_partial` and `stt_final` events.
- Review gate: partial updates visible before final transcript; error events standardized.

### Slice 4: Coach LLM Turn Generation
- Deliverable: reply + score generation with conditional correction/explanation.
- Review gate: contract compliance (`reply/score` always present), bilingual explanation formatting.

### Slice 5: Strict Mistake Extraction + Persistence
- Deliverable: normalize and persist mistake objects per turn/session.
- Review gate: all detected issues stored and queryable; no silent dropping.

### Slice 6: Model Selection + Fallback Guard
- Deliverable: strongest-model default, per-session override, latency-based downgrade + `model_fallback` event.
- Review gate: fallback is deterministic and logged; user-visible reason returned.

### Slice 7: React Coach Shell Integration
- Deliverable: React/Vite coach app mounted at `/coach`, legacy pages unaffected.
- Review gate: no regressions to existing `frontend/index.html` or `studio.html`.

### Slice 8: Push-to-Talk UI + Live Transcript
- Deliverable: mic hold/release flow, live transcript panel, final transcript lock.
- Review gate: interaction reliable across repeated turns.

### Slice 9: Feedback and Session Review UI
- Deliverable: per-turn feedback cards, strict corrections list, replayable review timeline.
- Review gate: full history is readable and tied to stored turn data.

### Slice 10: Progress Dashboard
- Deliverable: trend charts + mistake frequency + session-level progression.
- Review gate: metrics match persisted aggregates.

### Slice 11: Privacy Controls + MVP Acceptance
- Deliverable: transcript-only default, audio retention toggle, acceptance checklist run.
- Review gate: all MVP done criteria pass end-to-end.

## 5. Out of Scope for MVP

- Always-on VAD conversation mode.
- TTS voice output.
- Pronunciation scoring beyond transcript-based proxy scoring.
- Multi-user org features or cloud sync.
