# Interview transport: from polling to EventSource (SSE)

This document describes **today’s** HTTP patterns for the interview API, how **`GET /{session_id}/stream`** already uses SSE, why it is still **poll-based internally**, and what to change so clients can rely on **EventSource** as the primary mechanism for start/respond completion and real-time updates.

Companion doc for product/data model: [INTERVIEW_TEST_ID_AND_LEGACY_MIGRATION.md](./INTERVIEW_TEST_ID_AND_LEGACY_MIGRATION.md).

---

## 1. Executive summary

| Aspect | Today | Target |
| --- | --- | --- |
| **Start flow** | `POST /start` → `task_id` → poll `GET /start-status/{task_id}` | Keep `POST /start` for enqueue; clients **open SSE first** (or immediately after) and receive **`start_complete`** (or errors) **without** polling Celery |
| **Respond flow** | `POST /respond` → `task_id` → poll `GET /respond-status/{task_id}` | Same: **`ai_response` / `status`** delivered on SSE when worker finishes |
| **`GET /stream`** | SSE, but server **loops every 1s** reading Redis | **Push** model: workers publish events to Redis Pub/Sub (or Redis Streams); SSE handler **subscribes** and forwards |
| **Feedback** | `POST /end` → optional `task_id` → poll `GET /feedback-status/{task_id}` | Optional second phase: `feedback_progress` / `feedback_complete` events on same or separate SSE channel |

---

## 2. Current API surface (relevant endpoints)

Source: `api/routes/interview.py`

### 2.1 Polling endpoints (Celery task state + Redis session snapshot)

| Method | Path | Role |
| --- | --- | --- |
| `POST` | `/api/v1/interview/start` | Returns `task_id`; queues `process_interview_start` |
| `GET` | `/api/v1/interview/start-status/{task_id}` | Maps Celery state + loads Redis session for greeting / transcript / completion flags |
| `POST` | `/api/v1/interview/{session_id}/respond` | Returns `task_id`; queues `process_user_response_with_transcription` |
| `GET` | `/api/v1/interview/{session_id}/respond-status/{task_id}` | Same pattern for user turn |
| `GET` | `/api/v1/interview/feedback-status/{task_id}` | Feedback task polling |

### 2.2 Existing SSE endpoint

| Method | Path | Auth |
| --- | --- | --- |
| `GET` | `/api/v1/interview/{session_id}/stream` | Query params: `token` (Bearer JWT), `interview_access_token` (same as `X-Interview-Access-Token`) — **required** because browser `EventSource` cannot set headers |

### 2.3 Events emitted today (`stream`)

From the generator in `stream_interview_status`:

| SSE `event` | When |
| --- | --- |
| `error` | Auth failure, session missing, unauthorized, fatal worker/stream errors |
| `status` | Redis `session:{id}:status` changed |
| `ai_response` | Status is `ai_responded` and a **new** `timestamp` on stored response; then status reset to `waiting_for_response` |
| `transcription` | Transcript key non-empty (then cleared) |
| `quality_warning` | Video quality / engagement warnings |
| `complete` | Status → `completed` |

**Implementation detail:** The handler **polls Redis every 1 second** (`await asyncio.sleep(1)`). It does **not** receive push notifications from Celery. Under load, this adds latency (up to ~1s) and constant Redis reads per open connection.

---

## 3. Why clients still poll today

Historical reasons:

1. **Celery result readiness** is exposed only through `AsyncResult(task_id)` on poll endpoints.
2. **SSE** was added for “live” UX but **duplicates** information also returned by poll responses (`interview_ai_response`, etc.).
3. **EventSource** cannot send `Authorization` / `X-Interview-Access-Token` headers — query params were already solved for SSE.

---

## 4. Target design: EventSource-first

### 4.1 Principles

1. **One long-lived SSE connection per active interview session** (per tab/device).
2. **All state transitions** that today appear in poll JSON should be representable as **named SSE events** with a small JSON payload.
3. **Workers publish** when state is ready; **SSE** waits on subscription, not on a 1s loop (except heartbeat).
4. **Polling endpoints** become **deprecated** (410/Warning header) or thin wrappers for non-browser clients.

### 4.2 Recommended event vocabulary (extended)

Reuse existing names where possible; add:

| SSE `event` | Purpose | Example `data` shape |
| --- | --- | --- |
| `connected` | Stream ready, session bound | `{"session_id":"..."} ` |
| `heartbeat` | Keep proxies from closing idle connections | `{"t": 1730000000}` |
| `celery_start` | Start task accepted | `{"task_id":"...","phase":"start"}` |
| `start_progress` | Optional: map Celery PROGRESS meta | `{"progress":40,"message":"..."}` |
| `start_complete` | Start task SUCCESS — include same fields as today’s poll `interview_ai_response` + session snapshot | `{"task_id":"...","interview_ai_response":{...},"interview_transcript":...}` |
| `start_failed` | Start task FAILURE | `{"task_id":"...","error":"..."}` |
| `celery_respond` | Respond task queued | `{"task_id":"..."}` |
| `respond_progress` | Optional transcription / LLM stages | `{"progress":60,"message":"..."}` |
| `respond_complete` | Same payload family as current `respond-status` success | `{...}` |
| `respond_failed` | Respond task failure | `{"error":"..."}` |
| `ai_response` | **Keep** — AI turn visible (can mirror `respond_complete` or be the canonical one) | (existing) |
| `transcription` | **Keep** | (existing) |
| `status` | **Keep** | (existing) |
| `quality_warning` | **Keep** | (existing) |
| `complete` | **Keep** | (existing) |
| `feedback_queued` | After `POST /end` | `{"feedback_task_id":"..."}` |
| `feedback_complete` | Feedback task done | `{"feedback":{...}}` |

Exact payloads should match **current** `InterviewStartStatusResponse` / `RespondTaskStatusResponse` fields to minimize frontend churn, then slim down in a v2.

### 4.3 Redis Pub/Sub sketch

- **Channel:** `interview:session:{session_id}:events`
- **Publishers:**  
  - `process_interview_start` (success/failure/progress)  
  - `process_user_response_with_transcription`  
  - Optional: `generate_feedback` task  
- **Subscriber:** FastAPI SSE generator: `async for message in pubsub.listen(): ... yield f"event: ...\n\n"`

Use **connection pooling**; ensure Celery workers use a **sync** Redis client `publish()` after session writes so ordering is: Redis session updated → publish.

**Alternative:** Redis Streams with consumer groups if you need replay after reconnect (more complex; better for “missed events”).

### 4.4 Heartbeat and timeouts

- Emit `heartbeat` every **15–30s**.
- Align with reverse proxy timeouts (nginx `proxy_read_timeout`, Render/load balancer idle timeout).
- Document client **reconnect with Last-Event-ID** only if you implement stream replay (Streams); otherwise **reconnect + poll once** as fallback during transition.

### 4.5 Authentication (unchanged contract)

- **Query params:** `token`, `interview_access_token` on `GET .../stream`.
- **CORS:** Ensure `Access-Control-Allow-Origin` for web app if ever using `fetch` stream; native `EventSource` is GET and same-origin or CORS for credentialed cases — your current setup is query-based to avoid CORS preflight issues.

---

## 5. Frontend migration guide

### 5.1 Today (polling)

```text
POST /start → task_id
loop: GET /start-status/{task_id} until status === "completed"
POST /respond → task_id
loop: GET /respond-status/{session_id}/{task_id} until completed
```

Optional parallel `EventSource` on `/stream` for incremental UX.

### 5.2 Target (EventSource-first)

1. Obtain JWTs as today.
2. **Open** `EventSource` to `/api/v1/interview/{session_id}/stream?token=...&interview_access_token=...` **before or immediately after** `POST /start` (same `session_id`).
3. **POST /start**; ignore `task_id` for UX if `start_complete` will carry results (still log `task_id` for support).
4. On **`start_complete`**, render greeting / audio from payload (same mapping as current poll response).
5. On user turn: **POST /respond**; listen for **`respond_complete`** or **`ai_response`**.
6. On **`complete`**, call **`POST /end`** if required; listen for **`feedback_*`** or keep short poll for feedback only during transition.

### 5.3 Client implementation notes

- **Single `EventSource` instance** per session; close on unmount.
- **Reconnect:** on `error` event or `onerror`, exponential backoff; after reconnect, **optionally** one `GET start-status` if you still expose it (transition).
- **Base64 audio:** unchanged; large payloads over SSE are acceptable but watch **proxy buffer** settings (`X-Accel-Buffering: no` already set).
- **Mobile / React Native:** use a library that supports SSE over GET with query params; not all HTTP clients support SSE equally well.

---

## 6. Other services (BFF, Nest, DRF)

| Consumer | Change |
| --- | --- |
| **Browser app** | Prefer SSE; remove poll loops when backend emits `start_complete` / `respond_complete`. |
| **BFF** | If BFF proxies interview API, either **proxy SSE** (chunked) or expose WebSocket bridge; simplest is **transparent proxy** with long timeouts. |
| **Mobile native** | Same as web; ensure load balancer idle timeout > heartbeat interval. |
| **Automated tests** | Use `httpx`/`curl` streaming client or keep poll endpoints until deprecated. |

---

## 7. Backend implementation checklist (FastAPI + Celery)

| # | Task |
| --- | --- |
| 1 | Add small `services/interview_events.py`: `publish_session_event(session_id, event_name, dict)`. |
| 2 | Call `publish_session_event` from `process_interview_start` on PROGRESS, SUCCESS, FAILURE (after Redis session writes). |
| 3 | Same for `process_user_response_with_transcription`. |
| 4 | Refactor `stream_interview_status` to **subscribe** to `interview:session:{session_id}:events` + **short** Redis poll only for backward-compat fields not yet published (temporary). |
| 5 | Add `heartbeat` asyncio task cancelled on disconnect. |
| 6 | Deprecate `start-status` / `respond-status` in OpenAPI (`deprecated=True`) with description pointing to SSE. |
| 7 | Load-test: concurrent SSE + Celery workers + Redis pub/sub throughput. |
| 8 | (Optional) Feedback task → publish `feedback_complete` to same channel. |

---

## 8. Observability and ops

- **Metrics:** open SSE connections per instance, events/sec published, Redis pub/sub latency.
- **Logs:** correlate `session_id`, `task_id`, Celery `task_id` in publish path.
- **Rate limiting:** protect `GET /stream` from connection exhaustion (per user / IP).

---

## 9. Rollout strategy

1. **Phase 1:** Implement publish + consume; keep poll endpoints; frontend opt-in flag uses SSE for completion.
2. **Phase 2:** Default frontend to SSE; polls for monitoring only.
3. **Phase 3:** Remove or hard-deprecate poll endpoints in API v2.

---

*Based on `api/routes/interview.py` as of the migration planning pass.*
