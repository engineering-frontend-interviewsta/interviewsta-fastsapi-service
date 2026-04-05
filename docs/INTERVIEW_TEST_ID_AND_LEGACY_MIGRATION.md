# Interview test ID as source of truth + legacy interview code

This document describes the **current** FastAPI interview stack, a **live snapshot** of the `interview_tests` table (via Prisma), the mapping to `FastApiInterviewType`, and a concrete plan to make **`interview_test_id` (UUID)** the primary input for creating and conducting interviews while moving today’s per-type LangGraph workflows under **`legacy`**.

Companion doc for transport: [INTERVIEW_EVENTSOURCE_MIGRATION.md](./INTERVIEW_EVENTSOURCE_MIGRATION.md).

---

## 1. Executive summary

| Topic | Today | Target |
| --- | --- | --- |
| **Routing** | `fastapiInterviewType` in JWT selects a hard-coded graph in `workflows/*.py` via `InterviewAgentService.get_graph()` | Load **`interview_tests`** row by UUID; derive type (and metadata) from DB; run **one** conductor (`workflows/interview/phase_engine.py`) |
| **Phase definitions** | Encoded inside each Python workflow file | Rows in **`interview_phases`** linked through **`interview_test_interview_phases`** (ordering) — **currently empty in production DB** (see §3) |
| **JWT contract** | `interviewTestId` + `fastapiInterviewType` both used | **`interviewTestId` required**; `fastapiInterviewType` becomes **optional** (validation: must match DB if both sent) or **removed** from client |
| **Legacy code** | `workflows/technical.py`, `hr.py`, `coding.py`, etc. | Move under e.g. `workflows/legacy_interviews/` (or `legacy/`) and keep only adapters/fixtures until removed |

---

## 2. `FastApiInterviewType` alignment (frontend ↔ Postgres ↔ FastAPI)

### 2.1 TypeScript (your enum)

```ts
export enum FastApiInterviewType {
  TECHNICAL = 'Technical',
  HR = 'HR',
  COMPANY = 'Company',
  SUBJECT = 'Subject',
  CASE_STUDY = 'CaseStudy',
  COMMUNICATION = 'Communication',
  ROLE_BASED = 'Role-Based Interview',
  DEBATE = 'Debate',
}
```

### 2.2 PostgreSQL (Prisma enum `interview_tests_fastapi_interview_type_enum`)

Values stored in the database (string form):

`Technical`, `HR`, `Company`, `Subject`, `CaseStudy`, `Communication`, `Role_Based_Interview` (Prisma name) → DB maps to **`Role-Based Interview`**, `Debate`.

**Important:** The TS key `CASE_STUDY` maps to the value **`CaseStudy`** (no underscore). This matches the DB enum variant `CaseStudy`, not `Case_Study`.

### 2.3 FastAPI / Pydantic

`schemas/interview.py` allows the same string literals as JWT values (`INTERVIEW_TYPES`).

`services/interview_agent.py` defines `INTERVIEW_TYPES` and `INTERRUPT_NODES` per **string** interview type — this entire matrix is **legacy** once the phase engine owns the graph.

---

## 3. Live database snapshot: `interview_tests` (Prisma introspection)

**Queried from the connected Postgres** (local dev DB as of generation time). Re-run:

```bash
python3 scripts/report_interview_tests.py
# or use Prisma query_raw / SQL in your own tooling
```

### 3.1 Row counts

- **Total rows:** 64  
- **Rows with `fastapi_interview_type` NULL:** 0  

### 3.2 Counts by `fastapi_interview_type`

| `fastapi_interview_type` | Count |
| --- | --- |
| Company | 26 |
| Subject | 19 |
| Technical | 10 |
| Role-Based Interview | 5 |
| Communication | 2 |
| CaseStudy | 1 |
| Debate | 1 |
| **HR** | **0** |

So the **HR** enum value is valid everywhere, but **there is currently no `interview_tests` row** of type HR in this database.

### 3.3 Phase linkage (`interview_test_interview_phases`)

**Critical finding:** In this database, **`interview_test_interview_phases` has 0 rows** — no interview test is linked to canonical phase rows in `interview_phases`.

Until this is populated (admin tooling, migration seed, or sync from Nest), a **DB-driven** `phase_engine` graph cannot be selected purely from `interview_test_id` without a fallback (e.g. temporary mapping table or “use legacy graph for this UUID”).

### 3.4 Full listing (all 64 rows)

Grouped by `fastapi_interview_type` for readability. Columns: `id` (UUID), `title`, `is_active`, `code`.

#### CaseStudy (1)

| id | title | is_active | code |
| --- | --- | --- | --- |
| 2f3a3c41-9f8e-49dd-ad1d-7b800cc4b759 | Case Study Interview | true | — |

#### Communication (2)

| id | title | is_active | code |
| --- | --- | --- | --- |
| 9cdda9e0-3d4f-4487-a71b-ae36e5bbeedd | Behavioral Interview | true | — |
| c35d19e0-f5a0-40aa-8a15-a87a212a7080 | Communication Interview | true | — |

#### Company (26)

| id | title |
| --- | --- |
| 3035572d-66bd-4a8e-a6c9-8b44b6b497f8 | Amazon |
| 04e8f946-ed5c-43c6-9051-4bf9341b4bf3 | Accenture |
| 11fb152d-cf90-4852-9740-bedd72e110af | Byju's |
| 5873ddaf-7dbf-4bd3-a21e-50bd73106fd4 | Capgemini |
| 3f273583-0e24-49b6-9f5f-4006a4a5ffbd | Cognizant |
| a1e97a25-0ab5-4f37-9525-ff19953a292a | Deloitte |
| dd1ab10e-6bbc-4cc4-b294-e148ed7b899d | EY |
| 7a37a9a5-d6a6-4a02-99d8-78a00a3e745b | Flipkart |
| 2b1ad34d-f912-4d97-9c4c-fec063e411dd | Google |
| 590fd735-ef34-4b47-bf5e-ee5ef24703fa | IBM |
| 1abe1e37-0089-446b-b70c-a338b59d8504 | Infosys |
| 1782700f-315e-4815-8597-561aa1ab14b7 | Intel |
| 56d56a82-9c7d-4b43-806f-dfe3c232a73c | KPMG |
| 6ed0e076-ad29-4903-83d4-da36eef5f661 | LinkedIn |
| fe3dbc47-14eb-4cc6-a99e-fe71ce7f7dd8 | Microsoft |
| 3dba324a-4fff-4030-ac64-9fa6772b9648 | Netflix |
| 4f762c09-e9cc-4cb8-85d3-2b2750bb0db9 | Ola |
| 48607403-d2f7-46f1-af18-1909b6f08364 | Oracle |
| e87bd23e-60b3-4041-809b-a925817d116f | Paytm |
| d649ee29-1ba2-4697-bd8f-8e363d3e75b2 | PhonePe |
| edb5e790-b1b5-413d-b930-2f906a4f518e | PwC |
| 90d9d04c-414b-4eb2-ace3-12f3d4707515 | Salesforce |
| bea16594-0747-45a6-b9bc-67e11d690d74 | SAP |
| d7f09b04-8601-42c8-9702-667e22c4b872 | Swiggy |
| b878ebbb-b754-4223-a4c9-54c4dcaf691f | Uber |
| 0c0be52c-2765-466a-b0d4-b1a6f2a919cc | Zomato |

#### Debate (1)

| id | title | is_active | code |
| --- | --- | --- | --- |
| b5d8e2ee-f6e1-4b2b-a40a-0cfac1e78871 | Debate and Argumentation Interview | true | — |

#### Role-Based Interview (5)

| id | title | is_active | code |
| --- | --- | --- | --- |
| 4a40f1db-c929-4a58-9d83-9fd7f09e614e | Frontend Development | true | — |
| ef2dfa08-81cb-42f5-8d95-201706e1d093 | Backend Development | true | — |
| 5c5f9240-9fe7-4ab4-8cc8-8865865ff5be | UI/UX Design | true | — |
| 1f7f9947-cb62-45f1-b79f-2fe76d9153bf | AI/ML | true | — |
| 95256ab0-9fa6-4b39-97fc-b6f1e42e1db0 | Data Science | true | — |

#### Subject (19)

| id | title |
| --- | --- |
| ef3517d6-cef4-4f7a-bf26-de6f13234220 | Arrays - 1 |
| a0a7e3c8-9649-43e5-99fd-722e7a8ef614 | Arrays - 2 |
| 510552f8-e492-475e-835e-800cb2ad9f62 | Arrays - 3 |
| 40a4b065-6122-4bdc-a016-e9e2e661193f | Graphs - 1 |
| 052d9572-30ea-4546-a5db-330a056ab3ca | Graphs - 2 |
| df47540b-002b-4d4b-a586-450c7468051b | Dynamic Programming - 1 |
| 02ffb0a9-1ff8-467a-98b3-0a7eb1867a36 | Dynamic Programming - 2 |
| 926bf150-459b-4452-b5ca-0503a5833bcc | Linked Lists - 1 |
| 4f976c02-32dc-496c-9d62-15eaf0631000 | Linked Lists - 2 |
| 604ee19a-29fc-4a3e-bd6e-aaaef7a15524 | Heaps - 1 |
| f5df5ff4-84d1-49d0-9772-792012288e74 | Heaps - 2 |
| 9ac5ae9a-56c9-4bcf-b6ea-e04e4b14818e | Stacks & Queues - 1 |
| 3a3c943e-c6c9-47f6-aefe-005e7cf34488 | Stacks & Queues - 2 |
| 022cbeac-22d3-459b-a9cc-1f5b02cd4abb | Strings - 1 |
| de0469fb-9641-4285-b910-ab187aedac1e | Strings - 2 |
| e5a6a717-5211-4cb3-a037-c9c803f656c3 | Strings - 3 |
| 1c6dec5d-1137-4c3b-9b29-edd25d5e585d | Trees - 1 |
| ba03f191-bd14-4c43-a242-ae5b2c44b715 | Trees - 2 |
| 5f3e8900-032b-4955-97df-c7668c971b0e | Trees - 3 |

#### Technical (10)

| id | title |
| --- | --- |
| 8ee11674-0589-4bb5-b37-5aab59e70129 | Maths 1 |
| 9f07709c-4df7-4890-b0d0-67a3e4da0d0b | Maths 2 |
| adb6f337-8e7e-4768-8e36-c4f496bf6134 | Engineering Chemistry |
| 7c1ea2e9-3a4a-42ae-b3ce-96e7ac13c072 | Engineering Physics |
| a5f0eeb1-ea62-43bd-a100-1551d53bf1f3 | Engineering Materials |
| 8af6ce3b-7172-4452-86b1-449acd89e68b | Manufacturing Processes |
| 0cad419f-a580-4364-9a63-dae142250f0c | Operating Systems |
| 39b018ea-9556-4d4a-9f56-56780f00eca8 | Database Management Systems |
| 5d376f8a-0ee2-4311-8850-c733d694b9cb | Microprocessors |
| 11fd7202-d3a5-4740-b039-2c6038e9b942 | Technical Interview |

All rows above: `is_active: true`, `code: null` in the snapshot DB. For an up-to-date TSV: `python3 scripts/report_interview_tests.py`.

---

## 4. Current FastAPI interview pipeline (legacy conductor)

### 4.1 HTTP layer

- Router: `api/routes/interview.py`
- Start: `POST /api/v1/interview/start` → enqueues **`process_interview_start`** Celery task with `(session_id, interview_type, user_id, payload)`.
- **`interview_type`** comes from `X-Interview-Access-Token` → `fastapiInterviewType` (or deprecated body override).
- **`interview_test_id`** is merged from the same JWT into `payload` and stored on the Redis session.

### 4.2 Celery task

- `tasks/interview_tasks.py` → `process_interview_start`:
  - Creates Redis session via `InterviewSessionManager`.
  - For **Subject** / **Company**: may call **DRF** (`services/drf_client.py`) for research text and fixed question sets.
  - Selects graph: `get_interview_agent().get_graph(interview_type, ...)`.
  - Builds initial state with `create_initial_state(interview_type, payload)`.
  - Invokes LangGraph, TTS, stores greeting in Redis.

User turns use **`process_user_response_with_transcription`** (same file) with the same graph/checkpointer keyed by `session_id`.

### 4.3 Graph registry (legacy)

- `services/interview_agent.py` imports:
  - `workflows.technical`, `workflows.hr`, `workflows.coding`, `workflows.companybuilder`, `workflows.case_study`, `workflows.communication`, `workflows.rolebased`, `workflows.debate`
- Each module exposes builders / graphs tuned to that product line.

### 4.4 New workflow folder (in progress)

- `workflows/interview/phase_engine.py` — generic phase-based LangGraph builder (`build_graph`, `PhaseConfig`, etc.).
- `workflows/interview/db_models.py` — SQLAlchemy models for **`interviews`** / **`interview_phases`** (integer `Interview.id`) used as a **dev harness** for the engine; this is **not** the same physical table as Nest’s **`interview_tests`** (UUID).
- `workflows/interview/main.py` — **currently inconsistent** (duplicate `get_interview_config`, invalid Prisma usage `prisma.interview`, undefined `PHASES`, wrong keys like `name` vs `title`). Treat as **WIP**, not production entrypoint.

**Implication:** Wiring “only `interview_test_id`” requires either:

1. **Hydrating `PhaseConfig` from Postgres** `interview_tests` + `interview_test_interview_phases` + `interview_phases`, or  
2. **Migrating** phase JSON into a structure the phase engine accepts, and  
3. **Unifying** the SQLAlchemy “interviews” dev tables with production schema **or** dropping them in favor of Prisma read models.

---

## 5. Target architecture: interview UUID only

### 5.1 Contract changes (frontend / BFF / Nest)

1. **Single required identifier:** `interviewTestId` = UUID string (matches `interview_tests.id`).
2. **`fastapiInterviewType` in JWT:**
   - **Option A (strict):** Remove from client; server loads row and reads `fastapi_interview_type`.
   - **Option B (transitional):** Keep in JWT; server **must validate** it equals `interview_tests.fastapi_interview_type` or return **400** (prevents mismatched contracts).
3. **`title`, `duration`, `credits`, `feedbackItemId`:** Can remain in JWT for UX and feedback pipeline, or be loaded from DB / feedback_items linkage — product decision.

### 5.2 `POST /start` body

- Keep `session_id` (client-generated UUID).
- **`payload`** shrinks to session-specific data only, for example:
  - `resume` (Technical / HR if reintroduced)
  - `company` / `Company` (company-specific UX; may still override DB)
  - `Tags`, research fields where LLM needs user-supplied context
- **Do not require** clients to send `interview_type` (already discouraged).

### 5.3 DRF and fixed questions

Today, **Company** and **Subject** pull question bundles from DRF using `interview_test_id`. That can remain **even in the new world**, as long as the Celery/phase task receives the UUID. Alternatively, move question lists fully into Postgres (`questions` / join tables) and drop DRF for those paths.

### 5.4 Backend implementation checklist

| Step | Action |
| --- | --- |
| 1 | Add `load_interview_test(uuid) -> InterviewTestDTO` (Prisma or SQLAlchemy read-only). |
| 2 | Load ordered phases: `interview_test_interview_phases` JOIN `interview_phases` ORDER BY `phase_order`. |
| 3 | Map rows → `PhaseConfig` (same shape as `hydrate_phase` in `workflows/interview/db_models.py`). |
| 4 | If phase list empty: **feature flag** → call existing `get_graph(fastapi_interview_type)` (legacy) until data backfilled. |
| 5 | Replace `create_initial_state(interview_type, ...)` with state built from DB metadata + payload (one `BaseInterviewState` or typed state). |
| 6 | Move `workflows/technical.py` … `debate.py` into `workflows/legacy_interviews/`; update imports; narrow `InterviewAgentService` to legacy-only shim. |
| 7 | Align `interview_tests` JSON fields (`topics`, `subjects`, `greeting_prompt`, etc.) with what `phase_engine` needs for greetings and filters. |
| 8 | Update OpenAPI + [INTERVIEW_AND_FEEDBACK_API.md](./INTERVIEW_AND_FEEDBACK_API.md) (or replace with these docs). |

### 5.5 Other services

- **Nest / monolith:** Ensure issued JWT uses **string UUID** for `interviewTestId` (Pydantic already types it as `str`).
- **DRF:** Continue to accept UUID for interview test id in URLs; confirm no remaining **integer-only** paths for production tests.
- **Analytics / billing:** Key events on `interview_tests.id` + `session_id`.

---

## 6. Risks and open questions

1. **Empty `interview_test_interview_phases`:** Blocks pure DB-driven graphs until seeded.
2. **`workflows/interview/db_models.py` `Interview` table vs `interview_tests`:** Two concepts; merging avoids confusion.
3. **HR:** Enum supported in code; no DB row yet — add tests when product enables HR.
4. **Role-Based Interview:** Legacy graph uses `payload["role"]`; DB titles (e.g. “Frontend Development”) may replace or supplement that — define mapping.
5. **Feedback:** Still keyed by `feedback_item_id` + history; unchanged by interview-id routing except better consistency of `interview_test_id` on session.

---

## 7. Suggested legacy directory layout (after move)

```
workflows/
  interview/           # phase_engine + DB hydration + future single entry
  legacy_interviews/
    technical.py
    hr.py
    coding.py
    companybuilder.py
    case_study.py
    communication.py
    rolebased.py
    debate.py
```

Update `services/interview_agent.py` imports accordingly; add a deprecation note in the package docstring.

---

*Generated from repository inspection and a live Prisma query against `interview_tests`. Re-run queries before relying on counts in production.*
