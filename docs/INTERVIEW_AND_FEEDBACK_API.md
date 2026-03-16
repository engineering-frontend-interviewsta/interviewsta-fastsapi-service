# Interview & Feedback API – Frontend Documentation

**Live Swagger UI:** `GET /docs` (e.g. `https://your-service.com/docs`) – interactive OpenAPI docs.  
**ReDoc:** `GET /redoc`

Base URL for all endpoints below: **`/api/v1/interview`** (interview) and **`/api/v1/feedback`** (feedback).  
Example: `https://your-service.com/api/v1/interview/start`

---

## 1. Authentication & Required Headers

All interview endpoints require **two** tokens:

| Header | Required | Description |
|--------|----------|-------------|
| `Authorization` | Yes | `Bearer <JWT>` – User auth token (same secret as your auth service `JWT_SECRET`). Payload: `{ sub, email, name, roles }`. |
| `X-Interview-Access-Token` | Yes | JWT signed with the **same secret** as the Bearer token. Encodes the interview contract (test id, title, feedback schema, etc.). |

**Exception:** The SSE stream endpoint (`GET /{session_id}/stream`) cannot send custom headers (EventSource). Use **query parameters** instead: `token` (Bearer token) and `interview_access_token` (same value as `X-Interview-Access-Token`).

**Start interview:** The request body must **not** include `interview_type` or `user_id`. Both are derived from the two headers (see §3.1).

### X-Interview-Access-Token payload (encode this as JWT)

```ts
interface InterviewAccessTokenPayload {
  sub: string;              // userId
  interviewTestId: number;   // Interview test id (used for feedback and DRF)
  title: string;             // Interview title
  credits?: number;         // default 1
  duration?: number;         // e.g. minutes
  fastapiInterviewType?: string;  // One of: "Technical" | "HR" | "Company" | "Subject" | "CaseStudy" | "Communication" | "Role-Based Interview" | "Debate"
  feedbackItemId?: string;   // e.g. "fi-coding-i" – drives feedback schema and SaveFeedbackDto (see §5)
}
```

Valid `fastapiInterviewType` values:  
`Technical`, `HR`, `Company`, `Subject`, `CaseStudy`, `Communication`, `Role-Based Interview`, `Debate`.

Valid `feedbackItemId` values (from backend `feedback_items.json`):  
`fi-coding-i`, `fi-faang`, `fi-product-based`, `fi-mass-hiring`, `fi-communication`, `fi-debate`, `fi-role-based`.

---

## 2. Interview flow (high level)

```
1. POST /api/v1/interview/start     → get task_id
2. GET  /api/v1/interview/start-status/{task_id}  → poll until status is "completed"
3. (Optional) GET /api/v1/interview/{session_id}/stream?token=...&interview_access_token=...  → SSE for real-time updates
4. For each user turn:
   - POST /api/v1/interview/{session_id}/respond  → get task_id
   - GET  /api/v1/interview/{session_id}/respond-status/{task_id}  → poll until "completed"
5. POST /api/v1/interview/end       → optionally get feedback task_id
6. If feedback task_id returned:
   - GET /api/v1/interview/feedback-status/{task_id}  → poll until status is "completed", then use result
```

---

## 3. Interview API reference (Swagger-style)

### 3.1 Start interview

**POST** `/api/v1/interview/start`

**Headers**

- `Authorization: Bearer <user_jwt>`
- `X-Interview-Access-Token: <interview_contract_jwt>`
- `Content-Type: application/json`

**Request body (only `session_id` and `payload`; do not send `interview_type` or `user_id`)**

`interview_type` and `user_id` are **decoded from the two headers** and must not be sent in the body (they are encoded/encrypted in the JWTs):

- **interview_type** is taken from **X-Interview-Access-Token** → `fastapiInterviewType`. It must be set in that token; if missing, the API returns 400.
- **user_id** is taken from **Authorization** (Bearer) → `sub` (or `uid`, then `email` as fallback). Session ownership uses this value.

```json
{
  "session_id": "uuid-v4-string",
  "payload": {
    "resume": "...",
    "interview_test_id": 68,
    "Tags": ["Arrays", "DP"],
    "company": "Microsoft",
    "QuestionResearch": "..."
  }
}
```

- **session_id** (required): Unique session id (e.g. UUID v4).
- **payload** (optional): Type-specific data (e.g. Company/Subject: `company`, `subject`, `Tags`, `Questions`, `interview_test_id`; Coding: `resume`, `TechnicalResearch`, `CodingResearch`). `interview_test_id` and `feedback_item_id` can also be merged from the interview token by the backend if present in the token.

**Response** `200`

```json
{
  "task_id": "celery-task-uuid",
  "session_id": "uuid-v4-string",
  "status": "queued",
  "message": "Interview initialization queued"
}
```

---

### 3.2 Poll start status

**GET** `/api/v1/interview/start-status/{task_id}`

**Headers**

- `Authorization: Bearer <user_jwt>`
- `X-Interview-Access-Token: <interview_contract_jwt>`

**Response** `200`

```json
{
  "task_id": "...",
  "session_id": "uuid or null until completed",
  "status": "queued | processing | completed | failed",
  "progress": 0,
  "message": "Ready",
  "result": { "status": "ai_responded", "message": "...", "last_node": "..." },
  "error": null,
  "interview_status": "ai_responded",
  "interview_ai_response": {
    "message": "Hello, I'm your interviewer...",
    "audio": "base64_mp3...",
    "audio_base64": "base64_mp3...",
    "last_node": "Greeting",
    "timestamp": "2025-02-27T12:00:00.000Z",
    "question_number": 1,
    "total_questions": 5,
    "question_raw_content": "// starter code if coding question",
    "interview_questions": [ { "question_title": "...", "question_description": "...", "question_raw_content": "..." } ]
  },
  "interview_transcript": null,
  "interview_is_complete": false,
  "interview_warning": null
}
```

When `status === "completed"`, use `interview_ai_response` (and optional `interview_questions` for Company/Subject) for the first AI message and any question list.

---

### 3.3 Submit user response

**POST** `/api/v1/interview/{session_id}/respond`

**Headers**

- `Authorization: Bearer <user_jwt>`
- `X-Interview-Access-Token: <interview_contract_jwt>`
- `Content-Type: application/json`

**Request body**

- At least one of `audio_data` or `text_response` is required.
- `audio_data`: base64-encoded audio (for speech; will be transcribed).
- `text_response`: plain text (e.g. Communication writing/comprehension, or text-only answers).
- `code_input`: optional code submission for coding phases.

```json
{
  "audio_data": "base64_wav_or_mp3...",
  "text_response": null,
  "code_input": null,
  "skip_audio": false
}
```

**Response** `200`

```json
{
  "task_id": "celery-task-uuid",
  "session_id": "uuid-v4-string",
  "status": "processing"
}
```

---

### 3.4 Poll respond status

**GET** `/api/v1/interview/{session_id}/respond-status/{task_id}`

**Headers**

- `Authorization: Bearer <user_jwt>`
- `X-Interview-Access-Token: <interview_contract_jwt>`

**Response** `200`

Same shape as start-status; when `status === "completed"`, `interview_ai_response` contains the next AI message, `question_number`, `total_questions`, `question_raw_content` (for coding), and for Company/Subject `interview_questions`.

---

### 3.5 SSE stream (real-time updates)

**GET** `/api/v1/interview/{session_id}/stream?token={bearer_jwt}&interview_access_token={interview_contract_jwt}`

- **Query params:** `token` (user Bearer JWT), `interview_access_token` (same as header `X-Interview-Access-Token`).
- **Content-Type:** `text/event-stream`.

**Events**

- `status` – `{ "status": "waiting_for_response" | "processing" | "ai_responded" | "completed" }`
- `ai_response` – same object as `interview_ai_response` above (message, audio, question_number, question_raw_content, etc.)
- `transcription` – `{ "text": "user's transcribed speech" }`
- `quality_warning` – `{ "type": "...", "message": "..." }` (e.g. face detection, engagement)
- `complete` – interview finished
- `error` – `{ "error": "message", "fatal": true|false }`

---

### 3.6 Video quality (optional)

**POST** `/api/v1/interview/{session_id}/video-quality`

**Headers:** Same as above.

**Body:** `{ "face": "ok", "gaze": 0.8, "confidence": 0.7, "nervousness": 0.2, "engagement": 85, "distraction": 10 }`

---

### 3.7 Video telemetry (optional)

**POST** `/api/v1/interview/{session_id}/video-telemetry`  
**GET** `/api/v1/interview/{session_id}/video-telemetry`

Same headers; POST body includes `face`, `gaze`, `confidence`, `nervousness`, `engagement`, `distraction`, optional `big5_features`.

---

### 3.8 End interview

**POST** `/api/v1/interview/end`

**Headers**

- `Authorization: Bearer <user_jwt>`
- `X-Interview-Access-Token: <interview_contract_jwt>`
- `Content-Type: application/json`

**Request body**

```json
{
  "session_id": "uuid-v4-string",
  "interview_type": "Company",
  "interview_test_id": 68,
  "duration": 1200,
  "session_finished": true
}
```

**Response** `200`

```json
{
  "task_id": "feedback-celery-task-uuid or null",
  "status": "ended",
  "session_id": "uuid-v4-string",
  "message": "Interview ended successfully"
}
```

- If there was conversation history, `task_id` is set; use it to poll feedback status. If no history, `task_id` is `null`.

---

### 3.9 Poll feedback status (after end)

**GET** `/api/v1/interview/feedback-status/{task_id}`

**Headers**

- `Authorization: Bearer <user_jwt>`
- `X-Interview-Access-Token: <interview_contract_jwt>`

**Response** `200`

```json
{
  "task_id": "...",
  "session_id": "uuid",
  "status": "queued | processing | completed | failed",
  "progress": 100,
  "result": { ... },
  "error": null
}
```

When `status === "completed"`, `result` is the feedback object (see §4 and §5).

---

### 3.10 Delete session

**DELETE** `/api/v1/interview/{session_id}`

**Headers:** Same as above.  
**Response** `200`: `{ "status": "deleted", "session_id": "..." }`

---

## 4. Feedback result shape (legacy path)

When **no** `feedbackItemId` is provided (or the feedback-item pipeline is not used), the backend returns the **legacy** feedback structure in `result`:

```json
{
  "language_score": 75,
  "framework_score": 70,
  "algorithms_score": 80,
  "data_structures_score": 78,
  "approach_score": 82,
  "optimization_score": 72,
  "debugging_score": 68,
  "syntax_score": 85,
  "strengths": ["...", "...", "..."],
  "areas_of_improvements": ["...", "...", "..."],
  "interaction_log_feedback": {
    "answer_status": ["correct", "partially-correct", ...],
    "comment": "..."
  }
}
```

---

## 5. Feedback result shape (feedback-item path / SaveFeedbackDto)

When **`feedbackItemId`** is present in the X-Interview-Access-Token (and valid in backend), feedback is generated from `feedback_items.json` and stored via the **SaveFeedbackDto** schema. The **`result`** from `GET /feedback-status/{task_id}` then looks like:

```json
{
  "items": {
    "Problem Solving & Technical Logic": {
      "Requirement Clarification": 72,
      "Algorithmic Efficiency": 85,
      "Data Structure Selection": 78,
      "Edge Case Handling": -1
    },
    "Code Quality & Communication": {
      "Readability": 80,
      "Syntactic Fluency": 75,
      "Verbal Communication": 82,
      "Debugging": 70
    }
  },
  "strengths": ["...", "...", "..."],
  "areas_of_improvements": ["...", "...", "..."],
  "interaction_feedback": [
    { "status": "correct", "comment": "" },
    { "status": "partially-correct", "comment": "Consider edge case when n=0." }
  ],
  "interaction_status_logs": [
    { "status": "correct", "comment": "" },
    { "status": "partially-correct", "comment": "Consider edge case when n=0." }
  ],
  "interview_type_id": "fi-coding-i",
  "interview_title": "Coding - I"
}
```

- **`items`**: Scores by sleeve and metric (same structure as in `feedback_items.json`). `-1` = not tested.
- **`strengths`** / **`areas_of_improvements`**: Arrays of strings (e.g. length 3).
- **`interaction_feedback`** / **`interaction_status_logs`**: Per Q&A; `status` is one of `correct`, `incorrect`, `partially-correct`, `cross-question`.

The **internal save API** (DRF) expects this schema (matching SaveFeedbackDto):

| Field | Type | Description |
|-------|------|-------------|
| `interviewTestId` | string | Interview test id (uuid or string id). |
| `items` | FeedbackItemsMap | Sleeve → subkey → score (e.g. `{ "problem-solving": { "approach": 85 } }`). |
| `strengths` | string[] | Optional; default `[]`. |
| `duration` | string | `"HH:MM:SS"` (e.g. `"00:20:00"`). |
| `areasOfImprovements` | string[] | Optional; default `[]`. |
| `interactionLogs` | object[] | Optional; default `[]`. |
| `interactionStatusLogs` | object[] | Optional; default `[]` (e.g. `{ status, comment }`). |

---

## 6. Feedback API (alternative entrypoint)

You can also trigger feedback via the feedback router (same auth as above if applicable):

**POST** `/api/v1/feedback/generate`

**Body:** `{ "session_id": "...", "interview_type": "Company", "user_id": "..." }`

**Response:** `{ "task_id": "...", "session_id": "...", "status": "queued" }`

**GET** `/api/v1/feedback/{task_id}/status`  
Same response shape as `GET /api/v1/interview/feedback-status/{task_id}`.

**GET** `/api/v1/feedback/session/{session_id}`  
Returns cached feedback for that session if available.

---

## 7. Error responses

- **401 Unauthorized** – Missing or invalid `Authorization` or `X-Interview-Access-Token`, or token expired.
- **403 Forbidden** – Session exists but does not belong to the authenticated user.
- **404 Not Found** – Session or task not found.
- **429 Too Many Requests** – Previous respond task still processing; wait and retry.

All errors return a JSON body like: `{ "detail": "message" }`.

---

## 8. Quick reference – Interview endpoints

| Method | Path | Purpose |
|--------|------|--------|
| POST | `/api/v1/interview/start` | Start interview, get `task_id` |
| GET | `/api/v1/interview/start-status/{task_id}` | Poll until first AI message ready |
| POST | `/api/v1/interview/{session_id}/respond` | Send user answer (audio/text/code) |
| GET | `/api/v1/interview/{session_id}/respond-status/{task_id}` | Poll until next AI message ready |
| GET | `/api/v1/interview/{session_id}/stream?token=...&interview_access_token=...` | SSE stream |
| POST | `/api/v1/interview/{session_id}/video-quality` | Send video/behavioral metrics |
| POST | `/api/v1/interview/{session_id}/video-telemetry` | Send telemetry (soft skills) |
| GET | `/api/v1/interview/{session_id}/video-telemetry` | Get stored telemetry |
| POST | `/api/v1/interview/end` | End session, optionally get feedback `task_id` |
| GET | `/api/v1/interview/feedback-status/{task_id}` | Poll feedback result |
| DELETE | `/api/v1/interview/{session_id}` | Delete session |

Every request (except health) must include **Authorization** and **X-Interview-Access-Token** (or, for stream, the same values in query params).
