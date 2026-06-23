# CommByAI FastAPI Service

AI-powered tutoring microservice for CommByAI. Handles LLM-based activity generation,
writing/speaking evaluation, in-character scenario simulation, and audio (TTS/STT).

## Architecture

- **Speech-to-Text (STT)**: Cartesia `ink-whisper`
- **Text-to-Speech (TTS)**: AWS Polly
- **LLM**: OpenAI (default `gpt-4o-mini`)
- **Auth**: Shared JWT secret with the NestJS backend

All endpoints require JWT auth via `Authorization: Bearer <token>`. The signing key
**must match** the NestJS backend's `JWT_SECRET` (or this service's `JWT_SIGNING_KEY`).

## Quick Start

```bash
cd comm-by-ai/fastapi-service

# Install dependencies
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env — at minimum set OPENAI_API_KEY and JWT_SIGNING_KEY

# Run
uvicorn main:app --reload --port 8000
```

OpenAPI docs: <http://localhost:8000/docs>

## Endpoints

All under `/api/v1/comm`:

- `POST /onboarding/assess` — Deterministic placement scoring (no LLM call)
- `POST /activity/generate` — LLM-generated activity (sentence reorder / word learn / reading / scenario / etc.)
- `POST /activity/evaluate/writing` — AI writing evaluation → `FeedbackReport`
- `POST /activity/evaluate/speaking` — AI speaking evaluation → `FeedbackReport`
- `POST /activity/evaluate/scenario` — In-character scenario turn
- `POST /activity/evaluate/scenario/end` — Full scenario feedback
- `POST /tts` — Text → speech (base64 MP3)
- `POST /stt` — Audio → text
- `POST /scenario/start` — Generate scenario + AI's first message with TTS
- `POST /scenario/respond` — User turn → AI response + TTS

## Project Structure

```
fastapi-service/
├── main.py              # FastAPI app + middleware + router mounting
├── config.py            # Settings (pydantic-settings)
├── api/
│   ├── routes/comm.py   # All CommByAI endpoints
│   └── dependencies.py  # JWT verification, Redis client
├── services/
│   ├── comm_agent.py    # LLM agent (OpenAI) for activity gen + evaluation
│   └── audio_processor.py  # Cartesia STT + AWS Polly TTS
└── schemas/comm.py      # Pydantic request/response models
```

## AWS Polly IAM

Your AWS user/role needs `polly:SynthesizeSpeech` on `*`.

## License

Proprietary — CommByAI
