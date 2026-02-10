# FastAPI Interview Microservice

High-performance microservice for AI-powered interview sessions, built with FastAPI, Celery, and Redis.

## Architecture

### Audio Processing
- **Speech-to-Text (STT)**: Cartesia API (ink-whisper model) for transcription
- **Text-to-Speech (TTS)**: AWS Polly for speech synthesis
  - Supports neural and standard voices
  - Configurable speech rate (SSML)
  - Lower latency and better pricing than ElevenLabs

### Task Queuing
Celery workers with dedicated queues:
- `interview` - LLM-based interview processing (4 workers)
- `audio` - STT/TTS operations (2 workers)
- `resume` - Resume analysis with OCR (2 workers)
- `feedback` - Interview feedback generation (2 workers)

### State Management
- Redis for session state, conversation history, and caching
- LangGraph with Redis checkpoint for workflow persistence
- Firebase for authentication

## Environment Variables

### Required
```bash
# AI/ML
GOOGLE_API_KEY=your_key          # Gemini LLM
TAVILY_API_KEY=your_key          # Web search

# Audio - STT (Cartesia)
CARTESIA_API_KEY=your_key         # Speech-to-Text only
CARTESIA_MODEL=ink-whisper       # Optional, defaults to ink-whisper
CARTESIA_API_VERSION=2025-04-16  # Optional, defaults to 2025-04-16

# Audio - TTS (AWS Polly)
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_key
AWS_REGION=us-east-1
AWS_POLLY_VOICE_ID=Joanna        # Voice selection
AWS_POLLY_ENGINE=neural          # neural or standard
AWS_POLLY_SPEECH_RATE=75%        # 20%-200% or slow/medium/fast

# Database
REDIS_URL=redis://redis:6379
DATABASE_URL=postgresql://user:pass@host:5432/db

# Auth
FIREBASE_CREDENTIALS_JSON=base64_encoded_json
```

### Optional
```bash
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/0
```

## Quick Start

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your credentials

# Run FastAPI server
uvicorn main:app --reload --port 8001

# Run Celery workers (separate terminals)
celery -A tasks.celery_app worker -Q interview --loglevel=info --concurrency=4
celery -A tasks.celery_app worker -Q audio --loglevel=info --concurrency=2
celery -A tasks.celery_app worker -Q resume --loglevel=info --concurrency=2
celery -A tasks.celery_app worker -Q feedback --loglevel=info --concurrency=2

# Run Celery Beat (scheduler)
celery -A tasks.celery_app beat --loglevel=info
```

### Docker
```bash
# From project root
docker-compose up -d fastapi celery-worker-interview celery-worker-audio
```

## API Endpoints

### Interview
- `POST /api/v1/interview/start` - Start interview session
- `POST /api/v1/interview/{session_id}/respond` - Submit user response
- `GET /api/v1/interview/{session_id}/status` - Poll status
- `GET /api/v1/interview/{session_id}/stream` - SSE stream (real-time)
- `DELETE /api/v1/interview/{session_id}` - End session

### Resume Analysis
- `POST /api/v1/resume/analyze` - Upload resume + job description
- `GET /api/v1/resume/{task_id}/status` - Check analysis status

### Feedback
- `POST /api/v1/feedback/generate` - Generate interview feedback
- `GET /api/v1/feedback/{task_id}/status` - Check feedback status

## AWS Polly Configuration

### Available Voices
See [AWS Polly Voice List](https://docs.aws.amazon.com/polly/latest/dg/voicelist.html)

**Popular Neural Voices** (recommended):
- **English (US)**: Joanna (female), Matthew (male), Ivy (female, child)
- **English (GB)**: Amy (female), Brian (male), Emma (female)
- **English (AU)**: Olivia (female)

**Standard Voices** (lower cost):
- Joanna, Matthew, Kendra, Justin, Salli, Joey

### Speech Rate Control
Configure via `AWS_POLLY_SPEECH_RATE`:
- **Percentage**: `50%` to `200%` (e.g., `75%` = slower, `150%` = faster)
- **Keywords**: `x-slow`, `slow`, `medium`, `fast`, `x-fast`

Default: `75%` (slower speech for clarity)

### Engine Selection
- **`neural`** (recommended): Higher quality, more natural, supports fewer voices
- **`standard`**: Lower cost, more voices available

### IAM Permissions
Your AWS user/role needs:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "polly:SynthesizeSpeech"
      ],
      "Resource": "*"
    }
  ]
}
```

## Testing Audio

### Test TTS (AWS Polly)
```python
from services.audio_processor import AudioProcessor
import os

processor = AudioProcessor(
    cartesia_api_key=os.getenv("CARTESIA_API_KEY"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    aws_region="us-east-1",
    polly_voice_id="Joanna",
    polly_speech_rate="75%"
)

# Generate speech
audio_bytes = processor.synthesize_speech("Hello, this is a test.")
with open("test.mp3", "wb") as f:
    f.write(audio_bytes)
```

### Test STT (Cartesia)
```python
import base64

# Read audio file
with open("audio.wav", "rb") as f:
    audio_base64 = base64.b64encode(f.read()).decode()

# Transcribe
text = processor.transcribe_audio(audio_base64)
print(text)
```

## Monitoring

### Celery Tasks
```bash
# View active tasks
celery -A tasks.celery_app inspect active

# View registered tasks
celery -A tasks.celery_app inspect registered

# Purge queue
celery -A tasks.celery_app purge
```

### Flower UI
```bash
# Start Flower
celery -A tasks.celery_app flower --port=5555

# Open browser
http://localhost:5555
```

## Project Structure
```
interview_service/
├── main.py                 # FastAPI application
├── config.py              # Settings & environment
├── api/
│   ├── routes/
│   │   ├── interview.py   # Interview endpoints
│   │   ├── resume.py      # Resume analysis endpoints
│   │   └── feedback.py    # Feedback endpoints
│   └── dependencies.py    # Firebase auth, Redis
├── tasks/
│   ├── celery_app.py      # Celery configuration
│   ├── interview_tasks.py # Interview processing
│   ├── audio_tasks.py     # STT/TTS (Cartesia + Polly)
│   ├── resume_tasks.py    # Resume OCR + analysis
│   └── feedback_tasks.py  # Feedback generation
├── services/
│   ├── audio_processor.py # Cartesia STT + AWS Polly TTS
│   └── interview_session.py # Redis session manager
├── workflows/
│   ├── technical.py       # Technical interview graph
│   ├── hr.py             # HR interview graph
│   ├── coding.py         # Coding interview graph
│   ├── case_study.py     # Case study interview graph
│   └── feedback/
│       ├── technical_feedback.py
│       ├── hr_feedback.py
│       └── case_study_feedback.py
└── schemas/
    ├── interview.py       # Pydantic models
    ├── resume.py
    └── feedback.py
```

## Performance Tuning

### Worker Scaling
```bash
# Scale workers based on load
docker-compose up -d --scale celery-worker-interview=8
docker-compose up -d --scale celery-worker-audio=4
```

### AWS Polly Optimization
- Use **standard engine** for cost savings (set `AWS_POLLY_ENGINE=standard`)
- Cache common responses in Redis
- Batch similar requests

### Redis Memory
- Monitor with: `redis-cli --stat`
- Set maxmemory: `redis-server --maxmemory 2gb`

## Troubleshooting

### AWS Polly Errors

**Error: "Access Denied"**
- Check IAM permissions (`polly:SynthesizeSpeech`)
- Verify AWS credentials are correct

**Error: "Invalid voice"**
- Check voice is available for selected engine (neural vs standard)
- List voices: `aws polly describe-voices --engine neural`

**Error: "Text too long"**
- Neural: 6000 char limit
- Standard: 3000 char limit
- Text is auto-truncated

### Cartesia STT Errors

**Error: "Invalid API key"**
- Verify `CARTESIA_API_KEY` is set correctly
- Get your API key from https://play.cartesia.ai/keys

**Error: "Audio file too large"**
- Cartesia supports various formats (WAV, MP3, etc.)
- Consider chunking very long audio files

**Error: "Request timeout"**
- Check network connectivity
- Verify Cartesia API is accessible

## Migration from ElevenLabs STT

If you previously used ElevenLabs for STT:

1. **Get Cartesia API key** - Sign up at https://play.cartesia.ai
2. **Replace `ELEVENLABS_API_KEY`** with `CARTESIA_API_KEY` in your environment
3. **Update env vars** - See `.env.example`
4. **Rebuild containers** - `docker-compose up -d --build`

Benefits of AWS Polly:
- Lower cost for high volume
- Lower latency
- More voice options
- No rate limiting issues
- SSML support for fine control

## Documentation

- FastAPI Docs: http://localhost:8001/docs
- Migration Guide: See `../MIGRATION_GUIDE.md`
- Docker Setup: See `../DOCKER_SETUP.md`

## License

Proprietary - InterviewSta
