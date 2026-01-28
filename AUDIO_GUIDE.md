# Audio Processing Guide

## Overview
The interview system uses:
- **ElevenLabs STT** (Speech-to-Text) for transcribing user audio
- **AWS Polly TTS** (Text-to-Speech) for generating AI responses

## User Speech → Text (ElevenLabs STT)

### How It Works

1. **Frontend captures user audio** (microphone recording)
2. **Convert to WAV format and base64 encode**
3. **Send to `/api/v1/interview/{session_id}/respond` endpoint**
4. **Backend automatically transcribes** using ElevenLabs
5. **Transcription is processed** by LangGraph interview workflow

### Frontend Implementation

```javascript
// 1. Record audio from microphone
const recordAudio = async () => {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const mediaRecorder = new MediaRecorder(stream);
  const audioChunks = [];

  mediaRecorder.addEventListener("dataavailable", event => {
    audioChunks.push(event.data);
  });

  mediaRecorder.addEventListener("stop", async () => {
    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
    
    // 2. Convert to base64
    const base64Audio = await blobToBase64(audioBlob);
    
    // 3. Send to backend
    await submitAudioResponse(sessionId, base64Audio);
  });

  mediaRecorder.start();
  
  // Stop after user finishes speaking
  setTimeout(() => mediaRecorder.stop(), 5000);
};

// Helper: Convert Blob to base64
const blobToBase64 = (blob) => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      // Remove data:audio/wav;base64, prefix
      const base64 = reader.result.split(',')[1];
      resolve(base64);
    };
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
};

// Submit audio to interview API
const submitAudioResponse = async (sessionId, audioBase64) => {
  const token = await auth.currentUser.getIdToken();
  
  const response = await axios.post(
    `${API_BASE_URL}/api/v1/interview/${sessionId}/respond`,
    {
      audio_data: audioBase64,  // Base64 WAV audio
      // text_response: null (don't send if using audio)
    },
    {
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
      }
    }
  );
  
  // Returns task_id for polling
  const { task_id } = response.data;
  
  // Poll for result
  pollInterviewStatus(sessionId, task_id);
};
```

### Backend Flow

```
User Audio (Base64 WAV)
    ↓
POST /api/v1/interview/{session_id}/respond
    ↓
Celery Task: transcribe_audio (audio queue)
    ↓
ElevenLabs STT API
    ↓
Transcribed Text
    ↓
Celery Task: process_user_response (interview queue)
    ↓
LangGraph Interview Workflow
    ↓
AI Response + Audio (Polly TTS)
```

## AI Response → Speech (AWS Polly TTS)

### How It Works

1. **AI generates text response** via LangGraph
2. **Backend automatically synthesizes audio** using AWS Polly
3. **Audio returned as base64 MP3** in response
4. **Frontend plays the audio**

### Frontend Implementation

```javascript
// Poll for interview status and get AI response
const pollInterviewStatus = async (sessionId, taskId) => {
  const token = await auth.currentUser.getIdToken();
  
  const checkStatus = async () => {
    const response = await axios.get(
      `${API_BASE_URL}/api/v1/interview/${sessionId}/status`,
      {
        params: { task_id: taskId },
        headers: { 'Authorization': `Bearer ${token}` }
      }
    );
    
    const { state, message, audio, last_node } = response.data;
    
    if (state === 'SUCCESS') {
      // Display AI message
      console.log('AI:', message);
      
      // Play audio response
      if (audio) {
        playAudio(audio);
      }
      
      return true; // Done
    } else if (state === 'FAILURE') {
      console.error('Interview task failed');
      return true; // Stop polling
    }
    
    // Continue polling
    return false;
  };
  
  // Poll every 1 second
  const interval = setInterval(async () => {
    const done = await checkStatus();
    if (done) {
      clearInterval(interval);
    }
  }, 1000);
};

// Play base64 MP3 audio
const playAudio = (audioBase64) => {
  const audio = new Audio(`data:audio/mp3;base64,${audioBase64}`);
  audio.play();
};
```

### Using SSE (Server-Sent Events) - Recommended

```javascript
const connectToInterviewStream = async (sessionId) => {
  const user = auth.currentUser;
  const token = await user.getIdToken();
  
  // Token as query parameter (EventSource doesn't support headers)
  const url = `${API_BASE_URL}/api/v1/interview/${sessionId}/stream?token=${encodeURIComponent(token)}`;
  const eventSource = new EventSource(url);

  eventSource.addEventListener('ai_response', (event) => {
    const data = JSON.parse(event.data);
    
    // Display AI message
    console.log('AI:', data.message);
    
    // Play audio if available
    if (data.audio) {
      const audio = new Audio(`data:audio/mp3;base64,${data.audio}`);
      audio.play();
    }
  });

  eventSource.addEventListener('transcription', (event) => {
    const data = JSON.parse(event.data);
    console.log('User transcription:', data.transcription);
  });

  eventSource.onerror = (error) => {
    console.error('SSE Error:', error);
    eventSource.close();
  };

  return {
    close: () => eventSource.close()
  };
};
```

## Environment Variables Required

```bash
# ElevenLabs (Speech-to-Text)
ELEVENLABS_API_KEY=your_elevenlabs_api_key

# AWS Polly (Text-to-Speech)
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
AWS_REGION=ap-south-1
AWS_POLLY_VOICE_ID=Joanna
AWS_POLLY_ENGINE=neural
AWS_POLLY_SPEECH_RATE=85%
```

## Audio Format Requirements

### User Audio (Input)
- **Format**: WAV
- **Encoding**: Base64
- **Sample Rate**: 16kHz or higher recommended
- **Channels**: Mono preferred
- **Max Size**: No hard limit, but keep under 10MB for performance

### AI Audio (Output)
- **Format**: MP3
- **Encoding**: Base64
- **Quality**: Neural engine (high quality)
- **Speech Rate**: Configurable (default 85% for slower, clearer speech)

## Troubleshooting

### "No transcription detected"
- Ensure audio is in WAV format
- Check audio file is not empty/silent
- Verify ELEVENLABS_API_KEY is set correctly

### "Audio synthesis failed"
- Check AWS credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
- Verify AWS_REGION is valid
- Check Polly service limits/quota

### Audio not playing in frontend
- Ensure `audio` field in response is not null
- Check browser console for playback errors
- Verify base64 audio is being decoded correctly

## Testing

### Test Transcription (CLI)
```python
from services.audio_processor import AudioProcessor
import base64

processor = AudioProcessor(
    elevenlabs_api_key="your_key"
)

with open("test_audio.wav", "rb") as f:
    audio_base64 = base64.b64encode(f.read()).decode()

text = processor.transcribe_audio(audio_base64)
print(f"Transcription: {text}")
```

### Test TTS (CLI)
```python
from services.audio_processor import AudioProcessor
import base64

processor = AudioProcessor(
    aws_access_key_id="your_key",
    aws_secret_access_key="your_secret",
    aws_region="ap-south-1"
)

audio_base64 = processor.synthesize_speech_base64(
    "Hello, this is a test of AWS Polly text to speech."
)

# Save to file
audio_bytes = base64.b64decode(audio_base64)
with open("output.mp3", "wb") as f:
    f.write(audio_bytes)
```

## Performance Notes

- **Transcription latency**: ~1-3 seconds for typical responses
- **TTS latency**: ~0.5-2 seconds depending on text length
- **Concurrent processing**: Audio and interview tasks run in separate queues
- **Celery workers**: 
  - Audio queue: 2 workers recommended
  - Interview queue: 4 workers recommended
