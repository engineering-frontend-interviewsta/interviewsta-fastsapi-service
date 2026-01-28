# SSE Troubleshooting Guide

## Issue: "Session not found" Error

### What Was the Problem?

The frontend was connecting to the SSE stream immediately after receiving the `session_id` from `/start`, but the Celery worker hadn't finished creating the session in Redis yet, causing a race condition.

### Timeline of Events

```
Time 0ms:   POST /start → Returns session_id immediately
Time 1ms:   Frontend connects to SSE stream
Time 2ms:   SSE checks Redis → Session not found ❌
Time 500ms: Celery worker creates session in Redis
```

### The Solution

The SSE endpoint now **waits up to 10 seconds** for the session to appear in Redis before giving up:

```python
# Wait for session to be created (with timeout)
session = None
max_wait = 10  # Wait up to 10 seconds
wait_interval = 0.5  # Check every 500ms

while waited < max_wait:
    session = session_manager.get_session(session_id)
    if session:
        break
    await asyncio.sleep(wait_interval)
    waited += wait_interval
```

Similarly, the `/status` endpoint waits up to 5 seconds.

## Proper Frontend Flow

### Option 1: SSE Streaming (Recommended)

```javascript
const startInterview = async (interviewType, payload) => {
  const token = await auth.currentUser.getIdToken();
  
  // 1. Start the interview
  const response = await axios.post(
    `${API_BASE_URL}/api/v1/interview/start`,
    {
      interview_type: interviewType,
      ...payload
    },
    {
      headers: { 'Authorization': `Bearer ${token}` }
    }
  );
  
  const { session_id, task_id } = response.data;
  
  // 2. Immediately connect to SSE stream
  // The backend will wait for the session to be created
  const stream = await connectToInterviewStream(session_id);
  
  return { session_id, stream };
};

const connectToInterviewStream = async (sessionId) => {
  const user = auth.currentUser;
  const token = await user.getIdToken();
  
  const url = `${API_BASE_URL}/api/v1/interview/${sessionId}/stream?token=${encodeURIComponent(token)}`;
  const eventSource = new EventSource(url);

  // Listen for greeting
  eventSource.addEventListener('ai_response', (event) => {
    const data = JSON.parse(event.data);
    console.log('AI:', data.message);
    
    // Play audio
    if (data.audio) {
      const audio = new Audio(`data:audio/mp3;base64,${data.audio}`);
      audio.play();
    }
  });

  // Listen for transcriptions
  eventSource.addEventListener('transcription', (event) => {
    const data = JSON.parse(event.data);
    console.log('You said:', data.transcription);
  });

  // Listen for completion
  eventSource.addEventListener('complete', (event) => {
    console.log('Interview completed!');
    eventSource.close();
  });

  // Handle errors
  eventSource.addEventListener('error', (event) => {
    try {
      const data = JSON.parse(event.data);
      console.error('SSE Error:', data.error);
    } catch {
      console.error('SSE Connection error');
    }
  });

  return {
    close: () => eventSource.close()
  };
};
```

### Option 2: Polling (Fallback)

```javascript
const pollInterviewStatus = async (sessionId) => {
  const token = await auth.currentUser.getIdToken();
  
  const poll = async () => {
    try {
      const response = await axios.get(
        `${API_BASE_URL}/api/v1/interview/${sessionId}/status`,
        {
          headers: { 'Authorization': `Bearer ${token}` }
        }
      );
      
      const { status, message, audio } = response.data;
      
      if (status === 'ai_responded') {
        console.log('AI:', message);
        
        if (audio) {
          const audioElement = new Audio(`data:audio/mp3;base64,${audio}`);
          audioElement.play();
        }
        
        return true; // Stop polling
      }
      
      return false; // Continue polling
    } catch (error) {
      if (error.response?.status === 404) {
        console.log('Session not yet created, retrying...');
        return false; // Continue polling
      }
      throw error;
    }
  };
  
  // Poll every second for up to 30 seconds
  const maxAttempts = 30;
  for (let i = 0; i < maxAttempts; i++) {
    const done = await poll();
    if (done) break;
    await new Promise(resolve => setTimeout(resolve, 1000));
  }
};
```

## Debugging Tips

### 1. Check Backend Logs

Look for these log messages in sequence:

```
[interview worker] Starting interview session_id for user user_id
[interview worker] Creating session session_id in Redis for user user_id
[interview worker] Session session_id created and marked as processing
[fastapi] SSE stream connecting for session session_id
[fastapi] SSE: Session session_id found after 0.0s
[fastapi] SSE: Stream established for session session_id
```

### 2. Check Redis

```bash
# Connect to Redis
redis-cli

# Check if session exists
GET session:YOUR_SESSION_ID

# Check session status
GET session:YOUR_SESSION_ID:status
```

### 3. Frontend Console

Add detailed logging:

```javascript
console.log('[SSE] Connecting to:', url);
console.log('[SSE] ReadyState:', eventSource.readyState);

eventSource.onopen = () => {
  console.log('[SSE] Connection opened');
};

eventSource.onerror = (error) => {
  console.error('[SSE] Connection error:', error);
  console.log('[SSE] ReadyState:', eventSource.readyState);
  console.log('[SSE] URL:', url);
};
```

### 4. Network Tab

Check the SSE connection in browser DevTools:
- Go to Network tab
- Find the `/stream` request
- Check if it's stuck in "pending" (good - streaming)
- Check the response headers: `Content-Type: text/event-stream`
- View the EventStream tab to see incoming events

## Common Issues

### Issue: SSE times out after 10 seconds

**Cause**: Celery worker is slow or stuck

**Solution**:
1. Check Celery worker logs for errors
2. Ensure worker has enough resources
3. Check if LangGraph workflow is stuck
4. Verify API keys (Google, Tavily) are valid

### Issue: SSE connects but no events received

**Cause**: Interview processing is stuck

**Solution**:
1. Check Celery worker logs
2. Verify workflow is progressing: `GET /status`
3. Check for LLM API rate limits
4. Look for errors in LangGraph execution

### Issue: "Unauthorized" error

**Cause**: Firebase token is invalid or expired

**Solution**:
1. Get a fresh token: `await user.getIdToken(true)`
2. Ensure token is properly URL-encoded in query param
3. Check Firebase authentication is working

### Issue: CORS error

**Cause**: Frontend domain not in CORS allowed origins

**Solution**:
Add to `config.py`:
```python
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "https://your-frontend-domain.com"
]
```

## Performance Optimization

### For Production

1. **Use SSE, not polling** - More efficient, real-time updates
2. **Connection pooling** - Reuse Redis connections
3. **CDN for audio** - Store generated audio in S3/CloudFront
4. **Cache responses** - Cache identical prompts
5. **Horizontal scaling** - Multiple Celery workers

### Monitoring

Track these metrics:
- SSE connection time (should be < 1s)
- Session creation time (should be < 500ms)
- Interview processing time (varies by workflow)
- Audio synthesis time (should be < 2s)

## Testing

### Test Session Creation

```bash
curl -X POST http://localhost:8001/api/v1/interview/start \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "interview_type": "Subject",
    "subject": "Python",
    "difficulty": "Easy"
  }'
```

### Test SSE Connection

```bash
curl -N "http://localhost:8001/api/v1/interview/SESSION_ID/stream?token=YOUR_TOKEN"
```

You should see:
```
event: status
data: {"status": "processing"}

event: ai_response
data: {"message": "...", "audio": "..."}
```
