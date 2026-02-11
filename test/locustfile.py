"""
Locust load test: resume analysis, start interview, and full interview (start + 4 respond cycles).
Each flow submits, polls status, and marks success/failure from the actual task outcome.
"""
import base64
import json
import os
import pickle
import time
import uuid
from locust import HttpUser, task, between

# Long-lived Firebase ID token for auth (replace or set via env in production)
FIREBASE_ID_TOKEN = os.getenv(
    "FIREBASE_ID_TOKEN",
    "eyJhbGciOiJSUzI1NiIsImtpZCI6IjRiMTFjYjdhYjVmY2JlNDFlOTQ4MDk0ZTlkZjRjNWI1ZWNhMDAwOWUiLCJ0eXAiOiJKV1QifQ.eyJuYW1lIjoiQXJyeXVhbm4gS2hhbm5hIiwicGljdHVyZSI6Imh0dHBzOi8vbGgzLmdvb2dsZXVzZXJjb250ZW50LmNvbS9hL0FDZzhvY0tGVkNTX0NhbXV3LXVIQ1ZxQ0V2N0huT1hOa3RfcHR3My1OTWduNWp5MVdxSzRyTEFRPXM5Ni1jIiwiaXNzIjoiaHR0cHM6Ly9zZWN1cmV0b2tlbi5nb29nbGUuY29tL2ludGVydmlzdGEtNzE1MjYiLCJhdWQiOiJpbnRlcnZpc3RhLTcxNTI2IiwiYXV0aF90aW1lIjoxNzcwNzE3MzM1LCJ1c2VyX2lkIjoiMkNCMlBKYTFxS2JSSkhCZ3lNOVdEU2N6M1g0MyIsInN1YiI6IjJDQjJQSmExcUtiUkpIQmd5TTlXRFNjejNYNDMiLCJpYXQiOjE3NzA3MTczMzUsImV4cCI6MTc3MDcyMDkzNSwiZW1haWwiOiJhcnlhbmtoYW5uYWNoZEBnbWFpbC5jb20iLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiZmlyZWJhc2UiOnsiaWRlbnRpdGllcyI6eyJnb29nbGUuY29tIjpbIjExMTQxMzc5NjE3NTkwNDkzOTA4NiJdLCJlbWFpbCI6WyJhcnlhbmtoYW5uYWNoZEBnbWFpbC5jb20iXX0sInNpZ25faW5fcHJvdmlkZXIiOiJnb29nbGUuY29tIn19.btmGj0C4IEn7Xes8EMKNJnSU_QsdW9V3qK7L8ho0-V2-uhwOuRTYE7Vst6FRVNjyrSU81R3LAv8O2P5cbVeVIn8fboTCauXlAWwN6ZFKZc-fLgwnWmliDmgtdjnKcV9VMfp46e_W9QCAd0-EtUHPD5WCjIhWD7sAxGUD_IGc2waG_2O7NxD3u0gjOsTJgSgMwyK_Fu_9Wl14Qq_w9wlH2V4JC3-sHdQ2z5JpmaDlv5zAWBaPyRX6TIZKH2VBaIl2AIC6_8ktJ-gJEpSCI_vuo6CYO8LF4KYeSiYgduNLZjsAsbexMUaWUU0Cvw1nletFGoDz61ZkbfiL1YPmnooUmg",
)

# PDF next to repo root (parent of test/)
PDF_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ArryuannKhannaFinalLatestResume.pdf")

# Status polling
POLL_INTERVAL_SEC = 2
MAX_POLL_WAIT_SEC = 120

# Start interview: user_id must match the Firebase token's uid
INTERVIEW_USER_ID = os.getenv("INTERVIEW_USER_ID", "2CB2PJa1qKbRJHBygM9WDScz3X43")

# Full interview: audio.pkl is a dict with keys 0,1,2,3 (4 interactions); values are audio bytes or base64 str
AUDIO_PKL_PATH = os.getenv("AUDIO_PKL_PATH", os.path.join(os.path.dirname(__file__), "audio.pkl"))
RESPOND_POLL_WAIT_SEC = 90  # per respond (transcribe + workflow)


class ResumeAnalysisUser(HttpUser):
    wait_time = between(2, 5)
    host = "https://interview-service-api.onrender.com"

    def on_start(self):
        self.client.headers.update({
            "Authorization": f"Bearer {FIREBASE_ID_TOKEN}",
        })

    @task(1)
    def upload_resume_and_verify_status(self):
        """POST analyze, then poll GET /{task_id}/status until completed/failed; mark failure from outcome."""
        if not os.path.exists(PDF_PATH):
            # Skip task so failure counts reflect only real API/status outcomes
            return

        with open(PDF_PATH, "rb") as pdf:
            files = {
                "resume": ("resume.pdf", pdf, "application/pdf"),
                "job_description": (
                    "job_description.txt",
                    "Fullstack Engineer with experience in Django and LangGraph.",
                    "text/plain",
                ),
            }
            data = {
                "resume_name": "Arryuann's Resume",
                "session_id": "locust_test_session_999",
            }

            with self.client.post(
                "/api/v1/resume/analyze",
                files=files,
                data=data,
                name="POST /resume/analyze",
                catch_response=True,
            ) as response:
                if response.status_code != 200:
                    response.failure(f"Analyze returned {response.status_code}: {response.text}")
                    return
                try:
                    body = response.json()
                except Exception as e:
                    response.failure(f"Invalid JSON: {e}")
                    return
                task_id = body.get("task_id")
                if not task_id:
                    response.failure("Response missing task_id")
                    return

        # Poll status until completed or failed
        status_url = f"/api/v1/resume/{task_id}/status"
        deadline = time.time() + MAX_POLL_WAIT_SEC
        last_status = None

        while time.time() < deadline:
            with self.client.get(status_url, name="GET /resume/{task_id}/status", catch_response=True) as resp:
                if resp.status_code != 200:
                    resp.failure(f"Status returned {resp.status_code}: {resp.text}")
                    return
                try:
                    status_body = resp.json()
                except Exception as e:
                    resp.failure(f"Status invalid JSON: {e}")
                    return

                last_status = status_body.get("status")
                if last_status == "completed":
                    result = status_body.get("result")
                    if result is None:
                        resp.failure("status=completed but result is null")
                    return
                if last_status == "failed":
                    error = status_body.get("error") or "Unknown error"
                    resp.failure(f"Task failed: {error}")
                    return

            time.sleep(POLL_INTERVAL_SEC)

        # Timeout: mark this user's flow as failure
        with self.client.get(status_url, name="GET /resume/{task_id}/status (timeout)", catch_response=True) as resp:
            resp.failure(f"Timeout waiting for completion (last status={last_status})")

    @task(1)
    def start_interview_and_verify_status(self):
        """POST /interview/start, then poll GET /interview/start-status/{task_id} until completed/failed."""
        session_id = f"locust_start_{uuid.uuid4().hex[:12]}"
        payload = {
            "interview_type": "Technical",
            "session_id": session_id,
            "user_id": INTERVIEW_USER_ID,
            "payload": {
                "resume": "Locust test candidate with Python and FastAPI experience.",
                "TechnicalResearch": "Backend services, REST APIs",
                "CodingResearch": "Python, LangGraph",
            },
        }

        with self.client.post(
            "/api/v1/interview/start",
            json=payload,
            name="POST /interview/start",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"Start returned {response.status_code}: {response.text}")
                return
            try:
                body = response.json()
            except Exception as e:
                response.failure(f"Invalid JSON: {e}")
                return
            task_id = body.get("task_id")
            if not task_id:
                response.failure("Response missing task_id")
                return

        status_url = f"/api/v1/interview/start-status/{task_id}"
        deadline = time.time() + MAX_POLL_WAIT_SEC
        last_status = None

        while time.time() < deadline:
            with self.client.get(
                status_url,
                name="GET /interview/start-status/{task_id}",
                catch_response=True,
            ) as resp:
                if resp.status_code != 200:
                    resp.failure(f"Status returned {resp.status_code}: {resp.text}")
                    return
                try:
                    status_body = resp.json()
                except Exception as e:
                    resp.failure(f"Status invalid JSON: {e}")
                    return

                last_status = status_body.get("status")
                if last_status == "completed":
                    print("[Start interview] Completed response body:", json.dumps(status_body, indent=2, default=str))
                    return
                if last_status == "failed":
                    err = status_body.get("error") or "Unknown error"
                    resp.failure(f"Start interview task failed: {err}")
                    return

            time.sleep(POLL_INTERVAL_SEC)

        with self.client.get(
            status_url,
            name="GET /interview/start-status/{task_id} (timeout)",
            catch_response=True,
        ) as resp:
            resp.failure(f"Timeout waiting for start completion (last status={last_status})")


def _load_audio_pkl(path: str):
    """Load audio.pkl; return dict of key -> base64 audio str, or None if missing/invalid."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    required = {0, 1, 2, 3}
    if not required.issubset(data.keys()):
        return None
    out = {}
    for k in required:
        v = data[k]
        if isinstance(v, bytes):
            out[k] = base64.b64encode(v).decode("utf-8")
        elif isinstance(v, str):
            out[k] = v
        else:
            return None
    return out


class FullInterviewUser(HttpUser):
    """
    Simulate a full interview: start session, then send 4 audio responses (0,1,2,3 from audio.pkl)
    and poll respond-status after each until completed.
    """
    wait_time = between(1, 3)
    host = "https://interview-service-api.onrender.com"

    def on_start(self):
        self.client.headers.update({"Authorization": f"Bearer {FIREBASE_ID_TOKEN}"})
        self.audio_clips = _load_audio_pkl(AUDIO_PKL_PATH)

    @task(1)
    def run_full_interview(self):
        """Start interview, then submit audio 0,1,2,3 in order; poll respond-status after each."""
        if not self.audio_clips:
            return  # skip if audio.pkl missing or invalid

        session_id = f"locust_full_{uuid.uuid4().hex[:12]}"
        start_payload = {
            "interview_type": "Technical",
            "session_id": session_id,
            "user_id": INTERVIEW_USER_ID,
            "payload": {
                "resume": "Locust full-interview candidate.",
                "TechnicalResearch": "Backend, REST APIs",
                "CodingResearch": "Python, LangGraph",
            },
        }

        # 1) Start interview
        with self.client.post(
            "/api/v1/interview/start",
            json=start_payload,
            name="POST /interview/start (full)",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"Start returned {response.status_code}: {response.text}")
                return
            try:
                body = response.json()
            except Exception as e:
                response.failure(f"Invalid JSON: {e}")
                return
            task_id = body.get("task_id")
            if not task_id:
                response.failure("Response missing task_id")
                return

        # Poll start-status until completed
        status_url = f"/api/v1/interview/start-status/{task_id}"
        deadline = time.time() + MAX_POLL_WAIT_SEC
        while time.time() < deadline:
            with self.client.get(
                status_url,
                name="GET /interview/start-status/{task_id} (full)",
                catch_response=True,
            ) as resp:
                if resp.status_code != 200:
                    resp.failure(f"Start status {resp.status_code}: {resp.text}")
                    return
                try:
                    status_body = resp.json()
                except Exception as e:
                    resp.failure(f"Start status invalid JSON: {e}")
                    return
                if status_body.get("status") == "completed":
                    break
                if status_body.get("status") == "failed":
                    resp.failure(f"Start failed: {status_body.get('error', 'Unknown')}")
                    return
            time.sleep(POLL_INTERVAL_SEC)
        else:
            with self.client.get(status_url, name="GET /interview/start-status (timeout)", catch_response=True) as r:
                r.failure("Timeout waiting for start")
            return

        # 2) Submit each of 4 audio responses and poll respond-status
        for i in range(4):
            audio_b64 = self.audio_clips.get(i)
            if not audio_b64:
                continue
            with self.client.post(
                f"/api/v1/interview/{session_id}/respond",
                json={"audio_data": audio_b64},
                name=f"POST /interview/{{session_id}}/respond (i={i})",
                catch_response=True,
            ) as response:
                if response.status_code != 200:
                    response.failure(f"Respond {i} returned {response.status_code}: {response.text}")
                    return
                try:
                    body = response.json()
                except Exception as e:
                    response.failure(f"Respond {i} invalid JSON: {e}")
                    return
                respond_task_id = body.get("task_id")
                if not respond_task_id:
                    response.failure(f"Respond {i} missing task_id")
                    return

            respond_status_url = f"/api/v1/interview/{session_id}/respond-status/{respond_task_id}"
            deadline_respond = time.time() + RESPOND_POLL_WAIT_SEC
            while time.time() < deadline_respond:
                with self.client.get(
                    respond_status_url,
                    name=f"GET /interview/{{session_id}}/respond-status (i={i})",
                    catch_response=True,
                ) as resp:
                    if resp.status_code != 200:
                        resp.failure(f"Respond status {i} returned {resp.status_code}: {resp.text}")
                        return
                    try:
                        status_body = resp.json()
                    except Exception as e:
                        resp.failure(f"Respond status {i} invalid JSON: {e}")
                        return
                    if status_body.get("status") == "completed":
                        break
                    if status_body.get("status") == "failed":
                        resp.failure(f"Respond {i} failed: {status_body.get('error', 'Unknown')}")
                        return
                time.sleep(POLL_INTERVAL_SEC)
            else:
                with self.client.get(
                    respond_status_url,
                    name="GET /respond-status (timeout)",
                    catch_response=True,
                ) as r:
                    r.failure(f"Timeout waiting for respond {i}")
                return

