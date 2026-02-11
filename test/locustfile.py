"""
Locust load test: resume analysis and start interview.
Each flow submits, polls status, and marks success/failure from the actual task outcome.
"""
import json
import os
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
                if last_status == "completed" and status_body.get("interview_ai_response") is None:
                    resp.failure("status=completed but interview_ai_response is null")
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

