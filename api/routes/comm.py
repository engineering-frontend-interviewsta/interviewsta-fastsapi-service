"""
CommByAI API endpoints — AI-only (LLM-powered).

Architecture (same as Interviewsta):
- NestJS backend (interviewsta-backend): auth, user management, CRUD, progress/XP,
  badges, leaderboard, dashboard, skill-tree data, onboarding persistence.
- FastAPI service (this file): AI-powered activity generation, writing/speaking
  evaluation, scenario simulation, TTS/STT — anything that calls the LLM or audio services.

All endpoints require JWT authentication via get_current_user dependency.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Optional
import logging
import asyncio
import uuid

from api.dependencies import get_current_user
from services.comm_agent import get_comm_agent
from services.audio_processor import AudioProcessor
from schemas.comm import (
    GenerateActivityRequest,
    OnboardingAssessRequest,
    EvaluateWritingRequest,
    EvaluateSpeakingRequest,
    ScenarioTurnRequest,
    ScenarioEndRequest,
    FeedbackReport,
    PlacementResult,
    ScenarioTurnResponse,
    ActivityType,
)
import os
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()


# --- Onboarding (deterministic scoring, but lives here alongside AI service) ---

@router.post("/onboarding/assess", response_model=PlacementResult)
async def assess_onboarding(
    request: OnboardingAssessRequest,
    user_info: Dict = Depends(get_current_user),
):
    """
    Submit onboarding assessment answers and receive placement levels.
    Scoring is deterministic (no LLM call). Results should be persisted
    by the NestJS backend after this call returns.
    """
    try:
        agent = get_comm_agent()
        result = await agent.run_onboarding_assessment(request)
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        logger.error(f"Onboarding assessment error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Assessment failed",
        )


# --- Activity Generation (AI-powered) ---

@router.post("/activity/generate")
async def generate_activity(
    request: GenerateActivityRequest,
    user_info: Dict = Depends(get_current_user),
):
    """Generate a new activity using AI. Returns activity JSON."""
    try:
        agent = get_comm_agent()
        result = await agent.generate_activity(request)
        return result
    except ValueError as e:
        # ValueError from generate_activity means the LLM failed to produce valid output
        # after retries. For activity types we have a curated fallback for, return that
        # so the user always gets a working activity; otherwise we surface 502 (Bad Gateway).
        logger.error(f"Activity generation LLM failure: {e}", exc_info=True)
        fallback = _fallback_for(request)
        if fallback is not None:
            logger.warning(f"Serving {request.activity_type.value} from fallback pool (LLM unavailable)")
            return fallback
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Activity generation failed: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Activity generation error: {e}", exc_info=True)
        fallback = _fallback_for(request)
        if fallback is not None:
            logger.warning(f"Serving {request.activity_type.value} from fallback pool (unexpected error)")
            return fallback
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate activity",
        )


def _fallback_for(request: GenerateActivityRequest) -> Optional[Dict]:
    """Dispatch to the appropriate fallback for the activity type, or None."""
    if request.activity_type == ActivityType.SENTENCE_REORDER:
        return _fallback_sentence_reorder(request)
    if request.activity_type == ActivityType.WORD_LEARN:
        return _fallback_word_learn(request)
    if request.activity_type == ActivityType.READING:
        return _fallback_reading(request)
    return None


# --- Fallback pool (used only when LLM generation fails) ---

_FALLBACK_SENTENCE_POOL = [
    # topic_hint, correct_sentence, words (shuffled)
    ("general", "Simple sentences are easier to read.", ["are", "simple", "to", "read", "sentences", "easier"]),
    ("general", "Clear writing helps your readers understand.", ["clear", "writing", "your", "helps", "understand", "readers"]),
    ("general", "Practice every day to improve your skills.", ["every", "improve", "practice", "day", "to", "your", "skills"]),
    ("communication", "Good listening is an important skill.", ["important", "an", "is", "good", "listening", "skill"]),
    ("communication", "Speak clearly so everyone can understand.", ["clearly", "everyone", "speak", "can", "understand", "so"]),
    ("communication", "Eye contact shows that you are listening.", ["contact", "shows", "that", "are", "eye", "you", "listening"]),
    ("grammar", "A verb expresses an action or state.", ["verb", "a", "expresses", "an", "action", "or", "state"]),
    ("grammar", "Every sentence contains some kind of verb.", ["sentence", "contains", "some", "every", "kind", "of", "verb"]),
    ("grammar", "Adjectives make your writing more descriptive.", ["adjectives", "your", "make", "writing", "more", "descriptive"]),
    ("writing", "Short sentences often have the most impact.", ["short", "often", "have", "the", "most", "impact", "sentences"]),
    ("writing", "Read your draft aloud to find weak spots.", ["aloud", "to", "read", "your", "draft", "find", "weak", "spots"]),
    ("writing", "Strong verbs make your sentences more vivid.", ["strong", "verbs", "your", "sentences", "make", "more", "vivid"]),
    ("speaking", "Pause briefly before an important point.", ["briefly", "an", "pause", "important", "before", "point"]),
    ("speaking", "Breathe slowly to calm your nerves.", ["slowly", "calm", "to", "breathe", "your", "nerves"]),
    ("speaking", "Confidence grows with every practice session.", ["with", "confidence", "every", "grows", "practice", "session"]),
    ("vocabulary", "Choose precise words for clearer meaning.", ["precise", "for", "choose", "clearer", "words", "meaning"]),
    ("vocabulary", "New words enrich your everyday conversation.", ["everyday", "new", "enrich", "your", "words", "conversation"]),
    ("scenarios", "Ask polite questions when you need help.", ["polite", "when", "ask", "questions", "you", "need", "help"]),
    ("scenarios", "Thank the person who helped you today.", ["the", "who", "thank", "helped", "person", "you", "today"]),
    ("scenarios", "Arrive a few minutes early for the meeting.", ["a", "early", "arrive", "few", "minutes", "for", "the", "meeting"]),
]


def _fallback_sentence_reorder(request: GenerateActivityRequest) -> Optional[Dict]:
    """Return a valid SENTENCE_REORDER activity synthesized from a curated pool.

    Used only when the LLM is unavailable or keeps producing malformed output.
    Returns ``None`` if no pool entry can satisfy the request (shouldn't happen).
    """
    import hashlib
    import random as _random

    topic = (request.unit_context or "").lower()
    # Best-effort topic match against the pool's hint tokens.
    candidates = [
        (hint, sentence, words)
        for (hint, sentence, words) in _FALLBACK_SENTENCE_POOL
        if hint in topic or topic in hint
    ]
    if not candidates:
        candidates = _FALLBACK_SENTENCE_POOL

    # Deterministic selection per request so retries return the same activity.
    seed_src = f"{request.unit_context}|{request.difficulty_tier.value}"
    seed = int(hashlib.sha1(seed_src.encode()).hexdigest(), 16)
    rng = _random.Random(seed)
    hint, correct_sentence, words = rng.choice(candidates)

    # Re-shuffle the words deterministically (avoid correct-order identity).
    shuffle_seed = int(hashlib.sha1(correct_sentence.encode()).hexdigest(), 16)
    rng2 = _random.Random(shuffle_seed)
    shuffled = list(words)
    rng2.shuffle(shuffled)
    # Guarantee: shuffled must not already equal words in the same order
    if shuffled == list(words):
        shuffled[0], shuffled[1] = shuffled[1], shuffled[0]

    # Build correctOrder: shuffled[i] -> position in correct_sentence
    sentence_tokens = correct_sentence.split()
    correct_order = []
    used = set()
    for tok in sentence_tokens:
        norm_tok = tok.lower().rstrip(",.?!;:")
        chosen = None
        for i, w in enumerate(shuffled):
            if i in used:
                continue
            if w.lower().rstrip(",.?!;:") == norm_tok:
                chosen = i
                break
        if chosen is None:
            # multiset mismatch — pick first unused
            for i in range(len(shuffled)):
                if i not in used:
                    chosen = i
                    break
        correct_order.append(chosen if chosen is not None else 0)
        used.add(chosen if chosen is not None else 0)

    return {
        "activityId": str(uuid.uuid4()),
        "type": request.activity_type.value,
        "domain": request.skill_domain.value,
        "difficultyTier": request.difficulty_tier.value,
        "xpValue": 8,
        "question": "Arrange these words to form a correct sentence:",
        "words": shuffled,
        "correctOrder": correct_order,
        "correctSentence": correct_sentence,
        "topic": request.unit_context,
        "explanation": "Notice how the words change meaning when reordered.",
    }


# --- WordLearn fallback pool ---

_WORD_LEARN_POOL = [
    {
        "word": "concise",
        "definition": "using few words and giving information clearly",
        "exampleSentence": "Her concise email told the whole team exactly what to do.",
        "correct": "Please write a concise summary so the manager can read it in a minute.",
        "wrong1": "The teacher's concise lecture lasted for three full hours.",
        "wrong2": "We were concise with our friends at the big outdoor picnic.",
        "wrong3": "The novelist's concise novel contained over eight hundred pages of detail.",
        "explanation": "Concise means brief and to the point. The summary sentence fits this meaning; a three-hour lecture is the opposite. 'Concise with friends' uses the word as if it were a verb, which is incorrect.",
        "topic": "Precise vocabulary",
    },
    {
        "word": "reluctant",
        "definition": "not willing or wanting to do something",
        "exampleSentence": "She was reluctant to leave the warm cafe on a cold morning.",
        "correct": "He felt reluctant to share his password with anyone, even his best friend.",
        "wrong1": "The sun was reluctant to set over the calm lake at the end of the long day.",
        "wrong2": "Her reluctant dog wagged its tail and barked with great excitement.",
        "wrong3": "After winning the award, she felt reluctant and completely overjoyed.",
        "explanation": "Reluctant means unwilling. Sharing a password is something a person might hesitate to do. The sun cannot have feelings, a wagging dog is eager (the opposite), and 'reluctant' does not mean 'happy'.",
        "topic": "Emotion words",
    },
    {
        "word": "negotiate",
        "definition": "to discuss terms in order to reach an agreement",
        "exampleSentence": "We negotiated the price of the car for over an hour before signing.",
        "correct": "The union and the company negotiated a new contract that raised wages.",
        "wrong1": "She negotiated the book off the shelf and handed it to her little brother.",
        "wrong2": "The chef negotiated the soup by adding salt and tasting it again.",
        "wrong3": "We negotiated our way home by bus after the train was cancelled.",
        "explanation": "Negotiate means to bargain or discuss to reach an agreement. Unions and companies do this over contracts. The other sentences misuse the word as a synonym for 'took,' 'seasoned,' or simply 'traveled.'",
        "topic": "Action words",
    },
    {
        "word": "ambiguous",
        "definition": "having more than one possible meaning; unclear",
        "exampleSentence": "The directions were ambiguous, so we got lost twice before finding the cafe.",
        "correct": "His ambiguous answer left us unsure whether he agreed or politely declined.",
        "wrong1": "The sunny morning was ambiguous, brightening everyone's mood on the way to school.",
        "wrong2": "Her ambiguous essay was fifty pages long and clearly proved a single point.",
        "wrong3": "After the loud crash, the room fell ambiguous and everyone stopped talking.",
        "explanation": "Ambiguous means unclear or open to more than one interpretation. An answer that could mean either yes or no fits this. Sunny weather, a long single-point essay, and a silent room are not ambiguous in this way.",
        "topic": "Descriptive vocabulary",
    },
]


def _fallback_word_learn(request: GenerateActivityRequest) -> Optional[Dict]:
    """Synthesize a valid WORD_LEARN activity from a curated pool."""
    import hashlib
    import random as _random

    topic = (request.unit_context or "").lower()
    candidates = [
        e for e in _WORD_LEARN_POOL
        if any(tok in topic for tok in e["topic"].lower().split())
    ]
    if not candidates:
        candidates = _WORD_LEARN_POOL

    seed_src = f"word_learn|{request.unit_context}|{request.difficulty_tier.value}"
    seed = int(hashlib.sha1(seed_src.encode()).hexdigest(), 16)
    entry = _random.Random(seed).choice(candidates)

    options = [
        {"id": "a", "text": entry["correct"], "isCorrect": True},
        {"id": "b", "text": entry["wrong1"], "isCorrect": False},
        {"id": "c", "text": entry["wrong2"], "isCorrect": False},
        {"id": "d", "text": entry["wrong3"], "isCorrect": False},
    ]
    # Shuffle option order deterministically and remap ids
    opt_seed = int(hashlib.sha1(f"{entry['word']}-opts".encode()).hexdigest(), 16)
    rng = _random.Random(opt_seed)
    indices = [0, 1, 2, 3]
    rng.shuffle(indices)
    shuffled = [options[i] for i in indices]
    ids = ["a", "b", "c", "d"]
    for new_id, opt in zip(ids, shuffled):
        opt["id"] = new_id

    return {
        "activityId": str(uuid.uuid4()),
        "type": request.activity_type.value,
        "domain": request.skill_domain.value,
        "difficultyTier": request.difficulty_tier.value,
        "xpValue": 10,
        "word": entry["word"],
        "definition": entry["definition"],
        "exampleSentence": entry["exampleSentence"],
        "question": f'Which sentence uses the word "{entry["word"]}" correctly?',
        "options": shuffled,
        "explanation": entry["explanation"],
        "topic": entry["topic"],
    }


# --- Reading fallback pool ---

_READING_POOL = [
    {
        "title": "The Library Robot",
        "topic_hint": "library",
        "passage": (
            "The Glenview Public Library has a new helper, but it does not look like a librarian. "
            "It is a small gray robot named Pip, and it rolls quietly between the tall shelves, "
            "scanning labels with a soft blue light. Pip was introduced last spring, when the library "
            "noticed that many of its older books were being placed on the wrong shelves. "
            "Now, every time a book is returned, Pip glides over, scans the spine, and gently beeps "
            "if the book is in the wrong place. If the book is in the right place, Pip simply blinks "
            "its light twice and moves on to the next shelf. Children who visit the library often "
            "stop to watch Pip at work, and some even try to guess which book Pip will check next. "
            "The librarians say that since Pip arrived, the number of misplaced books has dropped by half. "
            "For the first time in years, every book on the shelves is exactly where it should be."
        ),
        "questions": [
            {
                "id": "q1",
                "question": "What is the main idea of the passage?",
                "skill": "main_idea",
                "correct": "A robot named Pip helps keep library books in the correct place.",
                "wrong1": "Children enjoy watching the librarians read stories out loud.",
                "wrong2": "The library was closed last spring for many months.",
                "wrong3": "Robots will soon replace all human workers in every library.",
                "explanation": "The passage focuses on Pip, a robot that scans and sorts library books. The other options either describe minor details or are not supported by the passage.",
            },
            {
                "id": "q2",
                "question": "What does Pip do when a book is in the right place?",
                "skill": "detail",
                "correct": "It blinks its light twice and moves to the next shelf.",
                "wrong1": "It gently beeps to warn the librarian.",
                "wrong2": "It picks the book up and carries it away.",
                "wrong3": "It asks the children to guess the next book.",
                "explanation": "The passage says, 'If the book is in the right place, Pip simply blinks its light twice and moves on to the next shelf.'",
            },
            {
                "id": "q3",
                "question": "What can you infer about the librarians' feelings about Pip?",
                "skill": "inference",
                "correct": "They are pleased that Pip has reduced the number of misplaced books.",
                "wrong1": "They are worried that Pip will replace them.",
                "wrong2": "They find Pip annoying because it beeps too often.",
                "wrong3": "They believe Pip is dangerous to the children.",
                "explanation": "The librarians' positive comment about the misplaced books dropping by half suggests they are happy with Pip's work.",
            },
        ],
        "topic": "Informational reading",
    },
    {
        "title": "A Morning Run",
        "topic_hint": "running",
        "passage": (
            "Maya laced up her shoes before the sun had risen. By the time she stepped outside, "
            "the streets were still empty and the air was cool. She started slowly, watching her breath "
            "form small clouds in front of her face. After a few minutes, she passed the bakery on the corner, "
            "where the smell of fresh bread was just beginning to drift out through the open door. "
            "Maya smiled and picked up her pace. She liked running in the morning because the world was so quiet. "
            "Only the birds and the soft tap of her shoes on the pavement broke the silence. "
            "Halfway through her run, she stopped briefly at a small park to stretch her legs. "
            "A jogger she had never met waved hello, and she waved back. "
            "By the time Maya reached her front door again, the sun was fully up, and her neighbors were beginning to leave for work."
        ),
        "questions": [
            {
                "id": "q1",
                "question": "What is the main idea of the passage?",
                "skill": "main_idea",
                "correct": "Maya enjoys an early-morning run through her quiet neighborhood.",
                "wrong1": "Maya is training for an important race.",
                "wrong2": "The bakery is the best place in town to buy bread.",
                "wrong3": "Maya meets a new friend in the park.",
                "explanation": "The passage follows Maya on a single morning run and describes what she sees and feels. The other options are not supported.",
            },
            {
                "id": "q2",
                "question": "Why does Maya like running in the morning?",
                "skill": "detail",
                "correct": "Because the world is quiet at that hour.",
                "wrong1": "Because the bakery is open early.",
                "wrong2": "Because she is meeting a friend at the park.",
                "wrong3": "Because the sun is already bright and warm.",
                "explanation": "The passage says, 'She liked running in the morning because the world was so quiet.'",
            },
            {
                "id": "q3",
                "question": "What does the small detail about the jogger waving suggest?",
                "skill": "inference",
                "correct": "People in the neighborhood are friendly to strangers.",
                "wrong1": "Maya and the jogger are old friends.",
                "wrong2": "The jogger was lost and needed directions.",
                "wrong3": "Maya was running too fast.",
                "explanation": "A simple wave between strangers suggests a friendly neighborhood culture. The passage never says they know each other.",
            },
        ],
        "topic": "Narrative reading",
    },
]


def _fallback_reading(request: GenerateActivityRequest) -> Optional[Dict]:
    """Synthesize a valid READING activity from a curated pool."""
    import hashlib
    import random as _random

    topic = (request.unit_context or "").lower()
    candidates = [
        e for e in _READING_POOL
        if e["topic_hint"] in topic or topic in e["topic_hint"]
    ]
    if not candidates:
        candidates = _READING_POOL

    seed_src = f"reading|{request.unit_context}|{request.difficulty_tier.value}"
    seed = int(hashlib.sha1(seed_src.encode()).hexdigest(), 16)
    entry = _random.Random(seed).choice(candidates)

    # Pick number of questions based on difficulty tier
    tier = request.difficulty_tier.value if hasattr(request.difficulty_tier, "value") else int(request.difficulty_tier)
    if tier <= 2:
        chosen_questions = entry["questions"][:2]
    else:
        chosen_questions = entry["questions"][:3]

    # Shuffle question order and remap ids
    q_seed = int(hashlib.sha1(f"{entry['title']}-qs".encode()).hexdigest(), 16)
    rng = _random.Random(q_seed)
    qs = list(chosen_questions)
    rng.shuffle(qs)
    new_questions = []
    for i, q in enumerate(qs):
        # Shuffle options for this question
        opts = [
            {"id": "a", "text": q["correct"], "isCorrect": True},
            {"id": "b", "text": q["wrong1"], "isCorrect": False},
            {"id": "c", "text": q["wrong2"], "isCorrect": False},
            {"id": "d", "text": q["wrong3"], "isCorrect": False},
        ]
        opt_seed = int(hashlib.sha1(f"{entry['title']}-{q['id']}-opts".encode()).hexdigest(), 16)
        rng2 = _random.Random(opt_seed)
        indices = [0, 1, 2, 3]
        rng2.shuffle(indices)
        shuffled = [opts[j] for j in indices]
        ids = ["a", "b", "c", "d"]
        for new_id, opt in zip(ids, shuffled):
            opt["id"] = new_id
        new_questions.append({
            "id": f"q{i+1}",
            "question": q["question"],
            "options": shuffled,
            "explanation": q["explanation"],
            "skill": q["skill"],
        })

    return {
        "activityId": str(uuid.uuid4()),
        "type": request.activity_type.value,
        "domain": request.skill_domain.value,
        "difficultyTier": request.difficulty_tier.value,
        "xpValue": 12,
        "title": entry["title"],
        "passage": entry["passage"],
        "questions": new_questions,
        "topic": entry["topic"],
    }


# --- Evaluation Endpoints (AI-powered) ---

@router.post("/activity/evaluate/writing", response_model=FeedbackReport)
async def evaluate_writing(
    request: EvaluateWritingRequest,
    user_info: Dict = Depends(get_current_user),
):
    """Evaluate a writing response using AI and return a FeedbackReport."""
    try:
        agent = get_comm_agent()
        result = await agent.evaluate_writing(request)
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        logger.error(f"Writing evaluation error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to evaluate writing",
        )


@router.post("/activity/evaluate/speaking", response_model=FeedbackReport)
async def evaluate_speaking(
    request: EvaluateSpeakingRequest,
    user_info: Dict = Depends(get_current_user),
):
    """Evaluate a speaking transcription using AI and return a FeedbackReport."""
    try:
        agent = get_comm_agent()
        result = await agent.evaluate_speaking(request)
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        logger.error(f"Speaking evaluation error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to evaluate speaking",
        )


@router.post("/activity/evaluate/scenario", response_model=ScenarioTurnResponse)
async def evaluate_scenario_turn(
    request: ScenarioTurnRequest,
    user_info: Dict = Depends(get_current_user),
):
    """Submit a dialogue turn and get AI in-character response."""
    try:
        agent = get_comm_agent()
        result = await agent.scenario_turn(request)
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        logger.error(f"Scenario turn error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process scenario turn",
        )


@router.post("/activity/evaluate/scenario/end", response_model=FeedbackReport)
async def evaluate_scenario_end(
    request: ScenarioEndRequest,
    user_info: Dict = Depends(get_current_user),
):
    """End a scenario and get full conversation feedback from AI."""
    try:
        agent = get_comm_agent()
        result = await agent.scenario_end_feedback(request)
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        logger.error(f"Scenario end feedback error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to evaluate scenario",
        )


# --- TTS/STT Endpoints (reusing existing AudioProcessor) ---

class TTSRequest(BaseModel):
    text: str
    voice_id: str = "Joanna"
    speed: str = "85%"


class STTRequest(BaseModel):
    audio_base64: str  # Base64 encoded WAV audio


def _get_audio_processor() -> AudioProcessor:
    """Get AudioProcessor instance with env config."""
    return AudioProcessor(
        cartesia_api_key=os.getenv("CARTESIA_API_KEY", ""),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "") or None,
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "") or None,
        aws_region=os.getenv("AWS_REGION", "ap-south-1"),
        polly_voice_id=os.getenv("AWS_POLLY_VOICE_ID", "Joanna"),
        polly_engine=os.getenv("AWS_POLLY_ENGINE", "neural"),
        polly_speech_rate=os.getenv("AWS_POLLY_SPEECH_RATE", "85%"),
        cartesia_model=os.getenv("CARTESIA_MODEL", "ink-whisper"),
        cartesia_api_version=os.getenv("CARTESIA_API_VERSION", "2025-04-16"),
    )


@router.post("/tts")
async def text_to_speech(
    request: TTSRequest,
    user_info: Dict = Depends(get_current_user),
):
    """Convert text to speech using AWS Polly. Returns base64 MP3 audio."""
    try:
        processor = _get_audio_processor()
        audio_base64 = await asyncio.to_thread(
            processor.synthesize_speech_base64,
            request.text,
            request.voice_id,
            request.speed,
        )
        return {"status": "success", "audio_base64": audio_base64}
    except Exception as e:
        logger.error(f"TTS error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Text-to-speech failed: {str(e)}",
        )


@router.post("/stt")
async def speech_to_text(
    request: STTRequest,
    user_info: Dict = Depends(get_current_user),
):
    """Transcribe audio using Cartesia ink-whisper STT. Accepts base64 WAV."""
    try:
        processor = _get_audio_processor()
        transcription = await asyncio.to_thread(
            processor.transcribe_audio,
            request.audio_base64,
        )
        return {"status": "success", "transcription": transcription}
    except Exception as e:
        logger.error(f"STT error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Speech-to-text failed: {str(e)}",
        )


# --- Scenario with TTS (AI speaks first, returns audio) ---

@router.post("/scenario/start")
async def start_scenario(
    request: GenerateActivityRequest,
    user_info: Dict = Depends(get_current_user),
):
    """
    Start a scenario: generates the scenario setup AND the AI's first message with TTS audio.
    Returns the scenario config + first AI message + audio.
    """
    try:
        agent = get_comm_agent()
        result = await agent.generate_activity(request)

        # If Gemini returned a list, take first item
        if isinstance(result, list):
            result = result[0] if result else {}
        if not isinstance(result, dict):
            result = {}

        first_message = result.get("firstMessage") or result.get("first_message") or "Hello! Let's begin our conversation."

        # Generate TTS for the first message
        audio_base64 = None
        try:
            processor = _get_audio_processor()
            audio_base64 = await asyncio.to_thread(
                processor.synthesize_speech_base64,
                first_message,
                None,
                "85%",
            )
        except Exception as tts_err:
            logger.warning(f"TTS failed for scenario start (non-fatal): {tts_err}")

        return {
            "scenarioRole": result.get("scenarioRole") or result.get("scenario_role") or "conversation partner",
            "scenarioDescription": result.get("scenarioDescription") or result.get("scenario_description") or request.unit_context,
            "totalTurns": result.get("totalTurns") or result.get("total_turns") or 6,
            "firstMessage": first_message,
            "firstMessageOptions": result.get("firstMessageOptions") or result.get("first_message_options") or [],
            "audio_base64": audio_base64,
        }
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        logger.error(f"Scenario start error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start scenario",
        )


@router.post("/scenario/respond")
async def scenario_respond(
    request: ScenarioTurnRequest,
    user_info: Dict = Depends(get_current_user),
):
    """
    Submit user's turn in scenario, get AI response with TTS audio.
    """
    try:
        agent = get_comm_agent()
        result = await agent.scenario_turn(request)

        # Generate TTS for AI response
        audio_base64 = None
        try:
            processor = _get_audio_processor()
            audio_base64 = await asyncio.to_thread(
                processor.synthesize_speech_base64,
                result.ai_response,
                None,
                "85%",
            )
        except Exception as tts_err:
            logger.warning(f"TTS failed for scenario respond (non-fatal): {tts_err}")

        return {
            "ai_response": result.ai_response,
            "is_complete": result.is_complete,
            "audio_base64": audio_base64,
        }
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        logger.error(f"Scenario respond error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process scenario turn",
        )
