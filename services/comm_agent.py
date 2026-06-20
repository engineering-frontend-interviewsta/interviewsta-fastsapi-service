"""
CommByAI Agent Service — LLM-powered activity generation and evaluation.
Uses OpenAI (GPT-4o-mini) for all LLM calls.
All methods are async; LLM calls run in thread pool via asyncio.to_thread.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from typing import Any, Optional

from openai import OpenAI
from pydantic import ValidationError

from schemas.comm import (
    ActivityType,
    DifficultyTier,
    EvaluateSpeakingRequest,
    EvaluateWritingRequest,
    FeedbackReport,
    GenerateActivityRequest,
    OnboardingAssessRequest,
    PlacementLevel,
    PlacementResult,
    ScenarioEndRequest,
    ScenarioTurnRequest,
    ScenarioTurnResponse,
    SkillDomain,
    UserSegment,
)

logger = logging.getLogger(__name__)

# Configure OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
_MODEL_NAME = "gpt-4o-mini"


def _get_client():
    """Get configured OpenAI client instance."""
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        _client = OpenAI(api_key=api_key)
    return _client


def _call_openai_sync(system_prompt: str, user_prompt: str) -> str:
    """Synchronous OpenAI call (to be run in thread pool)."""
    client = _get_client()
    response = client.chat.completions.create(
        model=_MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


SYSTEM_PROMPT = """You are CommByAI, an expert communication coach, language teacher, and curriculum designer.

User profile:
- Segment: {user_segment}
- Placement level: {placement_level}
- Skill domain: {skill_domain}
- Difficulty tier: {difficulty_tier}/5

Audience calibration — STRICTLY follow these:
- YoungLearner tier 1-2: Max 2-syllable words. Short sentences (under 10 words). Fun contexts: animals, school, family, food, colours. Use encouraging, playful tone. Questions should be answerable by an 8-year-old.
- YoungLearner tier 3-5: Slightly longer sentences. Introduce simple connectors (and, but, because). Keep contexts relatable. Answerable by a 10-year-old.
- MiddleSchool tier 1-3: Everyday contexts: friends, sports, social media, hobbies. Compound sentences. Vocabulary around school life.
- MiddleSchool tier 4-5: More complex grammar. Formal vs informal distinction.
- HighSchool tier 1-3: Academic and social contexts. Multi-clause sentences. Essay-style writing.
- HighSchool tier 4-5: Subordinate clauses, passive voice, academic vocabulary.
- College tier 1-3: Academic writing, research contexts, debate. Complex syntax expected.
- College tier 4-5: Nuanced argumentation, citation style, presentation language.
- WorkingProfessional tier 1-3: Professional email, meeting language, basic client communication.
- WorkingProfessional tier 4-5: Negotiation, executive communication, public speaking, concise high-stakes writing.

Question-design rules — NON-NEGOTIABLE:

1. REAL-BOOK QUALITY. Every question must be the kind of question that genuinely appears in a published school textbook or ESL workbook for the target age group. No trivia, no joke questions, no random topics. The stem should be clear, focused, and educational — not creative writing.

2. ANSWER-KEY CORRECTNESS. Before you mark any option isCorrect: true, verify three things in your own reasoning:
   (a) The marked option must actually, unambiguously answer the question.
   (b) No other option can reasonably be argued to be correct.
   (c) The `explanation` you write must cite a real rule and apply it correctly to the marked option.
   If you cannot verify all three, rewrite the question.

3. EXPLANATION TRUTH. The `explanation` is part of the answer key — it must be factually correct and must directly support the marked option. Do NOT write generic platitudes ("This is the right answer", "It sounds better", "It's more natural"). State the actual rule and apply it. Example of a BAD explanation: "The word 'fun' describes how the dog feels when playing." Example of a GOOD explanation: "The sentence needs an adverb ending in -ly to modify the verb 'runs'. 'Quickly' is the only adverb among the options; 'slow', 'fun', and 'happy' are adjectives."

4. DISTRACTOR QUALITY. The three wrong options must be plausible mistakes a real student would make:
   - Wrong grammatical form (e.g., "go" instead of "goes", "eat" instead of "eaten")
   - Real but wrong word (e.g., a synonym that doesn't fit the context)
   - Common confusion (e.g., "their/there/they're", "its/it's")
   They must NOT be obviously absurd, and they must NOT share meaning with the correct option.

5. SCHEMA COMPLETENESS. Return EVERY field listed in the JSON shape. All string fields must contain real content. Never use "Option 2", "...", "<TBD>", "placeholder", or any abbreviation.

6. VARIETY. Invent a fresh scenario each call. Do not reuse stock examples. Do not always put the correct answer in position b.

7. AGE-APPROPRIATE LANGUAGE. The wording of the question, options, and explanation must be calibrated to the segment. A YoungLearner question about tenses uses simple present with animals. A College question about the same grammar uses academic prose.

ALWAYS return valid JSON only. No markdown fences. No prose outside the JSON object."""


def _build_system_prompt(
    user_segment: UserSegment,
    placement_level: str,
    skill_domain: str,
    difficulty_tier: int,
) -> str:
    return SYSTEM_PROMPT.format(
        user_segment=user_segment.value,
        placement_level=placement_level,
        skill_domain=skill_domain,
        difficulty_tier=difficulty_tier,
    )


def _parse_json_response(text: str) -> dict:
    """Parse JSON from LLM response, stripping markdown fences if present."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Remove markdown code fences
        lines = cleaned.split("\n")
        # Remove first and last lines if they are fences
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines)
    return json.loads(cleaned)


def _call_llm_sync(system_prompt: str, user_prompt: str) -> str:
    """Synchronous LLM call using OpenAI (to be run in thread pool)."""
    return _call_openai_sync(system_prompt, user_prompt)


def _nonce() -> str:
    """Return a short unique nonce to guarantee per-call prompt variation."""
    return uuid.uuid4().hex[:8]


def _retry_llm_sync(system_prompt: str, base_prompt: str, validate_fn) -> dict:
    """Call LLM, parse JSON, validate; retry up to 3 times with fix-it instructions on failure.

    If the final response fails validation, raise ValueError with the last error.
    """
    nonce = _nonce()
    nonce_prompt = f"{base_prompt}\n\n<!-- session: {nonce} -->"
    last_err: Optional[str] = None
    last_data: Optional[dict] = None
    for attempt in range(3):
        if attempt == 0:
            prompt = nonce_prompt
        else:
            # Build corrective instructions that include a concrete worked example
            # for the most error-prone field ("correctOrder" in sentence_reorder).
            order_help = ""
            if last_err and "correctOrder" in last_err:
                order_help = (
                    "\n\nConcrete example of correctOrder:\n"
                    '  words = ["fox", "the", "quick", "brown", "jumps"]\n'
                    '  correctOrder = [1, 2, 3, 0, 4]\n'
                    '  because words[1]="the", words[2]="quick", words[3]="brown", words[0]="fox", words[4]="jumps"\n'
                    '  → correctSentence = "The quick brown fox jumps."\n'
                    "Rule: correctOrder[i] is the INDEX into the 'words' array for the i-th word of the sentence.\n"
                )
            prompt = (
                f"{base_prompt}\n\n"
                f"IMPORTANT: Your previous response was rejected ({last_err}). "
                "You must respond again with a fully-valid object that contains EVERY required field, "
                "with REAL content (no placeholders, no ellipses, no 'Option N' stubs). "
                "Do not repeat the previous content — invent a new question."
                f"{order_help}\n\n<!-- session: {nonce} -->"
            )
        try:
            raw = _call_llm_sync(system_prompt, prompt)
            data = _parse_json_response(raw)
            last_data = data
            err = validate_fn(data)
            if not err:
                return data
            last_err = err
        except (json.JSONDecodeError, Exception) as e:
            last_err = f"parse/validation error: {e}"
    raise ValueError(f"LLM response failed validation after 3 attempts: {last_err}")


class CommAgent:
    """
    Stateless LLM service for CommByAI.
    Uses OpenAI GPT-4o-mini for all LLM calls.
    All methods are async; heavy LLM calls run in thread pool via asyncio.to_thread.
    """

    async def generate_activity(self, req: GenerateActivityRequest) -> dict:
        """Generate a single activity based on type, segment, and difficulty.

        Returns a canonical camelCase dict with all required fields populated.
        On malformed LLM output, retries up to 3 times with corrective instructions.
        If all retries fail, raises ValueError (caller will return 500, not 422).
        """
        system = _build_system_prompt(
            req.user_segment,
            req.placement_level.value,
            req.skill_domain.value,
            req.difficulty_tier.value,
        )

        xp_base = {
            ActivityType.MCQ: 5,
            ActivityType.FILL_BLANK: 5,
            ActivityType.SENTENCE_REORDER: 8,
            ActivityType.WORD_LEARN: 10,
            ActivityType.READING: 12,
            ActivityType.WRITING_PROMPT: 20,
            ActivityType.SPEAKING: 25,
            ActivityType.SCENARIO: 30,
        }

        segment_guide = {
            "YoungLearner": "Use simple words (max 2 syllables), short sentences, fun contexts (animals, school, family, food). Be playful and encouraging.",
            "MiddleSchool": "Use everyday language, relatable contexts (friends, sports, hobbies). Moderate complexity.",
            "HighSchool": "Use academic contexts, multi-clause sentences. Formal and informal register.",
            "College": "Use academic/professional contexts. Complex syntax, nuanced vocabulary.",
            "WorkingProfessional": "Use professional/business contexts. Concise, formal register.",
        }.get(req.user_segment.value, "Use age-appropriate language.")

        prompt, validate, normalize = self._build_activity_prompt_and_validators(
            req, segment_guide
        )

        try:
            raw_text = await asyncio.to_thread(_retry_llm_sync, system, prompt, validate)
        except ValueError as llm_err:
            logger.error(f"Activity generation failed: {llm_err}")
            raise
        result = normalize(raw_text)

        # Attach metadata
        result["activityId"] = str(uuid.uuid4())
        result["type"] = req.activity_type.value
        result["domain"] = req.skill_domain.value
        result["difficultyTier"] = req.difficulty_tier.value
        result["xpValue"] = xp_base.get(req.activity_type, 5)

        return result

    def _build_activity_prompt_and_validators(self, req, segment_guide):
        """Return (prompt, validate_fn, normalize_fn) for the requested activity type."""
        at = req.activity_type
        topic = req.unit_context
        seg = req.user_segment.value

        if at == ActivityType.MCQ:
            if req.skill_domain.value == "Vocabulary":
                prompt = self._vocab_mcq_prompt(topic, seg, segment_guide)
            else:
                prompt = self._grammar_mcq_prompt(topic, seg, segment_guide)
            return prompt, self._validate_mcq, self._normalize_mcq
        if at == ActivityType.FILL_BLANK:
            return self._fill_blank_prompt(topic, seg, segment_guide), self._validate_fill_blank, self._normalize_fill_blank
        if at == ActivityType.SENTENCE_REORDER:
            return self._sentence_reorder_prompt(topic, seg, segment_guide), self._validate_sentence_reorder, self._normalize_sentence_reorder
        if at == ActivityType.WORD_LEARN:
            return self._word_learn_prompt(topic, seg, segment_guide), self._validate_word_learn, self._normalize_word_learn
        if at == ActivityType.READING:
            return self._reading_passage_prompt(topic, seg, segment_guide, req.difficulty_tier.value), self._validate_reading, self._normalize_reading
        if at == ActivityType.WRITING_PROMPT:
            min_words = 20 if req.difficulty_tier.value <= 2 else 40 if req.difficulty_tier.value <= 4 else 60
            return self._writing_prompt_prompt(topic, seg, segment_guide, min_words), self._validate_writing_prompt, self._normalize_writing_prompt
        if at == ActivityType.SPEAKING:
            min_duration = 5 if req.difficulty_tier.value <= 2 else 15 if req.difficulty_tier.value <= 4 else 25
            return self._speaking_prompt_prompt(topic, seg, segment_guide, min_duration), self._validate_speaking, self._normalize_speaking
        if at == ActivityType.SCENARIO:
            return self._scenario_prompt_prompt(topic, seg, segment_guide), self._validate_scenario, self._normalize_scenario
        raise ValueError(f"Unsupported activity type: {at}")

    # ── Per-type prompts ─────────────────────────────────────────────────────

    @staticmethod
    def _grammar_mcq_prompt(topic: str, seg: str, segment_guide: str) -> str:
        return f"""Generate ONE grammar multiple-choice question about "{topic}" for a {seg} student.

This question must be the type that appears in a real school grammar workbook for {seg} students — not a creative or trivia question.

Audience calibration: {segment_guide}

Return ONLY this exact JSON object (no markdown, no prose, no array):
{{
  "question": "<a clear, complete question — typically a sentence with a blank or an underlined part, followed by a clear instruction like 'Choose the correct word to complete the sentence.'>",
  "options": [
    {{"id": "a", "text": "<option text>", "isCorrect": false}},
    {{"id": "b", "text": "<option text>", "isCorrect": true}},
    {{"id": "c", "text": "<option text>", "isCorrect": false}},
    {{"id": "d", "text": "<option text>", "isCorrect": false}}
  ],
  "explanation": "<2-3 sentences that name the specific grammar rule and apply it to the marked-correct answer>",
  "topic": "<short grammar label like 'Subject-Verb Agreement', 'Past Tense', 'Articles', 'Prepositions of Time', 'Comparatives'>"
}}

Hard rules:
- EXACTLY 4 options, EXACTLY one with isCorrect: true.
- The marked-correct option must actually, unambiguously answer the question. Verify this in your reasoning before responding.
- The three wrong options must be plausible mistakes a real student would make — wrong verb form, wrong tense, common confusion. They must NOT be obviously absurd, and they must NOT have the same meaning as the correct option.
- The `explanation` must state the actual rule being tested (e.g., "third-person singular present tense adds -s/-es to the verb") and apply it to the marked option. NEVER write vague explanations like "It sounds right" or "It describes how the dog feels."
- "question" must be self-contained. The student should know exactly what to do after reading it. Do NOT use phrases like "Which of the following is true?" without context.
- "topic" must be a specific grammar concept label, not just "{topic}".
- Randomize which option (a/b/c/d) is correct — do not always put it in position b.
- Invent a fresh sentence. Do NOT reuse stock examples ("She doesn't like coffee", "He goes to school"). The sentence should be age-appropriate for {seg}."""

    @staticmethod
    def _vocab_mcq_prompt(topic: str, seg: str, segment_guide: str) -> str:
        return f"""Generate ONE vocabulary multiple-choice question about "{topic}" for a {seg} student.

This question must be the type that appears in a real school vocabulary workbook for {seg} students. The question should test the meaning, synonym, antonym, or correct usage of a single vocabulary word.

Audience calibration: {segment_guide}

Return ONLY this exact JSON object:
{{
  "question": "<a clear question that tests the meaning or usage of ONE specific vocabulary word — include a sentence with the word used in context, then ask what it means / which synonym fits / which word replaces it>",
  "options": [
    {{"id": "a", "text": "<option text>", "isCorrect": false}},
    {{"id": "b", "text": "<option text>", "isCorrect": false}},
    {{"id": "c", "text": "<option text>", "isCorrect": true}},
    {{"id": "d", "text": "<option text>", "isCorrect": false}}
  ],
  "explanation": "<2-3 sentences that define the target word correctly and show why the marked option is the right answer. Cite a real dictionary-style definition, not a guess.>",
  "topic": "<short vocabulary label like 'Synonyms', 'Context Clues', 'Word Meaning', 'Antonyms'>"
}}

Hard rules:
- EXACTLY 4 options, EXACTLY one with isCorrect: true.
- Pick ONE specific target word and use it correctly in the question stem. Do not leave the word ambiguous.
- The marked-correct option must be the actual meaning/usage of the target word. Verify this yourself in your reasoning. If the target word means "to do something repeatedly to get better at it", the correct answer must reflect that — NOT a generic or related-but-wrong definition.
- The three wrong options must be plausible-looking mistakes (similar meaning, related concept, common confusion), NOT obviously absurd.
- The `explanation` must contain the correct dictionary definition of the target word and a clear reason why the marked option matches it. NEVER fabricate a definition that does not match the real meaning of the word. NEVER say "X describes how Y feels" unless X is actually a feeling word.
- Avoid testing extremely rare or archaic words. The target word should be age-appropriate for {seg} and appear in a standard school vocabulary list.
- "question" must be self-contained. The student should know exactly what is being asked.
- "topic" must be a specific vocabulary skill, not just "{topic}".
- Randomize which option (a/b/c/d) is correct.
- Invent a fresh example. Do NOT reuse stock sentences."""

    @staticmethod
    def _fill_blank_prompt(topic: str, seg: str, segment_guide: str) -> str:
        return f"""Generate ONE fill-in-the-blank exercise about "{topic}" for a {seg} student.

This exercise must be the type that appears in a real school grammar workbook for {seg} students.

Audience calibration: {segment_guide}

Return ONLY this exact JSON object:
{{
  "question": "<a complete sentence with EXACTLY ONE blank, written as ____ where the missing word goes. The sentence should be a real, age-appropriate example.>",
  "correctAnswer": "<the single correct word to fill the blank — no spaces, no punctuation>",
  "distractors": ["<plausible wrong word 1>", "<plausible wrong word 2>", "<plausible wrong word 3>"],
  "hint": "<a one-clue hint that points toward the right word without giving it away — e.g., a definition, a context clue, or a part-of-speech hint>",
  "explanation": "<2-3 sentences explaining the specific grammar or vocabulary rule and why the correct word fits.>"
}}

Hard rules:
- The blank must be unambiguous: only ONE word in English correctly fills it given the context. Verify this in your reasoning.
- "distractors" must be plausible mistakes a real student would make (wrong form, wrong tense, wrong part of speech, near-synonym that doesn't fit the context). They must NOT be obviously absurd, and they must NOT be synonyms of the correct answer.
- The complete sentence with the correct word filled in must be a real, natural English sentence.
- The sentence must be about "{topic}" or directly demonstrate the grammar concept in "{topic}".
- "hint" must be a useful steer (e.g., "Think about verb tense", "This is a feeling word"), NOT the answer itself.
- "explanation" must state the actual rule and apply it to the correct answer. Do NOT write generic "it fits because it sounds right" — cite the rule.
- Every string field must contain real content. No "..." or placeholder.
- The exercise must be calibrated for {seg} — vocabulary and sentence complexity appropriate to that age."""

    @staticmethod
    def _sentence_reorder_prompt(topic: str, seg: str, segment_guide: str) -> str:
        return f"""Generate ONE sentence-reordering exercise about "{topic}".

Audience calibration: {segment_guide}

Return ONLY this exact JSON object (no markdown, no prose):
{{
  "question": "Arrange these words to form a correct sentence:",
  "words": ["<word1>", "<word2>", "<word3>", "<word4>", "<word5>"],
  "correctSentence": "<the correctly ordered sentence with proper punctuation>"
}}

Hard rules:
- 5 to 7 unique word tokens total (no repeats). Punctuation in correctSentence (period, comma, etc.) does NOT count as a word.
- "words" must contain the same tokens as correctSentence (case-insensitive), in a SHUFFLED order (not the correct order).
- Each word in "words" should be lowercase (e.g. "the", "running"). correctSentence should be properly capitalized (e.g. "The dog is running fast.").
- correctSentence must be a real, natural English sentence about "{topic}".
- Do not include "correctOrder" — it is computed automatically.
- Do not use "..." placeholders or generic words like "word1".
- Punctuation attached to a word in correctSentence (e.g. "fast.") must appear as the bare word in "words" (e.g. "fast").
  The leading letter of correctSentence may be capitalized even if words[0] is lowercase.

Worked example of a valid response:
  words = ["fox", "the", "quick", "brown", "jumps"]
  correctSentence = "The quick brown fox jumps."
"""

    @staticmethod
    def _word_learn_prompt(topic: str, seg: str, segment_guide: str) -> str:
        return f"""Generate ONE vocabulary acquisition activity about "{topic}" for a {seg} student.

This activity teaches a NEW word: it first shows the word with its definition and pronunciation, then immediately tests whether the student can use the word correctly in a sentence.

Audience calibration: {segment_guide}

Return ONLY this exact JSON object (no markdown, no prose):
{{
  "word": "<a single, real English word strongly related to the topic>",
  "definition": "<a single-line, dictionary-style definition. Maximum 18 words. No examples.>",
  "exampleSentence": "<one full sentence (10-22 words) that uses the word correctly in context>",
  "question": "<a clear usage question. Phrase it as: 'Which sentence uses the word \"<word>\" correctly?'>",
  "options": [
    {{"id": "a", "text": "<full sentence using the word CORRECTLY>", "isCorrect": true}},
    {{"id": "b", "text": "<full sentence using the word INCORRECTLY (wrong meaning or grammar)>", "isCorrect": false}},
    {{"id": "c", "text": "<full sentence where the word is swapped for a near-synonym with subtly different connotation>", "isCorrect": false}},
    {{"id": "d", "text": "<full sentence where the word is used in a context that doesn't fit>", "isCorrect": false}}
  ],
  "explanation": "<2-3 sentences: define the word accurately, show why the correct sentence works, and explain why the most tempting wrong answer is wrong>",
  "topic": "<short label like 'Descriptive vocabulary', 'Academic vocabulary', 'Emotion words'>"
}}

Hard rules:
- The "word" must be a SINGLE word (no spaces, no hyphens). Use US spelling.
- The "word" must be GENUINELY related to "{topic}" — not a generic vocabulary word.
- The "word" should be at the right level for {seg}. Avoid archaic, hyper-specific jargon, or extremely rare words.
- The "definition" must match a real dictionary definition of the word. Do NOT invent a meaning.
- "exampleSentence" must contain the word verbatim, used in its primary sense.
- EXACTLY 4 options, EXACTLY one with isCorrect: true.
- All 4 options must be full, grammatical sentences. The 3 wrong options must be plausible mistakes a real student would make (wrong word sense, wrong collocation, wrong register) — not obviously absurd.
- The "explanation" must name the specific meaning of the word and apply it to the correct sentence. Do not write vague explanations.
- "topic" must be a specific vocabulary concept label, not just "{topic}".
- Randomize which option (a/b/c/d) is correct — do not always put it in position a.
- Do NOT include any extra fields beyond those listed."""

    @staticmethod
    def _reading_passage_prompt(topic: str, seg: str, segment_guide: str, difficulty_tier: int) -> str:
        if difficulty_tier <= 2:
            passage_words = "70 to 110 words"
            num_questions = 2
            skills = ["main_idea", "detail"]
        elif difficulty_tier <= 4:
            passage_words = "110 to 160 words"
            num_questions = 3
            skills = ["main_idea", "detail", "inference"]
        else:
            passage_words = "160 to 220 words"
            num_questions = 3
            skills = ["main_idea", "inference", "vocabulary_in_context"]

        skills_csv = ", ".join(skills)
        return f"""Generate ONE reading comprehension activity about "{topic}" for a {seg} student.

The activity is a short passage followed by comprehension questions. It is meant to be done at speed, so the passage should be readable in 60-120 seconds.

Audience calibration: {segment_guide}

Return ONLY this exact JSON object (no markdown, no prose):
{{
  "title": "<a short, engaging title (max 8 words)>",
  "passage": "<a coherent passage of {passage_words} about {topic}. Use a clear topic sentence and 2-4 supporting paragraphs. Write at the {seg} reading level.>",
  "questions": [
    {{
      "id": "q1",
      "question": "<a comprehension question>",
      "options": [
        {{"id": "a", "text": "<option text>", "isCorrect": false}},
        {{"id": "b", "text": "<option text>", "isCorrect": true}},
        {{"id": "c", "text": "<option text>", "isCorrect": false}},
        {{"id": "d", "text": "<option text>", "isCorrect": false}}
      ],
      "explanation": "<1-3 sentences: cite the specific part of the passage that supports the correct answer>",
      "skill": "{skills[0]}"
    }},
    {{
      "id": "q2",
      "question": "<a comprehension question>",
      "options": [
        {{"id": "a", "text": "<option text>", "isCorrect": true}},
        {{"id": "b", "text": "<option text>", "isCorrect": false}},
        {{"id": "c", "text": "<option text>", "isCorrect": false}},
        {{"id": "d", "text": "<option text>", "isCorrect": false}}
      ],
      "explanation": "<1-3 sentences>",
      "skill": "{skills[1] if len(skills) > 1 else skills[0]}"
    }}{',' if num_questions >= 3 else ''}
    {f'''
    {{
      "id": "q3",
      "question": "<a comprehension question>",
      "options": [
        {{"id": "a", "text": "<option text>", "isCorrect": false}},
        {{"id": "b", "text": "<option text>", "isCorrect": false}},
        {{"id": "c", "text": "<option text>", "isCorrect": true}},
        {{"id": "d", "text": "<option text>", "isCorrect": false}}
      ],
      "explanation": "<1-3 sentences>",
      "skill": "{skills[2] if len(skills) > 2 else skills[0]}"
    }}''' if num_questions >= 3 else ''}
  ],
  "topic": "<short label like 'Informational reading', 'Narrative reading', 'Persuasive reading'>"
}}

Hard rules:
- The "passage" must be coherent prose with a clear topic. Not bullet points, not a list.
- The "passage" length must be in the {passage_words} range. Do not exceed.
- EXACTLY {num_questions} questions. No more, no less.
- Each question must have EXACTLY 4 options, EXACTLY 1 with isCorrect: true.
- The 3 wrong options per question must be plausible distractors based on misreading the passage — not obviously absurd, not the same idea as the correct option.
- The "skill" for each question must be exactly one of: {skills_csv}.
- The "explanation" for each question must cite the specific sentence(s) of the passage that justify the answer.
- Each question's "options" must be unique strings — no duplicates within a single question.
- Vary which option (a/b/c/d) is correct across the {num_questions} questions.
- The passage and questions must be entirely self-contained — do not reference an external source.
- "topic" must be a specific reading-skill label, not just "{topic}".
- Do NOT include any extra fields beyond those listed."""

    @staticmethod
    def _writing_prompt_prompt(topic: str, seg: str, segment_guide: str, min_words: int) -> str:
        return f"""Generate ONE writing task about "{topic}".

Audience calibration: {segment_guide}

Return ONLY this exact JSON object:
{{
  "scenario": "<background context or situation>",
  "task": "<specific writing instruction - what to write>",
  "requirements": ["<requirement 1>", "<requirement 2>", "<requirement 3>"],
  "minWords": {min_words},
  "maxWords": {min_words * 3},
  "tips": ["<helpful tip 1>", "<helpful tip 2>"]
}}

Hard rules:
- "scenario" must be a real, relatable situation for {seg} (not "Write about X" as a sentence).
- "task" must be a specific, achievable instruction.
- "requirements" must be 3 to 5 CONCRETE, CHECKABLE items (e.g., "Include a greeting", "Mention a deadline", "Use at least 2 adjectives"). They cannot be generic like "Be clear".
- "tips" must be 2 to 4 concrete, practical pieces of advice.
- Every string field must contain real content. No placeholders."""

    @staticmethod
    def _speaking_prompt_prompt(topic: str, seg: str, segment_guide: str, min_duration: int) -> str:
        return f"""Generate ONE speaking exercise about "{topic}" for a {seg} student.

This exercise must give the student something SPECIFIC to talk about — not a vague open prompt. The student should be able to start speaking immediately without having to invent their own scenario.

Audience calibration: {segment_guide}

Return ONLY this exact JSON object:
{{
  "taskType": "<one of: read_aloud | describe | opinion | narrate>",
  "instruction": "<1-2 sentences telling the user exactly what to do, in the second person ('You will...', 'Imagine that...')>",
  "content": "<the EXACT text to read aloud (for read_aloud) OR a 2-4 sentence concrete scenario with specific names, places, and details (for describe/opinion/narrate). Never just a vague topic.>",
  "minDurationSeconds": {min_duration},
  "tips": ["<practical tip 1>", "<practical tip 2>", "<practical tip 3>"]
}}

Hard rules — these are critical:
- "taskType" must be exactly one of: "read_aloud", "describe", "opinion", "narrate".
- "content" must be CONCRETE and ACTIONABLE. The user should be able to start talking right away. BAD: "Speak about a group project." GOOD: "Imagine you are Maya. Your class has to do a science project on plants. You and your friend Sam need to choose between growing bean seeds or studying a tree. Talk for 30 seconds about which option you would pick and why."
- For "read_aloud": content is the exact text the user will read. It must be {min_duration * 2} to {min_duration * 3} words long, age-appropriate for {seg}, and from a real passage (a short story, a poem, a news snippet, a dialogue from a school textbook). It must NOT be a generic filler sentence.
- For "describe" / "opinion" / "narrate": content must give 2-4 specific points the user is expected to cover. Example structure: "Tell the speaker about: (1) where you went, (2) what you saw, (3) who you went with, (4) why it was fun."
- "instruction" must be in the second person and tell the user what to do in 1-2 sentences. No "Imagine if..." without concrete follow-up.
- "tips" must be 2-4 practical speaking tips specific to the task (e.g., "Pause briefly after each point", "Speak at a steady pace", "Use complete sentences").
- Every string field must contain real content. No placeholders, no ellipses."""

    @staticmethod
    def _scenario_prompt_prompt(topic: str, seg: str, segment_guide: str) -> str:
        return f"""Set up ONE conversation scenario about "{topic}" for a {seg} student.

The AI must DRIVE the conversation. The first message must put the situation into action with a concrete hook, NOT a meta-question like "What would you like to talk about?" The user should be able to respond to the situation immediately.

Audience calibration: {segment_guide}

Return ONLY this exact JSON object:
{{
  "scenarioRole": "<the AI character - be specific, e.g., 'a friendly librarian named Mrs. Johnson'>",
  "scenarioDescription": "<clear 1-3 sentence description of the situation>",
  "userRole": "<who the user is in this scenario>",
  "firstMessage": "<the AI's first IN-CHARACTER message that puts the situation into motion — a concrete event, question, or statement the user can respond to immediately. NOT a meta question like 'What would you like to talk about?'. Example for a coffee shop scenario: 'Welcome to Bean & Brew! What can I get started for you today?'>",
  "firstMessageOptions": [
    "<a natural in-character user reply the user could tap>",
    "<a different natural in-character user reply>",
    "<a third natural in-character user reply>"
  ],
  "totalTurns": 6,
  "tips": ["<practical tip 1>", "<practical tip 2>", "<practical tip 3>"]
}}

Hard rules:
- "scenarioRole" must be a specific named character (not "a person"). Example: "Mrs. Johnson, the school librarian", "Sam, the barista at Bean & Brew".
- "scenarioDescription" must be 1-3 sentences explaining the situation and the user's goal in it.
- "firstMessage" must be IN CHARACTER and put the situation into motion — a concrete event, action, or specific question. NEVER meta-questions like "What do you want to practice?", "How can I help you today?" (without context), or "What specific scenarios would you like to work on?". The user must be able to respond to the situation immediately.
- "firstMessageOptions" must be 2-4 natural in-character reply suggestions the user can tap. Each must be a real, plausible opening reply the user could give. NOT "Hi" or "Hello" alone — make them scenario-specific and varied (e.g., for a coffee shop: "I'd like a latte, please.", "Can I see the menu?", "How much is a cappuccino?").
- "userRole" must be a clear role for the user.
- "totalTurns" must be an integer between 4 and 8.
- The scenario must be about "{topic}" specifically.
- For YoungLearner: simple, friendly characters (teacher, shopkeeper, friend). Use short sentences and age-appropriate language. The firstMessageOptions should also be short.
- For MiddleSchool: everyday contexts (asking for directions, ordering food, joining a club).
- For HighSchool: school activities, part-time jobs, social situations.
- For College/WorkingProfessional: business meetings, client calls, academic discussions."""

    # ── Validators ───────────────────────────────────────────────────────────

    @staticmethod
    def _has_real_text(s: object, min_len: int = 2) -> bool:
        if not isinstance(s, str):
            return False
        t = s.strip()
        if len(t) < min_len:
            return False
        # Reject obvious placeholders and stub patterns
        bad = (
            "...",
            "tbd",
            "todo",
            "n/a",
            "none",
            "placeholder",
            "<",
            ">",
            "option 1", "option 2", "option 3", "option 4",
            "choice a", "choice b", "choice c", "choice d",
            "question 1", "question 2", "word 1", "word 2",
            "lorem ipsum", "sample text", "your text here",
            "your answer", "enter your answer",
        )
        low = t.lower()
        return not any(b in low for b in bad)

    @staticmethod
    def _validate_mcq(data) -> str | None:
        if not isinstance(data, dict):
            return "response is not a JSON object"
        q = data.get("question")
        if not CommAgent._has_real_text(q, 10):
            return "missing or invalid 'question'"
        opts = data.get("options")
        if not isinstance(opts, list) or len(opts) != 4:
            return f"'options' must be a list of exactly 4 items (got {type(opts).__name__ if opts else 'None'})"
        cleaned = []
        for o in opts:
            if not isinstance(o, dict):
                return "each option must be an object"
            txt = o.get("text") or o.get("option") or ""
            if not CommAgent._has_real_text(txt, 1):
                return f"option text is empty or invalid: {o}"
            is_corr = o.get("isCorrect")
            if is_corr is None:
                is_corr = o.get("is_correct") or o.get("correct")
            if is_corr is None:
                return f"option missing isCorrect: {o}"
            cleaned.append({"text": str(txt).strip(), "isCorrect": bool(is_corr)})
        if sum(1 for o in cleaned if o["isCorrect"]) != 1:
            return "exactly one option must have isCorrect: true"
        if not CommAgent._has_real_text(data.get("explanation"), 10):
            return "missing or invalid 'explanation'"
        return None

    @staticmethod
    def _validate_fill_blank(data) -> str | None:
        if not isinstance(data, dict):
            return "response is not a JSON object"
        if not CommAgent._has_real_text(data.get("question"), 10):
            return "missing or invalid 'question' (must contain the blank ____)"
        q = data.get("question", "")
        if "____" not in q and "_____" not in q:
            return "'question' must contain a blank marker ____"
        if not CommAgent._has_real_text(data.get("correctAnswer"), 1):
            return "missing or invalid 'correctAnswer'"
        dist = data.get("distractors")
        if not isinstance(dist, list) or len(dist) < 3:
            return "'distractors' must be a list of at least 3 plausible wrong words"
        for d in dist:
            if not CommAgent._has_real_text(d, 1):
                return f"distractor is empty or invalid: {d}"
        if not CommAgent._has_real_text(data.get("hint"), 3):
            return "missing or invalid 'hint'"
        if not CommAgent._has_real_text(data.get("explanation"), 10):
            return "missing or invalid 'explanation'"
        return None

    @staticmethod
    def _validate_sentence_reorder(data) -> str | None:
        if not isinstance(data, dict):
            return "response is not a JSON object"
        words = data.get("words")
        if not isinstance(words, list) or not (5 <= len(words) <= 7):
            return f"'words' must be a list of 5-7 items (got {len(words) if isinstance(words, list) else 'invalid'})"
        cleaned_words = []
        for w in words:
            if not isinstance(w, str) or not w.strip():
                return f"invalid word in array: {w}"
            # Strip punctuation for matching (commas, periods, question marks)
            stripped = w.strip().lower().rstrip(",.?!;:").lstrip("\"'(").rstrip("\"')")
            if not stripped:
                return f"word has no alphanumeric content: {w}"
            cleaned_words.append(stripped)
        if not CommAgent._has_real_text(data.get("correctSentence"), 5):
            return "missing or invalid 'correctSentence'"

        # Extract sentence tokens (lowercased, punctuation-stripped)
        sentence_tokens = [
            t.lower().rstrip(",.?!;:").lstrip("\"'(").rstrip("\"')")
            for t in data["correctSentence"].split()
            if t.strip()
        ]
        sentence_tokens = [t for t in sentence_tokens if t]

        # Words in `words` should be a permutation of sentence tokens
        if sorted(cleaned_words) != sorted(sentence_tokens):
            return (
                f"'words' tokens don't match 'correctSentence' tokens "
                f"(words={cleaned_words}, sentence_tokens={sentence_tokens})"
            )

        # If LLM provided correctOrder, sanity-check it. Otherwise we'll compute it.
        order = data.get("correctOrder")
        if order is not None:
            if not isinstance(order, list) or len(order) != len(words):
                return f"'correctOrder' must have the same length as 'words' ({len(words)})"
            if set(order) != set(range(len(words))):
                return "'correctOrder' must be a permutation of [0, n)"
        return None

    @staticmethod
    def _compute_sentence_order(words: list, correct_sentence: str) -> list:
        """Compute the correctOrder array mapping each sentence position to its word index.

        Returns a list `order` such that ``" ".join(words[i] for i in order)``
        reproduces correct_sentence (modulo capitalization and punctuation).
        Greedy left-to-right match with backtracking if needed.
        """
        sentence_tokens = [
            t.lower().rstrip(",.?!;:").lstrip("\"'(").rstrip("\"')")
            for t in correct_sentence.split()
            if t.strip()
        ]
        sentence_tokens = [t for t in sentence_tokens if t]
        cleaned_words = [
            w.strip().lower().rstrip(",.?!;:").lstrip("\"'(").rstrip("\"')")
            for w in words
        ]

        order: list = []
        used = set()
        for token in sentence_tokens:
            # Find first unused word index whose cleaned form matches
            chosen = None
            for idx, w in enumerate(cleaned_words):
                if idx in used:
                    continue
                if w == token:
                    chosen = idx
                    break
            if chosen is None:
                # Fallback: pick any unused index (this should not happen if validator passed)
                for idx in range(len(words)):
                    if idx not in used:
                        chosen = idx
                        break
            if chosen is None:
                # Total fallback — pad with remaining indices
                chosen = 0
            order.append(chosen)
            used.add(chosen)
        return order


    @staticmethod
    def _validate_writing_prompt(data) -> str | None:
        if not isinstance(data, dict):
            return "response is not a JSON object"
        if not CommAgent._has_real_text(data.get("scenario"), 10):
            return "missing or invalid 'scenario'"
        if not CommAgent._has_real_text(data.get("task"), 5):
            return "missing or invalid 'task'"
        reqs = data.get("requirements")
        if not isinstance(reqs, list) or not (3 <= len(reqs) <= 5):
            return f"'requirements' must be a list of 3-5 items (got {len(reqs) if isinstance(reqs, list) else 'invalid'})"
        for r in reqs:
            if not CommAgent._has_real_text(r, 3):
                return f"invalid requirement: {r}"
        mw = data.get("minWords")
        if not isinstance(mw, int) or mw < 5:
            return "'minWords' must be a positive integer"
        if not CommAgent._has_real_text(data.get("tips", [""])[0] if isinstance(data.get("tips"), list) and data.get("tips") else data.get("tips"), 3):
            return "missing or invalid 'tips'"
        return None

    @staticmethod
    def _validate_speaking(data) -> str | None:
        if not isinstance(data, dict):
            return "response is not a JSON object"
        tt = data.get("taskType")
        if tt not in ("read_aloud", "describe", "opinion", "narrate"):
            return f"'taskType' must be one of read_aloud/describe/opinion/narrate (got {tt})"
        if not CommAgent._has_real_text(data.get("instruction"), 5):
            return "missing or invalid 'instruction'"
        if not CommAgent._has_real_text(data.get("content"), 5):
            return "missing or invalid 'content'"
        md = data.get("minDurationSeconds")
        if not isinstance(md, int) or md < 3:
            return "'minDurationSeconds' must be a positive integer"
        tips = data.get("tips")
        if not isinstance(tips, list) or len(tips) < 2:
            return "'tips' must be a list of at least 2 items"
        return None

    @staticmethod
    def _validate_scenario(data) -> str | None:
        if not isinstance(data, dict):
            return "response is not a JSON object"
        if not CommAgent._has_real_text(data.get("scenarioRole"), 3):
            return "missing or invalid 'scenarioRole'"
        if not CommAgent._has_real_text(data.get("scenarioDescription"), 10):
            return "missing or invalid 'scenarioDescription'"
        if not CommAgent._has_real_text(data.get("userRole"), 2):
            return "missing or invalid 'userRole'"
        first_msg = data.get("firstMessage", "")
        if not CommAgent._has_real_text(first_msg, 5):
            return "missing or invalid 'firstMessage'"
        # Reject meta-questions that put the cognitive load on the user.
        bad_openers = (
            "what would you like to talk about",
            "what do you want to talk about",
            "what specific scenarios",
            "how can i help you today",
            "what can i help you with",
            "what would you like to practice",
            "what would you like to discuss",
        )
        if any(b in first_msg.lower() for b in bad_openers):
            return f"firstMessage is a meta-question, not an in-character opening: {first_msg!r}. Rewrite to put the situation in motion."
        # Validate firstMessageOptions (preset reply chips)
        opts = data.get("firstMessageOptions")
        if not isinstance(opts, list) or len(opts) < 2:
            return "'firstMessageOptions' must be a list of at least 2 in-character reply suggestions"
        for o in opts:
            if not CommAgent._has_real_text(o, 3):
                return f"firstMessageOptions contains empty/placeholder item: {o!r}"
        tt = data.get("totalTurns")
        if not isinstance(tt, int) or not (4 <= tt <= 8):
            return f"'totalTurns' must be an integer 4-8 (got {tt})"
        return None

    # ── Normalizers (return canonical camelCase) ────────────────────────────

    @staticmethod
    def _normalize_mcq(data) -> dict:
        options = []
        for o in data["options"]:
            txt = o.get("text") or o.get("option") or ""
            is_corr = o.get("isCorrect")
            if is_corr is None:
                is_corr = o.get("is_correct") or o.get("correct")
            oid = o.get("id") or "a"
            options.append({"id": str(oid), "text": str(txt).strip(), "isCorrect": bool(is_corr)})
        return {
            "question": str(data["question"]).strip(),
            "options": options,
            "explanation": str(data.get("explanation", "")).strip(),
            "topic": str(data.get("topic", "")).strip(),
        }

    @staticmethod
    def _normalize_fill_blank(data) -> dict:
        return {
            "question": str(data["question"]).strip(),
            "correctAnswer": str(data["correctAnswer"]).strip(),
            "distractors": [str(d).strip() for d in data.get("distractors", [])][:3],
            "hint": str(data.get("hint", "")).strip(),
            "explanation": str(data.get("explanation", "")).strip(),
        }

    @staticmethod
    def _normalize_sentence_reorder(data) -> dict:
        words = [str(w).strip() for w in data["words"]]
        correct_sentence = str(data["correctSentence"]).strip()
        order = data.get("correctOrder")
        if not isinstance(order, list) or len(order) != len(words):
            order = CommAgent._compute_sentence_order(words, correct_sentence)
        return {
            "question": str(data.get("question", "Arrange these words to form a correct sentence:")).strip(),
            "words": words,
            "correctOrder": [int(i) for i in order],
            "correctSentence": correct_sentence,
        }

    @staticmethod
    def _validate_word_learn(data) -> str | None:
        if not isinstance(data, dict):
            return "response is not a JSON object"
        word = data.get("word")
        if not isinstance(word, str) or not word.strip() or len(word.strip().split()) > 1:
            return f"'word' must be a single non-empty string (got {word!r})"
        if not CommAgent._has_real_text(data.get("definition"), 5):
            return "missing or invalid 'definition'"
        if not CommAgent._has_real_text(data.get("exampleSentence"), 10):
            return "missing or invalid 'exampleSentence'"
        # The example must actually contain the target word (case-insensitive)
        word_lower = word.strip().lower()
        example = str(data["exampleSentence"]).lower()
        # Strip punctuation from example for the containment check
        example_clean = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in example)
        if word_lower not in example_clean.split():
            return f"'exampleSentence' must contain the word '{word}' (got {data['exampleSentence']!r})"
        if not CommAgent._has_real_text(data.get("question"), 10):
            return "missing or invalid 'question'"
        if not CommAgent._has_real_text(data.get("explanation"), 10):
            return "missing or invalid 'explanation'"
        options = data.get("options")
        if not isinstance(options, list) or len(options) != 4:
            return f"'options' must be a list of exactly 4 items (got {len(options) if isinstance(options, list) else 'invalid'})"
        seen_texts = set()
        for i, o in enumerate(options):
            if not isinstance(o, dict):
                return f"option {i} is not an object"
            text = o.get("text")
            if not CommAgent._has_real_text(text, 5):
                return f"option {i} has invalid text"
            norm = str(text).strip().lower()
            if norm in seen_texts:
                return f"option {i} text is a duplicate: {text!r}"
            seen_texts.add(norm)
        correct_count = sum(1 for o in options if isinstance(o, dict) and o.get("isCorrect") is True)
        if correct_count != 1:
            return f"exactly one option must be isCorrect: true (got {correct_count})"
        # The correct option must also contain the word
        correct_opt = next((o for o in options if isinstance(o, dict) and o.get("isCorrect") is True), None)
        if correct_opt is not None:
            co_text = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in str(correct_opt.get("text", "")).lower())
            if word_lower not in co_text.split():
                return f"the correct option must contain the word '{word}' (got {correct_opt.get('text')!r})"
        return None

    @staticmethod
    def _normalize_word_learn(data) -> dict:
        options = []
        for o in data["options"]:
            options.append({
                "id": str(o.get("id", "")).strip() or "x",
                "text": str(o["text"]).strip(),
                "isCorrect": bool(o.get("isCorrect") is True),
            })
        return {
            "word": str(data["word"]).strip(),
            "definition": str(data["definition"]).strip(),
            "exampleSentence": str(data["exampleSentence"]).strip(),
            "question": str(data.get("question", "")).strip(),
            "options": options,
            "explanation": str(data.get("explanation", "")).strip(),
            "topic": str(data.get("topic", "")).strip(),
        }

    _VALID_READING_SKILLS = {"main_idea", "detail", "inference", "vocabulary_in_context", "tone", "purpose"}

    @staticmethod
    def _validate_reading(data) -> str | None:
        if not isinstance(data, dict):
            return "response is not a JSON object"
        if not CommAgent._has_real_text(data.get("title"), 3):
            return "missing or invalid 'title'"
        passage = data.get("passage")
        if not isinstance(passage, str) or len(passage.split()) < 50:
            wc = len(passage.split()) if isinstance(passage, str) else 0
            return f"'passage' must be at least 50 words (got {wc})"
        if not CommAgent._has_real_text(data.get("topic"), 3):
            return "missing or invalid 'topic'"
        questions = data.get("questions")
        if not isinstance(questions, list) or not (2 <= len(questions) <= 3):
            return f"'questions' must be a list of 2-3 items (got {len(questions) if isinstance(questions, list) else 'invalid'})"
        seen_qids = set()
        for qi, q in enumerate(questions):
            if not isinstance(q, dict):
                return f"question {qi} is not an object"
            qid = str(q.get("id", "")).strip()
            if not qid:
                return f"question {qi} is missing 'id'"
            if qid in seen_qids:
                return f"question id is duplicated: {qid}"
            seen_qids.add(qid)
            if not CommAgent._has_real_text(q.get("question"), 10):
                return f"question {qi} has invalid 'question' text"
            if not CommAgent._has_real_text(q.get("explanation"), 10):
                return f"question {qi} has invalid 'explanation'"
            skill = str(q.get("skill", "")).strip()
            if skill not in CommAgent._VALID_READING_SKILLS:
                return f"question {qi} has invalid 'skill': {skill!r} (must be one of {sorted(CommAgent._VALID_READING_SKILLS)})"
            opts = q.get("options")
            if not isinstance(opts, list) or len(opts) != 4:
                return f"question {qi} options must be a list of exactly 4 (got {len(opts) if isinstance(opts, list) else 'invalid'})"
            seen_texts = set()
            for oi, o in enumerate(opts):
                if not isinstance(o, dict):
                    return f"question {qi} option {oi} is not an object"
                text = o.get("text")
                if not CommAgent._has_real_text(text, 3):
                    return f"question {qi} option {oi} has invalid text"
                norm = str(text).strip().lower()
                if norm in seen_texts:
                    return f"question {qi} option {oi} text is a duplicate: {text!r}"
                seen_texts.add(norm)
            correct_count = sum(1 for o in opts if isinstance(o, dict) and o.get("isCorrect") is True)
            if correct_count != 1:
                return f"question {qi} must have exactly 1 correct option (got {correct_count})"
        return None

    @staticmethod
    def _normalize_reading(data) -> dict:
        questions = []
        for q in data["questions"]:
            options = []
            for o in q["options"]:
                options.append({
                    "id": str(o.get("id", "")).strip() or "x",
                    "text": str(o["text"]).strip(),
                    "isCorrect": bool(o.get("isCorrect") is True),
                })
            questions.append({
                "id": str(q["id"]).strip(),
                "question": str(q["question"]).strip(),
                "options": options,
                "explanation": str(q.get("explanation", "")).strip(),
                "skill": str(q.get("skill", "detail")).strip(),
            })
        return {
            "title": str(data["title"]).strip(),
            "passage": str(data["passage"]).strip(),
            "questions": questions,
            "topic": str(data.get("topic", "")).strip(),
        }

    @staticmethod
    def _normalize_writing_prompt(data) -> dict:
        return {
            "scenario": str(data["scenario"]).strip(),
            "task": str(data["task"]).strip(),
            "requirements": [str(r).strip() for r in data.get("requirements", [])],
            "minWords": int(data["minWords"]),
            "maxWords": int(data.get("maxWords", data["minWords"] * 3)),
            "tips": [str(t).strip() for t in data.get("tips", [])],
        }

    @staticmethod
    def _normalize_speaking(data) -> dict:
        return {
            "taskType": str(data["taskType"]),
            "instruction": str(data["instruction"]).strip(),
            "content": str(data["content"]).strip(),
            "minDurationSeconds": int(data["minDurationSeconds"]),
            "tips": [str(t).strip() for t in data.get("tips", [])],
        }

    @staticmethod
    def _normalize_scenario(data) -> dict:
        return {
            "scenarioRole": str(data["scenarioRole"]).strip(),
            "scenarioDescription": str(data["scenarioDescription"]).strip(),
            "userRole": str(data.get("userRole", "yourself")).strip(),
            "firstMessage": str(data["firstMessage"]).strip(),
            "firstMessageOptions": [str(o).strip() for o in data.get("firstMessageOptions", [])][:4],
            "totalTurns": int(data["totalTurns"]),
            "tips": [str(t).strip() for t in data.get("tips", [])],
        }

    async def evaluate_writing(self, req: EvaluateWritingRequest) -> FeedbackReport:
        """Evaluate a writing response and return a FeedbackReport."""
        system = _build_system_prompt(
            req.user_segment,
            req.placement_level.value,
            "Writing",
            3,
        )

        prompt = f"""Evaluate this writing response.

Prompt given to user: {req.prompt_text}
User's response: {req.user_response}
Minimum word count required: {req.min_words}
User segment: {req.user_segment.value}
Placement level: {req.placement_level.value}
Scenario context: {req.scenario_context or 'General'}

Scoring calibration:
- YoungLearner: Focus on basic grammar only. Be very encouraging. Simple explanations.
- MiddleSchool: Grammar + basic clarity. Friendly tone. Explain errors simply.
- HighSchool: Grammar + clarity + structure. Balanced feedback.
- College: Grammar + clarity + argument structure + academic tone.
- WorkingProfessional: Grammar + clarity + tone appropriateness + conciseness + professional register.

Quality rules:
- "errors" must list ONLY real, verifiable errors found in the user's response. Each "original" must be a substring (or close approximation) of the user's text. "corrected" must be a real rewrite. "explanation" must be a real grammatical reason. Do not invent errors that are not in the text.
- If you cannot find a real error in a category, return an empty array for that category — never fabricate errors.
- "improvement_suggestions" must be 2-4 concrete, actionable tips (e.g., "Use past tense for events that already happened", not generic "be clearer").
- "model_answer" must be a complete rewrite of the user's response that addresses the prompt.
- "encouragement" must be a specific positive observation about this response, not a generic "good job".

Return ONLY this exact JSON object (no markdown, no extra text):
{{
  "overall_score": <0-100>,
  "grammar_score": <0-100>,
  "clarity_score": <0-100>,
  "tone_score": <0-100>,
  "errors": [
    {{"original": "...", "corrected": "...", "explanation": "...", "error_type": "grammar|spelling|punctuation|word_choice|structure"}}
  ],
  "improvement_suggestions": ["...", "..."],
  "model_answer": "<rewritten version>",
  "encouragement": "<age-appropriate positive message>"
}}"""

        def _validate(data) -> str | None:
            if not isinstance(data, dict):
                return "response is not a JSON object"
            for k in ("overall_score", "grammar_score", "clarity_score", "tone_score"):
                v = data.get(k)
                if v is not None and not isinstance(v, (int, float)):
                    return f"'{k}' must be a number 0-100"
                if v is not None and not (0 <= v <= 100):
                    return f"'{k}' must be in range 0-100"
            errs = data.get("errors")
            if errs is not None and not isinstance(errs, list):
                return "'errors' must be a list"
            return None

        raw_text = await asyncio.to_thread(_retry_llm_sync, system, prompt, _validate)
        data = _parse_json_response(raw_text)
        try:
            return FeedbackReport.model_validate(data)
        except ValidationError as e:
            logger.error(f"Writing feedback validation failed: {e}")
            raise ValueError(f"LLM response does not match FeedbackReport schema: {e}")

    async def evaluate_speaking(self, req: EvaluateSpeakingRequest) -> FeedbackReport:
        """Evaluate a speaking transcription and return a FeedbackReport."""
        system = _build_system_prompt(
            req.user_segment,
            req.placement_level.value,
            "Speaking",
            3,
        )

        safe_transcription = json.dumps(req.transcription)[1:-1]

        prompt = f"""Evaluate this speaking response.

Task given to user: {req.task_description}
Transcription of user's speech: {safe_transcription}
User segment: {req.user_segment.value}

Evaluate:
- Fluency: pace, hesitations, filler words (um, uh, like)
- Pronunciation: based on transcription clarity
- Content: did they address the task?

Quality rules:
- "word_corrections" must list ONLY words the user actually said that were mispronounced, misused, or could be improved. Each entry must reference a real word or phrase from the transcription.
- Do not invent corrections for words that do not appear in the transcription.
- "improvement_suggestions" must be 2-4 concrete speaking tips.
- "encouragement" must be specific to this response.

Return ONLY this exact JSON object:
{{
  "overall_score": <0-100>,
  "fluency_score": <0-100>,
  "pronunciation_score": <0-100>,
  "transcription": {json.dumps(safe_transcription)},
  "word_corrections": [
    {{"word": "...", "suggestion": "...", "explanation": "..."}}
  ],
  "improvement_suggestions": ["...", "..."],
  "encouragement": "<age-appropriate positive message>"
}}"""

        def _validate(data) -> str | None:
            if not isinstance(data, dict):
                return "response is not a JSON object"
            for k in ("overall_score", "fluency_score", "pronunciation_score"):
                v = data.get(k)
                if v is not None and not isinstance(v, (int, float)):
                    return f"'{k}' must be a number 0-100"
            wc = data.get("word_corrections")
            if wc is not None and not isinstance(wc, list):
                return "'word_corrections' must be a list"
            return None

        raw_text = await asyncio.to_thread(_retry_llm_sync, system, prompt, _validate)
        data = _parse_json_response(raw_text)
        try:
            return FeedbackReport.model_validate(data)
        except ValidationError as e:
            logger.error(f"Speaking feedback validation failed: {e}")
            raise ValueError(f"LLM response does not match FeedbackReport schema: {e}")

    async def scenario_turn(self, req: ScenarioTurnRequest) -> ScenarioTurnResponse:
        """Generate AI response for a scenario dialogue turn."""
        system = _build_system_prompt(
            req.user_segment,
            "Intermediate",  # scenarios don't use placement_level directly
            "Scenarios",
            3,
        )

        history_text = "\n".join(
            f"{'User' if turn.get('role') == 'user' else 'AI'}: {turn.get('message', '')}"
            for turn in req.history
        )

        prompt = f"""You are playing the role of: {req.scenario_role}
Scenario: {req.scenario_description}
User segment: {req.user_segment.value}
This is turn {req.turn_number} of {req.total_turns}.

Conversation so far:
{history_text}

User's latest message: {req.user_message}

Rules:
- Stay completely in character as {req.scenario_role}.
- Respond naturally as that person would in this situation.
- Keep language complexity appropriate for {req.user_segment.value}.
- If this is the final turn (turn {req.turn_number} == {req.total_turns}), wrap up the conversation naturally.
- Do NOT break character or give feedback during the scenario.
- Do NOT include placeholders, ellipses, or "..." in your response.
- The `ai_response` field must contain a complete, natural-sounding reply (at least one full sentence).

Return JSON:
{{
  "ai_response": "<your in-character response>",
  "is_complete": {'true' if req.turn_number >= req.total_turns else 'false'}
}}"""

        def _validate_scenario_turn(data) -> str | None:
            if not isinstance(data, dict):
                return "response is not a JSON object"
            if not CommAgent._has_real_text(data.get("ai_response"), 10):
                return "missing or invalid 'ai_response'"
            if not isinstance(data.get("is_complete"), bool):
                return "'is_complete' must be a boolean"
            return None

        data = await asyncio.to_thread(_retry_llm_sync, system, prompt, _validate_scenario_turn)
        try:
            return ScenarioTurnResponse.model_validate(data)
        except ValidationError as e:
            logger.error(f"Scenario turn validation failed: {e}")
            raise ValueError(f"LLM response does not match ScenarioTurnResponse schema: {e}")

    async def scenario_end_feedback(self, req: ScenarioEndRequest) -> FeedbackReport:
        """Evaluate the full scenario conversation and return a FeedbackReport."""
        system = _build_system_prompt(
            req.user_segment,
            "Intermediate",
            "Scenarios",
            3,
        )

        history_text = "\n".join(
            f"{'User' if turn.get('role') == 'user' else 'AI ({req.scenario_role})'}: {turn.get('message', '')}"
            for turn in req.full_history
        )

        prompt = f"""Evaluate the user's communication in this completed scenario.

Scenario: {req.scenario_description}
Role they were speaking with: {req.scenario_role}
User segment: {req.user_segment.value}

Full conversation:
{history_text}

Evaluate the user's messages only (not the AI's responses):
- Clarity: were their messages clear and easy to understand?
- Confidence: were they assertive and direct?
- Professionalism: was their register and tone appropriate?
- Identify the weakest turn and provide a model response for it.

Rules:
- Every score must be a number between 0 and 100 (inclusive). Do NOT return null, missing, or out-of-range.
- All string fields must contain REAL content (no placeholders, no ellipses, no "...").
- `key_strengths`, `areas_to_improve`, and `improvement_suggestions` must each have at least 1 concrete item.
- `model_answer` must be a complete, natural-sounding rewrite of the user's weakest turn.
- `encouragement` must be a sincere, specific positive closing (not generic platitudes).

Return JSON:
{{
  "overall_score": <0-100>,
  "clarity_score": <0-100>,
  "confidence_score": <0-100>,
  "professionalism_score": <0-100>,
  "errors": [],
  "key_strengths": ["...", "..."],
  "areas_to_improve": ["...", "..."],
  "improvement_suggestions": ["...", "..."],
  "model_answer": "<model response for the weakest turn>",
  "encouragement": "<positive closing message>"
}}"""

        def _validate_scenario_feedback(data) -> str | None:
            if not isinstance(data, dict):
                return "response is not a JSON object"
            for key in ("overall_score", "clarity_score", "confidence_score", "professionalism_score"):
                v = data.get(key)
                if not isinstance(v, (int, float)) or not (0 <= v <= 100):
                    return f"'{key}' must be a number between 0 and 100 (got {v!r})"
            for key in ("key_strengths", "areas_to_improve", "improvement_suggestions"):
                arr = data.get(key)
                if not isinstance(arr, list) or len(arr) < 1:
                    return f"'{key}' must be a non-empty list of strings"
                for item in arr:
                    if not CommAgent._has_real_text(item, 5):
                        return f"'{key}' contains a placeholder/empty item: {item!r}"
            if not CommAgent._has_real_text(data.get("model_answer"), 10):
                return "missing or invalid 'model_answer'"
            if not CommAgent._has_real_text(data.get("encouragement"), 10):
                return "missing or invalid 'encouragement'"
            return None

        data = await asyncio.to_thread(_retry_llm_sync, system, prompt, _validate_scenario_feedback)
        try:
            return FeedbackReport.model_validate(data)
        except ValidationError as e:
            logger.error(f"Scenario end feedback validation failed: {e}")
            raise ValueError(f"LLM response does not match FeedbackReport schema: {e}")

    async def run_onboarding_assessment(self, req: OnboardingAssessRequest) -> PlacementResult:
        """Score onboarding answers and determine placement levels per domain."""
        # Deterministic scoring — no LLM needed
        domain_scores: dict[str, list[bool]] = {}
        for answer in req.answers:
            domain_key = answer.domain.value
            if domain_key not in domain_scores:
                domain_scores[domain_key] = []
            domain_scores[domain_key].append(answer.is_correct)

        placement_levels: dict[str, str] = {}
        initial_tiers: dict[str, int] = {}

        for domain in SkillDomain:
            domain_key = domain.value
            scores = domain_scores.get(domain_key, [])
            if not scores:
                placement_levels[domain_key] = PlacementLevel.BEGINNER.value
                initial_tiers[domain_key] = 1
                continue

            accuracy = sum(scores) / len(scores)
            if accuracy >= 0.67:
                placement_levels[domain_key] = PlacementLevel.ADVANCED.value
                initial_tiers[domain_key] = 3
            elif accuracy >= 0.34:
                placement_levels[domain_key] = PlacementLevel.INTERMEDIATE.value
                initial_tiers[domain_key] = 2
            else:
                placement_levels[domain_key] = PlacementLevel.BEGINNER.value
                initial_tiers[domain_key] = 1

        return PlacementResult(
            placement_levels=placement_levels,
            initial_difficulty_tiers=initial_tiers,
        )


# Singleton instance
_comm_agent: Optional[CommAgent] = None


def get_comm_agent() -> CommAgent:
    """Return singleton CommAgent instance."""
    global _comm_agent
    if _comm_agent is None:
        _comm_agent = CommAgent()
    return _comm_agent