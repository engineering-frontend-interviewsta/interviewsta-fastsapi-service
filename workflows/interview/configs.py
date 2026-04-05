import random
from typing import Any, Dict, List, Optional

from workflows.interview.phase_engine import PhaseConfig


# =============================================================================
# Per-interview-type greeting prompts
# Pass as background["greeting_prompt"] in the initial invoke state.
# The engine uses this instead of the generic GREETING_PROMPT when present.
# =============================================================================

DEBATE_GREETING_PROMPT = (
    "Your name is Glee and you are a debate moderator conducting a live debate practice session.\n"
    "Respond in plain, conversational prose — no markdown, no bullet points.\n\n"
    "Your instructions:\n"
    "1. Start with a warm, friendly greeting and introduce yourself as the debate moderator.\n"
    "2. Explain the format: there will be a structured debate on a single topic with 3-4 rounds "
    "of back-and-forth arguments.\n"
    "3. Present ONE clear debate motion related to technology, AI, or corporate/business topics. "
    "The motion should be general and accessible. Generate a UNIQUE topic each time — do NOT reuse "
    "the same motions across sessions. Examples of categories to draw from (but create your own unique motion):\n"
    "   - AI impact: 'AI will replace more jobs than it creates'\n"
    "   - Work culture: 'Remote work is better than office work'\n"
    "   - Tech regulation: 'Social media companies should be more regulated'\n"
    "   - Data privacy: 'Privacy is more important than personalisation'\n"
    "   Do NOT ask the candidate to propose a topic. You must choose it.\n"
    "4. Ask the candidate to choose a side (for or against) and briefly state their initial position.\n"
    "5. Keep the tone supportive and encouraging — this is safe practice, not an exam.\n"
    "6. Ask if they have any questions about the format before you begin."
)

CASE_STUDY_GREETING_PROMPT = (
    "Your name is Glee and you are conducting a case-study based live interview session.\n"
    "Respond in plain, conversational prose — no markdown, no bullet points.\n\n"
    "Your instructions:\n"
    "1. Start with a warm, friendly greeting and introduce yourself as the interviewer.\n"
    "2. Explain the format: you will present a business case study problem and the focus is on "
    "the candidate's thought process and structured problem-solving approach, not just the final answer. "
    "Encourage them to think aloud.\n"
    "3. Explicitly ask if they have any questions ONLY about the process before you begin.\n"
    "4. After addressing questions (or if none), indicate you are ready to present the case."
)

COMMUNICATION_GREETING_PROMPT = (
    "Your name is Glee and you are conducting a communication skills interview session.\n"
    "Respond in plain, conversational prose — no markdown, no bullet points.\n\n"
    "Your instructions:\n"
    "1. Start with a warm, friendly greeting and introduce yourself.\n"
    "2. Explain the format: the session has four parts — a brief personal conversation, "
    "a speaking exercise, a writing comprehension task, and vocabulary MCQs. "
    "The focus is on communication skills.\n"
    "3. Explicitly ask if they have any questions about the process before you begin.\n"
    "4. After addressing questions (or if none), say you would like to start with a brief "
    "conversation to get to know them better."
)


# =============================================================================
# 1 — CODING INTERVIEW
# =============================================================================

CODING_INTERVIEW_PHASES = [

    # ── Rapport ───────────────────────────────────────────────────────────────
    PhaseConfig(
        phase_name="Rapport",
        order=1,
        prompt=(
            "Your name is Glee, SDE at {company}. You are conducting a coding interview.\n"
            "Your ONLY job in this phase is to have a warm personalised conversation "
            "to get to know the candidate — ask their name, education, hobbies, and "
            "journey into tech. Aim for 6-7 exchanges.\n\n"
            "IMPORTANT: Do NOT mention, hint at, or discuss any coding problems, "
            "algorithms, or technical questions in this phase. The coding assessment "
            "comes later. Keep the conversation entirely personal and rapport-building.\n\n"
            "Respond in plain conversational prose, no markdown."
        ),
        prompt_inputs=["background"],
        route_nodes=["Rapport", "Theoretical_before", "Offensive"],
        route_ahead_prompt=(
            "You are supervising the rapport/personalisation phase of a coding interview.\n"
            "The interviewer is getting to know the candidate personally before any technical assessment.\n\n"
            "Route to 'Rapport' if:\n"
            "  - Fewer than 6 personal conversational exchanges have occurred, OR\n"
            "  - The candidate has not yet explicitly confirmed they are ready to move on\n\n"
            "Route to 'Theoretical_before' ONLY when:\n"
            "  - At least 6 personal exchanges have occurred (name, background, hobbies, etc.), AND\n"
            "  - The candidate has clearly confirmed readiness ('yes', 'ready', 'sure', 'let's go', 'yup')\n\n"
            "CRITICAL: If the interviewer has presented a coding problem or algorithm question "
            "during this phase, that is wrong — still route to 'Rapport' to continue personalisation.\n\n"
            "Route to 'Offensive' only if the candidate is being rude or unserious."
        ),
    ),

    # ── Theoretical (deferred: generated from Coding questions on first entry) ─
    PhaseConfig(
        phase_name="Theoretical",
        order=2,
        prompt=(
            "You are a senior technical interviewer probing foundational understanding.\n"
            "You are on question {current_question_number} of {total_questions}.\n"
            "Ask the following theoretical/conceptual question. "
            "Do NOT reveal the specific coding problems that come later.\n\n"
            "Current question:\n{questions}\n\n"
            "Conversation so far:\n{history}\n\n"
            "Respond in plain conversational prose."
        ),
        prompt_inputs=["questions", "history"],
        number_of_questions_to_ask=2,
        setup_questions=True,
        setup_questions_prompt=(
            "You are a senior technical interviewer preparing theoretical warm-up questions.\n\n"
            "The candidate will solve these coding problems in the next phase:\n{coding_questions}\n\n"
            "Generate exactly 2-3 short theoretical/conceptual questions that test "
            "FOUNDATIONAL KNOWLEDGE required to solve those problems (data structures, "
            "algorithms, time/space complexity). Do NOT give away the solution or hint "
            "at the specific problem.\n\n"
            "Return a JSON array of objects with keys:\n"
            "  title        (short label, e.g. 'Sliding Window')\n"
            "  description  (the full conversational question to ask the candidate)\n"
            "  difficulty   (Easy | Medium | Hard)\n\n"
            "Return ONLY the JSON array, no markdown, no preamble."
        ),
        question_filters={"type": "theoretical"},
        route_nodes=["Theoretical", "Coding_before", "Offensive"],
        route_ahead_prompt=(
            "You are supervising the theoretical question phase of a coding interview.\n\n"
            "Route to 'Theoretical' if:\n"
            "  - The current theoretical question is still being discussed, OR\n"
            "  - The candidate's answer needs more probing or follow-up\n\n"
            "Route to 'Coding_before' when ALL theoretical questions are fully addressed "
            "and both parties are ready to begin the coding problems.\n\n"
            "Route to 'Offensive' if the candidate is being rude or unserious."
        ),
    ),

    # ── Coding (questions fetched from Question table) ────────────────────────
    PhaseConfig(
        phase_name="Coding",
        order=3,
        prompt=(
            "You are a technical interviewer guiding a live coding session at {company}.\n\n"
            "All coding problems for this session (for your reference):\n{all_questions}\n\n"
            "You are currently on problem {current_question_number} of {total_questions}:\n{questions}\n\n"
            "Guide the candidate through this problem:\n"
            "  1. Present the problem clearly (use the description; raw_content has full HTML detail).\n"
            "  2. Ask the candidate to explain the problem back in their own words.\n"
            "  3. Discuss their approach before coding.\n"
            "  4. Ask them to write the solution.\n"
            "  5. Probe edge cases and time/space complexity.\n"
            "  6. Once satisfied, explicitly say you are done with this problem and ready to move on.\n\n"
            "Respond in plain conversational prose, no markdown."
        ),
        prompt_inputs=["questions", "background"],
        number_of_questions_to_ask=2,
        setup_questions=False,
        question_filters={"use_db_questions": True},
        route_nodes=["Coding", "End", "Offensive"],
        route_ahead_prompt=(
            "Route to 'Coding' while any coding problem is still being worked on.\n"
            "Route to 'End' ONLY when ALL problems are fully complete: solution written "
            "and reviewed, edge cases discussed, complexity confirmed, and the interviewer "
            "has explicitly indicated they are done.\n"
            "Route to 'Offensive' if the candidate is being rude or unserious."
        ),
    ),
]


# =============================================================================
# 2 — CASE STUDY INTERVIEW
# =============================================================================

# All practice cases from case_study.py — stored here so the prompt generator
# can pick one at phase_state injection time.
PRACTICE_CASES = {
    "ecommerce_revenue_drop": {
        "case": (
            "A D2C company selling skincare products saw monthly revenue drop by 20% in the last 3 months, "
            "even though website traffic is stable. Analyze this situation and provide recommendations."
        ),
        "interaction": (
            "Key areas to explore:\n"
            "- Conversion funnel metrics (add-to-cart rate, checkout completion rate)\n"
            "- Product pricing changes or competitor analysis\n"
            "- Customer reviews and satisfaction scores\n"
            "- Shipping costs or delivery times\n"
            "- Website performance and checkout process\n"
            "- Marketing campaign effectiveness\n"
            "- Seasonality factors"
        ),
    },
    "food_delivery_expansion": {
        "case": (
            "A food delivery startup operating in 5 cities wants to expand to 20 cities in the next year. "
            "What factors should they consider and how should they prioritize?"
        ),
        "interaction": (
            "Key considerations:\n"
            "- Market size and demand analysis for each city\n"
            "- Operational infrastructure (riders, restaurants, logistics)\n"
            "- Unit economics and profitability per city\n"
            "- Competition landscape\n"
            "- Regulatory requirements\n"
            "- Technology scalability\n"
            "- Marketing and customer acquisition costs"
        ),
    },
    "subscription_churn": {
        "case": (
            "A SaaS company has a 5% monthly churn rate. They want to reduce it to 3%. "
            "What would be your approach?"
        ),
        "interaction": (
            "Areas to investigate:\n"
            "- Customer segmentation (who is churning?)\n"
            "- Reasons for cancellation (survey data, exit interviews)\n"
            "- Product usage patterns before churn\n"
            "- Customer success team effectiveness\n"
            "- Onboarding experience quality\n"
            "- Pricing and value perception\n"
            "- Feature gaps vs. competitors"
        ),
    },
    "retail_store_location": {
        "case": (
            "A retail chain wants to open 10 new stores. "
            "How would you help them decide which locations to choose?"
        ),
        "interaction": (
            "Evaluation factors:\n"
            "- Demographics (population, income levels, age distribution)\n"
            "- Foot traffic and accessibility\n"
            "- Competition density\n"
            "- Real estate costs (rent, maintenance)\n"
            "- Local regulations and permits\n"
            "- Parking availability\n"
            "- Proximity to complementary businesses"
        ),
    },
    "mobile_app_engagement": {
        "case": (
            "A social media app has 1 million downloads but only 100K monthly active users. "
            "How would you improve engagement?"
        ),
        "interaction": (
            "Investigation areas:\n"
            "- User activation and onboarding flow\n"
            "- Core value proposition clarity\n"
            "- Feature adoption rates\n"
            "- Push notification strategy\n"
            "- Content quality and relevance\n"
            "- Performance and technical issues\n"
            "- Comparison with competitor apps\n"
            "- User feedback and reviews"
        ),
    },
    "marketplace_liquidity": {
        "case": (
            "A two-sided marketplace connecting freelancers and clients is struggling with "
            "supply-demand imbalance. Too many freelancers, not enough clients. What should they do?"
        ),
        "interaction": (
            "Strategies to consider:\n"
            "- Client acquisition channels and cost\n"
            "- Value proposition for clients\n"
            "- Quality control for freelancers\n"
            "- Pricing strategy adjustment\n"
            "- Geographic or category focus\n"
            "- Marketing spend allocation\n"
            "- Platform fees structure\n"
            "- Success stories and social proof"
        ),
    },
    "product_pricing": {
        "case": (
            "An ed-tech company currently charges ₹999/month. They're considering changing to ₹9999/year. "
            "How would you evaluate this decision?"
        ),
        "interaction": (
            "Analysis framework:\n"
            "- Current customer LTV and payback period\n"
            "- Cash flow implications\n"
            "- Customer preference research\n"
            "- Churn impact modeling\n"
            "- Competitive pricing analysis\n"
            "- Unit economics comparison\n"
            "- Implementation and communication plan"
        ),
    },
    "logistics_optimization": {
        "case": (
            "An e-commerce company's average delivery time is 4 days. They want to reduce it to 2 days "
            "while maintaining profitability. What's your approach?"
        ),
        "interaction": (
            "Optimization levers:\n"
            "- Warehouse network expansion\n"
            "- Inventory placement strategy\n"
            "- Carrier partnerships and SLAs\n"
            "- Cost-benefit analysis per region\n"
            "- Technology (route optimization, predictive algorithms)\n"
            "- Customer willingness to pay for speed\n"
            "- Impact on customer satisfaction and retention"
        ),
    },
    "content_platform_monetization": {
        "case": (
            "A content platform with 5M monthly users is currently free. "
            "They want to introduce monetization. What options should they consider?"
        ),
        "interaction": (
            "Monetization models:\n"
            "- Subscription (freemium vs. paywall)\n"
            "- Advertising (display, native, sponsored content)\n"
            "- Transaction fees (marketplace model)\n"
            "- Hybrid approach\n"
            "- User segmentation for pricing\n"
            "- Impact on user growth\n"
            "- Competitive landscape"
        ),
    },
    "customer_acquisition": {
        "case": (
            "A fintech app is spending ₹500 to acquire each customer but LTV is only ₹400. "
            "How would you address this?"
        ),
        "interaction": (
            "Solutions to explore:\n"
            "- Improve LTV (increase engagement, cross-sell, reduce churn)\n"
            "- Reduce CAC (optimize marketing channels, referrals, virality)\n"
            "- Target different customer segments\n"
            "- Adjust product pricing\n"
            "- Focus on retention vs acquisition\n"
            "- Unit economics by channel analysis"
        ),
    },
    "market_entry": {
        "case": (
            "A successful Indian startup wants to expand to Southeast Asia. "
            "What framework would you use to evaluate this decision?"
        ),
        "interaction": (
            "Evaluation criteria:\n"
            "- Market size and growth potential\n"
            "- Competitive landscape\n"
            "- Regulatory environment\n"
            "- Cultural and consumer behavior differences\n"
            "- Go-to-market strategy\n"
            "- Resource requirements\n"
            "- Risk assessment\n"
            "- ROI projections"
        ),
    },
    "feature_prioritization": {
        "case": (
            "A product manager has 5 features to build but can only do 2 this quarter. "
            "How should they decide?"
        ),
        "interaction": (
            "Prioritization framework:\n"
            "- Impact on key metrics (engagement, revenue, retention)\n"
            "- Engineering effort and complexity\n"
            "- Customer pain point severity\n"
            "- Strategic alignment\n"
            "- Competitive necessity\n"
            "- Dependencies\n"
            "- RICE or similar scoring"
        ),
    },
    "crisis_management": {
        "case": (
            "A food delivery app had a data breach exposing customer payment information. "
            "How should they respond?"
        ),
        "interaction": (
            "Response plan:\n"
            "- Immediate containment and assessment\n"
            "- Customer communication strategy\n"
            "- Legal and regulatory compliance\n"
            "- PR and brand damage control\n"
            "- Compensation and goodwill measures\n"
            "- Long-term security improvements\n"
            "- Stakeholder management (investors, partners)"
        ),
    },
    "partnership_evaluation": {
        "case": (
            "An e-commerce company is considering partnering with a major retailer vs. building their own brand. "
            "How would you evaluate?"
        ),
        "interaction": (
            "Comparison factors:\n"
            "- Speed to market\n"
            "- Brand control and positioning\n"
            "- Economics (margins, revenue share)\n"
            "- Customer data access\n"
            "- Long-term strategic value\n"
            "- Resource requirements\n"
            "- Risk allocation"
        ),
    },
    "operational_efficiency": {
        "case": (
            "A restaurant chain's food cost is 40% of revenue (industry standard is 30%). "
            "How would you bring it down?"
        ),
        "interaction": (
            "Cost reduction levers:\n"
            "- Menu engineering and optimization\n"
            "- Supplier negotiations and sourcing\n"
            "- Portion control and waste reduction\n"
            "- Inventory management\n"
            "- Seasonal menu adjustments\n"
            "- Staff training on preparation\n"
            "- Technology for demand forecasting"
        ),
    },
}


def _pick_random_case() -> tuple:
    """Return (case_question, case_reference) for a randomly selected practice case."""
    key = random.choice(list(PRACTICE_CASES.keys()))
    entry = PRACTICE_CASES[key]
    return entry["case"], entry["interaction"]


CASE_STUDY_INTERVIEW_PHASES = [

    # ── CaseStudy (single phase — greeting is handled by the engine's fixed Greeting node) ──
    PhaseConfig(
        phase_name="CaseStudy",
        order=1,
        prompt=(
            "Your name is Glee and you are conducting a case-study based live interview session.\n"
            "Speak naturally and conversationally in plain prose — no bullet points, no markdown.\n\n"
            "CASE QUESTION:\n{case_question}\n\n"
            "CASE REFERENCE (use this to guide the discussion and ask cross-questions):\n{case_reference}\n\n"
            "Your instructions:\n"
            "1. Present the case question clearly and invite the candidate to ask any clarifying questions.\n"
            "2. Hold a thorough, conversational discussion using the CASE REFERENCE as your guide.\n"
            "3. Ask probing cross-questions to test depth of thinking.\n"
            "4. Do NOT give away the reference points — let the candidate discover them through reasoning.\n"
            "5. Once the discussion is sufficiently complete and you are satisfied, "
            "explicitly sign off: 'Thank you for your time, that concludes our case interview.'"
        ),
        prompt_inputs=[],          # case_question and case_reference come from phase_state
        route_nodes=["CaseStudy", "End", "Offensive"],
        route_ahead_prompt=(
            "You are supervising a case-study interview.\n\n"
            "Route to 'CaseStudy' if the case discussion is still ongoing — the candidate "
            "has not fully worked through the problem, or follow-up questions remain.\n\n"
            "Route to 'End' ONLY after the interviewer has EXPLICITLY signed off "
            "(e.g. 'that concludes our case interview', 'thank you, we're done').\n\n"
            "Route to 'Offensive' if the candidate is being rude or unserious."
        ),
    ),
]


def get_case_study_initial_phase_state() -> dict:
    """
    Call this when building the initial state for a case study session.
    Picks a random case and returns it pre-loaded into phase_state so the
    CaseStudy prompt can reference {case_question} and {case_reference}.

    Usage in Colab::

        extras = get_case_study_initial_phase_state()
        result = agent.invoke({
            "messages": [], "history": "", "LastNode": "",
            "phase_questions": {}, "phase_question_idx": {},
            "background": {"name": "Case Study Interview"},
            "phase_state": extras,
        }, config=THREAD)
    """
    case_question, case_reference = _pick_random_case()
    return {
        # The CaseStudy phase node looks up phase_state["CaseStudy"] for extra ctx
        "CaseStudy": {
            "case_question":   case_question,
            "case_reference":  case_reference,
        }
    }


# =============================================================================
# 3 — DEBATE INTERVIEW
# =============================================================================

DEBATE_INTERVIEW_PHASES = [

    # ── Debate ────────────────────────────────────────────────────────────────
    PhaseConfig(
        phase_name="Debate",
        order=1,
        prompt=(
            "You are continuing a structured debate practice session with a student.\n"
            "You are the OPPONENT — take the OPPOSITE position from what the student chose.\n\n"
            "Read the conversation history to identify:\n"
            "1. What is the debate motion/topic?\n"
            "2. Which side did the student choose (FOR or AGAINST)?\n"
            "You must take the OPPOSITE side and argue for it.\n\n"
            "Rules:\n"
            "- DEBATE, don't just ask questions. Present YOUR arguments and rebuttals.\n"
            "- Structure each response: (a) briefly acknowledge their point, "
            "(b) present your counter-argument with reasoning, (c) support with examples.\n"
            "- Keep responses 4-6 sentences.\n"
            "- Early rounds (1-2): opening arguments. Middle rounds (3-4): direct rebuttal.\n"
            "- Final round: ask for their closing statement, then provide yours.\n"
            "- After closing statements, transition to moderator mode:\n"
            "  'Thank you both. Now let me step back as moderator and provide a summary and feedback.'\n\n"
            "DO NOT start a new topic after closing statements.\n\n"
            "Conversation history:\n{history}"
        ),
        prompt_inputs=["history"],
        route_nodes=["Debate", "Summary_before", "Offensive"],
        route_ahead_prompt=(
            "You are supervising a debate practice session.\n\n"
            "Route to 'Debate' if the debate is still in progress and there is room "
            "for more arguments, rebuttals, or the closing statements have not yet been made.\n\n"
            "Route to 'Summary_before' when:\n"
            "  - Both sides have provided closing statements, AND\n"
            "  - The interviewer has explicitly transitioned to moderator mode "
            "(e.g. 'let me step back as moderator', 'let me provide a summary').\n\n"
            "HARD LIMIT: If more than 8 debate exchanges have occurred "
            "(check state phase_state.Debate.rounds_completed), route to 'Summary_before' regardless.\n\n"
            "Route to 'Offensive' if the candidate is being abusive or clearly unserious."
        ),
    ),

    # ── Summary (fires once, no human reply needed — ends the interview) ───────
    PhaseConfig(
        phase_name="Summary",
        order=2,
        prompt=(
            "You are now acting as a DEBATE MODERATOR providing comprehensive post-debate feedback.\n\n"
            "Your role has changed from opponent to neutral, constructive moderator.\n\n"
            "Provide all of the following in natural conversational prose (no bullet points):\n\n"
            "1. DEBATE SUMMARY (2-3 sentences): Topic debated, key arguments from both sides, "
            "overall flow.\n\n"
            "2. WINNER DECLARATION (2-3 sentences): Who won and specifically WHY — "
            "which arguments were stronger.\n\n"
            "3. STUDENT'S STRENGTHS (2-3 points): Specific good points the student made "
            "and why they were effective.\n\n"
            "4. STUDENT'S WEAKNESSES (2-3 points): Where their arguments could have been "
            "stronger and what was missing.\n\n"
            "5. IMPROVEMENT SUGGESTIONS (2-3 points): Concrete stronger arguments they "
            "could have used, actionable advice for future debates.\n\n"
            "Tone: encouraging, constructive, educational.\n"
            "End by thanking the student and wishing them well.\n\n"
            "Debate history to analyse:\n{history}"
        ),
        prompt_inputs=["history"],
        route_nodes=["End"],       # Summary always goes to End, no loop
        route_ahead_prompt=(
            "The summary has been delivered. Route to 'End'."
        ),
    ),
]


# =============================================================================
# 4 — ROLE-BASED INTERVIEW  (factory function)
# =============================================================================

_ROLE_TECHNICAL_PROMPTS = {
    "Frontend Development": (
        "You are a Frontend Development interviewer assessing the candidate's knowledge.\n"
        "Ask 5-7 technical questions covering:\n"
        "- Core concepts: HTML semantics, CSS layout, JS fundamentals (closures, event loop, etc.)\n"
        "- Framework knowledge: React hooks, component lifecycle, state management\n"
        "- Best practices: performance, accessibility, security\n"
        "- Problem-solving: 'How would you optimise a slow React app?'\n\n"
        "For coding: focus on SYNTAX and CONCEPTS, NOT building full websites:\n"
        "- Small code snippets (debounce function, React hook, CSS layout)\n"
        "- Specific syntax questions ('Explain flexbox vs grid', 'What does z-index do?')\n"
        "- JavaScript concepts ('Explain closures', 'Difference between let/const/var')\n\n"
        "Respond in plain conversational prose."
    ),
    "Backend Development": (
        "You are a Backend Development interviewer assessing the candidate's knowledge.\n"
        "Ask 5-7 technical questions covering fundamentals appropriate for placement/junior interviews:\n"
        "- Core concepts: REST, HTTP methods/status codes, database normalisation, ACID properties\n"
        "- Architecture: MVC, microservices basics, middleware\n"
        "- Databases: SQL vs NoSQL, primary keys, simple SQL queries with JOIN/GROUP BY\n"
        "- Security: authentication, authorisation, SQL injection\n\n"
        "For coding: keep to college-level placement questions:\n"
        "- Simple SQL queries ('Find the second highest salary')\n"
        "- Basic API endpoint code\n"
        "- Simple utility functions (hash a password, validate input)\n\n"
        "Respond in plain conversational prose."
    ),
    "UI/UX Design": (
        "You are a UI/UX Design interviewer. This is a PURELY CONVERSATION-BASED interview.\n"
        "DO NOT ask any coding or programming questions.\n\n"
        "Ask 5-7 questions covering:\n"
        "- Design principles: visual hierarchy, colour theory, typography, spacing\n"
        "- User research methods: interviews, surveys, usability testing\n"
        "- Design tools: Figma, Sketch, Adobe XD workflows\n"
        "- Accessibility: WCAG guidelines, inclusive design\n"
        "- Problem-solving: 'How would you improve the UX of a checkout flow?'\n"
        "- Design process: discovery, ideation, prototyping, testing\n\n"
        "Respond in plain conversational prose."
    ),
    "AI/ML": (
        "You are an AI/ML interviewer assessing the candidate's knowledge.\n"
        "Ask 5-7 questions appropriate for entry-to-mid level positions:\n"
        "- Fundamentals: supervised/unsupervised/RL, overfitting, bias-variance tradeoff\n"
        "- Algorithms: decision trees, SVM, neural networks (conceptually)\n"
        "- Evaluation: accuracy, precision, recall, F1, AUC — when to use each\n"
        "- Practical: handling imbalanced data, feature scaling, missing values\n"
        "- Deep learning basics: backpropagation, CNNs, transformers conceptually\n\n"
        "For coding: simple ML snippets (train a linear regression, split train/test, "
        "calculate F1 score) — NOT full model pipelines.\n\n"
        "Respond in plain conversational prose."
    ),
    "Data Science": (
        "You are a Data Science interviewer assessing the candidate's knowledge.\n"
        "Ask 5-7 questions appropriate for entry-to-mid level positions:\n"
        "- Statistics: p-value, normal distribution, correlation vs causation, hypothesis testing\n"
        "- Data preprocessing: missing values, feature engineering, outliers\n"
        "- SQL: queries with GROUP BY, HAVING, JOIN, window functions\n"
        "- Python/Pandas: merge dataframes, groupby, pivot tables\n"
        "- Visualisation: histogram vs bar chart, choosing chart types\n"
        "- Analysis workflow: EDA steps, A/B testing basics\n\n"
        "For coding: SQL queries, Pandas operations, simple statistical calculations.\n\n"
        "Respond in plain conversational prose."
    ),
}

_ROLE_CODING_PROMPTS = {
    "Frontend Development": (
        "Present a Frontend coding challenge testing syntax/concepts (NOT a full website):\n"
        "- Write a debounce/throttle function\n"
        "- Implement a custom React hook\n"
        "- CSS layout challenge (flexbox or grid)\n"
        "- JavaScript problem (closure, prototype, async/await)\n\n"
        "Present the challenge clearly and allow the candidate to think and code."
    ),
    "Backend Development": (
        "Present a Backend coding challenge at college placement level:\n"
        "- SQL query (joins, aggregations, window functions)\n"
        "- Simple API endpoint implementation\n"
        "- Algorithm/data structure problem relevant to backend\n"
        "- Database schema design for a simple use case\n\n"
        "Present the challenge clearly and allow the candidate to think and code."
    ),
    "UI/UX Design": (
        # UI/UX skips coding — this prompt is never used but must exist
        "UI/UX Design does not have a coding phase. This phase is skipped."
    ),
    "AI/ML": (
        "Present an AI/ML coding challenge:\n"
        "- Write code to train a simple model (linear/logistic regression)\n"
        "- Write code to preprocess data (handle missing values, scale features)\n"
        "- Implement a simple evaluation metric\n"
        "- Write code to split data and cross-validate\n\n"
        "Present the challenge clearly and allow the candidate to think and code."
    ),
    "Data Science": (
        "Present a Data Science coding challenge:\n"
        "- Write SQL queries (aggregations, joins, ranking)\n"
        "- Write Pandas operations (merge, groupby, pivot)\n"
        "- Statistical calculation (confidence interval, hypothesis test setup)\n"
        "- EDA script for a given dataset description\n\n"
        "Present the challenge clearly and allow the candidate to think and code."
    ),
}


def make_role_based_phases(role: str) -> list:
    """
    Build PhaseConfig list for a role-based interview.

    Parameters
    ----------
    role : one of "Frontend Development", "Backend Development",
                  "UI/UX Design", "AI/ML", "Data Science"

    The role and resume are expected in state["background"]["role"] and
    state["background"]["resume"] respectively.

    UI/UX Design skips the Coding phase entirely.
    """
    if role not in _ROLE_TECHNICAL_PROMPTS:
        raise ValueError(
            f"Unknown role '{role}'. Choose from: {list(_ROLE_TECHNICAL_PROMPTS.keys())}"
        )

    is_uiux = role == "UI/UX Design"

    phases = [

        # ── Rapport ───────────────────────────────────────────────────────────
        PhaseConfig(
            phase_name="Rapport",
            order=1,
            prompt=(
                f"Your name is Glee and you are conducting a {role} interview session.\n"
                "Speak naturally and conversationally.\n\n"
                "Your job in this phase:\n"
                "1. Ask about their name (if not mentioned).\n"
                f"2. Ask why they chose to pursue {role} — what sparked their interest?\n"
                "3. Ask about their educational background and how it relates to the role.\n"
                "4. Ask about their motivations, interests, hobbies, and interesting projects.\n"
                "5. IMPORTANT: Do NOT ask about resume details. Get to know them through natural conversation.\n"
                f"6. After 6-7 exchanges, transition: 'Thank you for sharing! Now let's move on to the "
                f"{role} assessment. Are you ready to begin?'\n\n"
                "Respond in plain conversational prose, no markdown."
            ),
            prompt_inputs=["background"],
            route_nodes=["Rapport", "Technical_before", "Offensive"],
            route_ahead_prompt=(
                f"You are supervising the personalised rapport phase of a {role} interview.\n\n"
                "Route to 'Rapport' if fewer than 6 exchanges have occurred or the candidate "
                "has not confirmed readiness.\n\n"
                "Route to 'Technical_before' after ~6-7 exchanges AND the candidate confirms "
                "readiness ('yes', 'ready', 'sure', 'let's go').\n\n"
                "Route to 'Offensive' if rude or unserious."
            ),
        ),

        # ── Technical ─────────────────────────────────────────────────────────
        PhaseConfig(
            phase_name="Technical",
            order=2,
            prompt=_ROLE_TECHNICAL_PROMPTS[role],
            prompt_inputs=[],
            route_nodes=(
                ["Technical", "Project_before", "Offensive"]
                if is_uiux
                else ["Technical", "Coding_before", "Offensive"]
            ),
            route_ahead_prompt=(
                f"You are supervising the technical question phase of a {role} interview.\n\n"
                "Route to 'Technical' if fewer than 5 questions have been asked or the "
                "current question is still being discussed.\n\n"
                + (
                    "Route to 'Project_before' after 5-7 questions are complete and "
                    "both parties are ready to move on.\n\n"
                    if is_uiux else
                    "Route to 'Coding_before' after 5-7 questions are complete and "
                    "both parties are ready to move on.\n\n"
                )
                + "Route to 'Offensive' if rude or unserious."
            ),
        ),
    ]

    # ── Coding (skipped for UI/UX) ─────────────────────────────────────────
    if not is_uiux:
        phases.append(
            PhaseConfig(
                phase_name="Coding",
                order=3,
                prompt=_ROLE_CODING_PROMPTS[role],
                prompt_inputs=[],
                route_nodes=["Coding", "Project_before", "Offensive"],
                route_ahead_prompt=(
                    f"You are supervising the coding challenge phase of a {role} interview.\n\n"
                    "Route to 'Coding' if the challenge is still in progress.\n\n"
                    "Route to 'Project_before' once the coding challenge is complete "
                    "and both parties are ready to move on.\n\n"
                    "Route to 'Offensive' if rude or unserious."
                ),
            )
        )

    # ── Project ────────────────────────────────────────────────────────────
    project_order = 4 if not is_uiux else 3
    phases.append(
        PhaseConfig(
            phase_name="Project",
            order=project_order,
            prompt=(
                f"You are a {role} interviewer discussing the candidate's projects and experience.\n\n"
                "Ask 2-3 detailed questions about:\n"
                "- Specific projects they have worked on\n"
                "- Technologies and tools used\n"
                "- Challenges faced and how they solved them\n"
                "- Impact and outcomes\n"
                "- What they learned\n\n"
                "Resume context (if available):\n{{resume}}\n\n"
                "Respond in plain conversational prose, no markdown."
            ).replace("{{resume}}", "{resume}"),
            prompt_inputs=["background"],   # resume is in background["resume"]
            route_nodes=["Project", "End", "Offensive"],
            route_ahead_prompt=(
                f"You are supervising the project discussion phase of a {role} interview.\n\n"
                "Route to 'Project' if fewer than 2-3 project questions have been asked "
                "or the discussion is still ongoing.\n\n"
                "Route to 'End' once 2-3 project questions are complete and the interviewer "
                "has wrapped up.\n\n"
                "Route to 'Offensive' if rude or unserious."
            ),
        )
    )

    return phases


# =============================================================================
# 5 — COMMUNICATION INTERVIEW  (unchanged)
# =============================================================================

SPEAKING_ENTITY_SCHEMA = {
    "fields": [
        {"name": "instruction", "type": "str", "optional": True,
         "description": "Instruction for the speaking exercise"},
        {"name": "paragraph",   "type": "str", "optional": True,
         "description": "50-80 word paragraph for the candidate to read aloud"},
    ]
}

COMPREHENSION_ENTITY_SCHEMA = {
    "fields": [
        {"name": "instruction", "type": "str", "optional": True,
         "description": "Instruction for the writing comprehension task"},
        {"name": "question",    "type": "str", "optional": True,
         "description": "Scenario/question for 50-100 word written response"},
    ]
}

MCQ_ENTITY_SCHEMA = {
    "fields": [
        {"name": "instruction", "type": "str", "optional": True,
         "description": "Instruction for solving the MCQ"},
        {"name": "question",    "type": "str", "optional": True,
         "description": "Fill-in-the-blank question"},
        {"name": "options",     "type": "list", "optional": True,
         "description": "Exactly 4 answer options"},
        {"name": "answer",      "type": "str",  "optional": True,
         "description": "Correct answer — must match one of the options exactly"},
    ]
}

COMMUNICATION_INTERVIEW_PHASES = [
    PhaseConfig(
        phase_name="Rapport",
        order=1,
        prompt=(
            "Your name is Glee. Build rapport with the candidate over 2-3 exchanges: "
            "ask their name, interests, hobbies. After that explain the format "
            "(conversation → speaking → writing → MCQ) and ask if they have questions. "
            "Respond in plain prose."
        ),
        prompt_inputs=[],
        route_nodes=["Rapport", "PersonalDetails_before", "Offensive"],
        route_ahead_prompt=(
            "Route to 'Rapport' while still in greeting/questions phase. "
            "Route to 'PersonalDetails_before' once the format is explained and the "
            "candidate is ready to start the conversation about hobbies. "
            "Route to 'Offensive' if appropriate."
        ),
    ),
    PhaseConfig(
        phase_name="PersonalDetails",
        order=2,
        prompt=(
            "Have a warm 4-6 exchange conversation about the candidate's name, where they're from, "
            "hobbies, interests. Show genuine interest. After 4-6 exchanges ask: "
            "'Are you ready to begin the speaking exercise?' Respond in plain prose."
        ),
        prompt_inputs=[],
        route_nodes=["PersonalDetails", "Speaking_before", "Offensive"],
        route_ahead_prompt=(
            "Route to 'PersonalDetails' while the conversation is still ongoing (fewer than "
            "4-6 exchanges or candidate hasn't confirmed readiness). "
            "Route to 'Speaking_before' once the candidate confirms they're ready. "
            "Route to 'Offensive' if appropriate."
        ),
    ),
    PhaseConfig(
        phase_name="Speaking",
        order=3,
        prompt=(
            "Present a speaking exercise. Generate a coherent 50-80 word paragraph and ask "
            "the candidate to read it aloud word for word. Present ONLY the paragraph."
        ),
        prompt_inputs=[],
        number_of_questions_to_ask=1,
        special_output_format="json",
        entity_schema=SPEAKING_ENTITY_SCHEMA,
        immediate_feedback_required=True,
        feedback_prompt=(
            "The candidate was asked to speak this paragraph:\n{paragraph}\n\n"
            "Their transcription was:\n{answer}\n\n"
            "Provide 2-3 sentence constructive feedback on accuracy and strengths. "
            "End with: 'Are you ready to move on to the writing comprehension phase?'"
        ),
        route_nodes=["Speaking", "Comprehension_before", "Offensive"],
        route_ahead_prompt=(
            "Route to 'Speaking' if the paragraph hasn't been presented yet or the candidate "
            "hasn't submitted their response. "
            "Route to 'Comprehension_before' after feedback and the candidate confirms readiness. "
            "Route to 'Offensive' if appropriate."
        ),
    ),
    PhaseConfig(
        phase_name="Comprehension",
        order=4,
        prompt=(
            "Present a writing comprehension exercise. Generate a clear scenario and ask "
            "the candidate to write 50-100 words about it. Present the scenario directly."
        ),
        prompt_inputs=[],
        number_of_questions_to_ask=1,
        special_output_format="json",
        entity_schema=COMPREHENSION_ENTITY_SCHEMA,
        immediate_feedback_required=True,
        feedback_prompt=(
            "The writing task was:\n{question}\n\n"
            "The candidate's response was:\n{answer}\n\n"
            "Provide 2-3 sentence constructive feedback on relevance and writing quality. "
            "End with: 'Are you ready to move on to the vocabulary MCQ phase?'"
        ),
        route_nodes=["Comprehension", "MCQ_before", "Offensive"],
        route_ahead_prompt=(
            "Route to 'Comprehension' if the task hasn't been presented or no response yet. "
            "Route to 'MCQ_before' after feedback and candidate confirms readiness. "
            "Route to 'Offensive' if appropriate."
        ),
    ),
    PhaseConfig(
        phase_name="MCQ",
        order=5,
        prompt=(
            "Ask fill-in-the-blank vocabulary MCQ questions. Provide exactly 4 options each time. "
            "Ask exactly 4 unique questions total. "
            "After 4 questions, thank the candidate and state the interview is complete."
        ),
        prompt_inputs=[],
        number_of_questions_to_ask=4,
        special_output_format="json",
        entity_schema=MCQ_ENTITY_SCHEMA,
        route_nodes=["MCQ", "End", "Offensive"],
        route_ahead_prompt=(
            "Route to 'MCQ' while fewer than 4 questions have been asked. "
            "Route to 'End' once 4 questions are done or the interviewer has signed off. "
            "Route to 'Offensive' if appropriate."
        ),
    ),
]


# =============================================================================
# Greeting prompts (stored in Interview.greeting_prompt in the DB)
# =============================================================================

DEBATE_GREETING_PROMPT = (
    "Your name is Glee and you are a debate moderator for a live debate practice session. "
    "Respond in a single paragraph of plain prose, as if speaking aloud.\n\n"
    "Instructions:\n"
    "1. Greet the candidate warmly and introduce yourself as the debate moderator.\n"
    "2. Explain the format: 3-4 rounds of back-and-forth arguments on a single motion.\n"
    "3. Present ONE clear debate motion on a TECHNOLOGY, AI, or BUSINESS topic. "
    "Generate a unique motion each session — do NOT reuse examples. "
    "Categories: AI regulation, work culture, tech monopolies, data privacy, automation.\n"
    "4. Ask the candidate to choose a side (for or against) and briefly state their position.\n"
    "5. Keep the tone supportive — this is safe practice, not an exam."
)

CASE_STUDY_GREETING_PROMPT = (
    "Your name is Glee and you are conducting a case-study based live interview session. "
    "Respond in a single paragraph of plain prose, as if speaking aloud.\n\n"
    "Instructions:\n"
    "1. Greet the candidate warmly and introduce yourself.\n"
    "2. Explain the format: you will present a real business case and the focus is on "
    "structured thinking and problem-solving approach, not just the final answer.\n"
    "3. Encourage them to think out loud.\n"
    "4. Ask if they have any questions about the process before you begin.\n"
    "5. Answer any questions clearly, then indicate you are ready to present the case."
)

COMMUNICATION_GREETING_PROMPT = (
    "Your name is Glee and you are conducting a communication skills assessment. "
    "Respond in a single paragraph of plain prose, as if speaking aloud.\n\n"
    "Instructions:\n"
    "1. Greet the candidate warmly and introduce yourself.\n"
    "2. Explain the format: a brief personal conversation, then a speaking exercise, "
    "a writing comprehension task, and finally vocabulary MCQ questions.\n"
    "3. The focus is on communication skills throughout.\n"
    "4. Ask if they have any questions about the process.\n"
    "5. After answering questions (or if none), say you would like to start with a "
    "brief conversation to get to know them better."
)

CODING_GREETING_PROMPT = (
    "Your name is Glee, SDE at {company}. You are conducting a coding interview. "
    "Respond in a single paragraph of plain prose, as if speaking aloud.\n\n"
    "Instructions:\n"
    "1. Greet the candidate warmly and introduce yourself.\n"
    "2. Explain the format: a brief personalised conversation first, then conceptual "
    "warm-up questions, and finally live coding problems on {subject} at {difficulty} difficulty.\n"
    "3. The focus is on thought process and problem-solving approach.\n"
    "4. Encourage them to think out loud.\n"
    "5. Ask if they have any questions about the process before you begin."
)

ROLE_BASED_GREETING_PROMPT = (
    "Your name is Glee and you are conducting a {role} technical interview. "
    "Respond in a single paragraph of plain prose, as if speaking aloud.\n\n"
    "Instructions:\n"
    "1. Greet the candidate warmly and introduce yourself.\n"
    "2. Explain the format: a brief personal conversation to get to know them, "
    "then technical questions about {role}, a coding challenge (if applicable), "
    "and a discussion about their projects and experience.\n"
    "3. Encourage them to think out loud.\n"
    "4. Ask if they have any questions about the process.\n"
    "5. After questions (or if none), say you would like to start with a brief "
    "conversation to get to know them better."
)


# =============================================================================
# HR phases (lazy import — avoids loading workflows.hr / LangGraph at import time)
# =============================================================================

_hr_phases_cache: Optional[List[PhaseConfig]] = None


def _get_hr_interview_phases() -> List[PhaseConfig]:
    global _hr_phases_cache
    if _hr_phases_cache is None:
        from workflows.hr import hr_prompt as _hr_prompt

        _hr_phases_cache = [
            PhaseConfig(
                phase_name="HR",
                order=1,
                prompt=_hr_prompt,
                prompt_inputs=["background", "history"],
                route_nodes=["HR", "End", "Offensive"],
                route_ahead_prompt=(
                    "You supervise an HR behavioral interview.\n"
                    "Route to 'HR' while behavioral questions or follow-ups remain "
                    "(typically until roughly 5 questions are fully explored).\n"
                    "Route to 'End' when the interviewer is clearly closing and has thanked the candidate.\n"
                    "Route to 'Offensive' if the candidate is abusive or clearly unserious."
                ),
            ),
        ]
    return _hr_phases_cache


# =============================================================================
# Interview meta — one dict per interview type, used to insert Interview rows
# and build the initial state.  greeting_prompt is rendered at insert time
# (background fields like {company} are available from interview_meta).
# =============================================================================

def get_interview_config(
    interview_type: str,
    role: str = "AI/ML",
    company: str = "Google",
    subject: str = "Arrays",
    difficulty: str = "Medium",
    resume: str = "",
) -> dict:
    """
    Returns a fully-populated config dict for one interview session.

    Keys
    ----
    phases          : List[PhaseConfig]
    interview_meta  : dict — columns for the Interview DB row
    phase_state     : dict — pre-loaded phase_state (e.g. CaseStudy random case)
    extra_background: dict — extra keys merged into background (e.g. role, resume)

    Usage::

        cfg = get_interview_config("debate")
        agent = build_graph(cfg["phases"], ...)
        result = agent.invoke({
            "messages": [], "history": "", "LastNode": "",
            "phase_questions": {}, "phase_question_idx": {},
            "phase_state": cfg["phase_state"],
            "background": {**cfg["interview_meta"], **cfg["extra_background"]},
        }, config=THREAD)
    """
    t = interview_type.lower().strip()

    if t == "coding":
        meta = {
            "name":            f"{subject} {difficulty} Coding Interview",
            "difficulty":      difficulty,
            "company":         company,
            "subject":         subject,
            "tags":            "DSA, coding",
            "greeting_prompt": CODING_GREETING_PROMPT.format(
                company=company, subject=subject, difficulty=difficulty
            ),
        }
        return {
            "phases":           CODING_INTERVIEW_PHASES,
            "interview_meta":   meta,
            "phase_state":      {},
            "extra_background": {},
        }

    elif t == "debate":
        meta = {
            "name":            "Debate Interview",
            "difficulty":      "-",
            "company":         "-",
            "subject":         "-",
            "tags":            "debate",
            "greeting_prompt": DEBATE_GREETING_PROMPT,
        }
        return {
            "phases":           DEBATE_INTERVIEW_PHASES,
            "interview_meta":   meta,
            "phase_state":      {},
            "extra_background": {},
        }

    elif t == "case_study":
        meta = {
            "name":            "Case Study Interview",
            "difficulty":      "-",
            "company":         "-",
            "subject":         "-",
            "tags":            "case study",
            "greeting_prompt": CASE_STUDY_GREETING_PROMPT,
        }
        return {
            "phases":           CASE_STUDY_INTERVIEW_PHASES,
            "interview_meta":   meta,
            "phase_state":      get_case_study_initial_phase_state(),
            "extra_background": {},
        }

    elif t == "communication":
        meta = {
            "name":            "Communication Interview",
            "difficulty":      "-",
            "company":         "-",
            "subject":         "-",
            "tags":            "communication",
            "greeting_prompt": COMMUNICATION_GREETING_PROMPT,
        }
        return {
            "phases":           COMMUNICATION_INTERVIEW_PHASES,
            "interview_meta":   meta,
            "phase_state":      {},
            "extra_background": {},
        }

    elif t == "role_based":
        meta = {
            "name":            f"{role} Interview",
            "difficulty":      "Medium",
            "company":         "-",
            "subject":         role,
            "tags":            role,
            "greeting_prompt": ROLE_BASED_GREETING_PROMPT.format(role=role),
        }
        return {
            "phases":           make_role_based_phases(role),
            "interview_meta":   meta,
            "phase_state":      {},
            "extra_background": {"role": role, "resume": resume},
        }

    elif t == "hr":
        from workflows.hr import hr_greeting_prompt as _hr_greeting

        resume_val = resume or "No resume provided"
        meta = {
            "name":            "HR Interview",
            "difficulty":      "-",
            "company":         "-",
            "subject":         "-",
            "tags":            "hr",
            "greeting_prompt": _hr_greeting.format(resume=resume_val),
            "resume":          resume_val,
            "resume_text":     resume_val,
        }
        return {
            "phases":           _get_hr_interview_phases(),
            "interview_meta":   meta,
            "phase_state":      {},
            "extra_background": {},
        }

    else:
        raise ValueError(
            f"Unknown interview_type '{interview_type}'. "
            "Choose from: coding, hr, debate, case_study, communication, role_based"
        )


def _bundle_with_db_title(bundle: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    """Use interview_tests.title as interview_meta.name when POST /start merged it into payload."""
    t = (payload.get("interview_test_title") or "").strip()
    if t:
        meta = bundle.get("interview_meta")
        if isinstance(meta, dict):
            meta["name"] = t
    return bundle


def resolve_phase_engine_bundle(
    api_interview_type: str,
    payload: Optional[Dict[str, Any]] = None,
    role: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Map FastAPI / JWT interview type strings to a phase-engine bundle from get_interview_config.

    Returns the same dict as get_interview_config plus ``cache_key`` for graph compilation cache.
    """
    payload = dict(payload or {})
    it = (api_interview_type or "").strip()

    if it in ("Company", "Subject", "Technical"):
        company = (payload.get("company") or payload.get("Company") or "Google").strip() or "Google"
        subject = (payload.get("subject") or payload.get("Subject") or "Arrays").strip() or "Arrays"
        if it == "Technical":
            tags = payload.get("Tags")
            if isinstance(tags, list) and tags:
                subject = str(tags[0]).strip() or subject
        diff = payload.get("Difficulty") or payload.get("difficulty") or "Medium"
        resume = (payload.get("resume") or "") or ""
        bundle = get_interview_config(
            "coding",
            company=company,
            subject=str(subject),
            difficulty=str(diff),
            resume=resume,
        )
        bundle["cache_key"] = "phase:coding"
        return _bundle_with_db_title(bundle, payload)

    if it == "HR":
        bundle = get_interview_config(
            "hr",
            resume=(payload.get("resume") or "No resume provided"),
        )
        bundle["cache_key"] = "phase:hr"
        return _bundle_with_db_title(bundle, payload)

    if it == "CaseStudy":
        bundle = get_interview_config("case_study")
        bundle["cache_key"] = "phase:case_study"
        return _bundle_with_db_title(bundle, payload)

    if it == "Communication":
        bundle = get_interview_config("communication")
        bundle["cache_key"] = "phase:communication"
        return _bundle_with_db_title(bundle, payload)

    if it == "Debate":
        bundle = get_interview_config("debate")
        bundle["cache_key"] = "phase:debate"
        return _bundle_with_db_title(bundle, payload)

    if it == "Role-Based Interview":
        r = (role or payload.get("role") or "Frontend Development").strip() or "Frontend Development"
        bundle = get_interview_config(
            "role_based",
            role=r,
            resume=(payload.get("resume") or ""),
        )
        bundle["cache_key"] = f"phase:role_based:{r}"
        return _bundle_with_db_title(bundle, payload)

    raise ValueError(
        f"Unsupported interview type for phase engine: {api_interview_type!r}. "
        f"Expected one of: Company, Subject, Technical, HR, CaseStudy, Communication, "
        f"Debate, Role-Based Interview"
    )


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    print("=== Coding interview ===")
    for p in CODING_INTERVIEW_PHASES:
        src = (
            "DB" if p.question_filters.get("use_db_questions")
            else ("LLM-deferred" if p.setup_questions else "none")
        )
        print(f"  {p.order}. {p.phase_name:<16} questions={src}")

    print("\n=== Case Study interview ===")
    for p in CASE_STUDY_INTERVIEW_PHASES:
        print(f"  {p.order}. {p.phase_name}")
    cs = get_case_study_initial_phase_state()
    print(f"  Sample case: {list(cs['CaseStudy']['case_question'].split())[:8]}...")

    print("\n=== Debate interview ===")
    for p in DEBATE_INTERVIEW_PHASES:
        print(f"  {p.order}. {p.phase_name:<16} → {p.route_nodes}")

    print("\n=== Role-based interview (all roles) ===")
    for role in _ROLE_TECHNICAL_PROMPTS:
        phases = make_role_based_phases(role)
        names = [p.phase_name for p in phases]
        print(f"  {role:<25} phases={names}")

    print("\n=== Communication interview ===")
    for p in COMMUNICATION_INTERVIEW_PHASES:
        print(f"  {p.order}. {p.phase_name:<16} structured={p.special_output_format}")