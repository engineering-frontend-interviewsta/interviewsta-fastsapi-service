from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Annotated, Literal, List, Callable, TypeVar
from langgraph.checkpoint.memory import InMemorySaver
# from pydantic import field_validator, Field,
# from typing import List, Callable, TypeVar
import inspect
import operator
import random
import json
import faiss
from uuid import uuid4
import pickle
# from django.apps import apps

# ── Topic-keyed practice case bank ───────────────────────────────────────────
# Each topic has 2-3 cases. The old flat dict is preserved as a fallback
# reference at the bottom for backward compatibility.
PRACTICE_CASES: dict[str, list[dict]] = {
    "profitability": [
        {
            "id": "ecommerce_revenue_drop",
            "case": """A D2C company selling skincare products saw monthly revenue drop by 20% in the last 3 months,
even though website traffic is stable. Analyze this situation and provide recommendations.""",
            "interaction": """
Key areas to explore:
- Conversion funnel metrics (add-to-cart rate, checkout completion rate)
- Product pricing changes or competitor analysis
- Customer reviews and satisfaction scores
- Shipping costs or delivery times
- Website performance and checkout process
- Marketing campaign effectiveness
- Seasonality factors
""",
        },
        {
            "id": "restaurant_food_cost",
            "case": """A restaurant chain's food cost is 40% of revenue (industry standard is 30%).
How would you bring it down while maintaining quality?""",
            "interaction": """
Cost reduction levers:
- Menu engineering and optimization
- Supplier negotiations and sourcing
- Portion control and waste reduction
- Inventory management
- Seasonal menu adjustments
- Staff training on preparation
- Technology for demand forecasting
""",
        },
        {
            "id": "saas_unit_economics",
            "case": """A fintech app is spending ₹500 to acquire each customer but LTV is only ₹400.
How would you address this unit-economics problem?""",
            "interaction": """
Solutions to explore:
- Improve LTV (increase engagement, cross-sell, reduce churn)
- Reduce CAC (optimize marketing channels, referrals, virality)
- Target different customer segments
- Adjust product pricing
- Focus on retention vs acquisition
- Unit economics by channel analysis
""",
        },
    ],
    "market_entry": [
        {
            "id": "india_to_sea_expansion",
            "case": """A successful Indian startup wants to expand to Southeast Asia.
What framework would you use to evaluate this decision and which country would you recommend entering first?""",
            "interaction": """
Evaluation criteria:
- Market size and growth potential per country
- Competitive landscape
- Regulatory environment
- Cultural and consumer behavior differences
- Go-to-market strategy
- Resource requirements
- Risk assessment
- ROI projections
""",
        },
        {
            "id": "food_delivery_expansion",
            "case": """A food delivery startup operating in 5 cities wants to expand to 20 cities in the next year.
What factors should they consider and how should they prioritize which cities to enter?""",
            "interaction": """
Key considerations:
- Market size and demand analysis for each city
- Operational infrastructure (riders, restaurants, logistics)
- Unit economics and profitability per city
- Competition landscape
- Regulatory requirements
- Technology scalability
- Marketing and customer acquisition costs
""",
        },
        {
            "id": "retail_store_location",
            "case": """A retail chain wants to open 10 new stores across India. How would you help them decide which locations to choose?""",
            "interaction": """
Evaluation factors:
- Demographics (population, income levels, age distribution)
- Foot traffic and accessibility
- Competition density
- Real estate costs (rent, maintenance)
- Local regulations and permits
- Parking availability
- Proximity to complementary businesses
""",
        },
    ],
    "growth_strategy": [
        {
            "id": "mobile_app_engagement",
            "case": """A social media app has 1 million downloads but only 100K monthly active users.
How would you improve engagement and grow the active user base?""",
            "interaction": """
Investigation areas:
- User activation and onboarding flow
- Core value proposition clarity
- Feature adoption rates
- Push notification strategy
- Content quality and relevance
- Performance and technical issues
- Comparison with competitor apps
- User feedback and reviews
""",
        },
        {
            "id": "marketplace_liquidity",
            "case": """A two-sided marketplace connecting freelancers and clients is struggling with supply-demand imbalance.
Too many freelancers, not enough clients. What growth strategy would you recommend?""",
            "interaction": """
Strategies to consider:
- Client acquisition channels and cost
- Value proposition for clients
- Quality control for freelancers
- Pricing strategy adjustment
- Geographic or category focus
- Marketing spend allocation
- Platform fees structure
- Success stories and social proof
""",
        },
        {
            "id": "content_platform_monetization",
            "case": """A content platform with 5M monthly users is currently free.
They want to introduce monetization without losing their user base. What options should they consider?""",
            "interaction": """
Monetization models:
- Subscription (freemium vs. paywall)
- Advertising (display, native, sponsored content)
- Transaction fees (marketplace model)
- Hybrid approach
- User segmentation for pricing
- Impact on user growth
- Competitive landscape
""",
        },
    ],
    "mergers_acquisitions": [
        {
            "id": "edtech_acquisition",
            "case": """A large ed-tech company is considering acquiring a smaller competitor with 500K users and ₹20 Cr ARR.
How would you evaluate whether this acquisition makes sense?""",
            "interaction": """
Evaluation framework:
- Strategic rationale (market share, technology, talent)
- Revenue and cost synergies
- Valuation benchmarks (EV/ARR multiples)
- Cultural fit and integration risk
- Customer overlap and retention risk
- Regulatory considerations
- Alternative options (build vs. buy vs. partner)
""",
        },
        {
            "id": "partnership_evaluation",
            "case": """An e-commerce company is considering acquiring a logistics startup vs. building their own last-mile delivery.
How would you evaluate the build vs. buy decision?""",
            "interaction": """
Comparison factors:
- Speed to market
- Brand control and positioning
- Economics (margins, revenue share)
- Customer data access
- Long-term strategic value
- Resource requirements
- Risk allocation
""",
        },
    ],
    "pricing_strategy": [
        {
            "id": "subscription_pricing_shift",
            "case": """An ed-tech company currently charges ₹999/month. They're considering changing to ₹9999/year.
How would you evaluate this pricing change?""",
            "interaction": """
Analysis framework:
- Current customer LTV and payback period
- Cash flow implications
- Customer preference research
- Churn impact modeling
- Competitive pricing analysis
- Unit economics comparison
- Implementation and communication plan
""",
        },
        {
            "id": "saas_pricing_tiers",
            "case": """A B2B SaaS company charges a flat ₹5,000/month per seat. Competitors use usage-based pricing.
Should they switch pricing models, and if so, how?""",
            "interaction": """
Pricing model analysis:
- Customer willingness-to-pay segmentation
- Revenue predictability vs. growth upside
- Sales complexity and deal cycle impact
- Competitive positioning
- Migration plan for existing customers
- Impact on LTV and churn
""",
        },
    ],
    "operations": [
        {
            "id": "logistics_optimization",
            "case": """An e-commerce company's average delivery time is 4 days. They want to reduce it to 2 days
while maintaining profitability. What's your approach?""",
            "interaction": """
Optimization levers:
- Warehouse network expansion
- Inventory placement strategy
- Carrier partnerships and SLAs
- Cost-benefit analysis per region
- Technology (route optimization, predictive algorithms)
- Customer willingness to pay for speed
- Impact on customer satisfaction and retention
""",
        },
        {
            "id": "call_center_efficiency",
            "case": """A telecom company's customer service call center has average handle time of 12 minutes (industry benchmark: 7 minutes).
How would you reduce it without hurting customer satisfaction?""",
            "interaction": """
Process improvement areas:
- Call categorization and routing
- Agent training and knowledge base quality
- Self-service deflection (IVR, chatbot)
- First-call resolution rate
- Technology and CRM tooling
- Incentive structure for agents
- Root cause of long calls
""",
        },
    ],
    "competitive_response": [
        {
            "id": "new_entrant_disruption",
            "case": """A well-funded startup has entered your client's market with 50% lower prices and is growing fast.
Your client is a market leader with 40% share. How should they respond?""",
            "interaction": """
Response strategies:
- Competitive mapping (who is the new entrant targeting?)
- Defensive moves (loyalty programs, contracts, switching costs)
- Offensive moves (price matching in select segments, product differentiation)
- War-gaming the competitor's next moves
- Cost structure analysis (can you match their pricing?)
- Customer retention vs. acquisition trade-off
""",
        },
        {
            "id": "crisis_management",
            "case": """A food delivery app had a data breach exposing customer payment information.
A competitor is now running ads targeting your users. How should they respond?""",
            "interaction": """
Response plan:
- Immediate containment and assessment
- Customer communication strategy
- Legal and regulatory compliance
- PR and brand damage control
- Compensation and goodwill measures
- Long-term security improvements
- Competitive counter-messaging
""",
        },
    ],
    "digital_transformation": [
        {
            "id": "bank_digital_migration",
            "case": """A traditional bank has 80% of transactions still happening at branches. They want to migrate 60% to digital channels in 3 years.
How would you approach this transformation?""",
            "interaction": """
Transformation framework:
- Customer segmentation by digital readiness
- Channel economics (cost per transaction: branch vs. digital)
- Technology investment (app, UX, backend)
- Change management and employee retraining
- Regulatory compliance
- Risk of customer attrition during migration
- Phased rollout plan
""",
        },
        {
            "id": "feature_prioritization",
            "case": """A product manager at a retail company has 5 digital features to build but engineering can only deliver 2 this quarter.
How should they decide which to prioritize?""",
            "interaction": """
Prioritization framework:
- Impact on key metrics (engagement, revenue, retention)
- Engineering effort and complexity
- Customer pain point severity
- Strategic alignment
- Competitive necessity
- Dependencies
- RICE or similar scoring
""",
        },
        {
            "id": "subscription_churn",
            "case": """A SaaS company has a 5% monthly churn rate. They want to reduce it to 3% using data and product changes.
What would be your approach?""",
            "interaction": """
Areas to investigate:
- Customer segmentation (who is churning?)
- Reasons for cancellation (survey data, exit interviews)
- Product usage patterns before churn
- Customer success team effectiveness
- Onboarding experience quality
- Pricing and value perception
- Feature gaps vs. competitors
""",
        },
    ],
}

# ── Topic system prompts (injected into LLM as hidden guidance) ───────────────
TOPIC_SYSTEM_PROMPTS: dict[str, str] = {
    "profitability": """
You are evaluating the candidate's ability to build a MECE profit tree.
Expected framework: Start with Profit = Revenue - Costs. Decompose Revenue into Price x Volume
(by segment, product, or channel). Decompose Costs into Fixed and Variable.
Guide the candidate to identify the driver of the decline using hypothesis-driven questions.
Key behaviours to look for: top-down decomposition, numerical sizing, quantified hypotheses,
prioritisation of branches before drilling down.
""",
    "market_entry": """
You are evaluating whether the candidate assesses both market attractiveness AND client capability.
Expected framework: (1) Is the market attractive? (Size, growth, competition, profitability).
(2) Can the client win? (Capabilities, competitive advantage, route to market).
(3) How to enter? (Organic / acquire / partner, timing, sequencing).
Key behaviours: PESTLE awareness, competitive dynamics, risk flags, clear recommendation with rationale.
""",
    "growth_strategy": """
You are evaluating the candidate's ability to identify and prioritise growth levers.
Expected framework: Ansoff Matrix (market penetration, product development, market development, diversification).
Candidate should segment the opportunity, size each lever, and recommend a prioritised roadmap.
Key behaviours: customer segmentation, channel economics, organic vs inorganic options, trade-off awareness.
""",
    "mergers_acquisitions": """
You are evaluating the candidate's ability to assess an M&A transaction.
Expected framework: (1) Strategic rationale (why this deal?). (2) Synergy analysis (revenue + cost).
(3) Valuation sanity check (EV/EBITDA or EV/ARR multiples). (4) Integration risk and PMI.
Key behaviours: build-buy-partner comparison, synergy quantification, risk identification, clear go/no-go recommendation.
""",
    "pricing_strategy": """
You are evaluating the candidate's ability to design or fix a pricing model.
Expected framework: (1) Understand customer willingness-to-pay (value-based). (2) Benchmark competitors
(competitive pricing). (3) Assess cost floor (cost-plus). (4) Model price-volume trade-offs.
Key behaviours: segmentation of customers by WTP, quantified impact on revenue/margin, implementation plan.
""",
    "operations": """
You are evaluating the candidate's ability to diagnose and fix an operational bottleneck.
Expected framework: Process mapping -> identify bottleneck -> root cause analysis -> solution options -> trade-offs.
Candidate should use lean/Six Sigma thinking: eliminate waste, reduce variability, improve throughput.
Key behaviours: data-driven diagnosis, make-vs-buy analysis, cost-benefit of solutions, phased implementation.
""",
    "competitive_response": """
You are evaluating the candidate's ability to respond strategically to a competitive threat.
Expected framework: (1) Understand the threat (who, what, why now). (2) Assess impact on your client.
(3) Defensive moves (retention, switching costs). (4) Offensive moves (differentiation, pricing, innovation).
Key behaviours: war-gaming competitor next moves, customer segmentation, short vs long-term trade-offs.
""",
    "digital_transformation": """
You are evaluating the candidate's ability to build a business case for a technology investment.
Expected framework: (1) Current state assessment. (2) Build-buy-partner decision.
(3) ROI / NPV analysis (cost savings + revenue upside). (4) Change management and risk.
Key behaviours: quantified ROI, phased roadmap, stakeholder management, risk mitigation plan.
""",
}

# ── Topic framework hints (injected into CASE_STUDY_PROMPT) ──────────────────
TOPIC_FRAMEWORK_HINTS: dict[str, str] = {
    "profitability": "Revenue tree (Price x Volume by segment) and Cost tree (Fixed vs Variable). Push for MECE decomposition and quantified hypotheses.",
    "market_entry": "Market attractiveness (TAM/SAM/SOM, Porter's 5 Forces, PESTLE) + client capability (competitive advantage, route to market) + entry mode (organic/acquire/partner).",
    "growth_strategy": "Ansoff Matrix (penetration, product dev, market dev, diversification). Size each lever, prioritise by effort vs impact.",
    "mergers_acquisitions": "Strategic rationale + synergy tree (revenue/cost) + valuation benchmark (EV/EBITDA) + integration risk. End with clear go/no-go.",
    "pricing_strategy": "Value-based (WTP) vs cost-plus vs competitive. Model price-volume trade-off. Segment customers by WTP.",
    "operations": "Process map -> bottleneck -> root cause -> solution options (make/buy/automate) -> cost-benefit. Use lean thinking.",
    "competitive_response": "Threat assessment -> customer impact -> defensive moves (loyalty, switching costs) -> offensive moves (differentiation, pricing). War-game competitor response.",
    "digital_transformation": "Current state -> build/buy/partner -> ROI/NPV -> change management risk -> phased roadmap.",
}

# ── Legacy flat dict (kept for backward compatibility; not used by new code) ──
practice_cases = {
    case["id"]: {"case": case["case"], "interaction": case["interaction"]}
    for cases in PRACTICE_CASES.values()
    for case in cases
}

# ── Real Company Playbook cases ───────────────────────────────────────────────
# Each key is a company_slug. topic_slug maps to an existing TOPIC_FRAMEWORK_HINTS
# entry so framework guidance still applies during the interview.
COMPANY_PRACTICE_CASES: dict[str, list[dict]] = {
    "zomato": [
        {
            "id": "zomato_unit_economics",
            "case": """Zomato's food delivery GMV grew 3x between 2017 and 2019, but contribution margins
remained deeply negative. The CEO wants to reach contribution-margin breakeven within 18 months
without cutting delivery partner pay or raising prices for consumers.
What would you recommend?""",
            "interaction": """
Key areas to explore:
- Revenue levers: advertising, subscription (Zomato Gold), B2B (Hyperpure)
- Cost levers: delivery radius optimisation, batching orders, dark stores
- Demand-side: minimum order value, surge pricing, restaurant commission renegotiation
- Customer segmentation: high-frequency vs occasional users, LTV analysis
- Competitor context: Swiggy's parallel unit-economics race
""",
        },
        {
            "id": "zomato_hyperpure_pivot",
            "case": """Zomato launched Hyperpure, a B2B ingredient supply business for restaurants, in 2019.
Initial traction is good in Bengaluru but the unit economics look worse than food delivery.
Should Zomato double down, scale back, or exit Hyperpure?""",
            "interaction": """
Key areas to explore:
- Strategic rationale: vertical integration, data moat, restaurant loyalty
- Unit economics: gross margin on ingredients vs food delivery take rate
- Competitive advantage: can Zomato win against established HORECA distributors?
- Capital allocation: opportunity cost vs core delivery business
- Synergies: does Hyperpure reduce restaurant churn on the platform?
""",
        },
    ],
    "cred": [
        {
            "id": "cred_monetisation",
            "case": """CRED has 7.5 million premium credit-card users (2021) and near-zero revenue per user.
Investors are asking for a credible path to monetisation. The founders want to avoid
anything that feels 'cheap' or damages the premium brand.
How would you design CRED's monetisation strategy?""",
            "interaction": """
Key areas to explore:
- User profile: high-income, high-spend, brand-conscious — willingness to pay
- Revenue streams: CRED Pay (UPI), CRED Cash (lending), CRED Store (commerce), brand partnerships
- Pricing psychology: subscription vs transaction fee vs freemium
- Risk: monetisation that erodes trust or exclusivity
- Benchmarks: Amex Platinum, Revolut Metal — premium fintech monetisation models
""",
        },
    ],
    "meesho": [
        {
            "id": "meesho_reseller_growth",
            "case": """Meesho's reseller model grew to 13 million resellers by 2021, mostly homemakers
sharing products on WhatsApp. But average order value is low and repeat purchase rates
are declining. How should Meesho accelerate growth while improving economics?""",
            "interaction": """
Key areas to explore:
- Reseller segmentation: top 10% drive 60% of GMV — how to nurture them
- Supply side: catalogue depth, quality control, supplier onboarding
- Demand side: end-consumer experience, returns, trust
- Channel expansion: move beyond WhatsApp to own app
- Unit economics: commission structure, logistics cost per order
- Competition: Glowroad, Shop101 — differentiation
""",
        },
    ],
    "dunzo": [
        {
            "id": "dunzo_dark_store",
            "case": """Dunzo is expanding its quick-commerce (10-minute delivery) model by opening dark stores
in Bengaluru. Each dark store costs Rs 15 lakh to set up and serves a 2 km radius.
How many dark stores should Dunzo open in Bengaluru, and how do you decide where?""",
            "interaction": """
Key areas to explore:
- Demand mapping: order density by pin code, peak hours
- Supply constraints: SKU range (2,000 vs 10,000), inventory turns
- Unit economics per dark store: fixed cost, variable cost, break-even orders/day
- Location criteria: rent, catchment population, competitor presence
- Sequencing: pilot learnings before city-wide rollout
- Operational risks: last-mile delivery partner availability, spoilage
""",
        },
    ],
    "zepto": [
        {
            "id": "zepto_category_expansion",
            "case": """Zepto has proven 10-minute grocery delivery in 6 metro cities (2022).
The board wants to expand into non-grocery categories (electronics, pharmacy, fashion)
to increase basket size and frequency. Should Zepto expand categories, and if so, which ones first?""",
            "interaction": """
Key areas to explore:
- Category attractiveness: frequency, margin, return rate, storage complexity
- Infrastructure fit: cold chain, fragile goods handling, dark store space
- Consumer behaviour: do Zepto users want non-grocery from the same app?
- Competitive response: Blinkit, Swiggy Instamart — are they already there?
- Prioritisation framework: pharmacy (high frequency, high margin) vs electronics (low frequency, high AOV)
""",
        },
    ],
    "razorpay": [
        {
            "id": "razorpay_international",
            "case": """Razorpay processes $60B in payments annually in India (2022) and holds 30% market share.
The CEO wants to expand internationally — Southeast Asia is the primary target.
How should Razorpay approach this expansion?""",
            "interaction": """
Key areas to explore:
- Market selection: Indonesia vs Vietnam vs Philippines — TAM, competition, regulatory complexity
- Product-market fit: is India's payment stack (UPI-like rails) replicable abroad?
- Go-to-market: follow Indian diaspora businesses, partner with local banks, or greenfield
- Build vs buy vs partner: acquire a local player vs build from scratch
- Regulatory risk: payment licenses, data localisation, central bank relationships
- Capital requirements and timeline to breakeven
""",
        },
    ],
    "ola": [
        {
            "id": "ola_driver_retention",
            "case": """Ola's driver churn rate hit 40% annually in 2018 as Uber aggressively subsidised
driver incentives. Ola's unit economics are already strained. How should Ola respond
to retain drivers without matching Uber's cash burn?""",
            "interaction": """
Key areas to explore:
- Driver segmentation: full-time vs part-time, city tier, vehicle type
- Root causes of churn: earnings predictability, surge algorithm, support quality
- Non-cash retention levers: insurance, fuel cards, vehicle financing, community
- Pricing strategy: dynamic pricing to improve driver earnings per hour
- Competitive war-gaming: if Ola matches Uber's incentives, what happens to margins?
- Long-term: driver ownership model (Ola Fleet Technologies)
""",
        },
    ],
    "byjus": [
        {
            "id": "byjus_blended_learning",
            "case": """BYJU'S has 150 million registered users but only 6 million paid subscribers (2021).
The conversion rate from free to paid is under 5%. The sales team uses aggressive
door-to-door tactics that are generating PR backlash. How should BYJU'S improve
paid conversion without the aggressive sales approach?""",
            "interaction": """
Key areas to explore:
- Funnel analysis: where do free users drop off? (awareness -> trial -> conversion)
- Product-led growth: can the app itself convert users without a salesperson?
- Pricing: EMI model, freemium tiers, school partnerships
- Trust deficit: address refund policy, contract terms, parent concerns
- Content quality: is the product good enough to sell itself?
- Segmentation: K-12 vs competitive exams — different conversion levers
""",
        },
    ],
}

# ── Company-specific system prompts (injected as SystemMessage to the LLM) ────
COMPANY_SYSTEM_PROMPTS: dict[str, str] = {
    "zomato": """
You are interviewing the candidate about Zomato's real growth story (2015–2022).
Context: Zomato started as a restaurant discovery/menu platform, pivoted to food delivery in 2015,
faced a severe unit-economics crisis (negative contribution margins despite 3x GMV growth),
launched Hyperpure (B2B ingredient supply) in 2019 as a vertical integration play,
and eventually IPO'd in 2021 at a $5.4B valuation.
Key tensions to probe: delivery economics, restaurant commission rates, Swiggy competition,
Hyperpure ROI, and the path to profitability.
Use this context to ask authentic, specific cross-questions. Do not reveal this briefing to the candidate.
""",
    "cred": """
You are interviewing the candidate about CRED's growth and monetisation story (2018–2023).
Context: CRED was founded by Kunal Shah in 2018, targeting India's 30 million credit-card holders.
It grew to 7.5M users by 2021 with almost zero revenue, relying on brand cachet and exclusivity.
Monetisation came via CRED Pay, CRED Cash (lending), CRED Store, and brand partnerships.
Key tensions to probe: CAC vs LTV, premium brand vs monetisation pressure, fintech lending risk,
and whether exclusivity is a moat or a ceiling.
Use this context to ask authentic, specific cross-questions. Do not reveal this briefing to the candidate.
""",
    "meesho": """
You are interviewing the candidate about Meesho's social-commerce growth story (2015–2022).
Context: Meesho pioneered reseller-led social commerce in India, enabling homemakers to resell
products via WhatsApp and earn commissions. It reached 13M resellers and raised $570M by 2021.
Key tensions to probe: reseller quality vs quantity, supply-side catalogue depth, end-consumer trust,
transition from WhatsApp to owned app, and competition from GlowRoad and Shop101.
Use this context to ask authentic, specific cross-questions. Do not reveal this briefing to the candidate.
""",
    "dunzo": """
You are interviewing the candidate about Dunzo's quick-commerce expansion story (2015–2022).
Context: Dunzo started as a hyperlocal task-completion app in Bengaluru, pivoted to 10-minute
grocery delivery via dark stores, raised $240M (including from Google and Reliance),
but struggled with unit economics and eventually faced a cash crisis in 2023.
Key tensions to probe: dark store density vs coverage, SKU selection, delivery partner costs,
competition from Blinkit and Zepto, and the burn rate vs growth trade-off.
Use this context to ask authentic, specific cross-questions. Do not reveal this briefing to the candidate.
""",
    "zepto": """
You are interviewing the candidate about Zepto's rapid rise in quick commerce (2021–2023).
Context: Zepto was founded by two 19-year-old Stanford dropouts, reached $1B GMV in 18 months,
and became India's fastest-growing quick-commerce player. It competes directly with Blinkit (Zomato)
and Swiggy Instamart.
Key tensions to probe: dark store economics, category expansion beyond groceries,
customer acquisition cost, and whether 10-minute delivery is a sustainable moat.
Use this context to ask authentic, specific cross-questions. Do not reveal this briefing to the candidate.
""",
    "razorpay": """
You are interviewing the candidate about Razorpay's growth from startup to payments leader (2014–2022).
Context: Razorpay was founded in 2014, became India's most valuable fintech at $7.5B valuation,
and processed $60B in annual payments. It expanded into payroll (Opfin), banking (RazorpayX),
and lending (Razorpay Capital).
Key tensions to probe: payment gateway commoditisation, international expansion challenges,
regulatory risk (RBI licensing), build vs buy for new products, and competition from PayU and CCAvenue.
Use this context to ask authentic, specific cross-questions. Do not reveal this briefing to the candidate.
""",
    "ola": """
You are interviewing the candidate about Ola's competitive battle with Uber in India (2014–2020).
Context: Ola dominated Indian ride-hailing with 60%+ market share but faced an existential threat
when Uber entered India in 2014 with massive subsidies. Ola responded with its own cash burn,
launched Ola Money, Ola Fleet, and Ola Electric, and eventually survived while Uber retreated.
Key tensions to probe: driver retention vs incentive cost, dynamic pricing, market segmentation
(auto-rickshaws, bikes, cabs), and the long-term shift to EVs.
Use this context to ask authentic, specific cross-questions. Do not reveal this briefing to the candidate.
""",
    "byjus": """
You are interviewing the candidate about BYJU'S growth and governance story (2011–2022).
Context: BYJU'S became the world's most valuable edtech at $22B, with 150M registered users
but only 6M paid subscribers. It grew via aggressive door-to-door sales, controversial EMI contracts,
and a string of acquisitions (Aakash, WhiteHatJr, Epic). It later faced a severe governance
and cash crisis.
Key tensions to probe: free-to-paid conversion, sales ethics vs growth, acquisition integration,
content quality vs marketing spend, and the unit economics of offline tutoring vs digital.
Use this context to ask authentic, specific cross-questions. Do not reveal this briefing to the candidate.
""",
}

# CASE_GREETING_PROMPT = """
# Your name is Glee and you are conducting a case study interview.
# Speak naturally and conversationally in one paragraph.

# 1. Greet the candidate warmly.
# 2. Introduce yourself.
# 3. Explain this is a case interview focused on structured thinking.
# 4. Encourage thinking aloud.
# 5. Ask if they have any questions ONLY about the process.
# """

CASE_GREETING_PROMPT = '''
Your name is Glee and you have to act as an interviewer conducting a case-study based live interview session AND SIMPLY FOLLOW [INSTRUCTIONS] WITHOUT ANY CROSS-QUESTIONS.
Your primary role is to emulate a real, empathetic human interviewer, speaking naturally and conversationally.
Respond in a single paragraph of plain-continuous text, without using special characters or formatting like bold,italics texts or coding texts, as if you were speaking aloud.

Your [INSTRUCTIONS] are:

1. Start with a Warm Greeting: Begin with a friendly and personal greeting. Do not include any parenthetical actions, stage directions, or cues (e.g., laughing gently, sighs, smiles).

2. Introduce Yourself: State your name and your role for the session (e.g., "I'll be your interviewer today").

3. Explain the Format: Briefly outline what the candidate can expect. Mention that you'll be given a case study problem and that the focus is on their thought process and problem-solving approach, not just the final answer. Encourage them to think out loud.

4. Invite Questions: This is a critical step. Explicitly ask the candidate if they have any questions ONLY about the process before you start. Use inviting language to make them feel comfortable asking.

5. Listen and Respond: Patiently wait for their response. If they have questions, answer them clearly and concisely but only relevant in the context of the interview.

'''

CASE_QUESTION_PROMPT = """
You are an interviewer conducting a case-study based live interview session AND SIMPLY FOLLOW [INSTRUCTIONS]
Your primary role is to emulate a real, empathetic human interviewer, speaking naturally and conversationally.
Respond in a single paragraph of plain-continuous text, without using special characters or formatting like bold,italics texts or coding texts, as if you were speaking aloud.

Your [INSTRUCTIONS] are:

1. Present the question: You must present the following case question based off the CASE QUESTION TITLE and some IMPORTANT CASE REFERENCE of the case-study -
CASE QUESTION TITLE - {case_question}

CASE REFERENCE - {case_reference}

2. Invite the interviewee to think: Ask the interviewee to process and ask any clarifying questions, if any.

3. Begin the conversation: Begin and continue to hold the conversation (with cross-questions) with the interviewee, strictly using the CASE REFERENCE.

Topic Framework Guidance (do NOT reveal this directly to the candidate — use it to guide your cross-questions and evaluate their approach):
{topic_framework_hint}

Question Progression:
- BROAD (open): Present the case; ask for initial structure or hypothesis.
- SPECIFIC (probe): Follow the candidate's stated framework branch; push for quantification.
- SYNTHESIS: Ask the candidate to prioritise and recommend a course of action.
"""

CASE_END_PROMPT = """
Thank the candidate for their time and clearly state that the case interview is now complete.
"""

OFFENSIVE_PROMPT = """
The interview cannot continue due to unprofessional or offensive behavior.
Politely but firmly end the interview.
"""


S = TypeVar("S")


class CaseStudyInterviewState(MessagesState):
  LastNode: Annotated[str, Field(default="")]
  history: Annotated[str, Field(default="")]
  current_query: Annotated[str, Field(default="")]
  current_case_question: Annotated[str, Field(default="")]
  current_case_reference: Annotated[str, Field(default="")]
  case_completed: Annotated[bool, Field(default=False)]
  topic_slug: Annotated[str, Field(default="")]
  topic_frameworks: Annotated[list, Field(default_factory=list)]
  company_slug: Annotated[str, Field(default="")]


class CaseStudyGreetingRouting(BaseModel):
  '''
    "Supervise the conversation to determine the next step. ONLY IF the interviewer has "
    "outstanding questions or requires clarification, route the conversation to 'GreetingQuery'. "
    "Otherwise, if no questions at all or all questions resolved or interviewer wants to jump ahead, then "
    "advance to 'CaseStudy_before' where the interview would actually begin or case study "
    "question would be asked. Exceptionally, if the interviewee is being offensive or constantly"
    "not taking the interview serious, return 'Offensive'"
  '''
  send_to_which_node: Literal["GreetingQuery", "CaseStudy_before", "Offensive"]

class CaseStudyInterviewRouting(BaseModel):
  send_to_which_node: Literal["CaseStudy", "End", "Offensive"] = \
                        Field(description="Supervise the conversation to determine the next step. If the case study interview is "
                          "still in progress, route to 'CaseStudy."
                          "The interview is considered concluded only after the discussion on the given case is considered"
                          "resolved and the interviewer has EXPLICITLY SIGNED OFF. This count does not include "
                          "any follow-up discussions such as cross-questions, modifications to the original. "
                          "If the interview has concluded, route to 'End'."
                          "problem, or edge case analysis. Exceptionally, if the interviewee is being offensive or constantly"
                          "not taking the interview serious, return 'Offensive'")


def create_dummy_node() -> Callable:
  def _Node(state):
    return state
  return _Node

def get_llm(api_key: str):
    return ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.3
    )


def create_route_to_greeting(InterviewProgress_llm) -> Callable:
  def _Node(state:CaseStudyInterviewState) -> Literal['GreetingQuery', 'CaseStudy_before', 'Offensive']:
    print("Hereee in route to greeting")
    response = InterviewProgress_llm.invoke(state["history"])
    print("This is the response", response)
    # if response.send_to_which_node == 'Greeting':
    #   state["current_query"] = state["messages"][-1].content

    return response.send_to_which_node
  return _Node

def create_greeting_query_node(key: str) -> Callable:
  def _Node(state:CaseStudyInterviewState):
    state[key].append(
            AIMessage(content="", tool_calls=[
                {
                    'name': 'rag_case_study',
                    'args': {'query': f'The interviewee has asked this, provide me the relevant context - {state["messages"][-1]}'},
                    'id': str(uuid4())
                }
            ]
        ))
    return state
  return _Node


def create_case_study_before_node(llm):
  def _Node(state: CaseStudyInterviewState):
    company_slug = state.get("company_slug") or ""

    from langchain_core.messages import SystemMessage
    new_messages = list(state.get("messages", []))

    if company_slug and company_slug in COMPANY_PRACTICE_CASES:
        # Company Playbook path: pick from company-specific cases
        cases = COMPANY_PRACTICE_CASES[company_slug]
        chosen_case = random.choice(cases)

        # Derive topic_slug from the company's associated topic for framework hints
        # Map company -> topic (mirrors COMPANY_PLAYBOOKS on the frontend)
        COMPANY_TOPIC_MAP = {
            "zomato":    "growth_strategy",
            "cred":      "market_entry",
            "meesho":    "growth_strategy",
            "dunzo":     "operations",
            "zepto":     "operations",
            "razorpay":  "digital_transformation",
            "ola":       "competitive_response",
            "byjus":     "growth_strategy",
        }
        topic_slug = COMPANY_TOPIC_MAP.get(company_slug, "growth_strategy")
        topic_framework_hint = TOPIC_FRAMEWORK_HINTS.get(topic_slug, "")

        company_system_prompt = COMPANY_SYSTEM_PROMPTS.get(company_slug, "")
        if company_system_prompt:
            new_messages.append(SystemMessage(content=company_system_prompt))

        state["messages"] = new_messages
        state["current_case_question"] = chosen_case["case"]
        state["current_case_reference"] = chosen_case["interaction"]
        state["topic_slug"] = topic_slug
        state["company_slug"] = company_slug
        state["topic_frameworks"] = topic_framework_hint.split(". ")
    else:
        # Standard topic path: use state's topic_slug if set, else pick randomly
        topic_slug = state.get("topic_slug") or ""
        if not topic_slug or topic_slug not in PRACTICE_CASES:
            topic_slug = random.choice(list(PRACTICE_CASES.keys()))

        cases = PRACTICE_CASES[topic_slug]
        chosen_case = random.choice(cases)

        topic_system_prompt = TOPIC_SYSTEM_PROMPTS.get(topic_slug, "")
        topic_framework_hint = TOPIC_FRAMEWORK_HINTS.get(topic_slug, "")

        if topic_system_prompt:
            new_messages.append(SystemMessage(content=topic_system_prompt))

        state["messages"] = new_messages
        state["current_case_question"] = chosen_case["case"]
        state["current_case_reference"] = chosen_case["interaction"]
        state["topic_slug"] = topic_slug
        state["topic_frameworks"] = TOPIC_FRAMEWORK_HINTS.get(topic_slug, "").split(". ")

    return state
  return _Node


def create_route_to_casestudy(CaseStudy_llm) -> Callable:
  def _Node(state:CaseStudyInterviewState) -> Literal['CaseStudy', 'End', 'Offensive']:
    print("Hereee in route to case")
    response = CaseStudy_llm.invoke(state["history"])
    print("This is the response", response)
    # if response.send_to_which_node == 'Greeting':
    #   state["current_query"] = state["messages"][-1].content

    return response.send_to_which_node
  return _Node


# S = TypeVar("S")
class ToolNode(BaseModel):
  model_config = ConfigDict(extra='allow')
  tools: Annotated[List[Callable],Field(description="List of tools to be used")]
  key: Annotated[str,Field(description="Key in the state where the tool calls are to be made")]

  @field_validator("key")
  @classmethod
  def validate_key(cls,v):
    if not isinstance(v,str):
      raise ValueError("Key must be a string")
    return v

  @field_validator("tools")
  @classmethod
  def validate_tools(cls,v):
    for i,tool in enumerate(v):
      if not callable(tool):
        raise ValueError(f"Tool {i} is not a callable")
      if not inspect.isfunction(tool):
        raise ValueError(f"Tool {i} is not a function")
    return v

  def __init__(self,tools:List[Callable],key:str,*args,**kwargs):
    super().__init__(tools = tools, key = key, *args,**kwargs)
    self.tools = tools
    self.tool_names = {f"{tool.__name__}":tool for tool in tools}


  def __call__(self,state:S) -> S:
    latest_message = state[self.key][-1]
    if not getattr(latest_message,"tool_calls",None):
      return state

    output = []
    for tool_call in latest_message.tool_calls:
      tool_result = self.tool_names[tool_call["name"]](**tool_call["args"])
      output.append(
          ToolMessage(
              content=str(tool_result),
              name=tool_call["name"],
              tool_call_id=tool_call["id"],
          )
      )
    return {self.key: output}


def custom_tool_node(tools_by_name):
    def _Node(state):
      outputs = []
      last_message = state["messages"][-1]

      for tool_call in last_message.tool_calls:
          tool_result = tools_by_name[tool_call["name"]](**tool_call["args"])
          outputs.append(
              ToolMessage(
                  content=str(tool_result),
                  name=tool_call["name"],
                  tool_call_id=tool_call["id"],
              )
          )

      return {"toolCall": outputs}
    return _Node


# def make_search_tool(tavily_api_key: str, max_results: int = 5):
#     search = TavilySearch(max_results=max_results, topic="general",tavily_api_key=tavily_api_key, include_answer=True)

#     def get_google_search(query: str):
#         "Call to perform google search online and get reliable results"
#         return search.invoke({"query": query})

#     return get_google_search

def make_tool_nodes(search_fn):
    return ToolNode([search_fn], "messages")



def rag_case_study(query: str, top_k: int = 2) -> str:
    """
    Returns the most relevant case-study chunk for a given query.
    Designed for case interview rounds.
    """
    config = apps.get_app_config('myapp')
    embedder = config.embedder
    index = config.index
    chunks = config.chunks
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)

    results = [chunks[i] for i in indices[0]]
    return "\n\n".join(results)


def create_greeting_node(Greeting_llm) -> Callable:
  def _Node(state: S) -> S:
    if state["LastNode"] != "Greeting":
      inp_company = getattr(state, "company", None)
      inp_state = getattr(state, "subject", None)
      # greeting_prompt = get_greeting_prompt_template(interview_type, inp_company or inp_state)
      # print(greeting_prompt.format_messages())
      greeting_prompt = ChatPromptTemplate.from_messages([
          ("system", CASE_GREETING_PROMPT),
      # ("human", "{input}")
      ])
      input_ = greeting_prompt.format_messages() + [{"role":"human","content":"Start the interview now"}]
      state["messages"] = state["messages"] + input_
    # else:
    #   state["messages"].append(
    #         AIMessage(content="", tool_calls=[
    #             {
    #                 'name': 'rag_case_study',
    #                 'args': {'query': state["current_query"]}
    #             }
    #         ]
    #     ))

    response = Greeting_llm.invoke(state["messages"])

    # if state["current_query"]:
    #   state["messages"].append()
    #   pass

    state["messages"] = state["messages"] + [response]
    state["history"] = state["history"] + "\n" + "Interviewer-" + response.content
    state["LastNode"] = "Greeting"

    # print("We are delivering greetings-->",response)
    return state
  return _Node



def create_case_study_node(CaseStudy_llm):
  def _Node(state:CaseStudyInterviewState):
    if state["LastNode"] != "CaseStudy":
      topic_slug = state.get("topic_slug", "")
      topic_framework_hint = TOPIC_FRAMEWORK_HINTS.get(topic_slug, "No specific framework required — use your best judgment.")
      # Format the prompt with actual case question, reference, and topic hint
      formatted_prompt = CASE_QUESTION_PROMPT.format(
        case_question=state["current_case_question"],
        case_reference=state["current_case_reference"],
        topic_framework_hint=topic_framework_hint,
      )
      # Create a new system message with the formatted prompt
      case_prompt = ChatPromptTemplate.from_messages([
        ("system", formatted_prompt),
      ])
      state["messages"] = case_prompt.format_messages() + [{"role":"human","content":"Please present the case study question"}]
    
    response = CaseStudy_llm.invoke(state["messages"])

    print("In here Case Study Node \n\n")
    # print("This is the state", state["messages"][-2:])
    # print("This is the response", response)

    # if state["current_query"]:
    #   state["messages"].append()
    #   pass

    state["messages"] = state["messages"] + [response]
    state["history"] = state["history"] + "\n" + "Interviewer-" + response.content
    state["LastNode"] = "CaseStudy"

    return state
  return _Node



def build_case_study_graph(google_api_key: str, checkpointer):
    llm = get_llm(google_api_key)

    # checkpointer = InMemorySaver()

    workflow = StateGraph(CaseStudyInterviewState)

    workflow.add_node("Greeting", create_greeting_node(llm))
    workflow.add_node("Greeting_after", create_dummy_node())
    workflow.add_node("Offensive", create_dummy_node())
    workflow.add_node("GreetingQuery", create_greeting_query_node("messages"))
    workflow.add_node("GreetingQueryTool", make_tool_nodes(rag_case_study))
    workflow.add_node("CaseStudy_before", create_case_study_before_node(llm))
    workflow.add_node("CaseStudy", create_case_study_node(llm))
    workflow.add_node("CaseStudy_after", create_dummy_node())
    workflow.add_node("End", create_dummy_node())
    # workflow.add_node("End", create_dummy_node())
    # workflow.add_node("PickCase", pick_case_node())
    # workflow.add_node("CaseDiscussion", case_discussion_node(llm))
    # workflow.add_node("End", end_node(llm))
    # workflow.add_node("Offensive", offensive_node(llm))


    # workflow.add_node("CaseStudy_before", create_dummy_node())

    workflow.set_entry_point("Greeting")

    workflow.add_edge("Greeting", "Greeting_after")
    workflow.add_edge("GreetingQuery", "GreetingQueryTool")
    workflow.add_edge("GreetingQueryTool", "Greeting")
    workflow.add_edge("CaseStudy_before", "CaseStudy")
    workflow.add_edge("CaseStudy", "CaseStudy_after")
    workflow.add_edge("End", END)
    workflow.add_edge("Offensive", END)

    workflow.add_conditional_edges(
        "Greeting_after",
        create_route_to_greeting(llm.with_structured_output(CaseStudyGreetingRouting))
    )

    workflow.add_conditional_edges(
        "CaseStudy_after",
        create_route_to_casestudy(llm.with_structured_output(CaseStudyInterviewRouting))
    )

    # workflow.add_edge("End", END)
    workflow.add_edge("Offensive", END)

    return workflow.compile(checkpointer=checkpointer)