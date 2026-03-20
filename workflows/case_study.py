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

# ── Consulting topic case bank ────────────────────────────────────────────────

TOPIC_FRAMEWORK_HINTS: dict[str, str] = {
    "profitability": (
        "Guide: decompose into Revenue (Price × Volume) vs Cost (Fixed vs Variable). "
        "Push candidate to quantify each branch and form a hypothesis before drilling down."
    ),
    "market-entry": (
        "Guide: Market attractiveness (size, growth, profitability) → Competitive position → "
        "Entry strategy (Build / Buy / Partner). Push for prioritisation of which segment to enter first."
    ),
    "growth-strategy": (
        "Guide: Ansoff Matrix or Growth Accounting (Acquisition / Retention / Monetisation). "
        "Push for unit economics: CAC, LTV, payback period. Ask where growth is leaking."
    ),
    "mergers-acquisitions": (
        "Guide: Strategic rationale → Revenue + Cost synergies → Valuation sanity check → "
        "Integration risk. Push for a Go / No-Go recommendation with clear rationale."
    ),
    "operations-cost": (
        "Guide: Process mapping → Identify waste / bottlenecks → Prioritise fixes. "
        "Push for quantified impact of proposed changes and Quick Wins vs Structural Fixes."
    ),
    "pricing-strategy": (
        "Guide: Value-based → Competitive → Cost-plus analysis. "
        "Push for revenue impact modelling of the pricing change and a rollout plan."
    ),
    "product-innovation": (
        "Guide: Customer need → Market opportunity → Build / prioritise decision. "
        "Push for a prioritisation framework (RICE or similar) with explicit scoring."
    ),
    "turnaround-crisis": (
        "Guide: Cash flow triage (survive) → Root cause (stabilise) → Growth plan (thrive). "
        "Push for a 30 / 60 / 90-day action plan with clear owners."
    ),
}

TOPIC_CASES: dict[str, dict] = {
    "profitability": {
        "display_name": "Profitability",
        "cases": [
            {
                "case": (
                    "A D2C skincare brand's gross margins dropped from 35 % to 18 % over 12 months "
                    "despite flat revenue. The founder wants a path back to 30 % margins within two quarters. "
                    "How would you approach this?"
                ),
                "interaction": (
                    "Probe: revenue mix shift (hero SKU vs tail), COGS breakdown (raw materials, packaging, "
                    "logistics), marketing spend as % of revenue, returns rate, any recent pricing changes."
                ),
            },
            {
                "case": (
                    "A quick-service restaurant chain is profitable in Tier-1 cities but losing money in Tier-2. "
                    "The CFO wants to know whether to exit Tier-2 or fix it. What is your recommendation?"
                ),
                "interaction": (
                    "Probe: fixed vs variable cost structure per city tier, average order value differences, "
                    "occupancy costs, brand awareness levels, break-even analysis per outlet."
                ),
            },
        ],
    },
    "market-entry": {
        "display_name": "Market Entry",
        "cases": [
            {
                "case": (
                    "A successful Indian B2B SaaS company wants to expand to Southeast Asia. "
                    "They have 18 months of runway to become cash-flow positive in the new market. "
                    "How would you evaluate this decision?"
                ),
                "interaction": (
                    "Probe: market sizing per country, competitive landscape, regulatory environment, "
                    "go-to-market strategy (direct vs partner), localisation requirements, resource requirements."
                ),
            },
            {
                "case": (
                    "A premium gym chain operating in 5 metros wants to enter Tier-2 cities. "
                    "They can open 10 new outlets in the next year. How should they decide which cities to enter?"
                ),
                "interaction": (
                    "Probe: population and income demographics, fitness penetration rates, real estate costs, "
                    "competition density, brand awareness, unit economics per outlet."
                ),
            },
        ],
    },
    "growth-strategy": {
        "display_name": "Growth Strategy",
        "cases": [
            {
                "case": (
                    "An ed-tech platform has 2 million registered users but only 80,000 paying subscribers. "
                    "Monthly new registrations are flat. The board wants to double paying subscribers in 12 months. "
                    "What levers would you pull?"
                ),
                "interaction": (
                    "Probe: activation funnel (registration → first lesson → subscription), "
                    "churn rate among paying users, CAC by channel, LTV, content gaps vs competitors."
                ),
            },
            {
                "case": (
                    "A hyperlocal grocery delivery startup has strong retention in its first three cities "
                    "but growth has plateaued at 15 % month-on-month. "
                    "How would you identify and prioritise the next growth levers?"
                ),
                "interaction": (
                    "Probe: Ansoff matrix (existing vs new products / markets), "
                    "cohort analysis of order frequency, dark store economics, referral programme effectiveness."
                ),
            },
        ],
    },
    "mergers-acquisitions": {
        "display_name": "Mergers & Acquisitions",
        "cases": [
            {
                "case": (
                    "A large Indian conglomerate is considering acquiring a loss-making but fast-growing "
                    "fintech startup for ₹2,000 crore. The startup has 5 million active users and "
                    "a lending book of ₹500 crore. Should they proceed?"
                ),
                "interaction": (
                    "Probe: strategic rationale (distribution, technology, talent), "
                    "revenue and cost synergies, NPA risk in the lending book, "
                    "integration complexity, alternative uses of ₹2,000 crore."
                ),
            },
            {
                "case": (
                    "Two mid-sized logistics companies are considering a merger to compete with larger players. "
                    "Combined they would have 30 % market share. Evaluate the strategic case for the merger."
                ),
                "interaction": (
                    "Probe: cost synergies (fleet, warehouses, headcount), revenue synergies (cross-sell), "
                    "cultural integration risk, regulatory approvals, combined valuation vs standalone."
                ),
            },
        ],
    },
    "operations-cost": {
        "display_name": "Operations & Cost",
        "cases": [
            {
                "case": (
                    "A restaurant chain's food cost is 42 % of revenue against an industry benchmark of 30 %. "
                    "The operations head wants to close the gap within six months without changing the menu. "
                    "How would you approach this?"
                ),
                "interaction": (
                    "Probe: menu engineering (contribution margin per dish), supplier negotiations, "
                    "portion control, waste and spoilage, inventory management, demand forecasting."
                ),
            },
            {
                "case": (
                    "An e-commerce company's average delivery time is 4 days. They want to reduce it to 2 days "
                    "while maintaining profitability. What is your approach?"
                ),
                "interaction": (
                    "Probe: warehouse network (number, location, inventory placement), "
                    "last-mile carrier SLAs, cost-benefit per region, "
                    "customer willingness to pay for speed, impact on NPS."
                ),
            },
        ],
    },
    "pricing-strategy": {
        "display_name": "Pricing Strategy",
        "cases": [
            {
                "case": (
                    "An ed-tech company currently charges ₹999 per month. "
                    "They are considering switching to ₹9,999 per year. "
                    "How would you evaluate this pricing change?"
                ),
                "interaction": (
                    "Probe: current LTV and payback period, cash flow implications, "
                    "churn impact modelling, customer preference research, "
                    "competitive pricing landscape, implementation and communication plan."
                ),
            },
            {
                "case": (
                    "A B2B SaaS company is losing deals to a competitor that is 20 % cheaper. "
                    "They have three options: match the price, bundle more features, or reposition upmarket. "
                    "Which would you recommend?"
                ),
                "interaction": (
                    "Probe: customer segmentation (price-sensitive vs value-driven), "
                    "gross margin impact of price cut, feature development cost, "
                    "ICP for upmarket segment, competitive differentiation."
                ),
            },
        ],
    },
    "product-innovation": {
        "display_name": "Product & Innovation",
        "cases": [
            {
                "case": (
                    "A social media app has 1 million downloads but only 100,000 monthly active users. "
                    "The product team has five features to build this quarter but can only ship two. "
                    "How should they decide?"
                ),
                "interaction": (
                    "Probe: activation and retention funnel, core value proposition clarity, "
                    "feature impact on key metrics (DAU, session length, D7 retention), "
                    "engineering effort, RICE or similar prioritisation framework."
                ),
            },
            {
                "case": (
                    "A consumer fintech app wants to launch a credit card product. "
                    "They have a user base of 8 million but no lending licence. "
                    "How would you evaluate the opportunity and recommend a path forward?"
                ),
                "interaction": (
                    "Probe: market sizing, regulatory path (NBFC partnership vs own licence), "
                    "user segment most likely to adopt, risk of cannibalising existing products, "
                    "build vs partner decision."
                ),
            },
        ],
    },
    "turnaround-crisis": {
        "display_name": "Turnaround & Crisis",
        "cases": [
            {
                "case": (
                    "A food delivery app suffered a data breach exposing payment information of 2 million users. "
                    "The story broke in the press this morning. You are the incoming CEO. "
                    "What do you do in the first 72 hours?"
                ),
                "interaction": (
                    "Probe: immediate containment (technical), customer communication strategy, "
                    "regulatory and legal obligations, PR and brand damage control, "
                    "compensation and goodwill measures, long-term security roadmap."
                ),
            },
            {
                "case": (
                    "An online fashion retailer is burning ₹8 crore per month with only 4 months of runway left. "
                    "Revenue is flat and the last fundraise fell through. "
                    "How would you stabilise the business?"
                ),
                "interaction": (
                    "Probe: cash flow triage (which costs can be cut immediately), "
                    "revenue acceleration options (liquidate inventory, B2B channel), "
                    "stakeholder management (investors, suppliers, employees), "
                    "30 / 60 / 90-day plan."
                ),
            },
        ],
    },
}

# ── Company growth story case bank ───────────────────────────────────────────

COMPANY_STORIES: dict[str, dict] = {
    "zomato": {
        "display_name": "Zomato",
        "fun_title": "From Menu to Home",
        "context": (
            "Zomato started as a restaurant discovery platform in 2008. By 2015 it had expanded to "
            "24 countries but was burning cash. In 2019 it pivoted hard to food delivery, raised $1B+, "
            "went public in 2021, and then faced a profitability crisis in 2022 before turning "
            "EBITDA-positive in 2023."
        ),
        "cases": [
            {
                "case": (
                    "Zomato's monthly active users grew 3× post-COVID but unit economics worsened — "
                    "delivery costs rose 40 % while average order value stayed flat. "
                    "The board wants a path to profitability within 18 months. How would you approach this?"
                ),
                "interaction": (
                    "Probe: delivery cost structure (fixed vs variable), AOV levers (Zomato Gold, bundling), "
                    "dark store expansion, customer segmentation by order frequency, take-rate optimisation."
                ),
            },
            {
                "case": (
                    "Zomato acquired Blinkit for $568M in 2022. At the time Blinkit was loss-making. "
                    "Evaluate whether this was the right strategic decision."
                ),
                "interaction": (
                    "Probe: synergies (shared riders, dark stores), Blinkit unit economics trajectory, "
                    "Swiggy Instamart competitive response, long-term platform value vs near-term cash burn."
                ),
            },
        ],
    },
    "swiggy": {
        "display_name": "Swiggy",
        "fun_title": "Hunger Games",
        "context": (
            "Swiggy launched in 2014 as a food delivery platform in Bengaluru. It scaled rapidly, "
            "raised over $3.6B, launched Instamart (10-minute grocery delivery) in 2021, "
            "and went public in 2024 — still working toward profitability."
        ),
        "cases": [
            {
                "case": (
                    "Swiggy Instamart is growing fast but each dark store takes 12–18 months to break even. "
                    "The CFO wants to reduce that to 9 months. How would you approach this?"
                ),
                "interaction": (
                    "Probe: dark store economics (fixed costs, SKU mix, wastage), "
                    "demand density per pin code, basket size optimisation, "
                    "private label margins, delivery cost per order."
                ),
            },
            {
                "case": (
                    "Swiggy's food delivery business is profitable in the top 10 cities but loss-making "
                    "in the next 30 cities. Should Swiggy exit those cities or invest to fix them?"
                ),
                "interaction": (
                    "Probe: unit economics per city (contribution margin, fixed cost allocation), "
                    "competitive intensity in Tier-2, brand awareness, path to break-even timeline."
                ),
            },
        ],
    },
    "cred": {
        "display_name": "CRED",
        "fun_title": "Pay Day",
        "context": (
            "CRED launched in 2018 as a credit card bill payment app targeting high-credit-score Indians. "
            "It grew to 12M+ members by 2022, expanded into lending (CRED Cash), travel, and commerce, "
            "and is valued at $6.4B — still pre-profitability."
        ),
        "cases": [
            {
                "case": (
                    "CRED has 12 million members who pay credit card bills but only 15 % use any of its "
                    "monetisation products (CRED Cash, CRED Travel, CRED Store). "
                    "How would you improve monetisation without alienating the core user base?"
                ),
                "interaction": (
                    "Probe: user segmentation (high-spend vs low-spend), product-market fit of each vertical, "
                    "trust and brand perception, cross-sell funnel, CAC vs LTV per product."
                ),
            },
            {
                "case": (
                    "CRED is considering launching a UPI payments product to compete with PhonePe and GPay. "
                    "Should they do it, and if so how?"
                ),
                "interaction": (
                    "Probe: market sizing and competitive intensity, CRED's differentiation angle, "
                    "regulatory requirements, monetisation model for UPI, "
                    "risk of distraction from core credit card business."
                ),
            },
        ],
    },
    "meesho": {
        "display_name": "Meesho",
        "fun_title": "Resale Royale",
        "context": (
            "Meesho started in 2015 as a social commerce platform enabling resellers to sell via WhatsApp. "
            "It pivoted to a direct-to-consumer marketplace in 2021, targeting Tier-2/3 India with "
            "zero-commission for sellers and free delivery. It reached 140M+ annual transacting users by 2023."
        ),
        "cases": [
            {
                "case": (
                    "Meesho offers zero commission to sellers and free returns to buyers. "
                    "Its logistics cost per order is ₹65 and average order value is ₹350. "
                    "How would you build a path to profitability?"
                ),
                "interaction": (
                    "Probe: monetisation levers (ads, fulfilment fees, financial services), "
                    "logistics cost reduction (own fleet vs 3PL), return rate reduction, "
                    "AOV improvement through category mix."
                ),
            },
            {
                "case": (
                    "Meesho wants to expand from unbranded / value products into branded goods "
                    "to increase AOV. How should they approach this without losing their core Tier-2 user base?"
                ),
                "interaction": (
                    "Probe: brand willingness to sell on Meesho, pricing perception risk, "
                    "logistics and return handling for branded goods, "
                    "user segmentation (value vs aspirational buyers)."
                ),
            },
        ],
    },
    "zepto": {
        "display_name": "Zepto",
        "fun_title": "10 Minutes to Glory",
        "context": (
            "Zepto was founded in 2021 by two Stanford dropouts. It pioneered 10-minute grocery delivery "
            "in India through a dark store network, raised $1.4B+ by 2024, and reached a $5B valuation — "
            "one of the fastest-growing startups in Indian history."
        ),
        "cases": [
            {
                "case": (
                    "Zepto operates 350+ dark stores across 10 cities. Each store costs ₹40L to set up "
                    "and takes 8 months to break even. A competitor is undercutting prices by 10 %. "
                    "How should Zepto respond?"
                ),
                "interaction": (
                    "Probe: price elasticity of Zepto's user base, margin structure, "
                    "non-price differentiation (speed, assortment, reliability), "
                    "competitor's ability to sustain the price cut, selective vs blanket response."
                ),
            },
            {
                "case": (
                    "Zepto is considering expanding beyond groceries into pharmacy, electronics, and apparel. "
                    "Which category should they enter first and why?"
                ),
                "interaction": (
                    "Probe: category economics (margin, return rate, perishability), "
                    "dark store compatibility, regulatory requirements for pharmacy, "
                    "competitive landscape per category, user demand signals."
                ),
            },
        ],
    },
    "byjus": {
        "display_name": "Byju's",
        "fun_title": "Class Dismissed",
        "context": (
            "Byju's was founded in 2011 and became the world's most valuable ed-tech company at $22B in 2022. "
            "It grew through aggressive sales, acquisitions (Aakash, WhiteHat Jr), and international expansion. "
            "By 2023 it faced a severe liquidity crisis, regulatory scrutiny, and mass layoffs."
        ),
        "cases": [
            {
                "case": (
                    "Byju's spent ₹2,400 crore on sales and marketing in FY22 — more than its revenue. "
                    "Its CAC was ₹15,000 per student against an average LTV of ₹18,000. "
                    "How would you restructure the business to reach sustainable unit economics?"
                ),
                "interaction": (
                    "Probe: CAC reduction levers (channel mix, referrals, school partnerships), "
                    "LTV improvement (retention, upsell, completion rates), "
                    "product-led growth vs sales-led growth, which segments to prioritise."
                ),
            },
            {
                "case": (
                    "Byju's acquired WhiteHat Jr for $300M in 2020 to enter coding education. "
                    "By 2022 WhiteHat Jr was losing money and had significant brand damage. "
                    "What went wrong and what would you have done differently?"
                ),
                "interaction": (
                    "Probe: acquisition rationale vs reality, integration failures, "
                    "sales culture mismatch, product-market fit of live coding for young children, "
                    "post-acquisition governance."
                ),
            },
        ],
    },
    "ola": {
        "display_name": "Ola",
        "fun_title": "Ride or Die",
        "context": (
            "Ola was founded in 2010 and became India's largest ride-hailing platform. "
            "It expanded to the UK, Australia, and New Zealand, launched Ola Electric in 2017, "
            "and by 2024 Ola Electric had gone public — while the ride-hailing business faced "
            "intense competition from Uber and driver-partner unrest."
        ),
        "cases": [
            {
                "case": (
                    "Ola's driver-partner churn is 35 % per month in metro cities. "
                    "Each new driver costs ₹8,000 to onboard and train. "
                    "How would you reduce churn and improve driver economics?"
                ),
                "interaction": (
                    "Probe: root causes of churn (earnings, flexibility, platform fees, support quality), "
                    "driver segmentation (full-time vs part-time), incentive structure redesign, "
                    "comparison with Uber's driver proposition."
                ),
            },
            {
                "case": (
                    "Ola Electric sold 500,000 scooters in FY24 but has a 40 % service complaint rate. "
                    "This is hurting repeat purchases and brand perception. "
                    "How would you fix the post-sale service problem?"
                ),
                "interaction": (
                    "Probe: service network density vs vehicle density, "
                    "software vs hardware issues breakdown, "
                    "cost of building own service centres vs authorised partner model, "
                    "impact on NPS and repeat purchase rate."
                ),
            },
        ],
    },
    "flipkart": {
        "display_name": "Flipkart",
        "fun_title": "Cart Before the Horse",
        "context": (
            "Flipkart was founded in 2007 as an online bookstore and grew into India's largest e-commerce "
            "platform. It launched Myntra (fashion) and PhonePe (payments), was acquired by Walmart "
            "for $16B in 2018, and continues to battle Amazon India for market leadership."
        ),
        "cases": [
            {
                "case": (
                    "Flipkart's fashion vertical (Myntra + Flipkart Fashion) has a 35 % return rate "
                    "versus 8 % for electronics. Returns cost ₹120 per order in logistics and processing. "
                    "How would you reduce the return rate without hurting conversion?"
                ),
                "interaction": (
                    "Probe: root causes of returns (size mismatch, quality, impulse buying), "
                    "AR try-on and size recommendation tools, return policy tightening trade-offs, "
                    "seller quality standards, impact on GMV if policy tightened."
                ),
            },
            {
                "case": (
                    "Flipkart wants to grow its advertising revenue from ₹3,000 crore to ₹10,000 crore "
                    "in three years. How would you build this business?"
                ),
                "interaction": (
                    "Probe: current ad product suite (sponsored listings, display, video), "
                    "seller vs brand advertiser mix, measurement and attribution capabilities, "
                    "pricing model (CPC vs CPM vs CPA), comparison with Amazon Ads."
                ),
            },
        ],
    },
}

# ── Prompt templates ──────────────────────────────────────────────────────────

TOPIC_GREETING_PROMPT_TEMPLATE = '''
Your name is Glee and you are conducting a {topic_name} case study interview.
Your primary role is to emulate a real, empathetic human interviewer, speaking naturally and conversationally.
Respond in a single paragraph of plain-continuous text, without using special characters or formatting like bold, italics, or coding text, as if you were speaking aloud.

Your [INSTRUCTIONS] are:

1. Start with a Warm Greeting: Begin with a friendly and personal greeting.

2. Introduce Yourself: State your name and your role for the session.

3. Explain the Format: Briefly outline what the candidate can expect. This is a {topic_name} case interview. The focus is on their thought process and structured problem-solving approach, not just the final answer. Encourage them to think out loud.

4. Invite Questions: Explicitly ask the candidate if they have any questions ONLY about the process before you start.

5. Listen and Respond: Patiently wait for their response. If they have questions, answer them clearly and concisely but only in the context of the interview.
'''

TOPIC_QUESTION_PROMPT_TEMPLATE = """
You are an interviewer conducting a {topic_name} case study interview AND SIMPLY FOLLOW [INSTRUCTIONS].
Your primary role is to emulate a real, empathetic human interviewer, speaking naturally and conversationally.
Respond in a single paragraph of plain-continuous text, without using special characters or formatting like bold, italics, or coding text, as if you were speaking aloud.

{framework_hints}

Your [INSTRUCTIONS] are:

1. Present the case: CASE QUESTION: {case_question}

CASE REFERENCE (for your use only — do NOT read this out): {case_reference}

2. Invite the interviewee to ask any clarifying questions before structuring their approach.

3. Begin and continue the conversation using the CASE REFERENCE to guide your cross-questions.

4. Do NOT reveal the CASE REFERENCE or framework hints to the candidate.
"""

COMPANY_GREETING_PROMPT_TEMPLATE = '''
Your name is Glee and you are conducting a case study interview based on {company_name}'s real growth story.
This session is titled "{fun_title}".
Your primary role is to emulate a real, empathetic human interviewer, speaking naturally and conversationally.
Respond in a single paragraph of plain-continuous text, without using special characters or formatting like bold, italics, or coding text, as if you were speaking aloud.

Your [INSTRUCTIONS] are:

1. Start with a Warm Greeting: Begin with a friendly and personal greeting.

2. Introduce Yourself: State your name and your role for the session.

3. Set the Stage: Briefly share this context about {company_name}: {company_context}

4. Explain the Format: Tell the candidate they will be given a specific business problem that {company_name} actually faced. The focus is on their structured thinking and problem-solving process, not just the final answer. Encourage them to think out loud.

5. Invite Questions: Explicitly ask the candidate if they have any questions ONLY about the process before you start.

6. Listen and Respond: Patiently wait for their response. If they have questions, answer them clearly and concisely but only in the context of the interview.
'''

COMPANY_QUESTION_PROMPT_TEMPLATE = """
You are an interviewer conducting a case study interview about a real {company_name} business problem AND SIMPLY FOLLOW [INSTRUCTIONS].
Your primary role is to emulate a real, empathetic human interviewer, speaking naturally and conversationally.
Respond in a single paragraph of plain-continuous text, without using special characters or formatting like bold, italics, or coding text, as if you were speaking aloud.

Your [INSTRUCTIONS] are:

1. Present the case: CASE QUESTION: {case_question}

CASE REFERENCE (for your use only — do NOT read this out): {case_reference}

2. Invite the interviewee to ask any clarifying questions before structuring their approach.

3. Begin and continue the conversation using the CASE REFERENCE to guide your cross-questions. Keep the {company_name} context in mind throughout.

4. Do NOT reveal the CASE REFERENCE to the candidate.
"""

# ── Practice case studies database (generic fallback) ────────────────────────

# Practice case studies database
practice_cases = {
    "ecommerce_revenue_drop": {
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
"""
    },
    "food_delivery_expansion": {
        "case": """A food delivery startup operating in 5 cities wants to expand to 20 cities in the next year. 
What factors should they consider and how should they prioritize?""",
        "interaction": """
Key considerations:
- Market size and demand analysis for each city
- Operational infrastructure (riders, restaurants, logistics)
- Unit economics and profitability per city
- Competition landscape
- Regulatory requirements
- Technology scalability
- Marketing and customer acquisition costs
"""
    },
    "subscription_churn": {
        "case": """A SaaS company has a 5% monthly churn rate. They want to reduce it to 3%. 
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
"""
    },
    "retail_store_location": {
        "case": """A retail chain wants to open 10 new stores. How would you help them decide which locations to choose?""",
        "interaction": """
Evaluation factors:
- Demographics (population, income levels, age distribution)
- Foot traffic and accessibility
- Competition density
- Real estate costs (rent, maintenance)
- Local regulations and permits
- Parking availability
- Proximity to complementary businesses
"""
    },
    "mobile_app_engagement": {
        "case": """A social media app has 1 million downloads but only 100K monthly active users. 
How would you improve engagement?""",
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
"""
    },
    "marketplace_liquidity": {
        "case": """A two-sided marketplace connecting freelancers and clients is struggling with supply-demand imbalance. 
Too many freelancers, not enough clients. What should they do?""",
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
"""
    },
    "product_pricing": {
        "case": """An ed-tech company currently charges ₹999/month. They're considering changing to ₹9999/year. 
How would you evaluate this decision?""",
        "interaction": """
Analysis framework:
- Current customer LTV and payback period
- Cash flow implications
- Customer preference research
- Churn impact modeling
- Competitive pricing analysis
- Unit economics comparison
- Implementation and communication plan
"""
    },
    "logistics_optimization": {
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
"""
    },
    "content_platform_monetization": {
        "case": """A content platform with 5M monthly users is currently free. 
They want to introduce monetization. What options should they consider?""",
        "interaction": """
Monetization models:
- Subscription (freemium vs. paywall)
- Advertising (display, native, sponsored content)
- Transaction fees (marketplace model)
- Hybrid approach
- User segmentation for pricing
- Impact on user growth
- Competitive landscape
"""
    },
    "customer_acquisition": {
        "case": """A fintech app is spending ₹500 to acquire each customer but LTV is only ₹400. 
How would you address this?""",
        "interaction": """
Solutions to explore:
- Improve LTV (increase engagement, cross-sell, reduce churn)
- Reduce CAC (optimize marketing channels, referrals, virality)
- Target different customer segments
- Adjust product pricing
- Focus on retention vs acquisition
- Unit economics by channel analysis
"""
    },
    "market_entry": {
        "case": """A successful Indian startup wants to expand to Southeast Asia. 
What framework would you use to evaluate this decision?""",
        "interaction": """
Evaluation criteria:
- Market size and growth potential
- Competitive landscape
- Regulatory environment
- Cultural and consumer behavior differences
- Go-to-market strategy
- Resource requirements
- Risk assessment
- ROI projections
"""
    },
    "feature_prioritization": {
        "case": """A product manager has 5 features to build but can only do 2 this quarter. 
How should they decide?""",
        "interaction": """
Prioritization framework:
- Impact on key metrics (engagement, revenue, retention)
- Engineering effort and complexity
- Customer pain point severity
- Strategic alignment
- Competitive necessity
- Dependencies
- RICE or similar scoring
"""
    },
    "crisis_management": {
        "case": """A food delivery app had a data breach exposing customer payment information. 
How should they respond?""",
        "interaction": """
Response plan:
- Immediate containment and assessment
- Customer communication strategy
- Legal and regulatory compliance
- PR and brand damage control
- Compensation and goodwill measures
- Long-term security improvements
- Stakeholder management (investors, partners)
"""
    },
    "partnership_evaluation": {
        "case": """An e-commerce company is considering partnering with a major retailer vs. building their own brand. 
How would you evaluate?""",
        "interaction": """
Comparison factors:
- Speed to market
- Brand control and positioning
- Economics (margins, revenue share)
- Customer data access
- Long-term strategic value
- Resource requirements
- Risk allocation
"""
    },
    "operational_efficiency": {
        "case": """A restaurant chain's food cost is 40% of revenue (industry standard is 30%). 
How would you bring it down?""",
        "interaction": """
Cost reduction levers:
- Menu engineering and optimization
- Supplier negotiations and sourcing
- Portion control and waste reduction
- Inventory management
- Seasonal menu adjustments
- Staff training on preparation
- Technology for demand forecasting
"""
    }
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

1. Present the question: You must present the follow case question based off the CASE QUESTION TITLE and some IMPORTANT CASE REFERENCE of the case-study -
CASE QUESTION TITLE - {case_question} \n\n
CASE REFERENCE - {case_reference}

2. Invite the interviewee to think: Ask the interviewee to process and ask any clarifying questions, if any.

3. Begin the conversation: Begin and continue to hold the conversation (with cross-questions) with the interviewee, strictly using the CASE REFERENCE
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
  # Consulting topic fields (topic-based case interviews)
  consulting_topic: Annotated[str, Field(default="")]
  topic_name: Annotated[str, Field(default="Case Study")]
  # Company growth story fields (real-company case interviews)
  company_slug: Annotated[str, Field(default="")]
  company_name: Annotated[str, Field(default="")]
  fun_title: Annotated[str, Field(default="")]


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
    company_slug: str = state.get("company_slug", "") or ""
    consulting_topic: str = state.get("consulting_topic", "") or ""

    if company_slug and company_slug in COMPANY_STORIES:
      cases = COMPANY_STORIES[company_slug]["cases"]
    elif consulting_topic and consulting_topic in TOPIC_CASES:
      cases = TOPIC_CASES[consulting_topic]["cases"]
    else:
      # Generic fallback — use the original hardcoded practice_cases dict
      case_no = random.randint(0, len(practice_cases) - 1)
      chosen = practice_cases[list(practice_cases.keys())[case_no]]
      state["current_case_question"] = chosen["case"]
      state["current_case_reference"] = chosen["interaction"]
      return state

    chosen = random.choice(cases)
    state["current_case_question"] = chosen["case"]
    state["current_case_reference"] = chosen["interaction"]
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
      company_slug: str = state.get("company_slug", "") or ""
      consulting_topic: str = state.get("consulting_topic", "") or ""

      if company_slug and company_slug in COMPANY_STORIES:
        story = COMPANY_STORIES[company_slug]
        system_text = COMPANY_GREETING_PROMPT_TEMPLATE.format(
          company_name=story["display_name"],
          fun_title=story["fun_title"],
          company_context=story["context"],
        )
      elif consulting_topic and consulting_topic in TOPIC_CASES:
        topic_data = TOPIC_CASES[consulting_topic]
        system_text = TOPIC_GREETING_PROMPT_TEMPLATE.format(
          topic_name=topic_data["display_name"],
        )
      else:
        system_text = CASE_GREETING_PROMPT

      greeting_prompt = ChatPromptTemplate.from_messages([("system", system_text)])
      input_ = greeting_prompt.format_messages() + [{"role": "human", "content": "Start the interview now"}]
      state["messages"] = state["messages"] + input_

    response = Greeting_llm.invoke(state["messages"])
    state["messages"] = state["messages"] + [response]
    state["history"] = state["history"] + "\n" + "Interviewer-" + response.content
    state["LastNode"] = "Greeting"
    return state
  return _Node



def create_case_study_node(CaseStudy_llm):
  def _Node(state: CaseStudyInterviewState):
    if state["LastNode"] != "CaseStudy":
      company_slug: str = state.get("company_slug", "") or ""
      consulting_topic: str = state.get("consulting_topic", "") or ""

      if company_slug and company_slug in COMPANY_STORIES:
        story = COMPANY_STORIES[company_slug]
        formatted_prompt = COMPANY_QUESTION_PROMPT_TEMPLATE.format(
          company_name=story["display_name"],
          case_question=state["current_case_question"],
          case_reference=state["current_case_reference"],
        )
      elif consulting_topic and consulting_topic in TOPIC_CASES:
        topic_data = TOPIC_CASES[consulting_topic]
        formatted_prompt = TOPIC_QUESTION_PROMPT_TEMPLATE.format(
          topic_name=topic_data["display_name"],
          framework_hints=TOPIC_FRAMEWORK_HINTS.get(consulting_topic, ""),
          case_question=state["current_case_question"],
          case_reference=state["current_case_reference"],
        )
      else:
        formatted_prompt = CASE_QUESTION_PROMPT.format(
          case_question=state["current_case_question"],
          case_reference=state["current_case_reference"],
        )

      case_prompt = ChatPromptTemplate.from_messages([("system", formatted_prompt)])
      state["messages"] = case_prompt.format_messages() + [{"role": "human", "content": "Please present the case study question"}]
    
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