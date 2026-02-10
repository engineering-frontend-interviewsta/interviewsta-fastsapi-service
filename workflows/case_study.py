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
  def _Node(state:CaseStudyInterviewState):
    case_no = random.randint(0,14)
    state['current_case_question'] = practice_cases[list(practice_cases.keys())[case_no]]['case']
    state['current_case_reference'] = practice_cases[list(practice_cases.keys())[case_no]]['interaction']
    # Removed timer logic as per requirements
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
      # Format the prompt with actual case question and reference
      formatted_prompt = CASE_QUESTION_PROMPT.format(
        case_question=state["current_case_question"],
        case_reference=state["current_case_reference"]
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