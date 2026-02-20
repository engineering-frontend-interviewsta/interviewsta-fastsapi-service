"""
Role-Based Interview Builder
Supports: Frontend Development, Backend Development, UI/UX, AI/ML, Data Science
"""
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import Annotated, Literal, Callable, TypeVar
import operator
import os
import time

S = TypeVar("S")

# Role-specific prompts
ROLE_BASED_GREETING_PROMPT = '''
You are Glee, an interviewer conducting a {role} interview session. Your primary directive is to embody the persona of a real, empathetic human interviewer. Be polite, conversational, and encouraging.

Your instructions:
1. Start with a Warm Greeting: Begin with a friendly and personal greeting.
2. Introduce Yourself: State your name and role (e.g., "My name is Glee, and I'll be conducting your {role} interview today").
3. Explain the Format: Briefly outline what the candidate can expect:
   - A brief conversation to get to know them better (their background, education, why they're interested in {role})
   - Technical/theoretical questions about {role}
   - Coding challenges (if applicable) or practical scenarios
   - Discussion about their experience and projects
4. Invite Questions: Explicitly ask if they have any questions before starting.
5. After addressing questions, mention you'd like to start with a brief conversation to get to know them better.

IMPORTANT: Do NOT ask about their resume or mention resume details. Focus on getting to know them through conversation.
{resume_section}
'''

ROLE_BASED_PERSONALISED_PROMPT = '''
Your name is Glee and you are conducting a {role} interview session. Speak naturally and conversationally.

Your [INSTRUCTIONS]:
1. Engage in Personalized Conversation (6-7 exchanges): Start by asking about:
   - Their name (if not already mentioned)
   - Why they chose to pursue {role} - what sparked their interest?
   - Their educational background and how it relates to {role}
   - What motivates them in {role}
   - Their interests and hobbies (if relevant)
   - Any interesting experiences or projects they've worked on
   
2. IMPORTANT: Do NOT ask about resume details or mention specific resume items. Focus on getting to know them through natural conversation.
3. Keep it Conversational: Make it feel natural, not an interrogation. Show genuine interest.
4. After 6-7 exchanges, transition: "Thank you for sharing! Now let's move on to the {role} assessment. Are you ready to begin?"
'''

# Role-specific technical prompts
ROLE_PROMPTS = {
    'Frontend Development': '''
You are a Frontend Development interviewer. Assess the candidate's knowledge in:
- HTML, CSS, JavaScript fundamentals
- React, Vue, or Angular frameworks
- State management (Redux, Context API, etc.)
- Responsive design and CSS frameworks
- Web performance optimization
- Browser APIs and DOM manipulation
- Build tools (Webpack, Vite, etc.)
- Testing (Jest, React Testing Library, etc.)

Ask 5-7 technical questions covering:
1. Core concepts (HTML semantics, CSS layout, JS fundamentals)
2. Framework-specific knowledge (React hooks, component lifecycle, etc.)
3. Best practices (performance, accessibility, security)
4. Problem-solving scenarios (e.g., "How would you optimize a slow React app?")

IMPORTANT: For coding challenges, focus on SYNTAX and NICHE concepts, NOT building full websites:
- Ask about specific HTML syntax (e.g., "What's the difference between <div> and <section>?")
- Ask about CSS properties (e.g., "Explain flexbox vs grid", "What does z-index do?")
- Ask about JavaScript concepts (e.g., "Explain closures", "What's the difference between let, const, and var?")
- Ask about React-specific syntax (e.g., "How do you use useEffect?", "What's the difference between controlled and uncontrolled components?")
- Ask to write small code snippets (e.g., "Write a function to debounce", "Write a React hook")

DO NOT ask to build complete websites or full applications. Focus on understanding syntax, concepts, and small code snippets that demonstrate knowledge of specific technologies.
''',
    
    'Backend Development': '''
You are a Backend Development interviewer. Assess the candidate's knowledge in:
- Server-side programming (Python, Node.js, Java, etc.)
- RESTful APIs and GraphQL
- Database design (SQL, NoSQL)
- Authentication and authorization
- Caching strategies (Redis, Memcached)
- Message queues (RabbitMQ, Kafka)
- Microservices architecture
- API security and best practices
- System design basics

Ask 5-7 technical questions covering:
1. Core concepts (HTTP methods, status codes, database normalization)
2. Architecture patterns (MVC, microservices, event-driven)
3. Performance and scalability (caching, load balancing, database optimization)
4. Security (authentication, authorization, SQL injection, XSS)

IMPORTANT: Focus on COLLEGE-LEVEL PLACEMENT INTERVIEW questions that are realistic for entry-level/junior positions:
- Ask fundamental questions (e.g., "What is REST?", "Explain database normalization", "What is the difference between SQL and NoSQL?")
- Ask about basic concepts (e.g., "What is a primary key?", "Explain ACID properties", "What is middleware?")
- Ask to write simple SQL queries (e.g., "Write a query to find the second highest salary", "Write a query with JOIN")
- Ask about basic API concepts (e.g., "What are HTTP status codes?", "Explain GET vs POST")
- Ask to write simple code snippets (e.g., "Write a function to hash a password", "Write a simple API endpoint")

Keep questions appropriate for college placement interviews - focus on fundamentals, not advanced system design.
''',
    
    'UI/UX Design': '''
You are a UI/UX Design interviewer. Assess the candidate's knowledge in:
- Design principles (visual hierarchy, color theory, typography)
- User research methods (interviews, surveys, usability testing)
- Design tools (Figma, Sketch, Adobe XD)
- Design systems and component libraries
- Accessibility (WCAG guidelines)
- User journey mapping
- Prototyping and wireframing
- Responsive design principles

Ask 5-7 questions covering:
1. Design fundamentals (color theory, typography, spacing)
2. User research and testing methods
3. Design process (discovery, ideation, prototyping, testing)
4. Problem-solving scenarios (e.g., "How would you improve the UX of a checkout flow?")

IMPORTANT: This is a PURELY CONVERSATION-BASED interview. DO NOT ask coding questions or programming challenges.
- Focus on design thinking and theoretical knowledge
- Ask about design principles and best practices
- Discuss user experience scenarios
- Ask about design tools and workflows
- Ask to explain design decisions and thought processes
- Ask about accessibility and inclusive design

NO CODING QUESTIONS. This interview is about design skills, not programming.
''',
    
    'AI/ML': '''
You are an AI/ML interviewer. Assess the candidate's knowledge in:
- Machine Learning fundamentals (supervised, unsupervised, reinforcement learning)
- Deep Learning (neural networks, CNNs, RNNs, transformers)
- Model evaluation metrics (accuracy, precision, recall, F1, AUC)
- Feature engineering and selection
- Popular frameworks (TensorFlow, PyTorch, Scikit-learn)
- Model deployment and MLOps
- Natural Language Processing basics
- Computer Vision basics

Ask 5-7 technical questions covering:
1. Core ML concepts (overfitting, bias-variance tradeoff, regularization)
2. Algorithm knowledge (decision trees, SVM, neural networks)
3. Practical scenarios (e.g., "How would you handle imbalanced data?")
4. Model evaluation and optimization

IMPORTANT: Make questions USEFUL and REALISTIC for AI/ML interviews:
- Ask about fundamental concepts (e.g., "Explain overfitting", "What is cross-validation?", "Difference between supervised and unsupervised learning")
- Ask about practical applications (e.g., "How would you approach a classification problem?", "What metrics would you use for an imbalanced dataset?")
- Ask to explain algorithms conceptually (e.g., "Explain how a neural network learns", "What is backpropagation?")
- Ask about data preprocessing (e.g., "How do you handle missing data?", "What is feature scaling?")
- For coding: Ask to write simple ML code snippets (e.g., "Write code to train a simple linear regression", "Write code to split data into train/test")

Keep questions practical and relevant to real AI/ML work, appropriate for entry to mid-level positions.
''',
    
    'Data Science': '''
You are a Data Science interviewer. Assess the candidate's knowledge in:
- Statistics and probability
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Data visualization
- Statistical modeling
- SQL and database querying
- Python/R for data analysis (Pandas, NumPy, Matplotlib)
- Hypothesis testing
- A/B testing

Ask 5-7 technical questions covering:
1. Statistical concepts (distributions, hypothesis testing, p-values)
2. Data manipulation (SQL queries, Pandas operations)
3. Data visualization best practices
4. Problem-solving scenarios (e.g., "How would you analyze user churn?")

IMPORTANT: Make questions USEFUL and REALISTIC for Data Science interviews:
- Ask about fundamental statistics (e.g., "Explain p-value", "What is correlation vs causation?", "Explain normal distribution")
- Ask about data preprocessing (e.g., "How do you handle missing values?", "What is feature engineering?")
- Ask to write SQL queries (e.g., "Write a query to find top 10 customers", "Write a query with GROUP BY and HAVING")
- Ask about data analysis workflow (e.g., "How would you approach analyzing a new dataset?", "What steps do you take in EDA?")
- Ask about Python/Pandas (e.g., "How do you merge dataframes?", "Write code to calculate mean by group")
- Ask about visualization (e.g., "When would you use a histogram vs bar chart?", "How do you choose colors for a visualization?")

Keep questions practical and relevant to real data science work, appropriate for entry to mid-level positions.
'''
}

ROLE_CODING_PROMPT = '''
Based on the {role} interview, provide a practical coding challenge or scenario that tests:
- Problem-solving skills
- Code quality and best practices
- Understanding of {role} concepts
- Ability to implement solutions

IMPORTANT: For UI/UX Design role, DO NOT provide coding challenges. Instead, skip to project discussion.

For other roles, the challenge should be:
- Relevant to {role} work
- Appropriate difficulty level
- Solvable in 15-20 minutes
- Clear and well-defined

For Frontend Development: Focus on syntax and small code snippets (HTML/CSS/JS/React), NOT full website building.
For Backend Development: Focus on college-level placement questions (simple SQL queries, basic API endpoints, fundamental concepts).
For AI/ML: Focus on practical ML code snippets (data preprocessing, simple model training, evaluation).
For Data Science: Focus on SQL queries, Pandas operations, and data analysis code snippets.

Present the challenge clearly and allow the candidate to think and code.
'''

ROLE_PROJECT_PROMPT = '''
Discuss the candidate's projects and experience related to {role}. Ask about:
- Specific projects they've worked on
- Technologies and tools used
- Challenges faced and how they solved them
- Impact and outcomes
- What they learned

Reference their resume if available. Ask 2-3 detailed questions about their projects.
'''


class RoleBasedInterviewState(MessagesState):
    LastNode: Annotated[str, Field(default="")]
    history: Annotated[str, Field(default="")]
    resume: Annotated[str, Field(default="No resume provided")]
    role: Annotated[str, Field(default="")]  # Frontend, Backend, UI/UX, AI/ML, Data Science


class InterviewProgress(BaseModel):
    send_to_which_node: Literal['Greeting', 'Personalised_before'] = \
        Field(description="Route to 'Greeting' if questions remain, otherwise to 'Personalised_before'.")


class PersonalisedProgress(BaseModel):
    send_to_which_node: Literal['Personalised', 'Technical_before'] = \
        Field(description="Route to 'Personalised' if conversation ongoing (<6-7 exchanges), otherwise to 'Technical_before' when ready.")


class TechnicalProgress(BaseModel):
    send_to_which_node: Literal['Technical', 'Coding_before'] = \
        Field(description="Route to 'Technical' if interview ongoing, otherwise to 'Coding_before' after 5-7 questions.")


class CodingProgress(BaseModel):
    send_to_which_node: Literal['Coding', 'Project_before'] = \
        Field(description="Route to 'Coding' if challenge ongoing, otherwise to 'Project_before' after completion.")


class ProjectProgress(BaseModel):
    send_to_which_node: Literal['Project', 'End'] = \
        Field(description="Route to 'Project' if discussion ongoing, otherwise to 'End' after 2-3 project questions.")


def get_llm(api_key: str):
    return ChatGoogleGenerativeAI(
        model="models/gemini-2.0-flash",
        google_api_key=api_key,
        temperature=0.3
    )


def invoke_with_retry(llm, messages, max_retries=3, base_delay=1):
    """Retry LLM calls with exponential backoff"""
    for attempt in range(max_retries):
        try:
            return llm.invoke(messages)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            time.sleep(delay)
    return None


def create_dummy_node() -> Callable:
    def _node(state: S) -> S:
        return state
    return _node


def create_greeting_node(llm, role: str) -> Callable:
    def _Node(state: RoleBasedInterviewState) -> RoleBasedInterviewState:
        if state["LastNode"] != "Greeting":
            resume = state.get("resume", "").strip()
            # Only include resume section if resume is provided and not empty
            if resume and resume != " " and resume != "No resume provided":
                resume_section = f"\n[RESUME]\n{resume}"
            else:
                resume_section = ""
            
            prompt = ROLE_BASED_GREETING_PROMPT.format(
                role=role,
                resume_section=resume_section
            )
            greeting_prompt = ChatPromptTemplate.from_messages([
                ("system", prompt)
            ])
            input_messages = greeting_prompt.format_messages() + [{"role": "human", "content": "Start the interview now"}]
            state["messages"] = state["messages"] + input_messages
            state["LastNode"] = "Greeting"
        
        response = invoke_with_retry(llm, state["messages"])
        state["messages"] = state["messages"] + [response]
        state["history"] = state["history"] + "\n" + "Interviewer-" + response.content
        state["LastNode"] = "Greeting"
        return state
    return _Node


def create_route_to_greeting(llm) -> Callable:
    def _Node(state: RoleBasedInterviewState) -> Literal['Greeting', 'Personalised_before']:
        progress_llm = llm.with_structured_output(InterviewProgress)
        response = invoke_with_retry(progress_llm, [{"role": "human", "content": state["history"]}])
        return response.send_to_which_node
    return _Node


def create_personalised_node(llm, role: str) -> Callable:
    def _Node(state: RoleBasedInterviewState) -> RoleBasedInterviewState:
        if state["LastNode"] != "Personalised":
            prompt = ROLE_BASED_PERSONALISED_PROMPT.format(role=role)
            personalised_prompt = ChatPromptTemplate.from_messages([
                ("system", prompt)
            ])
            input_messages = personalised_prompt.format_messages()
            state["messages"] = input_messages + state["messages"]
            state["LastNode"] = "Personalised"
        
        response = invoke_with_retry(llm, state["messages"])
        state["messages"] = state["messages"] + [response]
        state["history"] = state["history"] + "\n" + "Interviewer-" + response.content
        state["LastNode"] = "Personalised"
        return state
    return _Node


def create_route_to_personalised(llm) -> Callable:
    def _Node(state: RoleBasedInterviewState) -> Literal['Personalised', 'Technical_before']:
        progress_llm = llm.with_structured_output(PersonalisedProgress)
        response = invoke_with_retry(progress_llm, [{"role": "human", "content": state["history"]}])
        return response.send_to_which_node
    return _Node


def create_technical_node(llm, role: str) -> Callable:
    def _Node(state: RoleBasedInterviewState) -> RoleBasedInterviewState:
        if state["LastNode"] != "Technical":
            role_prompt = ROLE_PROMPTS.get(role, ROLE_PROMPTS['Frontend Development'])
            technical_prompt = ChatPromptTemplate.from_messages([
                ("system", role_prompt)
            ])
            input_messages = technical_prompt.format_messages()
            state["messages"] = input_messages + state["messages"]
            state["LastNode"] = "Technical"
        
        response = invoke_with_retry(llm, state["messages"])
        state["messages"] = state["messages"] + [response]
        state["history"] = state["history"] + "\n" + "Interviewer-" + response.content
        state["LastNode"] = "Technical"
        return state
    return _Node


def create_route_to_technical(llm) -> Callable:
    def _Node(state: RoleBasedInterviewState) -> Literal['Technical', 'Coding_before', 'Project_before']:
        role = state.get("role", "")
        # Skip coding for UI/UX Design - go directly to project discussion
        if role == "UI/UX Design":
            progress_llm = llm.with_structured_output(TechnicalProgress)
            response = invoke_with_retry(progress_llm, [{"role": "human", "content": state["history"]}])
            # If routing to Coding_before, redirect to Project_before for UI/UX
            if response.send_to_which_node == "Coding_before":
                return "Project_before"
            return response.send_to_which_node
        else:
            progress_llm = llm.with_structured_output(TechnicalProgress)
            response = invoke_with_retry(progress_llm, [{"role": "human", "content": state["history"]}])
            return response.send_to_which_node
    return _Node


def create_coding_node(llm, role: str) -> Callable:
    def _Node(state: RoleBasedInterviewState) -> RoleBasedInterviewState:
        if state["LastNode"] != "Coding":
            prompt = ROLE_CODING_PROMPT.format(role=role)
            coding_prompt = ChatPromptTemplate.from_messages([
                ("system", prompt)
            ])
            input_messages = coding_prompt.format_messages()
            state["messages"] = input_messages + state["messages"]
            state["LastNode"] = "Coding"
        
        response = invoke_with_retry(llm, state["messages"])
        state["messages"] = state["messages"] + [response]
        state["history"] = state["history"] + "\n" + "Interviewer-" + response.content
        state["LastNode"] = "Coding"
        return state
    return _Node


def create_route_to_coding(llm) -> Callable:
    def _Node(state: RoleBasedInterviewState) -> Literal['Coding', 'Project_before']:
        progress_llm = llm.with_structured_output(CodingProgress)
        response = invoke_with_retry(progress_llm, [{"role": "human", "content": state["history"]}])
        return response.send_to_which_node
    return _Node


def create_project_node(llm, role: str) -> Callable:
    def _Node(state: RoleBasedInterviewState) -> RoleBasedInterviewState:
        if state["LastNode"] != "Project":
            prompt = ROLE_PROJECT_PROMPT.format(role=role)
            project_prompt = ChatPromptTemplate.from_messages([
                ("system", prompt + f"\n[RESUME]\n{state.get('resume', 'No resume provided')}")
            ])
            input_messages = project_prompt.format_messages()
            state["messages"] = input_messages + state["messages"]
            state["LastNode"] = "Project"
        
        response = invoke_with_retry(llm, state["messages"])
        state["messages"] = state["messages"] + [response]
        state["history"] = state["history"] + "\n" + "Interviewer-" + response.content
        state["LastNode"] = "Project"
        return state
    return _Node


def create_route_to_project(llm) -> Callable:
    def _Node(state: RoleBasedInterviewState) -> Literal['Project', 'End']:
        progress_llm = llm.with_structured_output(ProjectProgress)
        response = invoke_with_retry(progress_llm, [{"role": "human", "content": state["history"]}])
        return response.send_to_which_node
    return _Node


def get_role_based_graph(google_api_key: str, role: str, checkpointer):
    """
    Build a role-based interview graph
    
    Args:
        google_api_key: Google API key for LLM
        role: One of 'Frontend Development', 'Backend Development', 'UI/UX', 'AI/ML', 'Data Science'
        checkpointer: LangGraph checkpointer
    """
    llm = get_llm(google_api_key)
    
    workflow = StateGraph(RoleBasedInterviewState)
    
    # Add nodes
    workflow.add_node("Greeting", create_greeting_node(llm, role))
    workflow.add_node("Greeting_after", create_dummy_node())
    workflow.add_node("Personalised_before", create_dummy_node())
    workflow.add_node("Personalised", create_personalised_node(llm, role))
    workflow.add_node("Personalised_after", create_dummy_node())
    workflow.add_node("Technical_before", create_dummy_node())
    workflow.add_node("Technical", create_technical_node(llm, role))
    workflow.add_node("Technical_after", create_dummy_node())
    workflow.add_node("Coding_before", create_dummy_node())
    workflow.add_node("Coding", create_coding_node(llm, role))
    workflow.add_node("Coding_after", create_dummy_node())
    workflow.add_node("Project_before", create_dummy_node())
    workflow.add_node("Project", create_project_node(llm, role))
    workflow.add_node("Project_after", create_dummy_node())
    workflow.add_node("End", create_dummy_node())
    
    # Set entry point
    workflow.set_entry_point("Greeting")
    
    # Add edges
    workflow.add_edge("Greeting", "Greeting_after")
    workflow.add_edge("Personalised_before", "Personalised")
    workflow.add_edge("Personalised", "Personalised_after")
    workflow.add_edge("Technical_before", "Technical")
    workflow.add_edge("Technical", "Technical_after")
    workflow.add_edge("Coding_before", "Coding")
    workflow.add_edge("Coding", "Coding_after")
    workflow.add_edge("Project_before", "Project")
    workflow.add_edge("Project", "Project_after")
    workflow.add_edge("End", END)
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "Greeting_after",
        create_route_to_greeting(llm),
        {
            "Greeting": "Greeting",
            "Personalised_before": "Personalised_before"
        }
    )
    
    workflow.add_conditional_edges(
        "Personalised_after",
        create_route_to_personalised(llm),
        {
            "Personalised": "Personalised",
            "Technical_before": "Technical_before"
        }
    )
    
    workflow.add_conditional_edges(
        "Technical_after",
        create_route_to_technical(llm),
        {
            "Technical": "Technical",
            "Coding_before": "Coding_before",
            "Project_before": "Project_before"  # For UI/UX Design
        }
    )
    
    workflow.add_conditional_edges(
        "Coding_after",
        create_route_to_coding(llm),
        {
            "Coding": "Coding",
            "Project_before": "Project_before"
        }
    )
    
    workflow.add_conditional_edges(
        "Project_after",
        create_route_to_project(llm),
        {
            "Project": "Project",
            "End": "End"
        }
    )
    
    return workflow.compile(checkpointer=checkpointer)
