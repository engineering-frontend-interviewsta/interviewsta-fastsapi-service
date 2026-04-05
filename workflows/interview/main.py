from workflows.interview.db_models import Interview, InterviewPhase
from workflows.interview.db_models import engine, Session
from workflows.interview.db_models import get_interview_config
from workflows.interview.db_models import INTERVIEW_TYPE
from workflows.interview.db_models import InMemorySaver
from workflows.interview.db_models import build_graph_for_interview
import os
import prisma

def get_interview_config(interview_id: str) -> dict:
    interview = prisma.interview.find_unique(where={"id": interview_id})
    if not interview:
        raise ValueError(f"Interview {interview_id} not found")
    return {
        "title": interview.title,
        "difficulty": interview.difficulty,
        "company": interview.company if interview.company else "",
        "subject": interview.subject if interview.subject else None,
        "description": interview.description if interview.description else None,
        "tags": interview.tags if interview.tags else None,
    }

def start_interview(interview_id: str):
    interview_meta = get_interview_config(interview_id)

    with Session(engine) as session:

        interview = Interview(
            name=interview_meta["name"],
            difficulty=interview_meta["difficulty"],
            company=interview_meta["company"],
            subject=interview_meta["subject"],
            tags=interview_meta["tags"],
            greeting_prompt=interview_meta.get("greeting_prompt"),  # stored in DB
        )
        session.add(interview)
        session.flush()

        for phase_cfg in PHASES:
            session.add(InterviewPhase(
                interview_id=interview.id,
                phase_name=phase_cfg.phase_name,
                order=phase_cfg.order,
                prompt=phase_cfg.prompt,
                prompt_inputs=phase_cfg.prompt_inputs,
                number_of_questions_to_ask=phase_cfg.number_of_questions_to_ask,
                setup_questions=phase_cfg.setup_questions,
                setup_questions_prompt=phase_cfg.setup_questions_prompt,
                question_filters=phase_cfg.question_filters,
                route_nodes=phase_cfg.route_nodes,
                route_ahead_prompt=phase_cfg.route_ahead_prompt,
                immediate_feedback_required=phase_cfg.immediate_feedback_required,
                feedback_prompt=phase_cfg.feedback_prompt,
                mcp_tools=phase_cfg.mcp_tools,
                tool_names=phase_cfg.tool_names,
                special_output_format=phase_cfg.special_output_format,
                entity_schema=phase_cfg.entity_schema,
            ))
        session.commit()
        print(f"[DB] Inserted interview id={interview.id}  "
            f"type={INTERVIEW_TYPE!r}  name={interview_meta['name']!r}")
        print(f"[DB] greeting_prompt='{'custom' if interview_meta.get('greeting_prompt') else 'generic engine default'}'")

        db_agent, initial_extras = build_graph_for_interview(
            interview_id=interview.id,
            db_session=session,
            google_api_key=os.environ["GOOGLE_API_KEY"],
            checkpointer=InMemorySaver(),
        )