# Interview Flow Verification (Final Check)

This document confirms that all 7 interview types are correctly wired in the FastAPI service: graph source, interrupt nodes, initial state, and state updates.

---

## 1. Graph source & interrupt nodes (interview_agent.py)

| Interview Type        | get_graph() source           | Workflow file     | *_after nodes in workflow | INTERRUPT_NODES | Match |
|-----------------------|-----------------------------|-------------------|---------------------------|-----------------|-------|
| **Technical**         | get_technical_graph          | technical.py      | Greeting_after, Technical_after, Coding_after, Project_after | Same 4 | Yes |
| **HR**                | get_hr_graph                | hr.py             | Greeting_after, HR_after | Same 2 | Yes |
| **Company**           | build_company_graph         | companybuilder.py | Greeting_after, Personalised_after, Conceptual_after, Project_after, ProductScenario_after, LogicalReasoning_after, Coding_after | Same 7 | Yes |
| **Subject**           | get_coding_graph("Subject")  | coding.py         | Greeting_after, Personalised_after, Coding_after | Same 3 | Yes |
| **CaseStudy**         | build_case_study_graph      | case_study.py     | Greeting_after, CaseStudy_after | Same 2 | Yes |
| **Communication**     | build_communication_graph  | communication.py  | Greeting_after, Rapport_after, PersonalDetails_after, Speaking_after, Speaking_feedback_after, Comprehension_after, Comprehension_feedback_after, MCQ_after | Same 8 | Yes |
| **Role-Based Interview** | get_role_based_graph(role) | rolebased.py      | Greeting_after, Personalised_after, Technical_after, Coding_after, Project_after | Same 5 | Yes |

All interrupt lists match the actual *_after nodes in each workflow. Company uses the full company graph (not the coding graph).

---

## 2. Conditional edges (routing after each interrupt)

Each workflow has `add_conditional_edges` for every *_after node listed above:

- **technical.py**: Greeting_after, Technical_after, Coding_after, Project_after
- **hr.py**: Greeting_after, HR_after
- **companybuilder.py**: Greeting_after, Personalised_after, Conceptual_after, Project_after, ProductScenario_after, LogicalReasoning_after, Coding_after
- **coding.py** (Subject only): Greeting_after, Personalised_after, Coding_after
- **case_study.py**: Greeting_after, CaseStudy_after
- **communication.py**: Greeting_after, Rapport_after, PersonalDetails_after, Speaking_after, MCQ_after (Comprehension_after is direct edge to Comprehension_feedback; interrupt_before still applies)
- **rolebased.py**: Greeting_after, Personalised_after, Technical_after, Coding_after, Project_after

---

## 3. Initial state (create_initial_state)

| Interview Type        | State class                    | Key payload fields | Notes |
|-----------------------|--------------------------------|--------------------|-------|
| Technical             | TechnicalInterviewState        | resume, TechnicalResearch, CodingResearch | OK |
| HR                    | HRInterviewState               | resume | OK |
| Company               | CompanyInterviewStateBuilder   | company, QuestionResearch, Difficulty, Tags, resume | Tags coerced to str if list |
| Subject               | SubjectInterviewState         | subject, QuestionResearch, Difficulty, Tags | Tags coerced to str if list |
| CaseStudy             | CaseStudyInterviewState        | (interview_type_id in payload) | OK |
| Communication         | CommunicationInterviewState    | (interview_type_id in payload) | OK |
| Role-Based Interview  | RoleBasedInterviewState        | resume, role, (interview_type_id) | OK |

Company uses state from workflows.companybuilder (includes resume). Subject uses state from workflows.coding.

---

## 4. State update on user response (update_workflow_state)

- **CaseStudy, Communication**: HumanMessage appended; Communication has MCQ/pending_mcq_answer handling.
- **Technical, HR, Company, Subject, Role-Based**: HumanMessage appended (no raw string), history updated. Same path for all five.

---

## 5. API & frontend alignment

- **Schemas** (schemas/interview.py): interview_type is Literal["Technical", "HR", "Company", "Subject", "CaseStudy", "Communication", "Role-Based Interview"].
- **Frontend** (InterviewInterface.jsx / VideoInterview.jsx): Sends the same strings (e.g. "Company", "Subject", "Role-Based Interview") when starting and when polling.

---

## 6. Summary

- All 7 interview types have the correct graph loaded in get_graph().
- INTERRUPT_NODES for each type exactly match the *_after nodes in that workflow.
- Company uses build_company_graph (full company graph); Subject uses get_coding_graph (subject-only graph).
- create_initial_state uses the correct state class per type; Company has resume; Tags is coerced to string for Company/Subject when frontend sends an array.
- update_workflow_state uses HumanMessage for all types and handles Communication/CaseStudy specifics.

Verification date: 2025-02-21.
