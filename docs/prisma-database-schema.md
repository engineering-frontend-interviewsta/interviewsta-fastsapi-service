# Database schema (from Prisma introspection)

This document lists **PostgreSQL `public` schema** tables and columns as reflected in [`prisma/schema.prisma`](../prisma/schema.prisma). It is generated from the live database via `npx prisma db pull`.

**Regenerate:**

```bash
npx prisma@5 db pull --schema=prisma/schema.prisma
python3 scripts/generate_prisma_schema_docs.py
```

**Summary:** 33 models (tables), 12 enums.

---

## Table of contents

- [announcements](#announcements)
- [assignment_files](#assignment-files)
- [assignment_submissions](#assignment-submissions)
- [assignments](#assignments)
- [classes](#classes)
- [companies](#companies)
- [credits](#credits)
- [feedback_items](#feedback-items)
- [feedbacks](#feedbacks)
- [institutions](#institutions)
- [interview_phases](#interview-phases)
- [interview_test_interview_phases](#interview-test-interview-phases)
- [interview_test_subjects](#interview-test-subjects)
- [interview_tests](#interview-tests)
- [parent_interview_tests](#parent-interview-tests)
- [question_companies](#question-companies)
- [question_subjects](#question-subjects)
- [questions](#questions)
- [resume_analysis](#resume-analysis)
- [student_classes](#student-classes)
- [subject_questions](#subject-questions)
- [subject_related_subjects](#subject-related-subjects)
- [subjects](#subjects)
- [subscription_tier_prices](#subscription-tier-prices)
- [subscription_tiers](#subscription-tiers)
- [subscriptions](#subscriptions)
- [time_slots](#time-slots)
- [transaction_logs](#transaction-logs)
- [user](#user)
- [user_institutions](#user-institutions)
- [user_parent_interview_scores](#user-parent-interview-scores)
- [user_parent_interview_weekly_scores](#user-parent-interview-weekly-scores)
- [user_sessions](#user-sessions)
- [Enums](#enums)

---

## `announcements`

| Field | Type and attributes |
| --- | --- |
| `id` | `String   @id(map: "PK_b3ad760876ff2e19d58e05dc8b0") @default(dbgenerated("uuid_generate_v4()")) @db.Uuid` |
| `class_id` | `String   @db.Uuid` |
| `teacher_id` | `String   @db.Uuid` |
| `title` | `String   @db.VarChar(200)` |
| `content` | `String` |
| `created_at` | `DateTime @default(now()) @db.Timestamp(6)` |
| `updated_at` | `DateTime @default(now()) @db.Timestamp(6)` |
| `classes` | `classes  @relation(fields: [class_id], references: [id], onDelete: Cascade, onUpdate: NoAction, map: "FK_8adc55f739d19d84f29e5ffcfb4")` |
| `user` | `user     @relation(fields: [teacher_id], references: [id], onDelete: Cascade, onUpdate: NoAction, map: "FK_a86942064f1e5eeb3bc8a84cb4a")` |

---

## `assignment_files`

| Field | Type and attributes |
| --- | --- |
| `id` | `String      @id(map: "PK_96f7fab55e3a114cc7a66e1c929") @default(dbgenerated("uuid_generate_v4()")) @db.Uuid` |
| `assignment_id` | `String      @db.Uuid` |
| `file_path` | `String      @db.VarChar(500)` |
| `uploaded_at` | `DateTime    @default(now()) @db.Timestamp(6)` |
| `assignments` | `assignments @relation(fields: [assignment_id], references: [id], onDelete: Cascade, onUpdate: NoAction, map: "FK_74f85d9ad270cea3b7822979e93")` |

---

## `assignment_submissions`

| Field | Type and attributes |
| --- | --- |
| `id` | `String      @id(map: "PK_0caedc49d0357bedac05ca5a806") @default(dbgenerated("uuid_generate_v4()")) @db.Uuid` |
| `assignment_id` | `String      @db.Uuid` |
| `student_id` | `String      @db.Uuid` |
| `submission_document_path` | `String?     @db.VarChar(500)` |
| `submission_link` | `String?     @db.VarChar(2048)` |
| `submitted_at` | `DateTime?   @db.Timestamptz(6)` |
| `submission_count` | `Int         @default(0)` |
| `grade` | `Float?` |
| `feedback` | `String      @default("")` |
| `graded_at` | `DateTime?   @db.Timestamptz(6)` |
| `graded_by_id` | `String?     @db.Uuid` |
| `status` | `String      @default("not_started") @db.VarChar(20)` |
| `assignments` | `assignments @relation(fields: [assignment_id], references: [id], onDelete: Cascade, onUpdate: NoAction, map: "FK_0c62b946a9e40285ac33fe970bb")` |
| `user_assignment_submissions_student_idTouser` | `user        @relation("assignment_submissions_student_idTouser", fields: [student_id], references: [id], onDelete: Cascade, onUpdate: NoAction, map: "FK_45b95e9a93646e79972f824a93f")` |
| `user_assignment_submissions_graded_by_idTouser` | `user?       @relation("assignment_submissions_graded_by_idTouser", fields: [graded_by_id], references: [id], onUpdate: NoAction, map: "FK_95883a3791ec0ef9f9f154ac3f3")` |

**Constraints / indexes:**
- `@@unique([assignment_id, student_id], map: "IDX_3dad9c5ba25b48cd6a82c8c676")`

---

## `assignments`

| Field | Type and attributes |
| --- | --- |
| `id` | `String                   @id(map: "PK_c54ca359535e0012b04dcbd80ee") @default(dbgenerated("uuid_generate_v4()")) @db.Uuid` |
| `class_assigned_id` | `String                   @db.Uuid` |
| `teacher_id` | `String                   @db.Uuid` |
| `title` | `String                   @db.VarChar(200)` |
| `description` | `String                   @default("")` |
| `instructions` | `String                   @default("")` |
| `deadline` | `DateTime?                @db.Timestamptz(6)` |
| `estimated_time_minutes` | `Int                      @default(0)` |
| `submission_type` | `String                   @default("document") @db.VarChar(20)` |
| `max_submissions` | `Int                      @default(1)` |
| `allow_late_submission` | `Boolean                  @default(true)` |
| `created_at` | `DateTime                 @default(now()) @db.Timestamp(6)` |
| `updated_at` | `DateTime                 @default(now()) @db.Timestamp(6)` |
| `assignment_files` | `assignment_files[]` |
| `assignment_submissions` | `assignment_submissions[]` |
| `user` | `user                     @relation(fields: [teacher_id], references: [id], onDelete: Cascade, onUpdate: NoAction, map: "FK_27322fa090b5deacc5785fcb94c")` |
| `classes` | `classes                  @relation(fields: [class_assigned_id], references: [id], onDelete: Cascade, onUpdate: NoAction, map: "FK_82f99ade8ca8284fdbbf65306a3")` |

---

## `classes`

| Field | Type and attributes |
| --- | --- |
| `id` | `String            @id(map: "PK_e207aa15404e9b2ce35910f9f7f") @default(dbgenerated("uuid_generate_v4()")) @db.Uuid` |
| `institution_id` | `String            @db.Uuid` |
| `name` | `String            @db.VarChar(100)` |
| `teacher_id` | `String            @db.Uuid` |
| `description` | `String            @default("")` |
| `join_code` | `String            @unique(map: "IDX_14f3ad560188f1babcbd3e746a") @db.VarChar(10)` |
| `created_at` | `DateTime          @default(now()) @db.Timestamp(6)` |
| `updated_at` | `DateTime          @default(now()) @db.Timestamp(6)` |
| `is_active` | `Boolean           @default(true)` |
| `announcements` | `announcements[]` |
| `assignments` | `assignments[]` |
| `user` | `user              @relation(fields: [teacher_id], references: [id], onDelete: Cascade, onUpdate: NoAction, map: "FK_b34c92e413c4debb6e0f23fed46")` |
| `institutions` | `institutions      @relation(fields: [institution_id], references: [id], onDelete: Cascade, onUpdate: NoAction, map: "FK_dbc6598b9a97fd324472e1189df")` |
| `student_classes` | `student_classes[]` |
| `time_slots` | `time_slots[]` |

**Constraints / indexes:**
- `@@unique([institution_id, teacher_id, name], map: "IDX_0b135e5103311b6fd8f2276a61")`

---

## `companies`

| Field | Type and attributes |
| --- | --- |
| `id` | `String                       @id(map: "PK_d4bc3e82a314fa9e29f652c2c22") @default(dbgenerated("uuid_generate_v4()")) @db.Uuid` |
| `name` | `String                       @unique(map: "IDX_3dacbb3eb4f095e29372ff8e13") @db.VarChar(100)` |
| `type` | `String                       @default("tech") @db.VarChar(20)` |
| `company_kind` | `companies_company_kind_enum?` |
| `interview_tests` | `interview_tests[]` |
| `question_companies` | `question_companies[]` |

**Constraints / indexes:**
- `@@index([type], map: "IDX_1e57cd6c6afae8f303847f159d")`

---

## `credits`

| Field | Type and attributes |
| --- | --- |
| `id` | `String   @id(map: "PK_45cea097fd0ee625d2e840ed99c") @default(dbgenerated("uuid_generate_v4()")) @db.Uuid` |
| `user_id` | `String   @unique(map: "IDX_47411fd61355a96b8e4d7aabb5") @db.Uuid` |
| `balance` | `Int      @default(0)` |
| `updated_at` | `DateTime @default(now()) @db.Timestamp(6)` |
| `user` | `user     @relation(fields: [user_id], references: [id], onDelete: Cascade, onUpdate: NoAction, map: "FK_47411fd61355a96b8e4d7aabb56")` |

---

## `feedback_items`

| Field | Type and attributes |
| --- | --- |
| `id` | `String            @id(map: "PK_a585b9e995d627b03882641bc8e") @default(dbgenerated("uuid_generate_v4()")) @db.Uuid` |
| `title` | `String            @db.VarChar` |
| `description` | `String            @default("")` |
| `strengths` | `Json?` |
| `areas_for_improvements` | `Json?` |
| `interaction_logs` | `Json              @default("[]")` |
| `interaction_status_logs` | `Json              @default("[]")` |
| `items` | `Json              @default("{}")` |
| `code` | `String?           @unique(map: "IDX_5ee92fdcd92bbedf6a52f33d57") @db.VarChar(120)` |
| `interview_tests` | `interview_tests[]` |

---

## `feedbacks`

| Field | Type and attributes |
| --- | --- |
| `id` | `String          @id(map: "PK_79affc530fdd838a9f1e0cc30be") @default(dbgenerated("uuid_generate_v4()")) @db.Uuid` |
| `user_id` | `String          @db.Uuid` |
| `interview_test_id` | `String          @db.Uuid` |
| `strengths` | `Json?` |
| `areas_for_improvements` | `Json?` |
| `interaction_logs` | `Json            @default("[]")` |
| `interaction_status_logs` | `Json            @default("[]")` |
| `items` | `Json            @default("{}")` |
| `overall_score` | `Float           @default(0)` |
| `duration` | `DateTime        @default(dbgenerated("'00:00:00'::time without time zone")) @db.Time(6)` |
| `saved_at` | `DateTime        @default(now()) @db.Timestamp(6)` |
| `updated_at` | `DateTime        @default(now()) @db.Timestamp(6)` |
| `session_id` | `String          @db.VarChar` |
| `interview_tests` | `interview_tests @relation(fields: [interview_test_id], references: [id], onDelete: NoAction, onUpdate: NoAction, map: "FK_1bbf686b0b474722b298bf46bc8")` |
| `user` | `user            @relation(fields: [user_id], references: [id], onDelete: NoAction, onUpdate: NoAction, map: "FK_4334f6be2d7d841a9d5205a100e")` |

---

## `institutions`

| Field | Type and attributes |
| --- | --- |
| `id` | `String              @id(map: "PK_0be7539dcdba335470dc05e9690") @default(dbgenerated("uuid_generate_v4()")) @db.Uuid` |
| `name` | `String              @db.VarChar(200)` |
| `slug` | `String              @unique(map: "IDX_b0e3a0e4d706e0baaa5a7c65f2") @db.VarChar(100)` |
| `description` | `String              @default("")` |
| `created_at` | `DateTime            @default(now()) @db.Timestamp(6)` |
| `updated_at` | `DateTime            @default(now()) @db.Timestamp(6)` |
| `is_active` | `Boolean             @default(true)` |
| `classes` | `classes[]` |
| `user_institutions` | `user_institutions[]` |

---

## `interview_phases`

| Field | Type and attributes |
| --- | --- |
| `id` | `Int                               @id(map: "PK_087ba7ca4c34b0de1bd3d88338d") @default(autoincrement())` |
| `phase_name` | `String                            @db.VarChar(100)` |
| `prompt` | `String` |
| `prompt_inputs` | `Json                              @default("[]")` |
| `number_of_questions_to_ask` | `Int                               @default(0)` |
| `setup_questions` | `Boolean                           @default(false)` |
| `setup_questions_prompt` | `String                            @default("")` |
| `question_filters` | `Json                              @default("{}")` |
| `route_nodes` | `Json                              @default("[]")` |
| `route_ahead_prompt` | `String                            @default("")` |
| `immediate_feedback_required` | `Boolean                           @default(false)` |
| `feedback_prompt` | `String                            @default("")` |
| `mcp_tools` | `Boolean                           @default(false)` |
| `tool_names` | `Json                              @default("[]")` |
| `special_output_format` | `String?                           @db.VarChar(20)` |
| `entity_schema` | `Json?` |
| `interview_test_interview_phases` | `interview_test_interview_phases[]` |

---

## `interview_test_interview_phases`

| Field | Type and attributes |
| --- | --- |
| `id` | `String            @id(map: "PK_5a6b68400fc0cface775f4ff690") @default(dbgenerated("uuid_generate_v4()")) @db.Uuid` |
| `phase_order` | `Int               @default(0)` |
| `interview_test_id` | `String?           @db.Uuid` |
| `interview_phase_id` | `Int?` |
| `interview_tests` | `interview_tests?  @relation(fields: [interview_test_id], references: [id], onDelete: Cascade, onUpdate: NoAction, map: "FK_60d3437e9ce21f9e6bfac7567ce")` |
| `interview_phases` | `interview_phases? @relation(fields: [interview_phase_id], references: [id], onDelete: Cascade, onUpdate: NoAction, map: "FK_a66430ec1ba703731e2b7fdb833")` |

**Constraints / indexes:**
- `@@unique([interview_test_id, interview_phase_id], map: "UQ_24b4c95136bdbccf2e68ff1ec86")`

---

## `interview_test_subjects`

| Field | Type and attributes |
| --- | --- |
| `interview_test_id` | `String          @db.Uuid` |
| `subject_id` | `String          @db.Uuid` |
| `interview_tests` | `interview_tests @relation(fields: [interview_test_id], references: [id], onDelete: Cascade, map: "FK_46d0dce443724c1b3f6cd1c2e2c")` |
| `subjects` | `subjects        @relation(fields: [subject_id], references: [id], onDelete: Cascade, map: "FK_67a1eac54fc028ffdca374444de")` |

**Constraints / indexes:**
- `@@id([interview_test_id, subject_id], map: "PK_2321f9ccc8d2e87c1a2d000fcaf")`
- `@@index([interview_test_id], map: "IDX_46d0dce443724c1b3f6cd1c2e2")`
- `@@index([subject_id], map: "IDX_67a1eac54fc028ffdca374444d")`

---

## `interview_tests`

| Field | Type and attributes |
| --- | --- |
| `id` | `String                                       @id(map: "PK_f8bae4c8da36cbb19319c78088a") @default(dbgenerated("uuid_generate_v4()")) @db.Uuid` |
| `title` | `String                                       @db.VarChar` |
| `difficulty` | `interview_tests_difficulty_enum?` |
| `questions` | `Int?` |
| `duration` | `Int` |
| `credits` | `Int                                          @default(1)` |
| `description` | `String` |
| `topics` | `Json                                         @default("[]")` |
| `parent_interview_test_id` | `String?                                      @db.Uuid` |
| `feedback_item_id` | `String?                                      @db.Uuid` |
| `subjects` | `Json                                         @default("[]")` |
| `company` | `String?                                      @db.VarChar(100)` |
| `subject` | `String?                                      @db.VarChar(100)` |
| `company_id` | `String?                                      @db.Uuid` |
| `is_active` | `Boolean                                      @default(true)` |
| `created_at` | `DateTime                                     @default(now()) @db.Timestamp(6)` |
| `updated_at` | `DateTime                                     @default(now()) @db.Timestamp(6)` |
| `fastapi_interview_type` | `interview_tests_fastapi_interview_type_enum?` |
| `thumbnail_key` | `String?                                      @db.VarChar(255)` |
| `thumbnail_url` | `String?` |
| `code` | `String?                                      @unique(map: "IDX_3153fc195efc06bc6adf14e991") @db.VarChar(120)` |
| `greeting_prompt` | `Json?` |
| `feedbacks` | `feedbacks[]` |
| `interview_test_interview_phases` | `interview_test_interview_phases[]` |
| `interview_test_subjects` | `interview_test_subjects[]` |
| `companies` | `companies?                                   @relation(fields: [company_id], references: [id], onDelete: NoAction, onUpdate: NoAction, map: "FK_1f6e4923a4c7f82203fe6a63f68")` |
| `feedback_items` | `feedback_items?                              @relation(fields: [feedback_item_id], references: [id], onDelete: NoAction, onUpdate: NoAction, map: "FK_5f9ef0877dfd8d5923eb29efe0c")` |
| `parent_interview_tests` | `parent_interview_tests?                      @relation(fields: [parent_interview_test_id], references: [id], onDelete: NoAction, onUpdate: NoAction, map: "FK_61aaaacced5f3b1c0391fcfe6c5")` |
| `time_slots` | `time_slots[]` |

**Constraints / indexes:**
- `@@index([subject], map: "IDX_a7e406a0edb0e4b1909a58579b")`
- `@@index([company], map: "IDX_d180f7837f8e7a686c3fc9f338")`
- `@@index([is_active], map: "IDX_eb7b7033264664d091d377fdc7")`

---

## `parent_interview_tests`

| Field | Type and attributes |
| --- | --- |
| `id` | `String                                @id(map: "PK_f2786f658213c4a160c192c9207") @default(dbgenerated("uuid_generate_v4()")) @db.Uuid` |
| `title` | `String                                @db.VarChar` |
| `description` | `String` |
| `type` | `parent_interview_tests_type_enum      @default(miscellaneous)` |
| `tags` | `Json                                  @default("[]")` |
| `created_at` | `DateTime                              @default(now()) @db.Timestamp(6)` |
| `updated_at` | `DateTime                              @default(now()) @db.Timestamp(6)` |
| `code` | `String?                               @unique(map: "IDX_3a2cf27fcb93d05f144d420104") @db.VarChar(120)` |
| `interview_tests` | `interview_tests[]` |
| `user_parent_interview_scores` | `user_parent_interview_scores[]` |
| `user_parent_interview_weekly_scores` | `user_parent_interview_weekly_scores[]` |

---

## `question_companies`

| Field | Type and attributes |
| --- | --- |
| `question_id` | `String    @db.Uuid` |
| `company_id` | `String    @db.Uuid` |
| `companies` | `companies @relation(fields: [company_id], references: [id], onDelete: Cascade, map: "FK_1853d02e4af97d894c831089438")` |
| `questions` | `questions @relation(fields: [question_id], references: [id], onDelete: Cascade, map: "FK_f0f65115f503a01b1b4b7388a39")` |

**Constraints / indexes:**
- `@@id([question_id, company_id], map: "PK_011ac434b52e6d2709bd8f3fe00")`
- `@@index([company_id], map: "IDX_1853d02e4af97d894c83108943")`
- `@@index([question_id], map: "IDX_f0f65115f503a01b1b4b7388a3")`

---

## `question_subjects`

| Field | Type and attributes |
| --- | --- |
| `question_id` | `String    @db.Uuid` |
| `subject_id` | `String    @db.Uuid` |
| `questions` | `questions @relation(fields: [question_id], references: [id], onDelete: Cascade, map: "FK_4fcad33a7f022217d9b7c7aee84")` |
| `subjects` | `subjects  @relation(fields: [subject_id], references: [id], onDelete: Cascade, map: "FK_cf94cf753e39162ac14afb76679")` |

**Constraints / indexes:**
- `@@id([question_id, subject_id], map: "PK_442ed6404b04ca44b5f4565b5b6")`
- `@@index([question_id], map: "IDX_4fcad33a7f022217d9b7c7aee8")`
- `@@index([subject_id], map: "IDX_cf94cf753e39162ac14afb7667")`

---

## `questions`

| Field | Type and attributes |
| --- | --- |
| `id` | `String               @id(map: "PK_08a6d4b0f49ff300bf3a0ca60ac") @default(dbgenerated("uuid_generate_v4()")) @db.Uuid` |
| `title` | `String               @db.VarChar(255)` |
| `source` | `String?              @db.VarChar(100)` |
| `url` | `String?              @db.VarChar(200)` |
| `raw_content` | `String?` |
| `description` | `String` |
| `difficulty` | `String               @db.VarChar(10)` |
| `example` | `String?` |
| `question_companies` | `question_companies[]` |
| `question_subjects` | `question_subjects[]` |

**Constraints / indexes:**
- `@@index([difficulty], map: "IDX_4549c6760c689a645f8539119e")`
- `@@index([title], map: "IDX_595b5fc355345b8c1dff5926b4")`

---

## `resume_analysis`

| Field | Type and attributes |
| --- | --- |
| `id` | `Int      @id(map: "PK_bdb67cc6d183f83ea737231a8f3") @default(autoincrement())` |
| `session_id` | `String   @default("__default") @db.VarChar(128)` |
| `resume_name` | `String   @default("Your_resume.pdf") @db.VarChar` |
| `company` | `String?  @db.VarChar(200)` |
| `role` | `String?  @db.VarChar(200)` |
| `job_match_score` | `Int      @default(0)` |
| `format_and_structure` | `Int      @default(0)` |
| `content_quality` | `Int      @default(0)` |
| `length_and_conciseness` | `Int      @default(0)` |
| `keywords_optimization` | `Int      @default(0)` |
| `found_keywords` | `String[] @default([])` |
| `not_found_keywords` | `String[] @default([])` |
| `top_3_keywords` | `String[] @default([])` |
| `required_skills` | `Int      @default(0)` |
| `preferred_skills` | `Int      @default(0)` |
| `experience` | `Int      @default(0)` |
| `education` | `Int      @default(0)` |
| `insights` | `Int      @default(0)` |
| `candidate_strengths` | `String[] @default([])` |
| `candidates_areas_of_improvements` | `String[] @default([])` |
| `userId` | `String?  @db.Uuid` |
| `user` | `user?    @relation(fields: [userId], references: [id], onDelete: NoAction, onUpdate: NoAction, map: "FK_4b719106e92a64e0dc55bbc7784")` |

---

## `student_classes`

| Field | Type and attributes |
| --- | --- |
| `id` | `String   @id(map: "PK_e6fcc2e4f8f79a5aa16a50c8f46") @default(dbgenerated("uuid_generate_v4()")) @db.Uuid` |
| `student_id` | `String   @db.Uuid` |
| `class_id` | `String   @db.Uuid` |
| `joined_at` | `DateTime @default(now()) @db.Timestamp(6)` |
| `user` | `user     @relation(fields: [student_id], references: [id], onDelete: Cascade, onUpdate: NoAction, map: "FK_09b94eccbdedd86b77d54daaeb8")` |
| `classes` | `classes  @relation(fields: [class_id], references: [id], onDelete: Cascade, onUpdate: NoAction, map: "FK_250de2754beaff18091a60a6654")` |

**Constraints / indexes:**
- `@@unique([student_id, class_id], map: "IDX_72cb908330b0e7eac3846c0998")`

---

## `subject_questions`

| Field | Type and attributes |
| --- | --- |
| `id` | `String   @id(map: "PK_a8db7ece63b57c1add32c3ac23f") @default(dbgenerated("uuid_generate_v4()")) @db.Uuid` |
| `subject_id` | `String   @db.Uuid` |
| `question` | `String` |
| `subjects` | `subjects @relation(fields: [subject_id], references: [id], onDelete: Cascade, onUpdate: NoAction, map: "FK_05067740ffc6e117e3e2f259308")` |

---

## `subject_related_subjects`

| Field | Type and attributes |
| --- | --- |
| `subject_id` | `String   @db.Uuid` |
| `related_subject_id` | `String   @db.Uuid` |
| `subjects_subject_related_subjects_related_subject_idTosubjects` | `subjects @relation("subject_related_subjects_related_subject_idTosubjects", fields: [related_subject_id], references: [id], onDelete: NoAction, onUpdate: NoAction, map: "FK_664fd8f321828e439fa08ceba41")` |
| `subjects_subject_related_subjects_subject_idTosubjects` | `subjects @relation("subject_related_subjects_subject_idTosubjects", fields: [subject_id], references: [id], onDelete: Cascade, map: "FK_960aacfc476f6282a153a8452c7")` |

**Constraints / indexes:**
- `@@id([subject_id, related_subject_id], map: "PK_1f2530417b8103a605a4351a66c")`
- `@@index([related_subject_id], map: "IDX_664fd8f321828e439fa08ceba4")`
- `@@index([subject_id], map: "IDX_960aacfc476f6282a153a8452c")`

---

## `subjects`

| Field | Type and attributes |
| --- | --- |
| `id` | `String                     @id(map: "PK_1a023685ac2b051b4e557b0b280") @default(dbgenerated("uuid_generate_v4()")) @db.Uuid` |
| `name` | `String                     @db.VarChar(100)` |
| `description` | `String?` |
| `type` | `String                     @default("general") @db.VarChar(20)` |
| `difficulty` | `String?                    @db.VarChar(10)` |
| `interview_test_subjects` | `interview_test_subjects[]` |
| `question_subjects` | `question_subjects[]` |
| `subject_questions` | `subject_questions[]` |
| `subject_related_subjects_subject_related_subjects_related_subject_idTosubjects` | `subject_related_subjects[] @relation("subject_related_subjects_related_subject_idTosubjects")` |
| `subject_related_subjects_subject_related_subjects_subject_idTosubjects` | `subject_related_subjects[] @relation("subject_related_subjects_subject_idTosubjects")` |

**Constraints / indexes:**
- `@@index([name], map: "IDX_47a287fe64bd0e1027e603c335")`
- `@@index([type], map: "IDX_5b0c336a7660d843b343a62b35")`

---

## `subscription_tier_prices`

| Field | Type and attributes |
| --- | --- |
| `id` | `String                                 @id(map: "PK_06edf063f4c08078be57ce335b8") @default(dbgenerated("uuid_generate_v4()")) @db.Uuid` |
| `tier_id` | `String                                 @db.Uuid` |
| `currency_code` | `String                                 @db.VarChar(3)` |
| `interval` | `subscription_tier_prices_interval_enum` |
| `amount_cents` | `Int` |
| `subscription_tiers` | `subscription_tiers                     @relation(fields: [tier_id], references: [id], onDelete: Cascade, onUpdate: NoAction, map: "FK_19cc476030f984ae5b630542cd9")` |

**Constraints / indexes:**
- `@@unique([tier_id, currency_code, interval], map: "IDX_198d2d85fc8606dd266767289b")`

---

## `subscription_tiers`

| Field | Type and attributes |
| --- | --- |
| `id` | `String                     @id(map: "PK_376aa3503bf3278d69af3d711b7") @default(dbgenerated("uuid_generate_v4()")) @db.Uuid` |
| `name` | `String                     @db.VarChar(100)` |
| `slug` | `String                     @unique(map: "IDX_038e7a87d224126077e0ccd02f") @db.VarChar(50)` |
| `description` | `String                     @default("")` |
| `is_active` | `Boolean                    @default(true)` |
| `created_at` | `DateTime                   @default(now()) @db.Timestamp(6)` |
| `updated_at` | `DateTime                   @default(now()) @db.Timestamp(6)` |
| `subscription_tier_prices` | `subscription_tier_prices[]` |
| `subscriptions` | `subscriptions[]` |

---

## `subscriptions`

| Field | Type and attributes |
| --- | --- |
| `id` | `String                              @id(map: "PK_a87248d73155605cf782be9ee5e") @default(dbgenerated("uuid_generate_v4()")) @db.Uuid` |
| `user_id` | `String                              @db.Uuid` |
| `tier_id` | `String                              @db.Uuid` |
| `status` | `subscriptions_status_enum` |
| `billing_interval` | `subscriptions_billing_interval_enum` |
| `currency_code` | `String                              @db.VarChar(3)` |
| `amount_cents` | `Int` |
| `started_at` | `DateTime                            @db.Timestamptz(6)` |
| `ends_at` | `DateTime                            @db.Timestamptz(6)` |
| `cancelled_at` | `DateTime?                           @db.Timestamptz(6)` |
| `external_subscription_id` | `String?                             @db.VarChar(255)` |
| `created_at` | `DateTime                            @default(now()) @db.Timestamp(6)` |
| `updated_at` | `DateTime                            @default(now()) @db.Timestamp(6)` |
| `subscription_tiers` | `subscription_tiers                  @relation(fields: [tier_id], references: [id], onUpdate: NoAction, map: "FK_c2d9de83b12e926d07ed0b083ec")` |
| `user` | `user                                @relation(fields: [user_id], references: [id], onDelete: Cascade, onUpdate: NoAction, map: "FK_d0a95ef8a28188364c546eb65c1")` |
| `transaction_logs` | `transaction_logs[]` |

**Constraints / indexes:**
- `@@index([user_id, started_at], map: "IDX_c5d5ac3fb014d40dddea029621")`

---

## `time_slots`

| Field | Type and attributes |
| --- | --- |
| `id` | `String          @id(map: "PK_f87c73d8648c3f3f297adba3cb8") @default(dbgenerated("uuid_generate_v4()")) @db.Uuid` |
| `class_id` | `String          @db.Uuid` |
| `created_by_id` | `String          @db.Uuid` |
| `scheduled_date` | `DateTime        @db.Date` |
| `start_time` | `DateTime        @db.Time(6)` |
| `end_time` | `DateTime        @db.Time(6)` |
| `title` | `String          @default("") @db.VarChar(200)` |
| `description` | `String          @default("")` |
| `interview_test_id` | `String          @db.Uuid` |
| `category` | `String?         @db.VarChar(50)` |
| `difficulty` | `String?         @db.VarChar(20)` |
| `status` | `String          @default("available") @db.VarChar(20)` |
| `created_at` | `DateTime        @default(now()) @db.Timestamp(6)` |
| `updated_at` | `DateTime        @default(now()) @db.Timestamp(6)` |
| `user` | `user            @relation(fields: [created_by_id], references: [id], onDelete: Cascade, onUpdate: NoAction, map: "FK_589da54526924d4c81c93f87838")` |
| `classes` | `classes         @relation(fields: [class_id], references: [id], onDelete: Cascade, onUpdate: NoAction, map: "FK_79d7eb7d23764a15f3b3cd0184f")` |
| `interview_tests` | `interview_tests @relation(fields: [interview_test_id], references: [id], onDelete: Cascade, onUpdate: NoAction, map: "FK_b303f033876033ab813ef64973e")` |

---

## `transaction_logs`

| Field | Type and attributes |
| --- | --- |
| `id` | `String                       @id(map: "PK_c7605f13413f4b5d06e53f2349b") @default(dbgenerated("uuid_generate_v4()")) @db.Uuid` |
| `user_id` | `String                       @db.Uuid` |
| `subscription_id` | `String?                      @db.Uuid` |
| `type` | `transaction_logs_type_enum` |
| `amount_cents` | `Int` |
| `currency_code` | `String                       @db.VarChar(3)` |
| `status` | `transaction_logs_status_enum` |
| `external_id` | `String?                      @db.VarChar(255)` |
| `metadata` | `Json?` |
| `created_at` | `DateTime                     @default(now()) @db.Timestamp(6)` |
| `subscriptions` | `subscriptions?               @relation(fields: [subscription_id], references: [id], onUpdate: NoAction, map: "FK_96adc92b5d72760584a003dc379")` |
| `user` | `user                         @relation(fields: [user_id], references: [id], onDelete: Cascade, onUpdate: NoAction, map: "FK_e52f46b707af549df677670d419")` |

**Constraints / indexes:**
- `@@index([subscription_id, created_at], map: "IDX_38a0de264f63b1a60e60e33459")`
- `@@index([user_id, created_at], map: "IDX_67b866d8057d2f367beb7751a2")`

---

## `user`

| Field | Type and attributes |
| --- | --- |
| `id` | `String                                @id(map: "PK_cace4a159ff9f2512dd42373760") @default(dbgenerated("uuid_generate_v4()")) @db.Uuid` |
| `name` | `String                                @db.VarChar` |
| `email` | `String                                @db.VarChar` |
| `roles` | `user_roles_enum[]                     @default([user])` |
| `password` | `String?                               @db.VarChar` |
| `auth_provider` | `user_auth_provider_enum               @default(local)` |
| `auth_provider_id` | `String?                               @db.VarChar` |
| `avatar_url` | `String?                               @db.VarChar` |
| `is_email_verified` | `Boolean                               @default(false)` |
| `is_whitelisted` | `Boolean                               @default(false)` |
| `refresh_token` | `String?                               @db.VarChar` |
| `announcements` | `announcements[]` |
| `assignment_submissions_assignment_submissions_student_idTouser` | `assignment_submissions[]              @relation("assignment_submissions_student_idTouser")` |
| `assignment_submissions_assignment_submissions_graded_by_idTouser` | `assignment_submissions[]              @relation("assignment_submissions_graded_by_idTouser")` |
| `assignments` | `assignments[]` |
| `classes` | `classes[]` |
| `credits` | `credits?` |
| `feedbacks` | `feedbacks[]` |
| `resume_analysis` | `resume_analysis[]` |
| `student_classes` | `student_classes[]` |
| `subscriptions` | `subscriptions[]` |
| `time_slots` | `time_slots[]` |
| `transaction_logs` | `transaction_logs[]` |
| `user_institutions` | `user_institutions[]` |
| `user_parent_interview_scores` | `user_parent_interview_scores[]` |
| `user_parent_interview_weekly_scores` | `user_parent_interview_weekly_scores[]` |
| `user_sessions` | `user_sessions[]` |

---

## `user_institutions`

| Field | Type and attributes |
| --- | --- |
| `id` | `String                      @id(map: "PK_af852ae7232e5a7e021e6395104") @default(dbgenerated("uuid_generate_v4()")) @db.Uuid` |
| `user_id` | `String                      @db.Uuid` |
| `institution_id` | `String                      @db.Uuid` |
| `role` | `user_institutions_role_enum` |
| `institutions` | `institutions                @relation(fields: [institution_id], references: [id], onDelete: Cascade, onUpdate: NoAction, map: "FK_d55bbbde2b43aa3e3f9ea72457b")` |
| `user` | `user                        @relation(fields: [user_id], references: [id], onDelete: Cascade, onUpdate: NoAction, map: "FK_d7a5a3abedf8e03a0cba655906f")` |

**Constraints / indexes:**
- `@@unique([user_id, institution_id], map: "IDX_cb715db1d831336ddbbc00b4be")`

---

## `user_parent_interview_scores`

| Field | Type and attributes |
| --- | --- |
| `id` | `String                 @id(map: "PK_15d2e20e9649afa8f6232359c12") @default(dbgenerated("uuid_generate_v4()")) @db.Uuid` |
| `user_id` | `String                 @db.Uuid` |
| `parent_interview_test_id` | `String                 @db.Uuid` |
| `average` | `Float                  @default(0)` |
| `count` | `Int                    @default(0)` |
| `duration` | `DateTime               @default(dbgenerated("'00:00:00'::time without time zone")) @db.Time(6)` |
| `user` | `user                   @relation(fields: [user_id], references: [id], onDelete: NoAction, onUpdate: NoAction, map: "FK_b5693bef2cd0d5754d781e8ac68")` |
| `parent_interview_tests` | `parent_interview_tests @relation(fields: [parent_interview_test_id], references: [id], onDelete: NoAction, onUpdate: NoAction, map: "FK_bc404e1485f88bf652641fc6253")` |

**Constraints / indexes:**
- `@@unique([user_id, parent_interview_test_id], map: "UQ_a330f68aafbdebefd01e2f0d8f3")`
- `@@index([average], map: "IDX_118f0b28a692fc52b69ade8d61")`

---

## `user_parent_interview_weekly_scores`

| Field | Type and attributes |
| --- | --- |
| `id` | `String                 @id(map: "PK_1b4ea3989ea5e6486caca3e5df9") @default(dbgenerated("uuid_generate_v4()")) @db.Uuid` |
| `user_id` | `String                 @db.Uuid` |
| `parent_interview_test_id` | `String                 @db.Uuid` |
| `week_no` | `Int` |
| `year` | `Int` |
| `average` | `Float                  @default(0)` |
| `count` | `Int                    @default(0)` |
| `duration` | `DateTime               @default(dbgenerated("'00:00:00'::time without time zone")) @db.Time(6)` |
| `created_at` | `DateTime               @default(now()) @db.Timestamp(6)` |
| `user` | `user                   @relation(fields: [user_id], references: [id], onDelete: NoAction, onUpdate: NoAction, map: "FK_b4662f50f0acb3ce8e12af0d909")` |
| `parent_interview_tests` | `parent_interview_tests @relation(fields: [parent_interview_test_id], references: [id], onDelete: NoAction, onUpdate: NoAction, map: "FK_db3b8704e715b3e716a973f5a1a")` |

**Constraints / indexes:**
- `@@index([average], map: "IDX_374ed6617668a7c64ba30255bb")`
- `@@index([user_id, parent_interview_test_id, week_no, year], map: "IDX_8b2eea95d7835aa6f10510fdca")`

---

## `user_sessions`

| Field | Type and attributes |
| --- | --- |
| `id` | `String   @id(map: "PK_e93e031a5fed190d4789b6bfd83") @default(dbgenerated("uuid_generate_v4()")) @db.Uuid` |
| `user_id` | `String   @db.Uuid` |
| `device_info` | `String?  @db.VarChar(255)` |
| `ip_address` | `String?  @db.VarChar(64)` |
| `user_agent` | `String?` |
| `refresh_token_hash` | `String?  @db.VarChar` |
| `is_revoked` | `Boolean  @default(false)` |
| `created_at` | `DateTime @default(now()) @db.Timestamp(6)` |
| `last_used_at` | `DateTime @default(now()) @db.Timestamp(6)` |
| `user` | `user     @relation(fields: [user_id], references: [id], onDelete: Cascade, onUpdate: NoAction, map: "FK_e9658e959c490b0a634dfc54783")` |

**Constraints / indexes:**
- `@@index([user_id], map: "IDX_e9658e959c490b0a634dfc5478")`

---

## Enums

### `companies_company_kind_enum`

- `Product_based @map("Product-based")`
- `Service_based @map("Service-based")`
- `Mass_hiring   @map("Mass-hiring")`
- `Startup`
- `FAANG`

### `interview_tests_difficulty_enum`

- `Easy`
- `Medium`
- `Hard`

### `interview_tests_fastapi_interview_type_enum`

- `Technical`
- `HR`
- `Company`
- `Subject`
- `CaseStudy`
- `Communication`
- `Role_Based_Interview @map("Role-Based Interview")`
- `Debate`

### `parent_interview_tests_type_enum`

- `technical`
- `case_study    @map("case-study")`
- `role_based    @map("role-based")`
- `debate`
- `behavioral`
- `specialised`
- `miscellaneous`

### `subscription_tier_prices_interval_enum`

- `monthly`
- `annual`

### `subscriptions_billing_interval_enum`

- `monthly`
- `annual`

### `subscriptions_status_enum`

- `active`
- `cancelled`
- `expired`
- `past_due`
- `trialing`

### `transaction_logs_status_enum`

- `success`
- `failed`
- `pending`
- `refunded`

### `transaction_logs_type_enum`

- `purchase`
- `renewal`
- `refund`
- `upgrade`
- `downgrade`
- `credit`
- `debit`

### `user_auth_provider_enum`

- `local`
- `google`
- `github`

### `user_institutions_role_enum`

- `student`
- `teacher`
- `admin`

### `user_roles_enum`

- `admin`
- `admin_staff`
- `developer`
- `teacher`
- `student`
- `user`

