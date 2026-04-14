# LLM Service Module — AI Career Advisor & Skill Assessment System

## Overview

The **LLM Service** is the core intelligence module of the AI Career Advisor system. It is responsible for generating personalized technical assessments, evaluating candidate answers, detecting skill gaps, and recommending learning paths — all powered by Large Language Models.

This module supports two primary user flows:

- **Flow A — Career Exploration (no JD):** A user uploads their CV. The NLP Engineer extracts skills, the ML Engineer recommends job roles, the user picks one, and **this module generates a tailored exam, evaluates answers, scores per skill, and detects gaps.**
- **Flow B — Job Preparation (with JD):** A user uploads their CV alongside a Job Description. Skills are extracted from the JD, and **this module generates scenario-based interview questions, evaluates answers, and identifies weaknesses relative to the job requirements.**

The module exposes a clean FastAPI REST interface that the Fullstack team consumes directly.

---

## Architecture

```
llm_service/
├── __init__.py                # Package init
├── config.py                  # LLM provider config, call_llm() wrapper
├── prompt_templates.py        # 6 prompt builder functions (no LLM calls)
├── quiz_generator.py          # Generates QUIZ_JSON from skills + job title
├── evaluator.py               # MCQ (rule-based) + essay (LLM-based) evaluation
├── interview_generator.py     # Generates scenario-based interview questions from JD
├── scorer.py                  # Skill scoring, gap detection, learning paths
├── service.py                 # FastAPI app with 4 endpoints
└── tests/
    ├── __init__.py
    ├── test_prompt_templates.py
    ├── test_quiz_generator.py
    ├── test_evaluator.py
    ├── test_interview_generator.py
    ├── test_scorer.py
    ├── test_service.py
    └── test_e2e_pipeline.py   # Full end-to-end integration tests
```

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- A Groq API key (free tier: https://console.groq.com/keys)

### Required Packages

```
fastapi>=0.100.0
uvicorn>=0.23.0
requests>=2.31.0
pydantic>=2.0.0
pytest>=7.0.0
httpx>=0.24.0
```

### Installation Steps

```bash
# 1. Clone the repository
git clone <repo-url>
cd DEPI-Graduation-Project

# 2. Install dependencies
pip install fastapi uvicorn requests pydantic pytest httpx

# 3. Set environment variables (optional — defaults are built in)
export GROQ_API_KEY="your_groq_api_key_here"
export LLM_PROVIDER="groq"        # or "ollama" for local testing
export GROQ_MODEL="llama-3.1-8b-instant"

# 4. Verify setup
python -m pytest llm_service/tests/ -v
```

### Running the Service

```bash
# Start the FastAPI server
uvicorn llm_service.service:app --host 0.0.0.0 --port 8000 --reload

# Verify it's running
curl http://localhost:8000/health
```

---

## API Reference

### GET /health

Returns service health status and the active LLM model.

**Example Response:**
```json
{
  "status": "ok",
  "provider": "groq",
  "model": "llama-3.1-8b-instant"
}
```

---

### POST /generate-quiz

Generates a personalized skills quiz from CV data and a target job title.

**Request Body:**
```json
{
  "cv_json": {
    "skills": ["Python", "SQL", "Machine Learning"],
    "projects": ["churn prediction model"],
    "experience": [{"role": "Data Analyst", "years": 2}],
    "confidence": 0.87
  },
  "job_title": "Data Scientist"
}
```

**Response:** `QUIZ_JSON` (see Data Contracts below)

**Notes:**
- Generates `n_per_skill=2` questions per skill (default)
- Maintains ~60% MCQ / ~40% essay ratio
- Returns 400 if `skills` list is empty
- Returns 422 if request body is malformed

---

### POST /generate-interview

Generates scenario-based interview questions from a Job Description.

**Request Body:**
```json
{
  "jd_json": {
    "required_skills": ["Docker", "REST APIs", "Python"]
  },
  "job_title": "Backend Engineer"
}
```

**Response:** `QUIZ_JSON` — with 5 scenario-based text questions

**Notes:**
- All questions are `type: "text"` (no MCQs)
- `skill_tag` values match the `required_skills` input
- Returns 422 if `required_skills` field is missing

---

### POST /evaluate

Evaluates quiz answers and returns full scoring with gap analysis.

**Request Body:**
```json
{
  "quiz": { "questions": [ ... ] },
  "answers": ["answer1", "answer2", "answer3"]
}
```

**Response:** `RESULT_JSON` (see Data Contracts below)

**Notes:**
- `answers` list length must match `questions` list length (returns 400 otherwise)
- MCQ questions are evaluated rule-based (exact match, no LLM call)
- Essay/text/coding questions are evaluated via LLM
- Returns per-skill scores, detected gaps, and overall feedback

---

## Data Contracts

### Input: CV_JSON

```json
{
  "skills": ["Python", "SQL", "Machine Learning"],
  "projects": ["churn prediction model", "NLP sentiment classifier"],
  "experience": [{"role": "Data Analyst", "years": 2}],
  "confidence": 0.87
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| skills | list[str] | Yes | Technical skills extracted from CV |
| projects | list[str] | No | Project names from CV |
| experience | list[dict] | No | Work experience entries |
| confidence | float | No | NLP extraction confidence score |

### Input: JD_JSON

```json
{
  "required_skills": ["Python", "TensorFlow", "Docker", "REST APIs"]
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| required_skills | list[str] | Yes | Skills required by the job description |

### Output: QUIZ_JSON

```json
{
  "questions": [
    {
      "question": "What is overfitting in machine learning?",
      "type": "text",
      "difficulty": "medium",
      "skill_tag": "Machine Learning",
      "options": null
    },
    {
      "question": "Which best describes a decision tree?",
      "type": "mcq",
      "difficulty": "easy",
      "skill_tag": "Machine Learning",
      "options": ["A", "B", "C", "D"],
      "correct_answer": "B"
    }
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| question | str | The question text |
| type | str | `"mcq"`, `"text"`, or `"coding"` |
| difficulty | str | `"easy"`, `"medium"`, or `"hard"` |
| skill_tag | str | Maps back to an input skill |
| options | list or null | 4 options for MCQ, null for others |
| correct_answer | str | Only present for MCQ questions |

### Output: RESULT_JSON

```json
{
  "score": 72,
  "feedback": "Good fundamental understanding, but room for improvement.",
  "skill_scores": {
    "Python": 90,
    "Machine Learning": 60,
    "SQL": 70
  },
  "question_results": [
    {
      "question": "What is overfitting?",
      "user_answer": "When model memorizes training data",
      "score": 4,
      "max_score": 5,
      "feedback": "Good definition, missing regularization techniques."
    }
  ],
  "gaps": ["model evaluation", "regularization"]
}
```

| Field | Type | Description |
|-------|------|-------------|
| score | int | Overall percentage score (0-100) |
| feedback | str | Summary feedback string |
| skill_scores | dict | Per-skill percentage scores |
| question_results | list[dict] | Detailed per-question results |
| gaps | list[str] | Skills scoring below 60% |

---

## Module Details

### config.py

Single source of truth for all LLM configuration. Supports two providers:

- **Groq** (cloud, default) — uses OpenAI-compatible API at `https://api.groq.com/openai/v1/chat/completions`
- **Ollama** (local) — uses `http://localhost:11434/api/chat`

Key function: `call_llm(system_prompt, user_prompt, expect_json=False)` — routes to the active provider. All other modules call this function exclusively.

All settings can be overridden via environment variables: `LLM_PROVIDER`, `GROQ_API_KEY`, `GROQ_MODEL`, `OLLAMA_BASE_URL`, `OLLAMA_MODEL`.

### prompt_templates.py

Six pure functions that construct `(system_prompt, user_prompt)` tuples. No LLM calls, no side effects. Each prompt includes:
- Explicit role assignment
- Exact JSON schema to return
- Constraint: "Return ONLY valid JSON"

| Function | Purpose |
|----------|---------|
| `build_mcq_prompt(skill, difficulty)` | Generate 1 MCQ question |
| `build_essay_prompt(skill, difficulty)` | Generate 1 essay/text question |
| `build_coding_prompt(skill, difficulty)` | Generate 1 coding question |
| `build_evaluation_prompt(question, answer, type)` | Evaluate a user answer |
| `build_interview_prompt(skills, job_title)` | Generate 5 interview questions |
| `build_learning_path_prompt(weak_skills)` | Generate markdown learning path |

### quiz_generator.py

Generates a full quiz with a ~60% MCQ / ~40% essay split per skill. For `n_per_skill=2`, each skill gets 1 MCQ and 1 essay question. Difficulty is randomly assigned from easy/medium/hard.

### evaluator.py

- **MCQ evaluation**: Pure rule-based exact string matching (case-insensitive). No LLM call needed.
- **Essay evaluation**: Sends the question + user answer to the LLM via `build_evaluation_prompt()`. Returns a score (0-5) and feedback. Includes graceful degradation — if the LLM call fails, returns score 0 with an error message instead of crashing.

### interview_generator.py

Generates scenario-based interview questions from JD skills. The prompt template produces 5 questions per call, so for `n > 5`, the function loops and concatenates results. Handles cases where the LLM wraps arrays in dict objects.

### scorer.py

- `compute_skill_scores()`: Groups question results by `skill_tag`, calculates percentage scores
- `detect_weak_skills()`: Filters skills below a threshold (default: 60%)
- `build_result_json()`: Assembles the complete `RESULT_JSON` contract
- `generate_learning_path()`: Calls LLM to produce markdown-formatted study recommendations

### service.py

FastAPI application wrapping all modules. Uses Pydantic models for request validation. Error handling:
- 400 for invalid input (empty skills, mismatched answer counts)
- 422 for malformed request bodies (automatic via Pydantic)
- 500 for LLM failures (with structured error detail, never a raw traceback)

---

## Testing

### Running All Tests

```bash
# Set PYTHONPATH first
export PYTHONPATH=/path/to/DEPI-Graduation-Project

# Run all tests
python -m pytest llm_service/tests/ -v
```

### Test Coverage Summary

| Test File | Tests | What It Covers |
|-----------|-------|----------------|
| test_prompt_templates.py | 7 | All 6 prompt builders + LLM integration |
| test_quiz_generator.py | 2 | Schema validation (mocked) + live generation |
| test_evaluator.py | 2 | MCQ rule-based logic + live essay evaluation |
| test_interview_generator.py | 2 | Chunking logic (mocked) + live generation |
| test_scorer.py | 2 | Score math verification + live learning path |
| test_service.py | 4 | All 4 endpoints + error handling |
| test_e2e_pipeline.py | 3 | Full Flow A + Flow B + all endpoints |
| **Total** | **22** | |

### Running Integration Tests Only

```bash
python -m pytest llm_service/tests/ -v -m integration
```

> Note: Integration tests require a valid Groq API key and internet access.

---

## Integration Guide for Team Members

### For Fullstack Engineers (Members 4 & 5)

Start the service with `uvicorn llm_service.service:app --port 8000`. Then call:

1. `POST /generate-quiz` with `cv_json` + `job_title` → get quiz questions to display
2. Collect user answers as a list of strings (same order as questions)
3. `POST /evaluate` with the quiz object + answers list → get scores, feedback, and gaps
4. Display `RESULT_JSON` fields in the dashboard

### For ML Engineer (Member 2)

- You provide the `job_title` string (e.g., "ML Engineer") from your role prediction model
- After evaluation, consume `RESULT_JSON.skill_scores` and `RESULT_JSON.gaps` for your gap analysis pipeline

### For NLP Engineer (Member 1)

- Your `CV_JSON` output (with `skills`, `projects`, `experience`, `confidence`) feeds directly into `/generate-quiz`
- Your `JD_JSON` output (with `required_skills`) feeds directly into `/generate-interview`
- No changes needed to your output schemas

---

## Configuration

### Switching LLM Providers

Edit `LLM_PROVIDER` in `config.py` or set the environment variable:

```bash
# Use Groq (cloud, fast, free tier)
export LLM_PROVIDER=groq

# Use Ollama (local, requires running Ollama server)
export LLM_PROVIDER=ollama
```

### Adjusting Quiz Parameters

In `quiz_generator.py`, the `generate_quiz()` function accepts:
- `n_per_skill` (default: 2) — number of questions per skill
- MCQ/essay ratio is hardcoded at 60/40 via `round(n_per_skill * 0.6)`

---

## Error Handling

| Scenario | Status Code | Response |
|----------|-------------|----------|
| Empty skills list | 400 | `{"error": "Invalid Input", "message": "..."}` |
| Missing required fields | 422 | Pydantic validation error |
| Answer count mismatch | 400 | `{"error": "Invalid Input", "message": "Length mismatch..."}` |
| LLM call failure | 500 | `{"error": "...", "message": "..."}` |
| LLM returns invalid JSON | Retried 2x | Falls back to structured error after 3 attempts |

All LLM calls include retry logic (max 2 retries). JSON responses are cleaned of `<think>` tags and markdown fences before parsing.

---

## Known Limitations

1. **MCQ evaluation is exact-match only** — if the LLM generates option text like `"A) Python"` and the user answers `"Python"`, it will be marked incorrect. The user must provide the exact option text.
2. **Rate limits** — Groq free tier has rate limits. High-frequency calls may trigger 429 errors. Consider adding a small delay between calls in production.
3. **`skill_tag` accuracy** — the LLM occasionally generates combined skill tags (e.g., `"Docker, Python"` instead of `"Docker"`). Consider post-processing validation in production.
4. **No persistent state** — the service is stateless. Quiz state must be managed by the frontend/backend integration layer.
5. **Learning path generation** is not part of any API endpoint — it's available as a Python function (`scorer.generate_learning_path()`) but not exposed via REST. The fullstack team can add an endpoint if needed.
