from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from llm_service.config import OLLAMA_MODEL, GROQ_MODEL, LLM_PROVIDER
from llm_service.quiz_generator import generate_quiz
from llm_service.interview_generator import generate_interview_questions
from llm_service.evaluator import evaluate_quiz
from llm_service.scorer import compute_skill_scores, detect_weak_skills, build_result_json

app = FastAPI(
    title="AI Career Advisor - LLM Service API",
    description="Manages prompt engineering, quiz generation, answer evaluation, and skill gap analysis."
)


class CVData(BaseModel):
    """Structured CV data following the CV_JSON contract."""
    skills: List[str]
    projects: Optional[List[str]] = None
    experience: Optional[List[dict]] = None
    confidence: Optional[float] = None


class GenerateQuizRequest(BaseModel):
    """Request body for POST /generate-quiz."""
    cv_json: CVData
    job_title: str


class JDData(BaseModel):
    """Structured JD data following the JD_JSON contract."""
    required_skills: List[str]


class GenerateInterviewRequest(BaseModel):
    """Request body for POST /generate-interview."""
    jd_json: JDData
    job_title: str


class EvaluateRequest(BaseModel):
    """Request body for POST /evaluate."""
    quiz: dict
    answers: List[str]


@app.get("/health")
def health_endpoint():
    """Returns service health status and active LLM model."""
    return {
        "status": "ok",
        "provider": LLM_PROVIDER,
        "model": GROQ_MODEL if LLM_PROVIDER == "groq" else OLLAMA_MODEL
    }


@app.post("/generate-quiz")
def endpoint_generate_quiz(req: GenerateQuizRequest):
    """Generate a skills-based quiz from CV data and a job title."""
    try:
        if not req.cv_json.skills:
            raise HTTPException(
                status_code=400,
                detail={"error": "Invalid Input", "message": "CV skills list must not be empty."}
            )

        quiz_json = generate_quiz(skills=req.cv_json.skills, job_title=req.job_title)
        return quiz_json
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": "Quiz Generation Failed", "message": str(e)}
        )


@app.post("/generate-interview")
def endpoint_generate_interview(req: GenerateInterviewRequest):
    """Generate scenario-based interview questions from JD skills."""
    try:
        quiz_json = generate_interview_questions(
            jd_json=req.jd_json.model_dump(),
            job_title=req.job_title,
            n=5
        )
        return quiz_json
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": "Interview Generation Failed", "message": str(e)}
        )


@app.post("/evaluate")
def endpoint_evaluate(req: EvaluateRequest):
    """Evaluate quiz answers and return skill scores with gap analysis."""
    try:
        # 1. Run evaluators
        question_results = evaluate_quiz(req.quiz, req.answers)

        # 2. Score breakdown
        skill_scores = compute_skill_scores(question_results, req.quiz)

        # 3. Detect weaknesses
        weak_skills = detect_weak_skills(skill_scores)

        # 4. Form result payload
        result_json = build_result_json(question_results, skill_scores, weak_skills)

        return result_json

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail={"error": "Invalid Input", "message": str(e)}
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": "Evaluation Workflow Failed", "message": str(e)}
        )
