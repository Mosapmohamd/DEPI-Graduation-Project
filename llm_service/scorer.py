from llm_service.config import call_llm
from llm_service.prompt_templates import build_learning_path_prompt


def compute_skill_scores(question_results: list[dict], quiz: dict) -> dict:
    """
    Groups questions by skill_tag mapping them from the original quiz.
    Calculates the percentage score per skill using actual scores vs max_scores.
    Returns: {"SkillName": 75, "AnotherSkill": 90, ...}
    """
    # Create a mapping from question text to its skill_tag
    q_to_skill = {}
    for q_obj in quiz.get("questions", []):
        q_to_skill[q_obj["question"]] = q_obj.get("skill_tag", "Unknown")

    skill_totals = {}

    for res in question_results:
        question_text = res.get("question")
        skill = q_to_skill.get(question_text, "Unknown")

        if skill not in skill_totals:
            skill_totals[skill] = {"earned": 0, "possible": 0}

        skill_totals[skill]["earned"] += res.get("score", 0)
        skill_totals[skill]["possible"] += res.get("max_score", 5)

    # Calculate percentages
    skill_scores = {}
    for skill, totals in skill_totals.items():
        if totals["possible"] > 0:
            pct = (totals["earned"] / totals["possible"]) * 100
            skill_scores[skill] = round(pct)
        else:
            skill_scores[skill] = 0

    return skill_scores


def detect_weak_skills(skill_scores: dict, threshold: int = 60) -> list[str]:
    """Filters skills falling below the threshold percentage."""
    return [skill for skill, score in skill_scores.items() if score < threshold]


def generate_learning_path(weak_skills: list[str]) -> str:
    """Uses LLM to generate Markdown learning path recommendations for weak skills."""
    if not weak_skills:
        return "Great job! No major skill gaps detected. Keep practicing your strengths."

    sys_p, usr_p = build_learning_path_prompt(weak_skills)
    try:
        return call_llm(sys_p, usr_p, expect_json=False)
    except Exception as e:
        return f"Failed to generate learning path: {e}"


def build_result_json(question_results: list[dict], skill_scores: dict, weak_skills: list[str]) -> dict:
    """Assembles the final RESULT_JSON object matching the contract exactly."""

    total_earned = sum(res.get("score", 0) for res in question_results)
    total_possible = sum(res.get("max_score", 5) for res in question_results)

    overall_score = round((total_earned / total_possible) * 100) if total_possible > 0 else 0

    if overall_score >= 80:
        feedback = "Excellent performance overall."
    elif overall_score >= 60:
        feedback = "Good fundamental understanding, but room for improvement in weaker areas."
    else:
        feedback = "Significant skill gaps detected. Recommend reviewing fundamental concepts."

    return {
        "score": overall_score,
        "feedback": feedback,
        "skill_scores": skill_scores,
        "question_results": question_results,
        "gaps": weak_skills
    }
