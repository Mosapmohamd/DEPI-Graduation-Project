import math
from llm_service.config import call_llm
from llm_service.prompt_templates import build_interview_prompt


def generate_interview_questions(jd_json: dict, job_title: str, n: int = 10) -> dict:
    """
    Generate scenario-based interview questions drawing from JD required skills.
    Because the prompt template dictates exactly 5 questions per call,
    this function loops if n > 5 and concatenates the results.
    Returns: A standard QUIZ_JSON dictionary structure.
    """
    required_skills = jd_json.get("required_skills", [])
    if not required_skills:
        raise ValueError("Invalid JD_JSON: 'required_skills' must not be empty.")

    runs_needed = math.ceil(n / 5)
    all_questions = []

    for _ in range(runs_needed):
        sys_p, usr_p = build_interview_prompt(required_skills, job_title)

        try:
            response_array = call_llm(sys_p, usr_p, expect_json=True)
        except Exception as e:
            raise RuntimeError(f"Failed to generate interview questions: {e}") from e

        if not isinstance(response_array, list):
            # Sometimes models wrap arrays in dicts. Attempt recovery.
            if isinstance(response_array, dict) and "questions" in response_array:
                response_array = response_array["questions"]
            else:
                raise ValueError("LLM did not return an array of questions as requested.")

        all_questions.extend(response_array)

    return {
        "questions": all_questions[:n]
    }
