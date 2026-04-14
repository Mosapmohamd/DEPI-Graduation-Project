import random
import json
from llm_service.config import call_llm
from llm_service.prompt_templates import build_mcq_prompt, build_essay_prompt


def generate_quiz(skills: list[str], job_title: str, n_per_skill: int = 2) -> dict:
    """
    Generate a full quiz for the given skills and job title.
    Maintains a ~60% MCQ / 40% Essay ratio per skill.
    Returns a QUIZ_JSON valid dictionary.
    """
    if not skills:
        raise ValueError("Skills list must not be empty.")

    quiz_json = {"questions": []}

    for skill in skills:
        # Calculate exactly how many of each type to generate for this skill
        n_mcq = round(n_per_skill * 0.6)
        n_essay = n_per_skill - n_mcq

        difficulties = ["easy", "medium", "hard"]

        # Generate MCQs
        for _ in range(n_mcq):
            difficulty = random.choice(difficulties)
            sys_p, usr_p = build_mcq_prompt(skill, difficulty)
            try:
                question_data = call_llm(sys_p, usr_p, expect_json=True)
                quiz_json["questions"].append(question_data)
            except Exception as e:
                raise RuntimeError(f"Failed to generate MCQ for skill '{skill}': {e}") from e

        # Generate Essays
        for _ in range(n_essay):
            difficulty = random.choice(difficulties)
            sys_p, usr_p = build_essay_prompt(skill, difficulty)
            try:
                question_data = call_llm(sys_p, usr_p, expect_json=True)
                quiz_json["questions"].append(question_data)
            except Exception as e:
                raise RuntimeError(f"Failed to generate essay for skill '{skill}': {e}") from e

    return quiz_json


if __name__ == "__main__":
    print(f"Generating sample Quiz for 'Data Analyst' with skills: Python, SQL...")
    quiz = generate_quiz(["Python", "SQL"], "Data Analyst", n_per_skill=2)
    print("\n\n--- QUIZ GENERATED ---")
    print(json.dumps(quiz, indent=2))
