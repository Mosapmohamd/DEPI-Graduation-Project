import json


def build_mcq_prompt(skill: str, difficulty: str) -> tuple[str, str]:
    """
    Build system and user prompts for generating a single MCQ question.
    Returns: (system_prompt, user_prompt)
    """
    system_prompt = """You are an expert technical interviewer and test creator.
Your goal is to generate a high-quality Multiple Choice Question (MCQ).
Return ONLY valid JSON. Do not include any explanations outside the JSON."""

    user_prompt = f"""Generate 1 Multiple Choice Question (MCQ) testing the following skill.
Skill: {skill}
Difficulty: {difficulty}

Follow this exact JSON contract:
{{
  "question": "The text of the question",
  "type": "mcq",
  "difficulty": "{difficulty}",
  "skill_tag": "{skill}",
  "options": ["A", "B", "C", "D"],
  "correct_answer": "One of the provided options exactly as written"
}}

Ensure there are exactly 4 options. Return ONLY valid JSON."""

    return system_prompt, user_prompt


def build_essay_prompt(skill: str, difficulty: str) -> tuple[str, str]:
    """
    Build system and user prompts for generating a single essay/text question.
    Returns: (system_prompt, user_prompt)
    """
    system_prompt = """You are an expert technical interviewer and test creator.
Your goal is to generate a high-quality free-text / essay question.
Return ONLY valid JSON. Do not include any explanations outside the JSON."""

    user_prompt = f"""Generate 1 essay or free-text question testing the following skill.
Skill: {skill}
Difficulty: {difficulty}

Follow this exact JSON contract:
{{
  "question": "The text of the question",
  "type": "text",
  "difficulty": "{difficulty}",
  "skill_tag": "{skill}",
  "options": null
}}

Return ONLY valid JSON."""

    return system_prompt, user_prompt


def build_coding_prompt(skill: str, difficulty: str) -> tuple[str, str]:
    """
    Build system and user prompts for generating a single coding question.
    Returns: (system_prompt, user_prompt)
    """
    system_prompt = """You are an expert technical interviewer and test creator.
Your goal is to generate a high-quality practical coding question.
Return ONLY valid JSON. Do not include any explanations outside the JSON."""

    user_prompt = f"""Generate 1 coding question testing the following skill.
Skill: {skill}
Difficulty: {difficulty}

Follow this exact JSON contract:
{{
  "question": "The text of the coding question or scenario",
  "type": "coding",
  "difficulty": "{difficulty}",
  "skill_tag": "{skill}",
  "options": null
}}

Return ONLY valid JSON."""

    return system_prompt, user_prompt


def build_evaluation_prompt(question: str, user_answer: str, q_type: str) -> tuple[str, str]:
    """
    Build system and user prompts for evaluating a candidate answer.
    Returns: (system_prompt, user_prompt)
    """
    system_prompt = """You are an expert technical interviewer evaluating a candidate's answer.
You must be fair and precise.
Return ONLY valid JSON. Do not include any explanations outside the JSON."""

    user_prompt = f"""Evaluate the candidate's answer to the following question.
Question Type: {q_type}
Question: {question}

Candidate Answer: {user_answer}

Follow this exact JSON contract:
{{
  "score": <integer from 0 to 5, where 5 is perfectly correct and 0 is completely wrong>,
  "max_score": 5,
  "feedback": "Specific feedback explaining the score and what the candidate missed or did well"
}}

Return ONLY valid JSON."""

    return system_prompt, user_prompt


def build_interview_prompt(jd_skills: list[str], job_title: str) -> tuple[str, str]:
    """
    Build system and user prompts for generating scenario-based interview questions.
    Returns: (system_prompt, user_prompt)
    """
    system_prompt = """You are an expert technical interviewer preparing for a live interview.
Your goal is to generate scenario-based and applied interview questions.
Return ONLY valid JSON. Do not include any explanations outside the JSON."""

    skills_str = ", ".join(jd_skills)
    user_prompt = f"""Generate scenario-based interview questions for the following role.
Job Title: {job_title}
Required Skills: {skills_str}

Follow this exact JSON contract (an array of question objects):
[
  {{
    "question": "The applied scenario-based text question",
    "type": "text",
    "difficulty": "medium",
    "skill_tag": "<must be one of the Required Skills exactly>",
    "options": null
  }}
]

Generate exactly 5 questions spanning the Required Skills. Return ONLY valid JSON."""

    return system_prompt, user_prompt


def build_learning_path_prompt(weak_skills: list[str]) -> tuple[str, str]:
    """
    Build system and user prompts for generating a learning path for weak skills.
    Returns: (system_prompt, user_prompt) — note: the LLM should return markdown, not JSON.
    """
    system_prompt = """You are an expert technical career coach.
Provide a concise, highly actionable learning path formatted in Markdown.
Do NOT return JSON. Return markdown text directly."""

    skills_str = ", ".join(weak_skills)
    user_prompt = f"""The candidate has demonstrated a skill gap in the following areas:
Weak Skills: {skills_str}

Please generate a professional, short learning path to help them improve.
Format your response in Markdown with bullet points, focusing on actionable steps, common concepts to master, and recommended types of resources (e.g., specific libraries, hands-on projects, documentation)."""

    return system_prompt, user_prompt


if __name__ == "__main__":
    # Smoke test for visually checking prompts over LLM
    from llm_service.config import call_llm

    print("Testing MCQ Prompt construction:")
    sys_p, usr_p = build_mcq_prompt("Python", "easy")
    print("SYSTEM:", sys_p)
    print("USER:\n", usr_p)

    print("\n\nTesting with chosen LLM...")
    try:
        res = call_llm(sys_p, usr_p, expect_json=True)
        print("Success! JSON Output:\n", json.dumps(res, indent=2))
    except Exception as e:
        print("Failed to call LLM:", e)
