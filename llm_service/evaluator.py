from llm_service.config import call_llm
from llm_service.prompt_templates import build_evaluation_prompt


def evaluate_mcq(question_obj: dict, user_answer: str) -> dict:
    """
    Rule-based evaluation for MCQ questions.
    Compares the user_answer against correct_answer without calling an LLM.
    Returns: {question, user_answer, score, max_score, feedback}
    """
    correct_answer = question_obj.get("correct_answer", "").strip()
    user_answer_clean = str(user_answer).strip()

    # Exact match (case-insensitive) for robust MCQ grading
    if user_answer_clean.lower() == correct_answer.lower():
        score = 5
        feedback = f"Correct. The answer is: {question_obj.get('correct_answer')}."
    else:
        score = 0
        feedback = f"Incorrect. The correct answer was: {question_obj.get('correct_answer')}."

    return {
        "question": question_obj.get("question"),
        "user_answer": user_answer,
        "score": score,
        "max_score": 5,
        "feedback": feedback
    }


def evaluate_essay(question_obj: dict, user_answer: str) -> dict:
    """
    LLM-based evaluation for essay or free-text questions.
    Calls chosen LLM to grade the user's text on a 0-5 scale.
    Returns: {question, user_answer, score, max_score, feedback}
    """
    sys_p, usr_p = build_evaluation_prompt(
        question=question_obj.get("question"),
        user_answer=user_answer,
        q_type=question_obj.get("type", "text")
    )

    try:
        eval_response = call_llm(sys_p, usr_p, expect_json=True)
    except Exception as e:
        # Graceful degradation: return 0 score with error feedback instead of crashing
        return {
            "question": question_obj.get("question"),
            "user_answer": user_answer,
            "score": 0,
            "max_score": 5,
            "feedback": f"Evaluation failed due to LLM error: {e}"
        }

    return {
        "question": question_obj.get("question"),
        "user_answer": user_answer,
        "score": eval_response.get("score", 0),
        "max_score": eval_response.get("max_score", 5),
        "feedback": eval_response.get("feedback", "No feedback provided.")
    }


def evaluate_quiz(quiz: dict, user_answers: list[str]) -> list[dict]:
    """
    End-to-End evaluator. Routes each quiz question to the right evaluator algorithm.
    Returns a list of 'question_results' mapping cleanly to the RESULT_JSON contract.
    """
    questions = quiz.get("questions", [])
    if len(questions) != len(user_answers):
        raise ValueError(f"Length mismatch: {len(questions)} questions vs {len(user_answers)} answers.")

    results = []

    for q_obj, u_ans in zip(questions, user_answers):
        if q_obj.get("type") == "mcq":
            res = evaluate_mcq(q_obj, u_ans)
        else:
            # Default to essay evaluation for "text" and "coding" type questions
            res = evaluate_essay(q_obj, u_ans)

        results.append(res)

    return results
