import pytest
from llm_service.evaluator import evaluate_mcq, evaluate_essay, evaluate_quiz
from llm_service.quiz_generator import generate_quiz

def test_evaluate_mcq():
    q_obj = {
        "question": "What is Python?",
        "type": "mcq",
        "correct_answer": "A programming language"
    }
    
    # Correct
    res = evaluate_mcq(q_obj, "a programming language")
    assert res["score"] == 5
    assert "Correct." in res["feedback"]
    
    # Incorrect
    res = evaluate_mcq(q_obj, "A snake")
    assert res["score"] == 0
    assert "Incorrect" in res["feedback"]
    
    # Schema check
    assert "question" in res
    assert "user_answer" in res
    assert "score" in res
    assert "max_score" in res
    assert "feedback" in res

@pytest.mark.integration
def test_evaluate_quiz_integration():
    """Live API integration test checking the result parsing schema."""
    # Generate a lightweight quiz
    quiz = generate_quiz(["SQL"], "Data Analyst", n_per_skill=1)
    
    # Generate mock answers: standard text to not test the LLM's grading IQ, just the pipeline
    answers = ["I am not entirely sure about this SQL question, perhaps SELECT *?"]
    
    results = evaluate_quiz(quiz, answers)
    
    assert len(results) == 1
    
    res = results[0]
    assert "question" in res
    assert "user_answer" in res
    assert res["user_answer"] == answers[0]
    assert "score" in res
    assert res["max_score"] == 5
    assert isinstance(res["feedback"], str)
