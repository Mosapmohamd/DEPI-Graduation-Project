import pytest
from llm_service.prompt_templates import (
    build_mcq_prompt,
    build_essay_prompt,
    build_coding_prompt,
    build_evaluation_prompt,
    build_interview_prompt,
    build_learning_path_prompt
)
from llm_service.config import call_llm

def test_build_mcq_prompt():
    sys_p, usr_p = build_mcq_prompt("Python", "easy")
    assert "expert technical interviewer" in sys_p
    assert "Python" in usr_p
    assert "4 options" in usr_p
    assert "correct_answer" in usr_p

def test_build_essay_prompt():
    sys_p, usr_p = build_essay_prompt("SQL", "hard")
    assert "text" in usr_p
    assert "SQL" in usr_p

def test_build_coding_prompt():
    sys_p, usr_p = build_coding_prompt("Machine Learning", "medium")
    assert "coding" in usr_p
    assert "Machine Learning" in usr_p

def test_build_evaluation_prompt():
    sys_p, usr_p = build_evaluation_prompt("What is OOP?", "Objects and classes", "text")
    assert "What is OOP?" in usr_p
    assert "Objects and classes" in usr_p
    assert "score" in usr_p

def test_build_interview_prompt():
    sys_p, usr_p = build_interview_prompt(["Docker", "K8s"], "DevOps Engineer")
    assert "Docker" in usr_p
    assert "DevOps Engineer" in usr_p

def test_build_learning_path_prompt():
    sys_p, usr_p = build_learning_path_prompt(["Regularization", "Pandas"])
    assert "Markdown" in sys_p
    assert "Regularization" in usr_p

@pytest.mark.integration
def test_llm_integration_mcq():
    """Smoke test to verify chosen LLM (Groq/Ollama) can handle the MCQ prompt."""
    sys_p, usr_p = build_mcq_prompt("Python", "easy")
    
    # Run 3 times to check reliability as required
    successes = 0
    for i in range(3):
        print(f"Run {i+1}...")
        try:
            res = call_llm(sys_p, usr_p, expect_json=True)
            assert isinstance(res, dict)
            assert "question" in res
            assert "options" in res
            assert "correct_answer" in res
            assert len(res["options"]) == 4
            successes += 1
            print("Run successful")
        except Exception as e:
            print(f"Run failed: {e}")
            pass
            
    assert successes >= 3, f"Failed to parse JSON reliably from LLM: {successes}/3 succeeded"
