import pytest
from unittest.mock import patch
from llm_service.interview_generator import generate_interview_questions

@patch("llm_service.interview_generator.call_llm")
def test_generate_interview_questions_mock(mock_call_llm):
    mock_questions = [
        {"question": "Q1", "type": "text", "difficulty": "medium", "skill_tag": "Docker", "options": None},
        {"question": "Q2", "type": "text", "difficulty": "medium", "skill_tag": "Python", "options": None},
        {"question": "Q3", "type": "text", "difficulty": "medium", "skill_tag": "Docker", "options": None},
        {"question": "Q4", "type": "text", "difficulty": "medium", "skill_tag": "REST APIs", "options": None},
        {"question": "Q5", "type": "text", "difficulty": "medium", "skill_tag": "Python", "options": None}
    ]
    mock_call_llm.return_value = mock_questions
    
    jd = {"required_skills": ["Docker", "REST APIs", "Python"]}
    quiz = generate_interview_questions(jd, "Backend Engineer", n=3)
    
    # Needs 1 run to get 5 questions, sliced down to 3
    assert mock_call_llm.call_count == 1
    assert "questions" in quiz
    assert len(quiz["questions"]) == 3

@pytest.mark.integration
def test_interview_generator_integration():
    """Live API test for Interview Generator."""
    jd = {"required_skills": ["Docker", "Python"]}
    # Requesting n=2 to keep the LLM inference limits low and fast
    quiz = generate_interview_questions(jd, "Backend Engineer", n=2)
    
    assert "questions" in quiz
    assert len(quiz["questions"]) == 2
    
    q1 = quiz["questions"][0]
    assert q1["type"] == "text"
    assert "skill_tag" in q1
    assert q1["skill_tag"] in jd["required_skills"]
