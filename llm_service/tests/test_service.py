import pytest
from fastapi.testclient import TestClient
from llm_service.service import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "model" in response.json()

def test_evaluate_endpoint_mock_workflow():
    """Testing the error handling robustness and contract ingestion."""
    mock_evaluate_request = {
        "quiz": {
            "questions": [
                {"question": "Q1", "type": "mcq", "correct_answer": "A", "skill_tag": "Docker"}
            ]
        },
        "answers": ["A"]
    }
    
    response = client.post("/evaluate", json=mock_evaluate_request)
    assert response.status_code == 200
    
    data = response.json()
    assert "score" in data
    assert "feedback" in data
    assert "skill_scores" in data
    assert data["score"] == 100
    assert data["skill_scores"]["Docker"] == 100

def test_generate_interview_validation_error():
    # Sending missing attributes
    mock_bad_req = {
        "jd_json": {"bad_format": True},
        "job_title": "Missing Skills Array"
    }
    response = client.post("/generate-interview", json=mock_bad_req)
    assert response.status_code == 422  # Handled automatically by FastAPI Pydantic Models

@pytest.mark.integration
def test_generate_quiz_integration_endpoint():
    """Live integration hitting the LLM model through the API boundaries."""
    req = {
        "cv_json": {
            "skills": ["Git"]
        },
        "job_title": "Developer"
    }
    
    response = client.post("/generate-quiz", json=req)
    assert response.status_code == 200
    assert "questions" in response.json()
    
