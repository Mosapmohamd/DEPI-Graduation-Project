import pytest
from llm_service.scorer import (
    compute_skill_scores,
    detect_weak_skills,
    build_result_json,
    generate_learning_path
)

def test_scorer_logic():
    # Mock data bridging Sprint 2 and 3 outputs
    mock_quiz = {
        "questions": [
            {"question": "Q1 Python", "skill_tag": "Python"},
            {"question": "Q2 Python", "skill_tag": "Python"},
            {"question": "Q1 SQL", "skill_tag": "SQL"}
        ]
    }
    
    mock_results = [
        {"question": "Q1 Python", "score": 5, "max_score": 5},
        {"question": "Q2 Python", "score": 2, "max_score": 5},  # 7 / 10 = 70%
        {"question": "Q1 SQL", "score": 0, "max_score": 5}       # 0 / 5 = 0%
    ]
    
    # Test skill aggregation
    scores = compute_skill_scores(mock_results, mock_quiz)
    assert "Python" in scores
    assert "SQL" in scores
    assert scores["Python"] == 70
    assert scores["SQL"] == 0
    
    # Test gap detection
    weak = detect_weak_skills(scores, threshold=60)
    assert len(weak) == 1
    assert weak[0] == "SQL"
    
    # Test final JSON formulation
    final = build_result_json(mock_results, scores, weak)
    assert final["score"] == round((7/15)*100)  # Total 7/15 = 47%
    assert final["skill_scores"] == scores
    assert final["gaps"] == weak
    assert len(final["question_results"]) == 3

@pytest.mark.integration
def test_learning_path_integration():
    """Live integration testing the markdown response block in learning path generation."""
    weak_skills = ["SQL joins", "Python Generators"]
    
    path_md = generate_learning_path(weak_skills)
    
    assert isinstance(path_md, str)
    assert len(path_md) > 50  # Must actually generate some valid markdown
    assert "no major skill gaps" not in path_md.lower()
