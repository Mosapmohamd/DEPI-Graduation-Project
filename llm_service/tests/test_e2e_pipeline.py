"""
PHASE 3 & 4 — End-to-End Pipeline Tests + FastAPI Endpoint Tests
Run this script to execute ALL integration pipeline verifications.
"""
import json
import sys
sys.path.insert(0, r"d:\DEPI-Graduation-Project")

from llm_service.quiz_generator import generate_quiz
from llm_service.interview_generator import generate_interview_questions
from llm_service.evaluator import evaluate_quiz
from llm_service.scorer import compute_skill_scores, detect_weak_skills, build_result_json, generate_learning_path
from llm_service.service import app
from fastapi.testclient import TestClient

SEPARATOR = "\n" + "=" * 70

def flow_a():
    """Flow A — Career Exploration (no JD)"""
    print(SEPARATOR)
    print("FLOW A — Career Exploration")
    print(SEPARATOR)

    # Step 1
    cv_json = {
        "skills": ["Python", "SQL", "Machine Learning"],
        "projects": ["churn prediction model", "NLP sentiment classifier"],
        "experience": [{"role": "Data Analyst", "years": 2}],
        "confidence": 0.87
    }
    job_title = "Data Scientist"
    print(f"\n[Step 1] CV Skills: {cv_json['skills']}, Job Title: {job_title}")

    # Step 2
    print("\n[Step 2] Generating quiz...")
    quiz = generate_quiz(cv_json["skills"], job_title, n_per_skill=2)
    print(json.dumps(quiz, indent=2))
    
    n_questions = len(quiz["questions"])
    n_mcq = sum(1 for q in quiz["questions"] if q.get("type") == "mcq")
    n_essay = n_questions - n_mcq
    print(f"\n  Total questions: {n_questions} (expected 6)")
    print(f"  MCQ: {n_mcq}, Essay/Text: {n_essay}")
    
    # Verify skill tags
    skill_tags = set(q.get("skill_tag") for q in quiz["questions"])
    print(f"  Skill tags found: {skill_tags}")
    assert n_questions == 6, f"Expected 6 questions, got {n_questions}"

    # Step 3 — Build answers matching the actual generated quiz
    user_answers = []
    for q in quiz["questions"]:
        if q.get("type") == "mcq":
            # Provide the correct answer for MCQs 
            user_answers.append(q.get("correct_answer", "A"))
        else:
            user_answers.append("I am not entirely sure about this topic, but I believe the answer involves using standard techniques and best practices in the field.")
    
    print(f"\n[Step 3] User answers prepared: {len(user_answers)} answers")

    # Step 4
    print("\n[Step 4] Evaluating answers...")
    question_results = evaluate_quiz(quiz, user_answers)
    for i, res in enumerate(question_results):
        print(f"  Q{i+1}: score={res['score']}/{res['max_score']} | {res['feedback'][:80]}...")
    assert len(question_results) == 6

    # Step 5
    print("\n[Step 5] Computing skill scores...")
    skill_scores = compute_skill_scores(question_results, quiz)
    print(f"  Skill scores: {json.dumps(skill_scores, indent=4)}")
    for skill in cv_json["skills"]:
        assert skill in skill_scores, f"Missing skill: {skill}"
        assert 0 <= skill_scores[skill] <= 100

    # Step 6
    print("\n[Step 6] Detecting weak skills (threshold=60)...")
    weak_skills = detect_weak_skills(skill_scores, threshold=60)
    print(f"  Weak skills: {weak_skills}")

    # Step 7
    print("\n[Step 7] Building RESULT_JSON...")
    result = build_result_json(question_results, skill_scores, weak_skills)
    print(json.dumps(result, indent=2, default=str))
    
    # Contract validation
    assert "score" in result and isinstance(result["score"], int)
    assert "feedback" in result and isinstance(result["feedback"], str)
    assert "skill_scores" in result and isinstance(result["skill_scores"], dict)
    assert "question_results" in result and isinstance(result["question_results"], list)
    assert "gaps" in result and isinstance(result["gaps"], list)
    print("\n  ✅ RESULT_JSON contract validated!")

    # Step 8
    print("\n[Step 8] Generating learning path for weak skills...")
    if weak_skills:
        learning_path = generate_learning_path(weak_skills)
        print(learning_path[:500])
    else:
        print("  No weak skills — generating for demo purposes...")
        learning_path = generate_learning_path(["SQL"])
        print(learning_path[:500])
    
    print("\n✅ FLOW A COMPLETE")
    return True


def flow_b():
    """Flow B — Job Preparation (with JD)"""
    print(SEPARATOR)
    print("FLOW B — Job Preparation")
    print(SEPARATOR)

    # Step 1
    jd_json = {"required_skills": ["Docker", "REST APIs", "Python", "PostgreSQL"]}
    job_title = "Backend Engineer"
    print(f"\n[Step 1] JD Skills: {jd_json['required_skills']}, Job Title: {job_title}")

    # Step 2
    print("\n[Step 2] Generating interview questions (n=4 to save tokens)...")
    quiz = generate_interview_questions(jd_json, job_title, n=4)
    print(json.dumps(quiz, indent=2))
    n_questions = len(quiz["questions"])
    print(f"\n  Total questions: {n_questions}")
    assert n_questions == 4

    # Step 3
    user_answers = [
        "I would use Docker Compose to define multi-container applications with a docker-compose.yml file.",
        "REST APIs follow the stateless client-server architecture using HTTP methods like GET, POST, PUT, DELETE.",
        "In Python I would use the requests library or FastAPI framework to build REST endpoints.",
        "PostgreSQL supports ACID transactions, indexing with B-trees, and complex queries with joins and CTEs."
    ]
    print(f"\n[Step 3] User answers prepared: {len(user_answers)} answers")

    # Step 4
    print("\n[Step 4] Evaluating answers...")
    question_results = evaluate_quiz(quiz, user_answers)
    for i, res in enumerate(question_results):
        print(f"  Q{i+1}: score={res['score']}/{res['max_score']} | {res['feedback'][:80]}...")
    assert len(question_results) == 4

    # Step 5
    print("\n[Step 5] Computing skill scores...")
    skill_scores = compute_skill_scores(question_results, quiz)
    print(f"  Skill scores: {json.dumps(skill_scores, indent=4)}")

    # Step 6
    print("\n[Step 6] Detecting weak skills...")
    weak_skills = detect_weak_skills(skill_scores, threshold=60)
    print(f"  Weak skills: {weak_skills}")

    # Step 7
    print("\n[Step 7] Building RESULT_JSON...")
    result = build_result_json(question_results, skill_scores, weak_skills)
    print(json.dumps(result, indent=2, default=str))
    assert "score" in result
    assert "skill_scores" in result
    assert "gaps" in result
    print("\n  ✅ RESULT_JSON contract validated!")

    # Step 8
    print("\n[Step 8] Generating learning path...")
    if weak_skills:
        learning_path = generate_learning_path(weak_skills)
        print(learning_path[:500])
    else:
        print("  All skills above threshold — no learning path needed!")
    
    print("\n✅ FLOW B COMPLETE")
    return True


def fastapi_endpoint_tests():
    """Phase 4 — FastAPI endpoint testing."""
    print(SEPARATOR)
    print("PHASE 4 — FastAPI Endpoint Tests")
    print(SEPARATOR)
    
    client = TestClient(app)
    
    # GET /health
    print("\n[Endpoint 1] GET /health")
    r = client.get("/health")
    print(f"  Status: {r.status_code}")
    print(f"  Response: {r.json()}")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    print("  ✅ PASSED")
    
    # POST /generate-quiz
    print("\n[Endpoint 2] POST /generate-quiz")
    r = client.post("/generate-quiz", json={
        "cv_json": {"skills": ["Python"]},
        "job_title": "Developer"
    })
    print(f"  Status: {r.status_code}")
    quiz_data = r.json()
    print(f"  Questions generated: {len(quiz_data.get('questions', []))}")
    assert r.status_code == 200
    assert "questions" in quiz_data
    print("  ✅ PASSED")
    
    # POST /generate-interview
    print("\n[Endpoint 3] POST /generate-interview")
    r = client.post("/generate-interview", json={
        "jd_json": {"required_skills": ["Docker", "Python"]},
        "job_title": "Backend Engineer"
    })
    print(f"  Status: {r.status_code}")
    interview_data = r.json()
    print(f"  Questions generated: {len(interview_data.get('questions', []))}")
    assert r.status_code == 200
    assert "questions" in interview_data
    print("  ✅ PASSED")
    
    # POST /evaluate (using quiz from endpoint 2)
    print("\n[Endpoint 4] POST /evaluate")
    answers = []
    for q in quiz_data["questions"]:
        if q.get("type") == "mcq":
            answers.append(q.get("correct_answer", "A"))
        else:
            answers.append("I believe the answer is related to standard best practices.")
    
    r = client.post("/evaluate", json={"quiz": quiz_data, "answers": answers})
    print(f"  Status: {r.status_code}")
    eval_data = r.json()
    print(f"  Overall score: {eval_data.get('score')}%")
    print(f"  Skill scores: {eval_data.get('skill_scores')}")
    print(f"  Gaps: {eval_data.get('gaps')}")
    assert r.status_code == 200
    assert "score" in eval_data
    assert "skill_scores" in eval_data
    assert "question_results" in eval_data
    assert "gaps" in eval_data
    print("  ✅ PASSED")
    
    # Error handling tests
    print("\n[Error Tests]")
    
    # Empty skills
    r = client.post("/generate-quiz", json={"cv_json": {"skills": []}, "job_title": "Dev"})
    print(f"  Empty skills → Status: {r.status_code}")
    assert r.status_code == 400
    print("  ✅ Returns 400 correctly")
    
    # Missing fields
    r = client.post("/generate-interview", json={"jd_json": {"bad": True}, "job_title": "Dev"})
    print(f"  Missing fields → Status: {r.status_code}")
    assert r.status_code == 422
    print("  ✅ Returns 422 correctly")
    
    # Mismatched answers count
    r = client.post("/evaluate", json={"quiz": quiz_data, "answers": ["only_one"]})
    print(f"  Mismatched answers → Status: {r.status_code}")
    assert r.status_code in [400, 500]
    print(f"  ✅ Returns {r.status_code} correctly")
    
    print("\n✅ ALL ENDPOINT TESTS PASSED")
    return True


if __name__ == "__main__":
    success_a = flow_a()
    success_b = flow_b()
    success_ep = fastapi_endpoint_tests()
    
    print(SEPARATOR)
    print("FINAL SUMMARY")
    print(SEPARATOR)
    print(f"  Flow A: {'✅ PASSED' if success_a else '❌ FAILED'}")
    print(f"  Flow B: {'✅ PASSED' if success_b else '❌ FAILED'}")
    print(f"  FastAPI: {'✅ PASSED' if success_ep else '❌ FAILED'}")
