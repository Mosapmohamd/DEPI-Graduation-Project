# AI Skill Gap Assessment & Interview Preparation System

## Overview
This project builds an AI system that evaluates technical skills from a CV and prepares users for interviews.

The system:
- analyzes CVs
- predicts job roles
- generates quizzes
- evaluates answers
- detects skill gaps

Target users:
- final-year students
- recent graduates

---

## Features

- CV parsing and skill extraction  
- Job role prediction  
- Personalized quiz generation  
- Interview simulation  
- Automated answer evaluation  
- Skill gap analysis  
- Learning recommendations  

---

## System Architecture

The system is divided into 4 main components:

### 1. NLP Engine
- Extracts data from CV and job description  
- Outputs structured JSON  

### 2. ML Engine
- Predicts job roles  
- Detects missing skills  

### 3. LLM Engine
- Generates quiz questions  
- Evaluates answers  
- Simulates interviews  

### 4. Fullstack System
- Backend APIs  
- Frontend UI  
- Integration layer  

---

## Workflow

1. User uploads CV  
2. System extracts skills  
3. System predicts job role  
4. User selects role  
5. System generates quiz  
6. User answers questions  
7. System evaluates answers  
8. System shows score and skill gaps  

---

## Project Structure

```
project/
│
├── nlp/
├── ml/
├── llm/
├── backend/
├── frontend/
├── data/
├── models/
└── app.py
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/ai-skill-gap-system.git
cd ai-skill-gap-system
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

---

## API Endpoints

- /upload_cv  
- /predict_role  
- /generate_quiz  
- /evaluate_answers  
- /get_results  

---

## Data Format

### CV JSON
```json
{
  "skills": [],
  "projects": [],
  "experience": []
}
```

### JD JSON
```json
{
  "required_skills": []
}
```

### Quiz JSON
```json
{
  "questions": [
    {
      "question": "",
      "type": ""
    }
  ]
}
```

### Result JSON
```json
{
  "score": 0,
  "gaps": []
}
```

---

## Team Responsibilities

### AI NLP Engineer
- CV parsing  
- Skill extraction  
- Job description parsing  

### ML Engineer
- Role prediction  
- Skill gap analysis  

### LLM Engineer
- Quiz generation  
- Answer evaluation  

### Fullstack Integration Engineer
- APIs  
- UI  
- System integration  

---

## Development Plan

Each week delivers a working system:

- Week 1: full flow with mock data  
- Week 2: CV parsing  
- Week 3: role prediction  
- Week 4: quiz generation  
- Week 5: evaluation and gaps  
- Week 6: optimization and deployment  

---

## Tech Stack

- Python  
- Streamlit  
- Scikit-learn  
- Transformers  
- LLM APIs  

---

## Future Improvements

- Real-time feedback  
- More accurate skill detection  
- Advanced interview simulation  
- Multi-language support  

---

## License

MIT License
