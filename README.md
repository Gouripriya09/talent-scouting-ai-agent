# 🤖 AI-Powered Talent Scouting Agent

An AI-powered decision system that not only matches candidates to jobs, but also evaluates their interest and generates actionable outreach - enabling recruiters to move from screening to decision instantly.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.56.0-FF4B4B?logo=streamlit&logoColor=white)
![OpenRouter](https://img.shields.io/badge/LLM-OpenRouter-6E56CF)
![Status](https://img.shields.io/badge/Status-Hackathon%20Ready-16A34A)

An AI-first recruiter assistant that parses job descriptions, identifies relevant candidates, evaluates technical fit and interest through simulated conversations, and generates a ranked shortlist with clear reasoning plus outreach-ready email.

## Problem

Recruiters often spend significant time on:

- Manually scanning job descriptions and mapping required skills
- Screening multiple candidate profiles with inconsistent criteria
- Estimating candidate interest before real outreach
- Writing personalized outreach from scratch for every role

This slows hiring cycles and makes shortlisting less consistent.

## Solution

This agent automates recruiter decision support in a clear pipeline:

1. Parse the job description and extract key skills.
2. Select the most relevant candidates from a candidate pool.
3. Evaluate each candidate on technical fit and conversation-driven interest.
4. Compute final ranking with confidence indicators.
5. Explain why the top candidate was selected.
6. Generate a personalized outreach email for immediate action.

## ⚡ How It Works (Quick View)

1. Input: Job Description  
2. AI extracts required skills  
3. Agent evaluates candidates  
4. Simulates conversation to assess interest  
5. Outputs ranked shortlist + reasoning + email

## Key Features

- JD parsing with AI-based key skill extraction
- Candidate discovery from predefined profile pool
- Hybrid match scoring (deterministic overlap + AI evaluation)
- Simulated recruiter-candidate interaction
- Ranked shortlist with explainable reasoning
- Outreach email generation for top candidate

## 🧠 Architecture (AI Decision Pipeline)

```text
Job Description Input
        ↓
Skill Extraction (AI)
        ↓
Candidate Selection
        ↓
Evaluation Engine
  - Skill Match
  - AI Evaluation
  - Conversation
        ↓
Scoring System
        ↓
Explainability Layer
        ↓
Streamlit UI
```

## Tech Stack

- Streamlit
- Python
- OpenRouter API
- Requests
- Pandas
- Python-Dotenv

## Scoring Logic

- **Match Score**: Technical role fit using skill overlap + AI technical judgment.
- **Interest Score**: Candidate engagement inferred from simulated conversation.
- **Final Score**: Weighted score using:
  - `Final Score = 0.7 * Match Score + 0.3 * Interest Score`
- **Confidence Score**: Stability indicator derived from match and interest signals.

## How to Run Locally

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start the app:

```bash
streamlit run app.py
```

## Environment Variables

Create a `.env` file in the project root:

```env
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=mistralai/mistral-7b-instruct
```

## 📌 Example Output

- 🏆 Recommended Candidate: Nina Gomez  
- 📊 Match Score: 78  
- 💬 Interest Score: 65  
- 📧 Outreach Email Generated  
- 🧠 Explanation Provided

## Demo

- Demo Video: [Add demo video link](https://example.com/demo-video)
- Live App: [Add deployed app link](https://example.com/live-app)

## Why This Matters

- Reduces manual recruiter screening effort
- Standardizes candidate evaluation criteria
- Surfaces top-fit talent faster
- Adds explainability to hiring decisions
- Speeds up personalized outreach

## Author

- Name: Gouripriya Kodam
- GitHub: https://github.com/Gouripriya09

This system transforms recruitment from manual screening into an intelligent, explainable decision-making workflow.
