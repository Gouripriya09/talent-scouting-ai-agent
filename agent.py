import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

from dotenv import load_dotenv
import google.api_core.exceptions as api_exceptions
import google.generativeai as genai

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise EnvironmentError(
        "Missing GOOGLE_API_KEY in environment. Add it to a .env file or your shell environment."
    )

genai.configure(api_key=API_KEY)
MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "models/gemini-2.5-flash")


def send_chat_prompt(prompt: str) -> str:
    model = genai.GenerativeModel(MODEL_NAME)
    chat = model.start_chat()
    try:
        response = chat.send_message(prompt)
    except api_exceptions.ResourceExhausted as exc:
        raise RuntimeError(
            "Gemini quota exceeded. "
            "Enable billing on your Google Cloud project for higher limits, "
            "switch to a paid model, or wait for quota renewal. "
            f"Model={MODEL_NAME}. Original error: {exc}"
        ) from exc

    if hasattr(response, "text") and response.text:
        return response.text.strip()
    if hasattr(response, "last") and getattr(response.last, "text", None):
        return response.last.text.strip()
    return str(response)


@dataclass
class Candidate:
    name: str
    title: str
    skills: List[str]
    experience: str


CANDIDATES: List[Candidate] = [
    Candidate(
        name="Avery Chen",
        title="Senior Backend Engineer",
        skills=["Python", "Django", "REST APIs", "AWS", "data pipelines"],
        experience="6 years building backend systems and APIs for mid-sized SaaS products.",
    ),
    Candidate(
        name="Jordan Kim",
        title="Machine Learning Engineer",
        skills=["Python", "machine learning", "data science", "TensorFlow", "SQL"],
        experience="5 years working on ML model development and analytics platforms.",
    ),
    Candidate(
        name="Maya Patel",
        title="Staff Software Engineer",
        skills=["Python", "FastAPI", "cloud architecture", "DevOps", "microservices"],
        experience="7 years designing scalable cloud services and deployment automation.",
    ),
    Candidate(
        name="Carlos Rivera",
        title="Full Stack Engineer",
        skills=["Python", "React", "full-stack development", "product collaboration", "CI/CD"],
        experience="4 years delivering customer-facing web applications and APIs.",
    ),
    Candidate(
        name="Nina Gomez",
        title="Data Engineer",
        skills=["Python", "SQL", "data engineering", "analytics", "Airflow"],
        experience="5 years building ETL workflows, reporting systems, and data integrations.",
    ),
]


def parse_json_response(text: str) -> Dict[str, str]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and start < end:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
    raise ValueError("Unable to parse JSON from Gemini response")


def score_candidate(job_description: str, candidate: Candidate) -> Tuple[int, str]:
    prompt = (
        "You are a recruiting assistant evaluating candidate fit for a job description. "
        "Return valid JSON with keys `score` and `explanation`. "
        "Score must be an integer from 0 to 100. Keep explanation brief and specific.\n\n"
        f"Job Description:\n{job_description}\n\n"
        f"Candidate:\nName: {candidate.name}\nTitle: {candidate.title}\n"
        f"Skills: {', '.join(candidate.skills)}\nExperience: {candidate.experience}\n"
        "Output only JSON."
    )

    content = send_chat_prompt(
        "You are a talent matching assistant.\n\n" + prompt
    )
    result = parse_json_response(content)
    score = int(result.get("score", 0))
    explanation = result.get("explanation", "No explanation provided.")
    return max(0, min(100, score)), explanation


def generate_outreach_email(job_description: str, candidate: Candidate) -> str:
    prompt = (
        "Write a short, polite outreach email to this candidate for the role described below. "
        "Mention the candidate's strongest fit areas and stay professional and friendly.\n\n"
        f"Candidate:\nName: {candidate.name}\nTitle: {candidate.title}\n"
        f"Skills: {', '.join(candidate.skills)}\nExperience: {candidate.experience}\n\n"
        f"Job Description:\n{job_description}"
    )

    content = send_chat_prompt(
        "You are a professional recruiting assistant.\n\n" + prompt
    )
    return content


def score_candidates(job_description: str, candidates: List[Candidate]) -> List[Dict[str, object]]:
    responses = []
    for candidate in candidates:
        score, explanation = score_candidate(job_description, candidate)
        responses.append(
            {
                "candidate": candidate,
                "score": score,
                "explanation": explanation,
            }
        )
        time.sleep(1)  # Rate limit: 1 second between requests to avoid quota exhaustion
    return sorted(responses, key=lambda item: item["score"], reverse=True)


def read_job_description() -> str:
    print("Paste the job description below. Finish input with an empty line:")
    lines: List[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if not line.strip():
            break
        lines.append(line)
    return "\n".join(lines).strip()


def main() -> None:
    if len(sys.argv) > 1:
        job_description = " ".join(sys.argv[1:]).strip()
    else:
        job_description = read_job_description()

    if not job_description:
        print("No job description provided. Exiting.")
        return

    print("\nScoring candidates...\n")
    scored = score_candidates(job_description, CANDIDATES)

    for result in scored:
        candidate = result["candidate"]
        print(
            f"{candidate.name} ({candidate.title}) - Score: {result['score']}\n"
            f"  Explanation: {result['explanation']}\n"
        )

    best = scored[0]
    email = generate_outreach_email(job_description, best["candidate"])

    print("---\nBest outreach email:\n---\n")
    print(email)


if __name__ == "__main__":
    main()
