import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

USE_RANDOMNESS = False
DEBUG = False


def get_candidate_personality() -> str:
    # Keep deterministic behavior in production mode.
    return "neutral"


@dataclass
class Candidate:
    name: str
    skills: List[str]
    bio: str


CANDIDATES: List[Candidate] = [
    Candidate(
        name="Avery Chen",
        skills=["Python", "REST APIs", "AWS", "data pipelines", "Django"],
        bio="Avery is a backend engineer who builds scalable API platforms and cloud-native data services.",
    ),
    Candidate(
        name="Jordan Kim",
        skills=["Python", "TensorFlow", "machine learning", "SQL", "data science"],
        bio="Jordan is an ML engineer skilled at turning models into production-ready analytics solutions.",
    ),
    Candidate(
        name="Maya Patel",
        skills=["FastAPI", "microservices", "cloud architecture", "DevOps", "Python"],
        bio="Maya has led distributed systems and automation projects for high-growth SaaS companies.",
    ),
    Candidate(
        name="Carlos Rivera",
        skills=["React", "Python", "full-stack development", "CI/CD", "product collaboration"],
        bio="Carlos builds customer-facing applications and bridges technical design with product goals.",
    ),
    Candidate(
        name="Nina Gomez",
        skills=["SQL", "Airflow", "data engineering", "analytics", "Python"],
        bio="Nina specializes in ETL pipelines, reporting systems, and turning raw data into business insights.",
    ),
]


def parse_json_response(text: str) -> object:
    def _extract_quoted_field(source: str, field: str) -> str:
        marker = f'"{field}"'
        marker_idx = source.find(marker)
        if marker_idx == -1:
            return ""
        colon_idx = source.find(":", marker_idx)
        if colon_idx == -1:
            return ""
        value_start = colon_idx + 1
        while value_start < len(source) and source[value_start].isspace():
            value_start += 1
        # Only parse this helper when the field actually starts with a quote.
        # Otherwise we might accidentally consume the next key name.
        if value_start >= len(source) or source[value_start] != '"':
            return ""
        first_quote_idx = value_start

        i = first_quote_idx + 1
        escaped = False
        value_chars: List[str] = []
        while i < len(source):
            ch = source[i]
            if escaped:
                value_chars.append(ch)
                escaped = False
            elif ch == "\\":
                escaped = True
                value_chars.append(ch)
            elif ch == '"':
                break
            else:
                value_chars.append(ch)
            i += 1

        if i >= len(source) or source[i] != '"':
            return ""
        raw_value = "".join(value_chars).strip()
        if not raw_value:
            return ""
        try:
            return json.loads(f'"{raw_value}"')
        except json.JSONDecodeError:
            return raw_value.replace('\\"', '"')

    def _salvage_candidate_evaluation(source: str) -> Dict[str, object]:
        salvaged: Dict[str, object] = {}

        tech_match = re.search(
            r'"technical_match_score"\s*:\s*(-?\d+)', source, flags=re.IGNORECASE
        )
        if tech_match:
            salvaged["technical_match_score"] = int(tech_match.group(1))

        interest_match = re.search(
            r'"interest_score"\s*:\s*(-?\d+)', source, flags=re.IGNORECASE
        )
        if interest_match:
            salvaged["interest_score"] = int(interest_match.group(1))

        technical_reason = _extract_quoted_field(source, "technical_reason")
        if technical_reason:
            salvaged["technical_reason"] = technical_reason

        interest_reason = _extract_quoted_field(source, "interest_reason")
        if interest_reason:
            salvaged["interest_reason"] = interest_reason

        conversation = ""
        conv_quoted = _extract_quoted_field(source, "conversation")
        if conv_quoted:
            conversation = conv_quoted
        else:
            conv_block = re.search(
                r'"conversation"\s*:\s*(.*?)(?:,\s*"interest_score"\s*:|\s*})',
                source,
                flags=re.DOTALL | re.IGNORECASE,
            )
            if conv_block:
                conversation = conv_block.group(1).strip().strip(",").strip()
                if conversation.startswith('"') and conversation.endswith('"'):
                    conversation = conversation[1:-1]
                conversation = conversation.replace('\\"', '"')
        if conversation:
            salvaged["conversation"] = conversation

        return salvaged

    def _attempt_json_parse(candidate_text: str) -> Optional[object]:
        try:
            return json.loads(candidate_text)
        except json.JSONDecodeError:
            return None

    parsed = _attempt_json_parse(text)
    if parsed is not None:
        return parsed

    # Strategy 2: extract likely JSON object regions and parse them.
    object_candidates = re.findall(r"\{[\s\S]*\}", text)
    for candidate_obj in object_candidates:
        parsed = _attempt_json_parse(candidate_obj)
        if parsed is not None:
            return parsed

    # Strategy 3: attempt repairs for common malformed output.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and start < end:
        snippet = text[start : end + 1]

        # Repair a common malformed pattern from model responses where
        # "conversation" is emitted with raw newlines or without quotes.
        unquoted_conversation_pattern = re.compile(
            r'("conversation"\s*:\s*)([^"\{\[].*?)(,\s*"interest_score"\s*:)',
            flags=re.DOTALL,
        )
        quoted_conversation_pattern = re.compile(
            r'("conversation"\s*:\s*")([\s\S]*?)(",\s*"interest_score"\s*:)',
            flags=re.DOTALL,
        )

        def _fix_unquoted_conversation(match: re.Match) -> str:
            prefix, body, suffix = match.groups()
            cleaned = body.strip()
            return f"{prefix}{json.dumps(cleaned)}{suffix}"

        def _fix_quoted_conversation(match: re.Match) -> str:
            prefix, body, suffix = match.groups()
            cleaned = body.strip()
            return f"{prefix[:-1]}{json.dumps(cleaned)}{suffix}"

        repaired = unquoted_conversation_pattern.sub(
            _fix_unquoted_conversation, snippet
        )
        repaired = quoted_conversation_pattern.sub(
            _fix_quoted_conversation, repaired
        )
        repaired = re.sub(r",\s*}", "}", repaired)
        repaired = re.sub(r",\s*]", "]", repaired)

        parsed = _attempt_json_parse(repaired)
        if parsed is not None:
            return parsed

        salvaged = _salvage_candidate_evaluation(repaired)
        if salvaged:
            if DEBUG:
                logger.debug("Recovered malformed model JSON via field-level salvage.")
            return salvaged

    salvaged = _salvage_candidate_evaluation(text)
    if salvaged:
        if DEBUG:
            logger.debug("Recovered malformed model response via fallback field extraction.")
        return salvaged

    if DEBUG:
        print("LLM RAW RESPONSE:", text)
    logger.warning("Unable to parse JSON response. Raw output: %s", text)
    return {"parsing_error": True, "raw_response": text.strip()}


def send_chat_prompt(prompt: str) -> str:
    provider = os.getenv("LLM_PROVIDER", "openrouter")

    if provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise EnvironmentError("Missing OPENROUTER_API_KEY")

        model = os.getenv("OPENROUTER_MODEL", "mistralai/mistral-7b-instruct")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
        }

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
        )

        if response.status_code != 200:
            raise RuntimeError(f"OpenRouter API error: {response.text}")

        result = response.json()["choices"][0]["message"]["content"].strip()
        if DEBUG:
            logger.debug("Raw LLM response: %s", result)
        time.sleep(1)
        return result

    else:
        raise ValueError("Unsupported provider")


@st.cache_data(show_spinner=False)
def extract_key_skills(job_description: str) -> List[str]:
    prompt = (
        "Extract the top 5 required or desired skills from the job description below. "
        "Return ONLY a valid JSON array of skill strings, with no additional text, explanations, or formatting.\n\n"
        f"Job Description:\n{job_description}\n\n"
        "Example output: [\"Python\", \"SQL\", \"Machine Learning\", \"Data Analysis\", \"AWS\"]"
    )
    content = send_chat_prompt(prompt)
    result = parse_json_response(content)
    if isinstance(result, list):
        return [str(item).strip() for item in result]
    if isinstance(result, dict) and "skills" in result:
        return [str(item).strip() for item in result["skills"]]
    raise ValueError("Unexpected skill extraction format from Gemini")


def retrieve_candidates(job_description: str, top_k: int = 3) -> List[Candidate]:
    """Use LLM to select top relevant candidates from the pool."""
    candidates_text = "\n".join(
        f"- {c.name}: {', '.join(c.skills)} | {c.bio}" for c in CANDIDATES
    )
    prompt = (
        f"Select the top {top_k} most relevant candidates for this job description from the list below.\n\n"
        f"Job Description:\n{job_description}\n\n"
        f"Candidates:\n{candidates_text}\n\n"
        "Output MUST be valid JSON only with no markdown and no extra commentary. "
        "If the output is not valid JSON, it will be rejected.\n"
        "Return exactly this schema: {\"selected_candidates\": [\"Name1\", \"Name2\", \"Name3\"]}"
    )
    content = send_chat_prompt(prompt)
    result = parse_json_response(content)
    if DEBUG:
        print("LLM RAW RESPONSE:", content)
        print("PARSED RESULT:", result)
    if isinstance(result, dict):
        selected_names = result.get("selected_candidates", [])
        selected = [c for c in CANDIDATES if c.name in selected_names][:top_k]
        return selected if selected else CANDIDATES[:top_k]
    return CANDIDATES[:top_k]


def get_candidate_status(final_score: float) -> str:
    if final_score > 50:
        return "Enthusiastic"
    if 40 <= final_score <= 50:
        return "Neutral"
    return "Rejected"


def compute_skill_match(jd_skills: List[str], candidate_skills: List[str]) -> float:
    jd_normalized = {skill.strip().lower() for skill in jd_skills if skill.strip()}
    candidate_normalized = {
        skill.strip().lower() for skill in candidate_skills if skill.strip()
    }
    if not jd_normalized:
        return 0.0
    overlap = len(jd_normalized.intersection(candidate_normalized))
    return round((overlap / len(jd_normalized)) * 100, 1)


def to_optional_int(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        parsed = int(value)
        return max(0, min(100, parsed))
    except (TypeError, ValueError):
        return None


def evaluate_candidate(
    job_description: str, candidate: Candidate, jd_skills: List[str]
) -> Dict[str, object]:
    """Evaluate candidate with technical match, conversation, and interest in one LLM call."""
    tone = get_candidate_personality()

    def normalize_conversation(raw_conversation: str, candidate_name: str) -> str:
        text = (raw_conversation or "").replace("\\n", "\n").strip()
        if not text:
            return (
                "Recruiter: Can you walk me through your recent work relevant to this role?\n"
                "Candidate: I recently worked on a data pipeline using Airflow and SQL, optimizing ETL performance by 30%.\n"
                "Recruiter: That’s great. How do you ensure data quality in such pipelines?\n"
                "Candidate: I implement validation checks and monitoring dashboards to catch anomalies early."
            )

        candidate_name_lower = candidate_name.lower()
        candidate_first_name = candidate_name.split()[0].lower() if candidate_name.split() else ""

        def _is_candidate_speaker(label: str) -> bool:
            normalized = label.strip().lower()
            return bool(normalized) and (
                normalized == candidate_name_lower
                or normalized == candidate_first_name
                or normalized in {"candidate", "applicant"}
            )

        cleaned_lines: List[str] = []
        for line in text.splitlines():
            line = line.strip().strip('",')
            if not line:
                continue
            if re.match(r'^"?[a-z_]+_score"?\s*:', line, flags=re.IGNORECASE):
                continue
            if re.match(r'^"?[a-z_]+_reason"?\s*:', line, flags=re.IGNORECASE):
                continue
            speaker_match = re.match(r"^([^:]{1,40})\s*:\s*(.+)$", line)
            if speaker_match:
                speaker = speaker_match.group(1).strip().strip('"')
                message = speaker_match.group(2).strip()
                speaker_lower = speaker.lower()
                if speaker_lower in {"recruiter", "interviewer", "hiring manager"}:
                    cleaned_lines.append(f"Recruiter: {message}")
                    continue
                if _is_candidate_speaker(speaker):
                    cleaned_lines.append(f"Candidate: {message}")
                    continue

        if cleaned_lines:
            return "\n".join(cleaned_lines)

        # Last resort: recover inline speaker segments from a single broken line.
        candidate_name_pattern = re.escape(candidate_name)
        candidate_first_pattern = re.escape(candidate_first_name) if candidate_first_name else ""
        candidate_aliases = (
            f"{candidate_name_pattern}|{candidate_first_pattern}|Candidate|Applicant"
            if candidate_first_pattern
            else f"{candidate_name_pattern}|Candidate|Applicant"
        )
        speaker_pattern = re.compile(
            rf"(Recruiter|Interviewer|Hiring Manager|{candidate_aliases})\s*:\s*([^:]+?)(?=(?:Recruiter|Interviewer|Hiring Manager|{candidate_aliases})\s*:|$)",
            flags=re.IGNORECASE,
        )
        recovered: List[str] = []
        for role, message in speaker_pattern.findall(text):
            role_clean = role.strip().lower()
            if role_clean in {"recruiter", "interviewer", "hiring manager"}:
                recovered.append(f"Recruiter: {message.strip()}")
            elif _is_candidate_speaker(role):
                recovered.append(f"Candidate: {message.strip()}")
        if recovered:
            return "\n".join(recovered)

        return (
            "Recruiter: Can you walk me through your recent work relevant to this role?\n"
            "Candidate: I recently worked on a data pipeline using Airflow and SQL, optimizing ETL performance by 30%.\n"
            "Recruiter: That’s great. How do you ensure data quality in such pipelines?\n"
            "Candidate: I implement validation checks and monitoring dashboards to catch anomalies early."
        )

    prompt = f"""
You are a senior technical recruiter evaluating role fit from a job description and candidate profile.
Return ONLY valid JSON. Do not include markdown or commentary.

FORMAT:
{{
  "technical_match_score": 0-100,
  "technical_reason": "specific explanation with missing/present skills",
  "conversation": "Recruiter: ...\nCandidate: ...\nRecruiter: ...\nCandidate: ...",
  "interest_score": 0-100,
  "interest_reason": "specific explanation",
  "combined_reason": "single concise explanation combining technical and interest rationale"
}}

RULES:
- Output MUST be valid JSON only. No markdown, no commentary, no preface, no suffix text.
- Include ALL required fields exactly as listed in FORMAT. Missing keys are invalid output.
- If JSON is invalid it will be rejected.
- Scores MUST depend on this exact JD and this exact candidate.
- Penalize missing required skills explicitly.
- Use the full 0-100 range realistically. Avoid clustering around mid-values.
- Do NOT return default values like 50.
- Keep conversation professional and realistic with short lines.
- Use actual new lines for each speaker line, not the literal "\\n" sequence.
- Make reasons concrete and non-generic.
- Conversation MUST have exactly 6-8 turns.
- Conversation MUST alternate strictly: Recruiter then Candidate on every line.
- Every line MUST start with "Recruiter:" or "Candidate:".
- Recruiter questions MUST be role-specific and grounded in JD skills/responsibilities.
- Candidate responses MUST use the candidate's own listed skills/bio and reference at least one concrete detail (tool, project, metric, scenario, or challenge).
- Candidate should mention realistic past work and one real challenge they handled.
- Do NOT use generic filler phrases like "I am excited", "happy to discuss", or "my experience aligns well".
- Do NOT repeat questions or answers; each recruiter question must be different.
- Candidate should show moderate enthusiasm (professional, not overexcited).
- Candidate MUST ask exactly one follow-up question in their final turn.
- Output only the conversation lines in this format:
  Recruiter: ...
  Candidate: ...
  Recruiter: ...
  Candidate: ...

Job Description:
{job_description}

JD Skills:
{', '.join(jd_skills)}

Candidate:
Name: {candidate.name}
Skills: {', '.join(candidate.skills)}
Bio: {candidate.bio}
"""

    try:
        content = send_chat_prompt(prompt)
    except Exception as exc:
        logger.warning("Candidate evaluation failed. Using fallback path. Error: %s", exc)
        content = ""

    result = parse_json_response(content) if content else {}
    if DEBUG:
        print("LLM RAW RESPONSE:", content)
        print("PARSED RESULT:", result)
    if not isinstance(result, dict):
        result = {"parsing_error": True}

    return {
        "technical_match_score": to_optional_int(result.get("technical_match_score")),
        "technical_reason": str(result.get("technical_reason", "")).strip(),
        "conversation": normalize_conversation(
            str(result.get("conversation", "")).strip(), candidate.name
        ),
        "interest_score": to_optional_int(result.get("interest_score")),
        "interest_reason": str(result.get("interest_reason", "")).strip(),
        "combined_reason": str(result.get("combined_reason", "")).strip(),
        "personality": tone,
        "parsing_issue": bool(result.get("parsing_error")),
    }


def run_agent(job_description: str, extracted_skills: List[str]) -> List[Dict[str, object]]:
    """AI agent that retrieves and evaluates candidates with decision-making and early stopping."""
    # Retrieve top candidates
    selected_candidates = retrieve_candidates(job_description, top_k=3)
    results = []

    for candidate in selected_candidates:
        # Evaluate candidate (technical + conversation + interest in one LLM call)
        eval_result = evaluate_candidate(job_description, candidate, extracted_skills)
        deterministic_score = compute_skill_match(extracted_skills, candidate.skills)

        llm_tech_score = eval_result.get("technical_match_score")
        if llm_tech_score is None:
            tech_score = deterministic_score
            technical_reason = (
                "Fallback used due to parsing issue. Technical score based on skill overlap only."
            )
        else:
            tech_score = round((0.6 * deterministic_score) + (0.4 * llm_tech_score), 1)
            technical_reason = eval_result.get("technical_reason") or (
                "Hybrid technical score combines JD skill overlap and LLM role-fit judgment."
            )

        interest_score = eval_result.get("interest_score")
        if interest_score is None:
            interest_score = round(deterministic_score * 0.6, 1)
            interest_reason = (
                "Fallback interest estimated from deterministic technical overlap due to missing LLM score."
            )
        else:
            interest_reason = eval_result.get("interest_reason") or (
                "Interest derived from candidate conversation signals."
            )

        final_score = round((0.7 * tech_score) + (0.3 * interest_score), 1)
        confidence_score = round(
            (0.6 * (tech_score / 100)) + (0.4 * (interest_score / 100)),
            2,
        )
        status = get_candidate_status(final_score)
        overall_fit = (
            "strong alignment for outreach"
            if final_score > 50
            else "moderate alignment with follow-up recommended"
            if final_score >= 40
            else "limited alignment for current role priorities"
        )
        combined_reason = eval_result.get("combined_reason") or (
            f"Technical match: {tech_score}/100 based on skill overlap. "
            f"Interest: {interest_score}/100 from conversation. "
            f"Overall fit: {overall_fit}."
        )

        results.append(
            {
                "name": candidate.name,
                "skills": ", ".join(candidate.skills),
                "bio": candidate.bio,
                "technical_match_score": tech_score,
                "interest_score": interest_score,
                "final_score": final_score,
                "confidence_score": confidence_score,
                "status": status,
                "conversation": eval_result["conversation"],
                "technical_reason": technical_reason,
                "interest_reason": interest_reason,
                "personality": eval_result.get("personality", "neutral"),
                "combined_reason": combined_reason,
            }
        )
        time.sleep(1)  # Rate limiting

    results.sort(key=lambda r: r["final_score"], reverse=True)
    return results

def generate_outreach_email(job_description: str, candidate: Candidate, skills: List[str]) -> str:
    prompt = (
        "Write a professional but conversational outreach email to this candidate for the role described below. "
        "Mention the key skills extracted from the JD and the candidate's experience. "
        "Keep it personalized, friendly, and no more than 5 short paragraphs.\n\n"
        f"Candidate:\nName: {candidate.name}\nSkills: {', '.join(candidate.skills)}\nBio: {candidate.bio}\n\n"
        f"Job Description Skills:\n{', '.join(skills)}\n\n"
        f"Job Description:\n{job_description}"
    )
    return send_chat_prompt(prompt)


def render_candidate_table(rows: List[Dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(
        [
            {
                "Name": row["name"],
                "Match Score": row["technical_match_score"],
                "Interest Score": row["interest_score"],
                "Status": row["status"].capitalize(),
                "Final Score": row["final_score"],
                "Skills": row["skills"],
                "Bio": row["bio"],
            }
            for row in rows
        ]
    )
    return df


def main() -> None:
    global USE_RANDOMNESS
    st.set_page_config(page_title="AI-Powered Talent Scouting Agent", layout="wide")
    st.markdown(
        """
        <div style="
            background: #f8fafc;
            padding: 24px 20px;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(15, 23, 42, 0.08);
            margin-bottom: 6px;
        ">
            <h1 style="
                text-align: center;
                font-size: 3rem;
                font-weight: 700;
                color: #2F5DA9;
                margin: 0;
                line-height: 1.2;
            ">
                AI Powered Talent Scouting Agent
            </h1>
            <p style="
                text-align: center;
                font-size: 1rem;
                font-style: italic;
                color: #6b7280;
                margin: 8px 0 0 0;
            ">
                By Deccan AI
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("## 🤖 AI Talent Dashboard")
    st.caption("Instant candidate discovery and engagement")

    st.sidebar.markdown("## 🎯 Recruiter Control Panel")
    use_randomness = st.sidebar.checkbox(
        "Enable realistic candidate behavior", value=USE_RANDOMNESS
    )
    # Production mode remains deterministic for stable ranking and scoring.
    USE_RANDOMNESS = False

    jd_input = st.sidebar.text_area("Paste the Job Description here", height=260)
    submitted = st.sidebar.button("Run Agent", use_container_width=True)

    if submitted:
        if not jd_input.strip():
            st.error("Please paste a job description before analyzing.")
            return

        with st.spinner("🔍 Analyzing job requirements..."):
            try:
                extracted_skills = extract_key_skills(jd_input)
                st.success("JD parsed successfully.")
                st.sidebar.success("JD parsed successfully.")
                st.sidebar.markdown("**Extracted Skills**")
                st.sidebar.caption(", ".join(extracted_skills))
            except Exception as exc:
                st.error(f"Failed to parse JD: {exc}")
                return

        with st.spinner("🧠 Evaluating candidate fit..."):
            try:
                agent_results = run_agent(jd_input, extracted_skills)
                st.success("Candidate evaluation completed.")
            except Exception as exc:
                st.error(f"Agent failed: {exc}")
                return

        top_candidate = sorted(agent_results, key=lambda x: x["final_score"], reverse=True)[0]
        selected_candidate = next(c for c in CANDIDATES if c.name == top_candidate["name"])

        with st.spinner("💬 Simulating recruiter interaction..."):
            conversation = top_candidate["conversation"].replace("\\n", "\n")

        with st.spinner("📧 Preparing outreach message..."):
            try:
                outreach_email = generate_outreach_email(jd_input, selected_candidate, extracted_skills)
                st.success("Outreach email generated.")
            except Exception as exc:
                st.error(f"Failed to generate outreach email: {exc}")
                return

        st.markdown("---")
        st.markdown("## 📄 Job Summary")
        st.caption("This role focuses on: " + ", ".join(extracted_skills))
        st.markdown(
            f"<div style='background:#f8fafc;padding:12px 14px;border-radius:10px;border:1px solid #e5e7eb;'>"
            f"<strong>Extracted Skills:</strong> {', '.join(extracted_skills)}</div>",
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.success(
            f"🎯 Recommended Hire: {top_candidate['name']} (Score: {top_candidate['final_score']}/100)"
        )
        top_skill_overlap = [
            skill for skill in extracted_skills if skill.lower() in top_candidate["skills"].lower()
        ][:2]
        strengths_text = ", ".join(top_skill_overlap) if top_skill_overlap else "strong role-relevant foundation"
        st.caption(f"💡 Insight: This candidate stands out due to {strengths_text}.")
        st.info(top_candidate.get("combined_reason", top_candidate["technical_reason"]))

        confidence_pct = int(top_candidate["confidence_score"] * 100)
        if confidence_pct > 75:
            st.success(f"Confidence Level: High Confidence ({confidence_pct}%)")
        elif confidence_pct >= 50:
            st.warning(f"Confidence Level: Moderate Confidence ({confidence_pct}%)")
        else:
            st.error(f"Confidence Level: Low Confidence ({confidence_pct}%)")

        st.markdown("## 🏆 Top Candidate")
        st.markdown(
            "<div style='border:1px solid #dbeafe;background:#f8fbff;padding:16px;border-radius:12px;'>"
            "<p style='margin:0;color:#1f3a68;font-weight:600;'>Top candidate spotlight</p></div>",
            unsafe_allow_html=True,
        )
        st.markdown(f"### {top_candidate['name']}")
        st.write(top_candidate["bio"])
        status = top_candidate["status"]
        if status == "Rejected":
            st.error("Status: Rejected")
        elif status == "Neutral":
            st.warning("Status: Neutral")
        else:
            st.success("Status: Enthusiastic")

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Final Score", f"{top_candidate['final_score']}/100")
        with m2:
            st.metric("Match Score", f"{top_candidate['technical_match_score']}/100")
        with m3:
            st.metric("Interest Score", f"{top_candidate['interest_score']}/100")
        with m4:
            st.metric("Confidence", f"{int(top_candidate['confidence_score'] * 100)}%")
        st.caption("Match Score reflects skill alignment with the job description.")
        st.caption("Interest Score reflects engagement level from simulated conversation.")

        st.markdown("---")
        st.markdown("## 📊 Score Breakdown")
        st.caption("Technical Match")
        st.progress(int(top_candidate["technical_match_score"]))
        st.caption("Interest")
        st.progress(int(top_candidate["interest_score"]))

        st.markdown("---")
        st.markdown("## 🧠 Why this candidate ranked highest")
        st.info(
            f"Combined reasoning: {top_candidate.get('combined_reason', top_candidate['technical_reason'])}"
        )
        st.write(f"Technical fit: {top_candidate['technical_reason']}")
        st.write(f"Interest from conversation: {top_candidate['interest_reason']}")

        st.markdown("---")
        st.markdown("## 💬 Real Candidate Interaction")
        st.caption("A recruiter-style conversation generated from candidate fit signals.")

        def build_chat_turns(raw_conversation: str) -> List[Dict[str, str]]:
            lines = [
                line.strip().rstrip("\\").strip('",')
                for line in (raw_conversation or "").splitlines()
                if line.strip()
            ]
            turns: List[Dict[str, str]] = []
            last_role = ""

            for line in lines:
                speaker_match = re.match(r"^([^:]{1,40})\s*:\s*(.+)$", line)
                if speaker_match:
                    speaker = speaker_match.group(1).strip().lower()
                    message = speaker_match.group(2).strip()
                    if speaker in {"recruiter", "interviewer", "hiring manager"}:
                        role = "assistant"
                    elif speaker in {"candidate", "applicant"}:
                        role = "user"
                    else:
                        candidate_first = top_candidate["name"].split()[0].lower()
                        role = "user" if speaker in {candidate_first, top_candidate["name"].lower()} else ""
                    if role and message:
                        turns.append({"role": role, "message": message})
                        last_role = role
                        continue

                role = "assistant" if last_role != "assistant" else "user"
                turns.append({"role": role, "message": line})
                last_role = role

            if not turns:
                turns = [
                    {"role": "assistant", "message": "Thanks for your time today."},
                    {"role": "user", "message": "Happy to discuss the role."},
                ]

            if turns[0]["role"] != "assistant":
                turns.insert(
                    0,
                    {
                        "role": "assistant",
                        "message": "Thanks for joining. I would like to quickly assess your fit for this role.",
                    },
                )

            while len(turns) < 6:
                next_role = "assistant" if turns[-1]["role"] == "user" else "user"
                filler = (
                    "Can you share how your recent work aligns with this position?"
                    if next_role == "assistant"
                    else "My recent work maps well to the role and I am interested in moving forward."
                )
                turns.append({"role": next_role, "message": filler})

            return turns[:8]

        chat_turns = build_chat_turns(conversation)
        for turn in chat_turns:
            with st.chat_message(turn["role"]):
                st.markdown(turn["message"])

        st.markdown("---")
        st.markdown("## 📧 Ready-to-Send Outreach")
        st.caption("Generated automatically based on candidate fit")
        st.text_area("Email output", outreach_email, height=300)

        st.markdown("---")
        st.markdown("## 👥 Other Considered Candidates")
        for row in agent_results[:3]:
            with st.container():
                c_name, c_score = st.columns([3, 1])
                with c_name:
                    st.markdown(f"### {row['name']}")
                with c_score:
                    st.metric("Final Score", f"{row['final_score']}/100")
                if row["status"] == "Rejected":
                    st.error("Status: Rejected")
                elif row["status"] == "Neutral":
                    st.warning("Status: Neutral")
                else:
                    st.success("Status: Enthusiastic")
                st.caption("Technical Match")
                st.progress(int(row["technical_match_score"]))
                st.caption(f"Confidence: {int(row['confidence_score'] * 100)}%")
            st.divider()

        st.markdown("---")
        st.markdown("## ❌ Why other candidates were not selected")
        for row in agent_results[1:]:
            candidate_skill_set = {
                skill.strip().lower() for skill in row["skills"].split(",") if skill.strip()
            }
            missing_skills = [
                skill for skill in extracted_skills if skill.strip().lower() not in candidate_skill_set
            ]
            st.markdown(f"### {row['name']}")
            reasons: List[str] = []
            if missing_skills:
                reasons.append(f"Missing key JD skills: {', '.join(missing_skills[:3])}.")
            if row["interest_score"] < top_candidate["interest_score"]:
                reasons.append(
                    f"Lower interest signal ({row['interest_score']}/100) than selected candidate ({top_candidate['interest_score']}/100)."
                )
            if row["final_score"] < top_candidate["final_score"]:
                reasons.append(
                    f"Weaker overall alignment score ({row['final_score']}/100 vs {top_candidate['final_score']}/100)."
                )
            if not reasons:
                reasons.append("Overall fit was lower compared with the selected top candidate.")
            for reason in reasons[:3]:
                st.markdown(f"- {reason}")
            st.divider()

        st.markdown("---")
        st.markdown("## 🚀 Why This Matters")
        st.caption("This agent reduces recruiter effort and accelerates hiring decisions.")
        st.markdown("- ⏱️ Saves recruiters hours of manual screening")
        st.markdown("- 🎯 Identifies best-fit candidates instantly")
        st.markdown("- 💬 Validates real interest before outreach")
        st.markdown("- 📧 Automates personalized communication")


if __name__ == "__main__":
    main()


