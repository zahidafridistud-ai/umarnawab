"""
ATS Resume Checker
------------------
Upload a resume (PDF / DOCX / TXT) and get:
  * an ATS score out of 100 (with a category breakdown)
  * strengths, problems, and concrete improvements
  * missing keywords (optionally matched against a job description)
  * rewritten bullet-point examples

UI: Streamlit  |  AI: Google Gemini Flash (google-genai SDK)
"""

import io
import json
import os
import random
import re
import time

import streamlit as st

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DEFAULT_MODEL = "gemini-3.5-flash"  # change in the sidebar or via GEMINI_MODEL
# Tried in order if the main model is overloaded (503) or rate limited (429)
FALLBACK_MODELS = ["gemini-2.5-flash", "gemini-3-flash-preview"]
RETRIES_PER_MODEL = 3               # attempts per model before moving on
BACKOFF_SECONDS = 2                 # waits 2s, 4s, ... between attempts
MAX_RESUME_CHARS = 20_000           # keeps the prompt small & fast
MAX_JD_CHARS = 8_000
MAX_FILE_MB = 5

# Category -> weight. Weights add up to 100, so the overall score is
# computed in code (consistent & explainable) rather than guessed by the AI.
CATEGORY_WEIGHTS = {
    "keywords_skills": 25,
    "experience_impact": 25,
    "formatting_structure": 20,
    "content_clarity": 15,
    "contact_sections": 15,
}
CATEGORY_LABELS = {
    "keywords_skills": "Keywords & Skills",
    "experience_impact": "Experience & Impact",
    "formatting_structure": "Formatting & Structure",
    "content_clarity": "Content & Clarity",
    "contact_sections": "Contact & Sections",
}

SECTION_PATTERNS = {
    "Experience": r"\b(work\s+experience|professional\s+experience|experience|employment(\s+history)?)\b",
    "Education": r"\beducation\b|\bacademic\b",
    "Skills": r"\bskills?\b|\btechnical\s+skills\b|\bcompetenc",
    "Summary": r"\b(summary|profile|objective|about\s+me)\b",
    "Projects": r"\bprojects?\b",
    "Certifications": r"\bcertifications?\b|\blicen[cs]es?\b",
}


# --------------------------------------------------------------------------- #
# File reading
# --------------------------------------------------------------------------- #
def extract_text(file_bytes: bytes, filename: str) -> str:
    """Return plain text from a PDF, DOCX or TXT file."""
    name = filename.lower()

    if name.endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(file_bytes))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                raise ValueError("This PDF is password-protected.")
        pages = [(page.extract_text() or "") for page in reader.pages]
        return "\n".join(pages).strip()

    if name.endswith(".docx"):
        from docx import Document

        doc = Document(io.BytesIO(file_bytes))
        parts = [p.text for p in doc.paragraphs if p.text.strip()]
        # Many resumes keep content inside tables (two-column layouts)
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text.strip():
                        parts.append(cell.text.strip())
        return "\n".join(parts).strip()

    if name.endswith(".txt"):
        return file_bytes.decode("utf-8", errors="ignore").strip()

    raise ValueError("Unsupported file type. Please upload a PDF, DOCX or TXT file.")


# --------------------------------------------------------------------------- #
# Quick rule-based checks (no AI needed)
# --------------------------------------------------------------------------- #
def basic_checks(text: str) -> dict:
    """Cheap, deterministic facts about the resume."""
    email = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
    phone = re.search(r"(\+?\d[\d\s().-]{8,}\d)", text)
    linkedin = re.search(r"linkedin\.com/", text, re.I)
    sections = {
        name: bool(re.search(pattern, text, re.I))
        for name, pattern in SECTION_PATTERNS.items()
    }
    words = len(re.findall(r"\w+", text))
    return {
        "email": bool(email),
        "phone": bool(phone),
        "linkedin": bool(linkedin),
        "sections": sections,
        "word_count": words,
    }


# --------------------------------------------------------------------------- #
# Gemini
# --------------------------------------------------------------------------- #
SYSTEM_INSTRUCTION = (
    "You are a strict, experienced ATS (Applicant Tracking System) specialist and "
    "senior technical recruiter. You evaluate resumes honestly - do not inflate "
    "scores. Respond with valid JSON only."
)


def build_prompt(resume_text: str, job_description: str, checks: dict) -> str:
    jd_block = (
        f"JOB DESCRIPTION (score keyword match against this):\n{job_description}\n"
        if job_description
        else "No job description was provided. Judge against general industry "
        "standards for the role this resume appears to target.\n"
    )
    return f"""Evaluate the resume below for ATS compatibility and quality.

{jd_block}
FACTS DETECTED BY CODE (trust these):
{json.dumps(checks)}

RESUME TEXT:
\"\"\"
{resume_text}
\"\"\"

Score each category from 0 to 100 (integers):
- keywords_skills: relevant hard skills, tools, industry keywords{" and match to the job description" if job_description else ""}
- experience_impact: quantified achievements, action verbs, results (not just duties)
- formatting_structure: standard section headings, consistent dates, parsable layout, sensible length
- content_clarity: concise, no filler, no typos, strong summary, readable bullets
- contact_sections: contact details present, and Experience / Education / Skills sections present

Return ONLY a JSON object with exactly this shape:
{{
  "target_role": "short guess of the role this resume targets",
  "scores": {{
    "keywords_skills": 0,
    "experience_impact": 0,
    "formatting_structure": 0,
    "content_clarity": 0,
    "contact_sections": 0
  }},
  "summary": "2-3 sentence overall verdict",
  "strengths": ["..."],
  "issues": [
    {{"severity": "high|medium|low", "problem": "...", "fix": "..."}}
  ],
  "missing_keywords": ["..."],
  "improved_bullets": [
    {{"original": "a weak bullet copied from the resume", "improved": "a stronger rewrite"}}
  ],
  "top_priorities": ["the 3-5 most valuable changes, in order"]
}}

Rules: base everything on the resume text only; never invent employers, degrees or numbers
that are not in the resume (if a metric is unknown, use a placeholder like [X%]).
Give 3-6 strengths, 4-8 issues, up to 15 missing keywords and 3-5 improved bullets."""


def get_api_key(sidebar_key: str = "") -> str:
    """Priority: sidebar input > Streamlit secrets > environment variable."""
    if sidebar_key.strip():
        return sidebar_key.strip()
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return str(st.secrets["GEMINI_API_KEY"]).strip()
    except Exception:
        pass  # no secrets file locally - that's fine
    return os.environ.get("GEMINI_API_KEY", "").strip()


def parse_json(raw: str) -> dict:
    """Parse model output into a dict, tolerating ```json fences / extra text."""
    raw = (raw or "").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            return json.loads(raw[start : end + 1])
        raise


def is_transient(err: Exception) -> bool:
    """True for temporary Google-side problems worth retrying (503, 500, 429...)."""
    low = str(err).lower()
    markers = (
        "503", "500", "504", "429", "unavailable", "overloaded", "high demand",
        "deadline", "resource_exhausted", "internal error", "timed out", "timeout",
    )
    return any(m in low for m in markers)


def call_gemini(prompt: str, api_key: str, model: str, on_status=None, sleep=time.sleep) -> dict:
    """
    Call Gemini and return parsed JSON.

    * Temporary errors (503 'high demand', 429, timeouts) are retried with
      exponential backoff, then the next model in FALLBACK_MODELS is tried.
    * Bad JSON is retried once per attempt.
    * Permanent errors (bad key, etc.) are raised immediately.
    """
    from google import genai
    from google.genai import types

    def say(msg: str) -> None:
        if on_status:
            on_status(msg)

    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        response_mime_type="application/json",
        temperature=0.2,
    )

    models = [model] + [m for m in FALLBACK_MODELS if m != model]
    last_err = None

    for model_index, name in enumerate(models):
        for attempt in range(1, RETRIES_PER_MODEL + 1):
            try:
                response = client.models.generate_content(
                    model=name, contents=prompt, config=config
                )
                try:
                    return parse_json(response.text)
                except (json.JSONDecodeError, ValueError, TypeError) as err:
                    last_err = ValueError(f"The AI returned an unreadable response ({err}).")
                    say("The AI response was malformed, retrying...")
                    continue
            except Exception as err:  # SDK raises various error classes
                last_err = err
                low = str(err).lower()
                if "404" in low or "not found" in low:
                    # Unknown model: if it's the user's main model, tell them; else skip it
                    if model_index == 0:
                        raise
                    break
                if not is_transient(err):
                    raise  # bad API key, invalid request, etc.
                if attempt < RETRIES_PER_MODEL:
                    wait = BACKOFF_SECONDS * (2 ** (attempt - 1)) + random.uniform(0, 1)
                    say(f"Gemini is busy ({name}). Retrying in {wait:.0f}s "
                        f"(attempt {attempt + 1}/{RETRIES_PER_MODEL})...")
                    sleep(wait)
        if model_index + 1 < len(models):
            say(f"{name} is overloaded. Switching to {models[model_index + 1]}...")

    raise last_err if last_err else RuntimeError("Gemini request failed.")


# --------------------------------------------------------------------------- #
# Scoring / cleaning
# --------------------------------------------------------------------------- #
def _clamp(value, low=0, high=100) -> int:
    try:
        return max(low, min(high, int(round(float(value)))))
    except (TypeError, ValueError):
        return 0


def _as_list(value) -> list:
    return value if isinstance(value, list) else []


def compute_overall(scores: dict) -> int:
    """Weighted average of the category scores -> 0-100."""
    total = sum(
        _clamp(scores.get(key, 0)) * weight for key, weight in CATEGORY_WEIGHTS.items()
    )
    return int(round(total / sum(CATEGORY_WEIGHTS.values())))


def normalise_result(data: dict) -> dict:
    """Make sure the AI output has every field we display, with safe types."""
    data = data if isinstance(data, dict) else {}
    scores = data.get("scores") if isinstance(data.get("scores"), dict) else {}
    clean_scores = {key: _clamp(scores.get(key, 0)) for key in CATEGORY_WEIGHTS}

    issues = []
    for item in _as_list(data.get("issues")):
        if isinstance(item, dict):
            sev = str(item.get("severity", "medium")).lower()
            issues.append(
                {
                    "severity": sev if sev in ("high", "medium", "low") else "medium",
                    "problem": str(item.get("problem", "")).strip(),
                    "fix": str(item.get("fix", "")).strip(),
                }
            )
    order = {"high": 0, "medium": 1, "low": 2}
    issues.sort(key=lambda i: order[i["severity"]])

    bullets = []
    for item in _as_list(data.get("improved_bullets")):
        if isinstance(item, dict) and item.get("improved"):
            bullets.append(
                {
                    "original": str(item.get("original", "")).strip(),
                    "improved": str(item.get("improved", "")).strip(),
                }
            )

    return {
        "target_role": str(data.get("target_role", "")).strip(),
        "scores": clean_scores,
        "overall": compute_overall(clean_scores),
        "summary": str(data.get("summary", "")).strip(),
        "strengths": [str(s) for s in _as_list(data.get("strengths"))],
        "issues": issues,
        "missing_keywords": [str(k) for k in _as_list(data.get("missing_keywords"))],
        "improved_bullets": bullets,
        "top_priorities": [str(p) for p in _as_list(data.get("top_priorities"))],
    }


def score_label(score: int) -> tuple:
    if score >= 80:
        return "Excellent", "🟢"
    if score >= 65:
        return "Good", "🟡"
    if score >= 50:
        return "Needs work", "🟠"
    return "Poor", "🔴"


def friendly_error(err: Exception) -> str:
    """Turn raw API exceptions into messages a user can act on."""
    msg = str(err)
    low = msg.lower()
    if "api key" in low or "api_key" in low or "401" in low or "403" in low or "permission" in low:
        return "Gemini rejected the API key. Check that it is correct and active."
    if "429" in low or "quota" in low or "resource_exhausted" in low or "rate limit" in low:
        return "Gemini rate limit / quota reached. Wait a minute and try again."
    if "503" in low or "unavailable" in low or "high demand" in low or "overloaded" in low:
        return ("Google's Gemini servers are overloaded right now (the app already retried and "
                "tried backup models). Please wait a minute and click Analyze again.")
    if "404" in low or "not found" in low:
        return "That model name was not found. Change the model in the sidebar (e.g. gemini-2.5-flash)."
    return f"Something went wrong: {msg}"


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def render_results(result: dict, checks: dict) -> None:
    overall = result["overall"]
    label, icon = score_label(overall)

    st.header("Your ATS Score")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Overall ATS Score", f"{overall} / 100")
        st.markdown(f"### {icon} {label}")
        if result["target_role"]:
            st.caption(f"Detected target role: **{result['target_role']}**")
    with col2:
        st.progress(overall / 100)
        if result["summary"]:
            st.write(result["summary"])

    st.subheader("Score breakdown")
    cols = st.columns(len(CATEGORY_WEIGHTS))
    for col, (key, weight) in zip(cols, CATEGORY_WEIGHTS.items()):
        with col:
            st.metric(CATEGORY_LABELS[key], f"{result['scores'][key]}")
            st.caption(f"Weight: {weight}%")
            st.progress(result["scores"][key] / 100)

    if result["top_priorities"]:
        st.subheader("🎯 Top priorities")
        for i, item in enumerate(result["top_priorities"], 1):
            st.markdown(f"{i}. {item}")

    left, right = st.columns(2)
    with left:
        st.subheader("✅ Strengths")
        for s in result["strengths"] or ["No strengths reported."]:
            st.markdown(f"- {s}")
    with right:
        st.subheader("🔑 Missing keywords")
        if result["missing_keywords"]:
            st.markdown(" ".join(f"`{k}`" for k in result["missing_keywords"]))
        else:
            st.write("No major keyword gaps found.")

    st.subheader("🛠️ Improvements")
    icons = {"high": "🔴 High", "medium": "🟠 Medium", "low": "🟡 Low"}
    if result["issues"]:
        for issue in result["issues"]:
            with st.expander(f"{icons[issue['severity']]} - {issue['problem']}"):
                st.markdown(f"**How to fix:** {issue['fix']}")
    else:
        st.write("No issues reported.")

    if result["improved_bullets"]:
        st.subheader("✍️ Bullet point rewrites")
        st.caption("Replace any [X] placeholders with your real numbers.")
        for b in result["improved_bullets"]:
            if b["original"]:
                st.markdown(f"**Before:** {b['original']}")
            st.markdown(f"**After:** {b['improved']}")
            st.divider()

    with st.expander("📋 Automatic checks (no AI)"):
        st.write(f"Word count: **{checks['word_count']}**")
        st.write(
            f"Email: {'✅' if checks['email'] else '❌'}  |  "
            f"Phone: {'✅' if checks['phone'] else '❌'}  |  "
            f"LinkedIn: {'✅' if checks['linkedin'] else '❌'}"
        )
        st.write(
            "Sections found: "
            + ", ".join(f"{n} {'✅' if ok else '❌'}" for n, ok in checks["sections"].items())
        )


def main() -> None:
    st.set_page_config(page_title="ATS Resume Checker", page_icon="📄", layout="wide")
    st.title("📄 ATS Resume Checker")
    st.write("Upload your resume to get an ATS score and clear steps to improve it.")

    with st.sidebar:
        st.header("Settings")
        key_input = st.text_input(
            "Gemini API key",
            type="password",
            help="Leave empty if the key is set in Streamlit secrets or GEMINI_API_KEY.",
        )
        model = st.text_input(
            "Gemini model",
            value=os.environ.get("GEMINI_MODEL", DEFAULT_MODEL),
            help="Any Gemini Flash model name, e.g. gemini-3.5-flash or gemini-2.5-flash.",
        )
        st.caption("Get a free key at https://aistudio.google.com/apikey")
        st.info("Your resume is sent to Google's Gemini API for analysis and is not stored by this app.")

    uploaded = st.file_uploader("Upload resume", type=["pdf", "docx", "txt"])
    job_description = st.text_area(
        "Job description (optional, for a tailored keyword match)",
        height=150,
        placeholder="Paste the job posting here...",
    )

    if st.button("Analyze resume", type="primary", disabled=uploaded is None):
        api_key = get_api_key(key_input)
        if not api_key:
            st.error("Please add your Gemini API key in the sidebar (or in Streamlit secrets).")
            st.stop()

        file_bytes = uploaded.getvalue()
        if len(file_bytes) > MAX_FILE_MB * 1024 * 1024:
            st.error(f"File is larger than {MAX_FILE_MB} MB.")
            st.stop()

        try:
            with st.spinner("Reading your resume..."):
                text = extract_text(file_bytes, uploaded.name)
        except Exception as err:
            st.error(f"Could not read the file: {err}")
            st.stop()

        if len(text) < 100:
            st.error(
                "Almost no text could be extracted. If your PDF is a scanned image or "
                "built from images, an ATS can't read it either - export a text-based "
                "PDF or DOCX and try again."
            )
            st.stop()

        checks = basic_checks(text)
        prompt = build_prompt(text[:MAX_RESUME_CHARS], job_description.strip()[:MAX_JD_CHARS], checks)

        try:
            status = st.empty()
            with st.spinner("Analyzing with Gemini..."):
                raw = call_gemini(
                    prompt, api_key, model.strip() or DEFAULT_MODEL,
                    on_status=lambda msg: status.info(msg),
                )
            status.empty()
            result = normalise_result(raw)
        except Exception as err:
            st.error(friendly_error(err))
            st.stop()

        render_results(result, checks)


if __name__ == "__main__":
    main()
