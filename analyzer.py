# analyzer.py
import re, time, logging, json
from typing import List, Dict
from groq import Groq
from config import CFG, SKILL_CATEGORIES, ALL_SKILLS
from models import Candidate

logger = logging.getLogger(__name__)

class ResumeAnalyzer:
    PHONE_PATTERNS = [
        r"\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
        r"\b\d{10}\b",
    ]
    EMAIL_PATTERN = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    YEAR_PATTERN = r"\b(19[89]\d|20[0-2]\d)\b"

    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)

    def analyze(self, text: str, filename: str = "", jd: str = "") -> Candidate:
        """Full analysis pipeline for a single resume."""
        raw_text_snippet = text[:CFG.max_resume_chars]
        jd_section = f"\n\nTarget Job Description:\n{jd[:3000]}" if jd else ""

        prompt = f"""You are an expert ATS (Applicant Tracking System). Analyze this resume carefully and return ONLY valid JSON.{jd_section}

Resume:
{raw_text_snippet}

Return this exact JSON structure (no markdown, no extra text):
{{
    "name": "Candidate's full name or null",
    "email": "email address or null",
    "phone": "10-digit phone number digits only or null",
    "skills": ["list", "of", "skills"],
    "education": "Highest degree - University name",
    "education_level": "High School | Associate | Bachelors | Masters | PhD | MBA",
    "experience_years": 0.0,
    "match_percentage": 0,
    "matching_skills": ["skills matching the JD"],
    "missing_skills": ["important JD skills not found"],
    "summary": "One powerful sentence pitch for this candidate",
    "certifications": ["cert1", "cert2"]
}}

Rules:
- experience_years: Sum ALL work experience precisely. Do NOT round up.
- match_percentage: Only set if a JD was provided. Be strict. 100% is extremely rare. Base on technical skill overlap, seniority, domain relevance.
- skills: Include ALL technical AND soft skills visible in the resume.
- If a field is missing, use null (not "N/A").

"""
        for attempt in range(CFG.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=CFG.groq_model,
                    temperature=CFG.temperature,
                    max_tokens=CFG.max_tokens,
                    response_format={"type": "json_object"},
                )
                raw = json.loads(resp.choices[0].message.content)
                return self._build_candidate(raw, text, filename, bool(jd))
            except json.JSONDecodeError:
                logger.warning(f"JSON decode failed on attempt {attempt + 1}")
            except Exception as e:
                logger.error(f"LLM call failed (attempt {attempt + 1}): {e}")
                if attempt < CFG.max_retries - 1:
                    time.sleep(CFG.retry_delay * (attempt + 1))

        return self._fallback_candidate(text, filename)

    def _build_candidate(self, raw: Dict, text: str, filename: str, has_jd: bool) -> Candidate:
        skills = [str(s).strip() for s in (raw.get("skills") or []) if s]
        categorized = self._categorize_skills(skills)

        phone = self._clean_phone(str(raw.get("phone") or "")) or self._extract_phone(text)
        email = raw.get("email") or self._extract_first(text, self.EMAIL_PATTERN)
        name = raw.get("name") or filename or "Unknown"
        exp = float(raw.get("experience_years") or 0) or self._estimate_experience(text)

        base_score = self._compute_base_score(name, email, phone, skills, exp, raw.get("education"))
        match_pct = int(raw.get("match_percentage") or 0) if has_jd else 0
        ats_score = int(base_score * 0.35 + match_pct * 0.65) if has_jd and match_pct else base_score

        return Candidate(
            name=name,
            email=email or "N/A",
            phone=phone or "N/A",
            skills=skills,
            education=raw.get("education") or "N/A",
            education_level=raw.get("education_level") or "N/A",
            experience_years=round(exp, 1),
            ats_score=min(100, ats_score),
            match_percentage=match_pct,
            matching_skills=[str(s) for s in (raw.get("matching_skills") or [])],
            missing_skills=[str(s) for s in (raw.get("missing_skills") or [])],
            summary=raw.get("summary") or "N/A",
            certifications=[str(c) for c in (raw.get("certifications") or [])],
            skill_categories=categorized,
            filename=filename,
        )

    def _categorize_skills(self, skills: List[str]) -> Dict[str, List[str]]:
        lower_skills = {s.lower() for s in skills}
        result = {}
        for category, keywords in SKILL_CATEGORIES.items():
            matched = [k.title() for k in keywords if k in lower_skills]
            if matched:
                result[category] = matched
        return result

    def _clean_phone(self, raw: str) -> str:
        digits = re.sub(r"\D", "", raw)
        return digits[-10:] if len(digits) >= 10 else ""

    def _extract_phone(self, text: str) -> str:
        for pattern in self.PHONE_PATTERNS:
            matches = re.findall(pattern, text[:2000])
            if matches:
                cleaned = self._clean_phone(matches[0])
                if cleaned:
                    return cleaned
        return ""

    def _extract_first(self, text: str, pattern: str) -> str:
        m = re.findall(pattern, text[:2000])
        if m:
            return m[0] if isinstance(m[0], str) else "".join(m[0])
        return ""

    def _estimate_experience(self, text: str) -> float:
        years = sorted({int(y) for y in re.findall(self.YEAR_PATTERN, text)})
        if len(years) >= 2:
            span = years[-1] - years[0]
            if 1 <= span <= 45:
                return float(span)
        return 0.0

    def _compute_base_score(self, name, email, phone, skills, exp, education) -> int:
        score = 0
        if name and name not in {"Unknown", "None"}: score += 10
        if email and "@" in email: score += 10
        if phone and len(phone) >= 10: score += 5
        score += min(len(skills) * 2, 35)
        score += min(int(exp) * 4, 20)
        if education and education != "N/A": score += 15
        if exp >= 2: score += 5
        return min(100, score)

    def _fallback_candidate(self, text: str, filename: str) -> Candidate:
        skills = self._extract_skills_fallback(text)
        return Candidate(
            name=filename or "Unknown",
            email=self._extract_first(text, self.EMAIL_PATTERN) or "N/A",
            phone=self._extract_phone(text) or "N/A",
            skills=skills,
            experience_years=self._estimate_experience(text),
            ats_score=15,
            filename=filename,
            skill_categories=self._categorize_skills(skills),
        )

    def _extract_skills_fallback(self, text: str) -> List[str]:
        text_lower = text.lower()
        return [s.title() for s in ALL_SKILLS if re.search(r"\b" + re.escape(s) + r"\b", text_lower)]