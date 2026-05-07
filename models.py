# models.py
from dataclasses import dataclass, field, asdict
from typing import List, Dict

@dataclass
class Candidate:
    name: str = "Unknown"
    email: str = "N/A"
    phone: str = "N/A"
    skills: List[str] = field(default_factory=list)
    education: str = "N/A"
    education_level: str = "N/A"
    experience_years: float = 0.0
    ats_score: int = 0
    match_percentage: int = 0
    matching_skills: List[str] = field(default_factory=list)
    missing_skills: List[str] = field(default_factory=list)
    summary: str = "N/A"
    certifications: List[str] = field(default_factory=list)
    skill_categories: Dict[str, List[str]] = field(default_factory=dict)
    raw_text: str = ""
    filename: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "Candidate":
        valid_fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in valid_fields})