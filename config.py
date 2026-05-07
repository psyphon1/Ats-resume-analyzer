# config.py
"""
Configuration module for ATS Resume Analyzer
Centralized settings for the entire application
"""

from dataclasses import dataclass
from typing import Dict, List

@dataclass
class Config:
    """Application configuration settings."""
    # LLM Configuration
    groq_model: str = "llama-3.1-8b-instant"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Processing Configuration
    chunk_size: int = 4000
    chunk_overlap: int = 200
    max_resume_chars: int = 8000
    max_tokens: int = 1500
    temperature: float = 0.1
    max_workers: int = 4
    retry_delay: float = 2.0
    max_retries: int = 3
    batch_size: int = 4

CFG = Config()

GOOGLE_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.readonly",
]

SKILL_CATEGORIES: Dict[str, List[str]] = {
    "Languages": [
        "python", "java", "javascript", "typescript", "c++", "c#", "php", "swift", 
        "kotlin", "go", "rust", "ruby", "scala", "r", "matlab", "perl"
    ],
    "Frontend": [
        "html", "css", "react", "angular", "vue", "svelte", "next.js", "tailwind", 
        "webpack", "babel", "jest", "typescript", "bootstrap", "material-ui"
    ],
    "Backend": [
        "node.js", "django", "flask", "fastapi", "spring", "express", "rails", 
        "graphql", "rest", "api", "microservices", "grpc", "soap"
    ],
    "Data/AI": [
        "machine learning", "deep learning", "tensorflow", "pytorch", "pandas", "numpy", 
        "scikit-learn", "nlp", "llm", "rag", "spark", "hadoop", "data science"
    ],
    "DevOps": [
        "docker", "kubernetes", "aws", "azure", "gcp", "ci/cd", "jenkins", "terraform", 
        "linux", "git", "github", "gitlab", "bitbucket", "ansible", "prometheus", "grafana"
    ],
    "Databases": [
        "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch", "dynamodb", 
        "firebase", "cassandra", "neo4j", "oracle", "sqlserver"
    ],
    "Soft Skills": [
        "leadership", "communication", "problem solving", "agile", "scrum", "teamwork", 
        "project management", "critical thinking", "analytical"
    ],
}

# Flatten all skills for quick lookups
ALL_SKILLS: List[str] = [s for skills in SKILL_CATEGORIES.values() for s in skills]

# Plotly theme configuration
PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#e2e8f0"),
    colorway=["#6366f1", "#818cf8", "#10b981", "#f59e0b", "#ef4444", "#06b6d4"],
)