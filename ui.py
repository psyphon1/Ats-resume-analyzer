# ui.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from typing import List, Optional
from models import Candidate
from config import PLOTLY_THEME
from extractors import DocumentExtractor, GoogleDriveHandler
from analyzer import ResumeAnalyzer
from rag import RAGPipeline
from main import process_file_worker


def get_api_key() -> str:
    if "groq_api_key" in st.secrets:
        return st.secrets["groq_api_key"]
    if os.environ.get("GROQ_API_KEY"):
        return os.environ.get("GROQ_API_KEY")
    return st.session_state.get("custom_api_key", "")

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg: #0d0f14;
    --surface: #161b25;
    --surface-2: #1e2535;
    --border: #2a3347;
    --accent: #6366f1;
    --accent-2: #818cf8;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
    --text: #e2e8f0;
    --text-muted: #64748b;
    --radius: 12px;
}

html, body, [data-testid="stApp"] {
    background-color: var(--bg) !important;
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700;
}

[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] * { color: var(--text) !important; }

.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: var(--surface);
    border-radius: var(--radius);
    padding: 4px;
    border: 1px solid var(--border);
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: 0.85rem;
    color: var(--text-muted) !important;
    background: transparent;
    border: none;
    padding: 8px 20px;
    transition: all 0.2s;
}

.stTabs [aria-selected="true"] {
    background: var(--accent) !important;
    color: white !important;
}

.stButton > button {
    background: var(--accent) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius) !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s ease !important;
    width: 100%;
}

.stButton > button:hover {
    background: var(--accent-2) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4) !important;
}

[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.2rem !important;
}

[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 2rem !important;
    font-weight: 800 !important;
    color: var(--accent-2) !important;
}

[data-testid="stMetricLabel"] { color: var(--text-muted) !important; }

.stTextInput input, .stTextArea textarea {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}

.stFileUploader {
    background: var(--surface-2) !important;
    border: 2px dashed var(--border) !important;
    border-radius: var(--radius) !important;
}

.stFileUploader:hover { border-color: var(--accent) !important; }

.candidate-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.2rem;
    margin-bottom: 0.8rem;
    transition: border-color 0.2s;
}

.candidate-card:hover { border-color: var(--accent); }

.score-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.85rem;
}

.skill-tag {
    display: inline-block;
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.3);
    color: var(--accent-2);
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.78rem;
    margin: 2px;
}

.missing-tag {
    background: rgba(239, 68, 68, 0.1);
    border-color: rgba(239, 68, 68, 0.3);
    color: #fca5a5;
}

.chat-user {
    background: var(--accent);
    color: white;
    border-radius: 16px 16px 4px 16px;
    padding: 0.8rem 1.2rem;
    margin-left: 20%;
    margin-bottom: 0.5rem;
}

.chat-bot {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 16px 16px 16px 4px;
    padding: 0.8rem 1.2rem;
    margin-right: 20%;
    margin-bottom: 0.5rem;
}

.stProgress > div > div {
    background: linear-gradient(90deg, var(--accent), var(--accent-2)) !important;
    border-radius: 4px !important;
}

.stSelectbox > div > div {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}

.stDataFrame { border: 1px solid var(--border); border-radius: var(--radius); }
div[data-testid="stExpander"] { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); }
.stAlert { border-radius: var(--radius) !important; }
hr { border-color: var(--border) !important; }
</style>
"""

def score_color(score: int) -> str:
    if score >= 80: return "#10b981"
    if score >= 60: return "#f59e0b"
    if score >= 40: return "#f97316"
    return "#ef4444"

def score_label(score: int) -> str:
    if score >= 80: return "Excellent"
    if score >= 60: return "Good"
    if score >= 40: return "Fair"
    return "Weak"

def make_score_bar_chart(candidates: List[Candidate]) -> go.Figure:
    names = [c.name for c in candidates]
    scores = [c.ats_score for c in candidates]
    colors = [score_color(s) for s in scores]
    fig = go.Figure(go.Bar(
        x=names, y=scores, marker_color=colors,
        text=scores, textposition="outside",
        hovertemplate="<b>%{x}</b><br>ATS Score: %{y}<extra></extra>",
    ))
    fig.update_layout(
        title="ATS Score Distribution",
        yaxis=dict(range=[0, 110], title="Score"),
        xaxis_tickangle=-30,
        **PLOTLY_THEME
    )
    return fig

def make_radar_chart(candidate: Candidate) -> go.Figure:
    from config import SKILL_CATEGORIES
    categories = list(SKILL_CATEGORIES.keys())
    values = [len(candidate.skill_categories.get(cat, [])) for cat in categories]
    max_val = max(values) if any(values) else 1
    norm = [v / max_val * 10 for v in values]
    fig = go.Figure(go.Scatterpolar(
        r=norm + [norm[0]],
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor="rgba(99,102,241,0.2)",
        line=dict(color="#6366f1"),
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10], color="#64748b")),
        showlegend=False,
        **PLOTLY_THEME,
    )
    return fig

def make_scatter_chart(candidates: List[Candidate]) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=[c.experience_years for c in candidates],
        y=[c.ats_score for c in candidates],
        mode="markers+text",
        text=[c.name for c in candidates],
        textposition="top center",
        marker=dict(
            size=[max(8, len(c.skills)) for c in candidates],
            color=[c.ats_score for c in candidates],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="ATS Score"),
        ),
        hovertemplate="<b>%{text}</b><br>Experience: %{x} yrs<br>Score: %{y}<extra></extra>",
    ))
    fig.update_layout(
        title="Experience vs. ATS Score (bubble size = skill count)",
        xaxis_title="Years of Experience",
        yaxis_title="ATS Score",
        **PLOTLY_THEME,
    )
    return fig

def make_jd_match_chart(candidates: List[Candidate]) -> Optional[go.Figure]:
    matched = [c for c in candidates if c.match_percentage > 0]
    if not matched:
        return None
    fig = go.Figure(go.Bar(
        x=[c.name for c in matched],
        y=[c.match_percentage for c in matched],
        marker_color="#10b981",
        text=[f"{c.match_percentage}%" for c in matched],
        textposition="outside",
    ))
    fig.update_layout(
        title="Job Description Match %",
        yaxis=dict(range=[0, 115]),
        xaxis_tickangle=-30,
        **PLOTLY_THEME,
    )
    return fig


def render_sidebar():
    from .main import get_api_key  # Import from main since it's a helper
    api_key = get_api_key()
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")

        if not api_key:
            st.warning("No Groq API key detected")
            new_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
            if st.button("💾 Save Key") and new_key:
                st.session_state["custom_api_key"] = new_key
                st.rerun()
        else:
            st.success("✅ API Key Active")
            if st.button("🔄 Change Key"):
                st.session_state.pop("custom_api_key", None)
                st.rerun()

        st.divider()
        st.markdown("### 📋 Job Description")
        st.session_state.jd_text = st.text_area(
            "Paste JD for match scoring",
            value=st.session_state.jd_text,
            height=220,
            placeholder="Senior Python Developer with 5+ years experience...",
        )

        st.divider()
        with st.expander("📖 How to Use"):
            st.markdown("""
1. Enter your **Groq API Key**
2. Paste a **Job Description** *(optional — enables match scoring)*
3. **Upload resumes** (PDF, DOCX, or images)
4. Click **🚀 Analyze Resumes**
5. Explore results in **Results** tab
6. Ask the **Chatbot** anything about candidates
""")

        st.divider()
        stats = st.session_state.candidates
        if stats:
            st.markdown(f"**{len(stats)} candidates loaded**")
            avg = sum(c.ats_score for c in stats) / len(stats)
            st.markdown(f"Avg score: **{avg:.1f}** / 100")

        if st.button("🗑️ Clear All"):
            st.session_state.clear()
            st.rerun()

    return api_key


def init_session():
    defaults = {
        "candidates": [],
        "rag_pipeline": None,
        "chat_history": [],
        "jd_text": "",
        "trigger_query": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    init_session()
    api_key = render_sidebar()

    st.markdown("# ðŸš€ ATS Resume Analyzer Pro")
    st.markdown("*AI-powered resume screening, scoring, and candidate intelligence*")
    st.divider()

    tab_upload, tab_results, tab_chat = st.tabs(["ðŸ“¥ Upload & Analyze", "ðŸ“Š Results & Insights", "ðŸ’¬ AI Chatbot"])

    with tab_upload:
        render_upload_tab(api_key)

    with tab_results:
        render_results_tab()

    with tab_chat:
        render_chatbot_tab()