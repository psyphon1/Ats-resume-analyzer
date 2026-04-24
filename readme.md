# 📋 ATS Resume Analyzer Pro (Upgraded)

A production-ready AI-powered resume parsing and analysis system with intelligent Job Description matching and candidate search capabilities.

## ✨ Features

- **🎯 Job Description Matching**: NEW! Match resumes against specific JD requirements for accurate scoring.
- **🚀 Premium UI/UX**: Sleek, modern interface with Inter font, glassmorphism elements, and interactive charts.
- **📊 Advanced Analytics**: Plotly-powered visualizations of ATS scores and candidate distribution.
- **💬 RAG Chatbot**: Chat with your candidate database to find the perfect fit using natural language.
- **📄 Multi-Format Support**: High-accuracy extraction from PDF, DOCX, and images (OCR).
- **☁️ Cloud Ready**: Optimized for Streamlit Cloud with Secret management and simplified configuration.
- **⚡ Parallel Processing**: Blazing fast AI analysis using ThreadPoolExecutor and Groq's high-speed inference.

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Groq API Key ([Get one here](https://console.groq.com))

### Installation
1. **Clone & Install**
```bash
git clone <your-repo-url>
cd Ats-resume-analyzer
pip install -r requirements.txt
```

2. **Run Locally**
```bash
streamlit run main.py
```

### ☁️ Streamlit Cloud Deployment
1. Push this repository to GitHub.
2. Go to [Streamlit Cloud](https://streamlit.io/cloud) and deploy.
3. In **Settings > Secrets**, add your Groq key:
```toml
groq_api_key = "your_gsk_key_here"
```

## 🛠️ Usage Guide

1. **Configuration**: Enter your Groq API key in the sidebar (if not using secrets).
2. **Target JD**: Paste the Job Description in the sidebar text area.
3. **Upload**: Drop your resumes in the "Upload" tab.
4. **Analyze**: Click "🚀 Process" and watch the AI work.
5. **Results**:
   - Check the **Results** tab for the leaderboards and charts.
   - Use the **Chatbot** to ask specific questions like "Who has the most experience in Python?".

## 📦 Requirements
Core dependencies include:
- `streamlit`
- `groq`
- `langchain` & `faiss-cpu`
- `PyMuPDF` & `RapidOCR`
- `plotly` & `pandas`

## 🤝 Contributing
Feel free to fork and submit PRs. For major changes, please open an issue first.

---
**Enhanced with ❤️ by Antigravity AI**