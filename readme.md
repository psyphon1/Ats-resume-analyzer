# � ATS Resume Analyzer Pro

A modern, AI-powered Applicant Tracking System (ATS) resume analyzer built with Flask, LangChain, and cutting-edge web technologies. Analyze resumes, match them against job descriptions, and search candidate profiles using natural language queries.

## ✨ Features

- **🚀 Fast Resume Processing** - Extract text from PDF, DOCX, and image formats
- **🤖 AI Analysis** - Powered by Groq LLaMA 3.1 for intelligent resume parsing
- **📊 Comprehensive Scoring** - ATS score, job description matching, and experience calculation
- **🔍 RAG Search** - Query your candidate database with natural language
- **💾 Batch Processing** - Process multiple resumes simultaneously with parallel workers
- **🎨 Modern UI/UX** - Beautiful, responsive interface built with vanilla HTML/CSS/JS
- **📤 Export** - Download candidate data as JSON
- **🔐 Secure** - No external dependencies for API credentials

## 🛠️ Tech Stack

- **Backend**: Flask 3.0+, Python 3.8+
- **LLM**: Groq (LLaMA 3.1)
- **Vector Store**: FAISS + Sentence Transformers
- **Document Processing**: PyMuPDF, python-docx, RapidOCR
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Architecture**: REST API with JSON responses

## 📦 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/psyphon1/Ats-resume-analyzer.git
cd Ats-resume-analyzer
```

### 2. Create Virtual Environment
```bash
# Using Python 3.8+
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (macOS/Linux)
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
```bash
# Copy the example env file
cp .env.example .env

# Edit .env with your Groq API key
# Get your key from: https://console.groq.com/keys
```

## 🚀 Quick Start

### Run the Server
```bash
python main.py
```

The application will start at `http://localhost:5000`

### Using the Application

1. **Upload Resumes**
   - Click the "Upload" tab
   - Drag and drop or select resume files (PDF, DOCX, PNG, JPG)
   - Optionally provide a job description for matching
   - Click "Upload Resume(s)"

2. **View Candidates**
   - Click the "Candidates" tab
   - See all processed candidates with scores
   - Sort by ATS Score, JD Match, or Experience
   - Click any card for detailed information

3. **Search with RAG**
   - Click the "RAG Search" tab
   - Provide your Groq API key
   - Click "Build RAG Pipeline"
   - Ask natural language questions about your candidates
   - Get AI-powered answers with relevant candidate references

4. **Export Data**
   - Click the "Settings" tab
   - Click "Export as JSON" to download candidate data
   - Use "Reset Application" to clear all data

## 📝 API Endpoints

### Candidates
- `GET /api/candidates` - List all candidates
- `GET /api/candidates/<id>` - Get candidate details
- `POST /api/upload` - Upload single resume
- `POST /api/batch-upload` - Upload multiple resumes

### RAG Pipeline
- `POST /api/rag/build` - Build search index
- `POST /api/rag/query` - Query candidates
- `GET /api/export` - Export all candidates
- `POST /api/clear` - Clear all data

### Configuration
- `GET /api/config` - Get app configuration

## 🔧 Configuration

Edit `config.py` to customize:

```python
# LLM Settings
groq_model = "llama-3.1-8b-instant"
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

# Processing Settings
max_resume_chars = 8000
max_tokens = 1500
temperature = 0.1
max_workers = 4  # Parallel processing threads

# Vector Store Settings
chunk_size = 4000
chunk_overlap = 200
```

## 📊 Scoring Algorithm

### ATS Score (0-100)
- Resume Information (25pts)
  - Name, email, phone presence
- Skills & Keywords (35pts)
  - Number and relevance of skills
- Experience (20pts)
  - Years of work experience
- Education (15pts)
  - Degree level and institution
- Recency (5pts)
  - Recent work history

### JD Match Percentage
- Calculated when job description is provided
- Based on skill overlap, seniority level, and domain relevance
- Weighted towards technical skill matching

## 🎯 Skill Categories Supported

- **Languages**: Python, Java, JavaScript, TypeScript, C++, Go, Rust, etc.
- **Frontend**: React, Angular, Vue, Next.js, Tailwind, etc.
- **Backend**: Django, Flask, FastAPI, Node.js, Spring, etc.
- **Data/AI**: ML, Deep Learning, TensorFlow, PyTorch, NLP, RAG, etc.
- **DevOps**: Docker, Kubernetes, AWS, Azure, GCP, CI/CD, etc.
- **Databases**: SQL, MongoDB, PostgreSQL, Redis, Elasticsearch, etc.
- **Soft Skills**: Leadership, Communication, Agile, Scrum, etc.

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'langchain.chains'"
**Solution**: Update langchain packages:
```bash
pip install --upgrade langchain langchain-community langchain-core
```

### Issue: "No module named 'groq'"
**Solution**: Install all dependencies:
```bash
pip install -r requirements.txt
```

### Issue: Slow resume processing
**Solution**: Reduce batch size or increase max_workers in config.py:
```python
max_workers = 8  # Increase from 4
```

### Issue: Poor resume extraction from images
**Solution**: Ensure image quality is good (300+ DPI). Update RapidOCR:
```bash
pip install --upgrade rapidocr-onnxruntime
```

## 📈 Performance Tips

- Use batch upload for multiple resumes (faster than one-by-one)
- For large batches (50+), increase `max_workers` in config.py
- Vector embeddings are cached - building RAG is faster after first build
- Consider using gzip compression for large exports

## 🔐 Security Notes

- API keys are not logged or stored
- All processing happens locally
- No candidate data is sent to third parties
- Temporary files are cleaned up automatically
- Use environment variables for sensitive configuration

## 📄 License

MIT License - see LICENSE file for details

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 💬 Support

For issues, questions, or suggestions, please open an issue on GitHub.

---

**Built with ❤️ using Flask, LangChain, and Groq API**

Visit [Groq Console](https://console.groq.com) to get your free API key and enjoy fast, cost-effective AI inference.

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
**Built By Chinmay Duse(psyphon1)**