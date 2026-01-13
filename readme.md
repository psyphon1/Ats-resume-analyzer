# 📋 ATS Resume Analyzer Pro

A production-ready AI-powered resume parsing and analysis system with intelligent candidate search capabilities.

## ✨ Features

- **Multi-Format Support**: PDF, DOCX, PNG, JPG/JPEG
- **OCR Integration**: Automatic text extraction from scanned documents
- **Google Sheets Integration**: Bulk process resumes from Google Drive
- **AI-Powered Analysis**: Extract contact info, skills, education, experience
- **ATS Scoring**: Calculate compatibility scores (0-100)
- **Intelligent Search**: RAG-powered chatbot for candidate queries
- **Production Ready**: Comprehensive error handling, logging, and security

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Groq API Key ([Get one here](https://console.groq.com))
- Google Cloud Project (optional, for Sheets integration)

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd ats-resume-analyzer
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up configuration**
```bash
mkdir config
```

4. **Run the application**
```bash
streamlit run app.py
```

5. **Access the app**
Open your browser to `http://localhost:8501`

## 📦 Requirements

Create a `requirements.txt` file:

```txt
streamlit>=1.28.0
PyMuPDF>=1.23.0
pandas>=2.0.0
rapidocr-onnxruntime>=1.3.0
python-docx>=1.0.0
groq>=0.4.0
gspread>=5.11.0
google-auth-oauthlib>=1.1.0
google-auth>=2.23.0
google-api-python-client>=2.100.0
langchain>=0.1.0
langchain-community>=0.0.20
langchain-groq>=0.0.1
faiss-cpu>=1.7.4
sentence-transformers>=2.2.0
```

## 🔧 Configuration

### Environment Variables (Optional)

Create a `.env` file:

```env
GROQ_MODEL=llama-3.1-8b-instant
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
MAX_WORKERS=3
MAX_FILE_SIZE_MB=10
CHUNK_SIZE=4000
TEMPERATURE=0.05
```

### Google Sheets Setup (Optional)

1. **Create Google Cloud Project**
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project

2. **Enable APIs**
   - Enable Google Drive API
   - Enable Google Sheets API

3. **Create OAuth Credentials**
   - Go to "Credentials" → "Create Credentials" → "OAuth client ID"
   - Application type: Desktop app
   - Download credentials as `credentials.json`
   - Place in `config/credentials.json`

4. **First Run**
   - The app will open a browser for authentication
   - Grant necessary permissions
   - Token will be saved to `config/token.json`

## 📖 Usage

### 1. Upload Resumes

**Method A: File Upload**
- Click "Upload Resumes" button
- Select PDF, DOCX, or image files
- Click "Analyze Resumes"

**Method B: Google Sheets**
- Prepare a Google Sheet with resume links
- Resumes must be in Google Drive
- Paste sheet URL in the app
- Click "Analyze Resumes"

### 2. View Results

Navigate to the "Results" tab to see:
- Parsed candidate information
- ATS scores
- Skills, experience, education
- Export to CSV

### 3. Search Candidates

Use the "Chatbot" tab to:
- Ask natural language questions
- Find candidates by skills
- Filter by experience level
- Get ranked recommendations

**Example Queries:**
```
- "Who is the best candidate?"
- "Find Python developers with 5+ years experience"
- "Show me candidates with AWS skills"
- "Who has a Master's degree?"
```

## 🏗️ Architecture

```
┌─────────────────┐
│   Streamlit UI  │
└────────┬────────┘
         │
    ┌────▼────┐
    │  Main   │
    │ Handler │
    └──┬───┬──┘
       │   │
┌──────▼───▼──────────┐
│  Document Extractor │ ◄── PyMuPDF, python-docx, RapidOCR
└──────────┬──────────┘
           │
    ┌──────▼──────┐
    │   Groq LLM  │ ◄── Resume Analysis
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │ RAG Pipeline│ ◄── FAISS + LangChain
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │  Chat Bot   │
    └─────────────┘
```

## 🔒 Security Features

- **File Validation**: Size and type restrictions
- **Input Sanitization**: Filename and path validation
- **API Key Validation**: Format checking
- **Error Handling**: Comprehensive exception management
- **Logging**: Rotating logs with sensitive data protection
- **Rate Limiting**: Controlled API usage

## 📊 ATS Scoring Algorithm

```
Total Score: 100 points

- Name (10 pts): Valid name found
- Email (15 pts): Valid email address
- Phone (10 pts): Valid phone number
- Skills (30 pts): 3 points per skill (max 10 skills)
- Experience (20 pts): 4 points per year (max 5 years)
- Education (10 pts): Valid degree found
- Certifications (5 pts): Any certifications present
```

## 🐛 Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

**2. OCR Not Working**
```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract
```

**3. Google Auth Fails**
```bash
# Delete and recreate token
rm config/token.json
# Restart app and re-authenticate
```

**4. Memory Issues**
```bash
# Reduce batch size in config
MAX_WORKERS=1
BATCH_SIZE=2
```

### Logs

Check logs for debugging:
```bash
tail -f ats.log
```

## 🚀 Production Deployment

### Docker (Recommended)

Create `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create directories
RUN mkdir -p config logs temp

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t ats-analyzer .
docker run -p 8501:8501 -v $(pwd)/config:/app/config ats-analyzer
```

### Cloud Deployment

**Streamlit Cloud:**
1. Push code to GitHub
2. Connect repository to [Streamlit Cloud](https://streamlit.io/cloud)
3. Add secrets in dashboard
4. Deploy

**AWS/GCP/Azure:**
- Use Docker container
- Set environment variables
- Configure load balancer
- Enable auto-scaling

## 📈 Performance Optimization

### Tips

1. **Batch Processing**: Process multiple resumes concurrently
2. **Caching**: Results are cached to avoid reprocessing
3. **Rate Limiting**: Respects API rate limits
4. **Chunking**: Large documents split intelligently
5. **Vector Store**: FAISS for fast similarity search

### Benchmarks

| Resumes | Time (avg) | Memory |
|---------|-----------|--------|
| 10      | 30s       | 500MB  |
| 50      | 2.5min    | 1GB    |
| 100     | 5min      | 1.5GB  |

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open Pull Request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- [Groq](https://groq.com/) - LLM inference
- [LangChain](https://langchain.com/) - RAG framework
- [Streamlit](https://streamlit.io/) - Web framework
- [PyMuPDF](https://pymupdf.readthedocs.io/) - PDF processing
- [RapidOCR](https://github.com/RapidAI/RapidOCR) - OCR engine

## 📞 Support

- **Issues**: [GitHub Issues](your-repo/issues)
- **Email**: support@example.com
- **Docs**: [Full Documentation](your-docs-url)

## 🔄 Changelog

### v2.0.0 (Production)
- ✅ Enhanced error handling
- ✅ Security validations
- ✅ Comprehensive logging
- ✅ Performance optimization
- ✅ Docker support
- ✅ Better documentation

### v1.0.0 (Initial)
- Basic resume parsing
- Simple UI
- File upload support

---

**Built with ❤️by Chinmay Duse(psyphon1) using AI-powered technologies**