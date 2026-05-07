"""
ATS Resume Analyzer Pro - Flask Backend Server
Modern production-ready REST API with HTML/CSS/JS frontend
"""

import os
import json
import logging
import tempfile
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

from config import CFG, SKILL_CATEGORIES, ALL_SKILLS
from models import Candidate
from extractors import DocumentExtractor, GoogleDriveHandler
from analyzer import ResumeAnalyzer
from rag import RAGPipeline

# ── Configuration ────────────────────────────────────────────────────────────
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.FileHandler("server.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ── Flask App ────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)

# Global state
CANDIDATES: List[Candidate] = []
RAG_PIPELINE: Optional[RAGPipeline] = None


def get_api_key() -> str:
    """Get Groq API key from environment or request."""
    return os.environ.get("GROQ_API_KEY", "")


def process_file_worker(args: Tuple) -> Optional[Candidate]:
    """Worker function for parallel resume processing."""
    text, filename, api_key, jd = args
    try:
        analyzer = ResumeAnalyzer(api_key)
        return analyzer.analyze(text, filename, jd)
    except Exception as e:
        logger.error(f"Worker failed for {filename}: {e}")
        return None


# ── API Routes ───────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def serve_index():
    """Serve the main HTML page."""
    return send_from_directory("static", "index.html")


@app.route("/static/<path:path>", methods=["GET"])
def serve_static(path):
    """Serve static files."""
    return send_from_directory("static", path)


@app.route("/api/config", methods=["GET"])
def get_config():
    """Return app configuration."""
    return jsonify({
        "skill_categories": SKILL_CATEGORIES,
        "groq_model": CFG.groq_model,
        "version": "2.0"
    })


@app.route("/api/upload", methods=["POST"])
def upload_resume():
    """
    Upload and process single resume.
    Expected: multipart/form-data with 'file' and optional 'jd'
    """
    api_key = request.form.get("api_key", "") or get_api_key()
    if not api_key:
        return jsonify({"error": "API key not provided"}), 400

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    jd = request.form.get("jd", "")

    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    try:
        # Extract file extension and text
        ext = Path(file.filename).suffix.lower().lstrip(".")
        
        # Save temporarily
        with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        # Extract text
        extractor = DocumentExtractor()
        text = extractor.extract_text(tmp_path, ext)
        os.unlink(tmp_path)

        if not text:
            return jsonify({"error": "Failed to extract text from file"}), 400

        # Analyze
        analyzer = ResumeAnalyzer(api_key)
        candidate = analyzer.analyze(text, file.filename, jd)
        
        CANDIDATES.append(candidate)

        return jsonify({
            "success": True,
            "candidate": candidate.to_dict(),
            "total_candidates": len(CANDIDATES)
        })

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/batch-upload", methods=["POST"])
def batch_upload():
    """
    Process multiple resumes in batch.
    Expected: multipart/form-data with 'files[]' and optional 'jd'
    """
    api_key = request.form.get("api_key", "") or get_api_key()
    if not api_key:
        return jsonify({"error": "API key not provided"}), 400

    files = request.files.getlist("files[]")
    jd = request.form.get("jd", "")

    if not files:
        return jsonify({"error": "No files provided"}), 400

    try:
        results = []
        extractor = DocumentExtractor()
        
        # Prepare worker arguments
        worker_args = []
        for file in files:
            ext = Path(file.filename).suffix.lower().lstrip(".")
            with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
                file.save(tmp.name)
                text = extractor.extract_text(tmp.name, ext)
                os.unlink(tmp.name)
                
                if text:
                    worker_args.append((text, file.filename, api_key, jd))

        # Process in parallel
        with ThreadPoolExecutor(max_workers=CFG.max_workers) as executor:
            futures = [executor.submit(process_file_worker, args) for args in worker_args]
            
            for idx, future in enumerate(as_completed(futures)):
                try:
                    candidate = future.result()
                    if candidate:
                        CANDIDATES.append(candidate)
                        results.append(candidate.to_dict())
                except Exception as e:
                    logger.error(f"Batch process error: {e}")

        return jsonify({
            "success": True,
            "processed": len(results),
            "candidates": results,
            "total_candidates": len(CANDIDATES)
        })

    except Exception as e:
        logger.error(f"Batch upload failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/candidates", methods=["GET"])
def list_candidates():
    """Get all processed candidates."""
    sort_by = request.args.get("sort", "ats_score")
    
    candidates_data = [c.to_dict() for c in CANDIDATES]
    
    if sort_by == "ats_score":
        candidates_data.sort(key=lambda x: x.get("ats_score", 0), reverse=True)
    elif sort_by == "match":
        candidates_data.sort(key=lambda x: x.get("match_percentage", 0), reverse=True)
    elif sort_by == "experience":
        candidates_data.sort(key=lambda x: x.get("experience_years", 0), reverse=True)

    return jsonify({
        "total": len(candidates_data),
        "candidates": candidates_data
    })


@app.route("/api/candidates/<int:idx>", methods=["GET"])
def get_candidate(idx):
    """Get detailed candidate info."""
    if not (0 <= idx < len(CANDIDATES)):
        return jsonify({"error": "Candidate not found"}), 404
    
    candidate = CANDIDATES[idx]
    return jsonify({
        "success": True,
        "candidate": candidate.to_dict()
    })


@app.route("/api/rag/build", methods=["POST"])
def build_rag():
    """Build RAG pipeline from current candidates."""
    global RAG_PIPELINE
    
    api_key = request.json.get("api_key", "") if request.json else ""
    api_key = api_key or get_api_key()
    
    if not api_key:
        return jsonify({"error": "API key required"}), 400
    
    if not CANDIDATES:
        return jsonify({"error": "No candidates to build RAG from"}), 400

    try:
        RAG_PIPELINE = RAGPipeline(api_key)
        RAG_PIPELINE.build(CANDIDATES)
        
        return jsonify({
            "success": True,
            "message": f"RAG pipeline built with {len(CANDIDATES)} candidates"
        })
    except Exception as e:
        logger.error(f"RAG build failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/rag/query", methods=["POST"])
def rag_query():
    """Query the RAG pipeline."""
    if not RAG_PIPELINE:
        return jsonify({"error": "RAG pipeline not initialized"}), 400
    
    data = request.json or {}
    question = data.get("question", "").strip()
    
    if not question:
        return jsonify({"error": "Question required"}), 400

    try:
        result = RAG_PIPELINE.query(question)
        return jsonify({
            "success": True,
            "answer": result["answer"],
            "sources": result["sources"]
        })
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/export", methods=["GET"])
def export_candidates():
    """Export all candidates as JSON."""
    candidates_data = [c.to_dict() for c in CANDIDATES]
    
    return jsonify({
        "export_date": datetime.now().isoformat(),
        "total_candidates": len(candidates_data),
        "candidates": candidates_data
    }), 200, {"Content-Disposition": "attachment; filename=candidates.json"}


@app.route("/api/clear", methods=["POST"])
def clear_candidates():
    """Clear all candidates and reset state."""
    global CANDIDATES, RAG_PIPELINE
    
    count = len(CANDIDATES)
    CANDIDATES.clear()
    RAG_PIPELINE = None
    
    return jsonify({
        "success": True,
        "message": f"Cleared {count} candidates"
    })


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors."""
    logger.error(f"Server error: {error}")
    return jsonify({"error": "Internal server error"}), 500


# ── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV") == "development"
    
    logger.info(f"Starting ATS Resume Analyzer on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
