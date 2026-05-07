#!/usr/bin/env python3
"""
ATS Resume Analyzer Pro - Application Launcher
Modern Flask-based web application with AI-powered resume analysis
"""

import os
import sys
import logging
from pathlib import Path

def main():
    """Main entry point for the application."""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    
    # Check required environment
    try:
        import flask
        import groq
        import langchain
    except ImportError as e:
        logger.error(f"Missing required package: {e}")
        logger.info("Run: pip install -r requirements.txt")
        sys.exit(1)
    
    # Check for API key
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY not set. You can provide it in the web interface.")
    
    # Run the Flask server
    logger.info("Starting ATS Resume Analyzer Pro...")
    
    try:
        from server import app
        port = int(os.environ.get("PORT", 5000))
        
        logger.info(f"🚀 Server running at http://localhost:{port}")
        logger.info("Press Ctrl+C to stop")
        
        app.run(
            host="0.0.0.0",
            port=port,
            debug=os.environ.get("FLASK_ENV") == "development"
        )
    except KeyboardInterrupt:
        logger.info("\n👋 Server stopped")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()



if __name__ == "__main__":
    main()
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


def render_sidebar():
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


def render_upload_tab(api_key: str):
    st.markdown("### 📥 Upload Resumes")
    col1, col2 = st.columns([3, 2])
    with col1:
        files = st.file_uploader(
            "Drag & drop resumes",
            type=["pdf", "docx", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
            help="Supports PDF, DOCX, and image files",
        )
    with col2:
        sheet_url = st.text_input("Or paste a Google Sheet URL", placeholder="https://docs.google.com/spreadsheets/...")

    if st.button("🚀 Analyze Resumes", type="primary"):
        if not api_key:
            st.error("Please add your Groq API key in the sidebar.")
            return
        if not files and not sheet_url:
            st.warning("Please upload files or provide a Google Sheet URL.")
            return

        _run_pipeline(files, sheet_url, api_key)


def _run_pipeline(files, sheet_url: str, api_key: str):
    extractor = DocumentExtractor()
    to_process: List[Tuple[str, str, str, str]] = []

    # Extract text from uploaded files
    if files:
        progress = st.progress(0, text="Extracting text…")
        for i, f in enumerate(files):
            suffix = f".{f.name.rsplit('.', 1)[-1]}"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(f.getvalue())
                tmp_path = tmp.name
            text = extractor.extract_text(tmp_path, suffix.lstrip("."))
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            if text.strip():
                to_process.append((text, f.name, api_key, st.session_state.jd_text))
            else:
                st.warning(f"Could not extract text from **{f.name}**")
            progress.progress((i + 1) / len(files), text=f"Extracted {i + 1}/{len(files)}")

    # Process Google Sheet
    if sheet_url:
        with st.spinner("Reading Google Sheet…"):
            handler = GoogleDriveHandler(extractor)
            for data, idx, total in handler.process_sheet(sheet_url):
                if "error" in data:
                    st.error(f"Sheet error: {data['error']}")
                    continue
                if data.get("text"):
                    to_process.append((data["text"], f"sheet_row_{idx}", api_key, st.session_state.jd_text))

    if not to_process:
        st.error("No usable text could be extracted from the provided files.")
        return

    # Run AI analysis in batches
    candidates: List[Candidate] = []
    with st.status(f"🧠 Analyzing {len(to_process)} resume(s)…", expanded=True) as status:
        for batch_start in range(0, len(to_process), CFG.batch_size):
            batch = to_process[batch_start: batch_start + CFG.batch_size]
            st.write(f"Processing batch {batch_start // CFG.batch_size + 1}…")
            with ThreadPoolExecutor(max_workers=CFG.max_workers) as executor:
                futures = {executor.submit(process_file_worker, item): item for item in batch}
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        candidates.append(result)
            if batch_start + CFG.batch_size < len(to_process):
                time.sleep(0.5)  # Rate limit buffer

        if not candidates:
            status.update(label="❌ No results", state="error")
            st.error("Analysis produced no results. Check your files and API key.")
            return

        # Sort by ATS score descending
        candidates.sort(key=lambda c: c.ats_score, reverse=True)
        st.session_state.candidates = candidates

        # Build RAG pipeline
        st.write("Building search index…")
        rag = RAGPipeline(api_key)
        rag.build(candidates)
        st.session_state.rag_pipeline = rag

        status.update(label=f"✅ Analyzed {len(candidates)} resume(s)!", state="complete")

    st.success(f"Successfully analyzed **{len(candidates)}** resume(s)!")
    st.balloons()


def render_results_tab():
    candidates: List[Candidate] = st.session_state.candidates
    if not candidates:
        st.info("No candidates yet. Upload resumes in the **Upload** tab.")
        return

    # ── KPI row ──
    st.markdown("### 📊 Overview")
    k1, k2, k3, k4 = st.columns(4)
    avg = sum(c.ats_score for c in candidates) / len(candidates)
    top = max(candidates, key=lambda c: c.ats_score)
    avg_exp = sum(c.experience_years for c in candidates) / len(candidates)
    k1.metric("Total Candidates", len(candidates))
    k2.metric("Avg ATS Score", f"{avg:.1f}")
    k3.metric("Top Scorer", f"{top.name} ({top.ats_score})")
    k4.metric("Avg Experience", f"{avg_exp:.1f} yrs")

    # ── Charts ──
    st.plotly_chart(make_score_bar_chart(candidates), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(make_scatter_chart(candidates), use_container_width=True)
    with col2:
        jd_fig = make_jd_match_chart(candidates)
        if jd_fig:
            st.plotly_chart(jd_fig, use_container_width=True)
        else:
            st.info("Provide a Job Description to see JD match scores.")

    # ── Data Table ──
    st.divider()
    st.markdown("### 📋 Candidate Summary")

    rows = []
    for c in candidates:
        rows.append({
            "Name": c.name,
            "ATS Score": c.ats_score,
            "JD Match %": c.match_percentage or "—",
            "Exp (yrs)": c.experience_years,
            "Education": c.education_level,
            "Skills Count": len(c.skills),
            "Email": c.email,
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Candidate Detail ──
    st.divider()
    st.markdown("### 🔍 Candidate Deep Dive")
    names = [c.name for c in candidates]
    selected = st.selectbox("Select candidate", names)
    cand = next((c for c in candidates if c.name == selected), None)
    if cand:
        _render_candidate_detail(cand)

    # ── Export ──
    st.divider()
    csv = df.to_csv(index=False)
    st.download_button("📥 Download CSV Report", csv, "ats_results.csv", "text/csv")


def _render_candidate_detail(c: Candidate):
    col1, col2, col3 = st.columns([1, 1, 2])
    color = score_color(c.ats_score)
    with col1:
        st.metric("ATS Score", c.ats_score)
        st.metric("JD Match", f"{c.match_percentage}%" if c.match_percentage else "N/A")
        st.metric("Experience", f"{c.experience_years} yrs")
    with col2:
        st.metric("Education", c.education_level)
        st.metric("Email", c.email)
        st.metric("Phone", c.phone)
    with col3:
        radar = make_radar_chart(c)
        st.plotly_chart(radar, use_container_width=True)

    if c.summary and c.summary != "N/A":
        st.info(f"**Summary:** {c.summary}")

    # Skills breakdown
    if c.skill_categories:
        st.markdown("**Skill Breakdown**")
        for cat, skills in c.skill_categories.items():
            tags = " ".join(f'<span class="skill-tag">{s}</span>' for s in skills)
            st.markdown(f"**{cat}:** {tags}", unsafe_allow_html=True)

    col_m, col_miss = st.columns(2)
    with col_m:
        if c.matching_skills:
            tags = " ".join(f'<span class="skill-tag">{s}</span>' for s in c.matching_skills)
            st.markdown(f"**✅ Matching Skills:**<br>{tags}", unsafe_allow_html=True)
    with col_miss:
        if c.missing_skills:
            tags = " ".join(f'<span class="skill-tag missing-tag">{s}</span>' for s in c.missing_skills)
            st.markdown(f"**❌ Missing Skills:**<br>{tags}", unsafe_allow_html=True)

    if c.certifications:
        st.markdown("**Certifications:** " + ", ".join(c.certifications))


def render_chatbot_tab():
    rag: Optional[RAGPipeline] = st.session_state.rag_pipeline
    if not rag:
        st.info("Process resumes first to enable the chatbot.")
        return

    st.markdown("### 💬 Ask About Candidates")
    st.markdown("Ask anything about the uploaded candidates — rankings, skill gaps, experience, and more.")

    # Quick questions
    quick_qs = [
        "Who is the best candidate overall?",
        "List Python developers sorted by experience",
        "Who has 5+ years of experience?",
        "Which candidates have machine learning skills?",
        "Who is missing the most required skills?",
    ]
    cols = st.columns(len(quick_qs))
    for i, q in enumerate(quick_qs):
        if cols[i].button(q, key=f"qq_{i}", use_container_width=True):
            st.session_state.chat_history.append({"role": "user", "content": q})
            st.session_state.trigger_query = True

    st.divider()

    # Chat input
    if prompt := st.chat_input("Ask a question about the candidates…"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.session_state.trigger_query = True

    # Handle query
    if st.session_state.trigger_query and st.session_state.chat_history:
        last = st.session_state.chat_history[-1]
        if last["role"] == "user":
            with st.spinner("Searching candidate database…"):
                result = rag.query(last["content"])
                answer = result["answer"]
                if result.get("sources"):
                    src_lines = "\n".join(
                        f"- **{s['name']}** (Score: {s['score']})" for s in result["sources"]
                    )
                    answer += f"\n\n**Referenced Candidates:**\n{src_lines}"
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.session_state.trigger_query = False
        st.rerun()

    # Render chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def main():
    st.set_page_config(
        page_title="ATS Resume Analyzer Pro",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    init_session()

    api_key = render_sidebar()

    st.markdown("# 🚀 ATS Resume Analyzer Pro")
    st.markdown("*AI-powered resume screening, scoring, and candidate intelligence*")
    st.divider()

    tab_upload, tab_results, tab_chat = st.tabs(["📥 Upload & Analyze", "📊 Results & Insights", "💬 AI Chatbot"])

    with tab_upload:
        render_upload_tab(api_key)

    with tab_results:
        render_results_tab()

    with tab_chat:
        render_chatbot_tab()


if __name__ == "__main__":
    main()