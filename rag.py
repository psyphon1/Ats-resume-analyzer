# rag.py
import logging
from typing import List, Dict, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document as LCDoc
from langchain_groq import ChatGroq
from langchain_core.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnablePassthrough
from config import CFG
from models import Candidate

logger = logging.getLogger(__name__)

class RAGPipeline:
    PROMPT_TEMPLATE = """You are an expert HR assistant with access to a database of candidate profiles.
Use the following context to answer the question accurately and concisely.
If you cannot find the answer in the context, say so clearly.

Context:
{context}

Question: {question}

Answer:"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.candidates: List[Candidate] = []
        self.vector_store: Optional[FAISS] = None
        self.qa_chain = None
        self._embeddings: Optional[HuggingFaceEmbeddings] = None

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(
                model_name=CFG.embedding_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        return self._embeddings

    def build(self, candidates: List[Candidate]):
        """Build vector store and QA chain from candidates."""
        self.candidates = candidates
        docs = [self._candidate_to_doc(i, c) for i, c in enumerate(candidates)]
        splitter = RecursiveCharacterTextSplitter(chunk_size=CFG.chunk_size, chunk_overlap=CFG.chunk_overlap)
        splits = splitter.split_documents(docs)
        self.vector_store = FAISS.from_documents(splits, self.embeddings)

        llm = ChatGroq(
            groq_api_key=self.api_key,
            model_name=CFG.groq_model,
            temperature=0.1,
            max_tokens=800,
        )
        
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 6})
        
        # Create prompt template
        prompt = PromptTemplate(
            template=self.PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )
        
        # Create chains using new langchain API
        combine_docs_chain = create_stuff_documents_chain(llm, prompt)
        self.qa_chain = create_retrieval_chain(retriever, combine_docs_chain)

    def _candidate_to_doc(self, idx: int, c: Candidate) -> LCDoc:
        skills_str = ", ".join(c.skills[:30])
        matching = ", ".join(c.matching_skills[:15])
        missing = ", ".join(c.missing_skills[:10])
        content = f"""CANDIDATE #{idx + 1}: {c.name}
ATS Score: {c.ats_score}/100 | JD Match: {c.match_percentage}%
Experience: {c.experience_years} years
Education: {c.education} ({c.education_level})
Skills: {skills_str}
Matching Skills: {matching}
Missing Skills: {missing}
Summary: {c.summary}
Certifications: {", ".join(c.certifications) or "None"}
Email: {c.email} | Phone: {c.phone}
"""
        return LCDoc(page_content=content, metadata={"idx": idx, "name": c.name, "score": c.ats_score})

    def query(self, question: str) -> Dict:
        """Query the RAG pipeline and return answer with sources."""
        if not self.qa_chain:
            return {"answer": "Pipeline not initialized. Please upload and process resumes first.", "sources": []}
        try:
            result = self.qa_chain.invoke({"input": question})
            sources = []
            
            # Extract context documents if available
            context_docs = result.get("context", [])
            seen = set()
            
            for doc in context_docs:
                idx = doc.metadata.get("idx")
                if idx is not None and idx not in seen and 0 <= idx < len(self.candidates):
                    seen.add(idx)
                    c = self.candidates[idx]
                    sources.append({"name": c.name, "score": c.ats_score, "idx": idx})
            
            return {
                "answer": result.get("output", result.get("answer", "")),
                "sources": sources[:3]
            }
        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            return {"answer": f"Query failed: {str(e)}", "sources": []}

    def get_candidate(self, idx: int) -> Optional[Candidate]:
        return self.candidates[idx] if 0 <= idx < len(self.candidates) else None