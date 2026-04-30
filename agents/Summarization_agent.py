"""
agents/summarization_agent.py

Summarization Agent — two paths:
  • PDF  → PyPDFLoader → semantic_chunk_text → per-chunk LLM summaries → LLM merge
  • CSV/Excel → pandas describe/sample → single LLM narrative

Returns a dict ready for the Flask /agent/summarize endpoint.
"""

import json
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyPDFLoader

from config.settings import DATASETS_DIR, GENERATION_MODEL_NAME, UPLOAD_DIR
from services.ingestion import semantic_chunk_text

load_dotenv()

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _get_llm():
    return init_chat_model(GENERATION_MODEL_NAME)


def _clean_json(text: str) -> str:
    """Strip markdown fences that models sometimes wrap around JSON."""
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        # parts[1] is the content between first pair of fences
        text = parts[1].lstrip("json").strip()
    return text


def _safe_parse(text: str, fallback_title: str) -> Dict:
    try:
        return json.loads(_clean_json(text))
    except Exception:
        return {
            "title": fallback_title,
            "executive_summary": text[:600],
            "key_points": [],
            "themes": [],
            "recommendation": "",
        }


# ─────────────────────────────────────────────────────────────
# PDF path
# ─────────────────────────────────────────────────────────────

def _summarize_chunk(llm, chunk_text: str, idx: int, total: int) -> str:
    """Ask the LLM to summarize a single chunk."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise document summarizer. "
                "Summarize the provided section concisely, preserving all key facts, "
                "figures, names, and insights. Output the summary only — no preamble, "
                "no labels, no markdown."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Section {idx} of {total}:\n\n{chunk_text}\n\n"
                "Write a concise summary of this section in 3–6 sentences."
            ),
        },
    ]
    return llm.invoke(messages).content.strip()


def _merge_pdf_summaries(
    llm, chunk_summaries: List[str], user_query: str, filename: str
) -> Dict:
    """Merge per-chunk summaries into a structured final report."""
    numbered = "\n\n".join(
        f"[Section {i + 1}]\n{s}" for i, s in enumerate(chunk_summaries)
    )
    focus = user_query if user_query else "Provide a comprehensive overview."

    prompt = f"""You are an expert analyst. Below are section-by-section summaries of "{filename}".
User's focus: {focus}

Section Summaries:
{numbered}

Produce a JSON response — absolutely no markdown, pure JSON — with this exact structure:
{{
  "title": "A concise, descriptive title for this document",
  "executive_summary": "A clear 3–5 sentence executive summary covering the most important points",
  "key_points": [
    "Key point 1",
    "Key point 2",
    "Key point 3",
    "Key point 4",
    "Key point 5"
  ],
  "themes": ["Theme 1", "Theme 2", "Theme 3"],
  "recommendation": "One actionable insight, conclusion, or next step"
}}"""

    response = llm.invoke([{"role": "user", "content": prompt}])
    return _safe_parse(response.content, filename)


def _run_pdf_summarization(llm, filename: str, user_query: str) -> Dict[str, Any]:
    pdf_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(pdf_path):
        return {"success": False, "error": f"'{filename}' not found in the knowledge base."}

    # Load all pages
    pages = PyPDFLoader(pdf_path).load()
    if not pages:
        return {"success": False, "error": "Could not extract any pages from the PDF."}

    full_text = "\n\n".join(p.page_content for p in pages if p.page_content.strip())
    if not full_text.strip():
        return {"success": False, "error": "PDF contains no extractable text."}

    total_words = len(full_text.split())

    # Use the same production-grade chunking already wired into the RAG pipeline
    chunks = semantic_chunk_text(full_text)
    if not chunks:
        return {"success": False, "error": "Text could not be split into chunks."}

    # ── Per-chunk summarisation ──
    chunk_records: List[Dict] = []
    for i, chunk in enumerate(chunks):
        summary = _summarize_chunk(llm, chunk, i + 1, len(chunks))
        chunk_records.append(
            {
                "section": i + 1,
                "summary": summary,
                "word_count": len(chunk.split()),
            }
        )

    # ── Merge all chunk summaries ──
    raw_summaries = [c["summary"] for c in chunk_records]
    merged = _merge_pdf_summaries(llm, raw_summaries, user_query, filename)

    return {
        "success": True,
        "file_type": "pdf",
        "filename": filename,
        "total_pages": len(pages),
        "total_chunks": len(chunks),
        "total_words": total_words,
        "chunk_summaries": chunk_records,
        **merged,
    }


# ─────────────────────────────────────────────────────────────
# CSV / Excel path
# ─────────────────────────────────────────────────────────────

def _run_tabular_summarization(llm, filename: str, user_query: str) -> Dict[str, Any]:
    import pandas as pd

    filepath = os.path.join(DATASETS_DIR, filename)
    if not os.path.exists(filepath):
        return {"success": False, "error": f"Dataset '{filename}' not found."}

    ext = filename.rsplit(".", 1)[-1].lower()
    try:
        df = pd.read_csv(filepath) if ext == "csv" else pd.read_excel(filepath)
    except Exception as exc:
        return {"success": False, "error": f"Could not read file: {exc}"}

    rows, cols_count = df.shape
    col_types = {col: str(dtype) for col, dtype in df.dtypes.items()}
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    stats_text = ""
    if numeric_cols:
        stats_text = (
            "\nNumeric column statistics:\n"
            + df[numeric_cols].describe().round(2).to_string()
        )

    cat_text = ""
    for col in cat_cols[:5]:
        top = df[col].value_counts().head(5).to_string()
        cat_text += f"\nTop values in '{col}':\n{top}\n"

    sample = df.head(5).to_string(index=False)
    focus = user_query if user_query else "Provide a comprehensive dataset overview."

    prompt = f"""You are an expert data analyst. Summarise the following dataset.

Filename: {filename}
Rows: {rows:,}  |  Columns: {cols_count}
Column names & types: {json.dumps(col_types, indent=2)}
{stats_text}
{cat_text}
Sample rows:
{sample}

User focus: {focus}

Respond with pure JSON — no markdown fences, no preamble:
{{
  "title": "A concise, descriptive title for this dataset",
  "executive_summary": "A 3–5 sentence overview: what the data represents, its scope, and standout characteristics",
  "key_points": [
    "Insight 1",
    "Insight 2",
    "Insight 3",
    "Insight 4",
    "Insight 5"
  ],
  "themes": ["Theme 1", "Theme 2", "Theme 3"],
  "recommendation": "One actionable recommendation or next analytical step"
}}"""

    response = llm.invoke([{"role": "user", "content": prompt}])
    parsed = _safe_parse(response.content, filename)

    return {
        "success": True,
        "file_type": "tabular",
        "filename": filename,
        "total_rows": rows,
        "total_cols": cols_count,
        "total_chunks": 0,
        "chunk_summaries": [],
        **parsed,
    }


# ─────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────

def run_summarization_agent(query: str, filename: str) -> Dict[str, Any]:
    """
    Dispatches to the correct summarisation path based on file extension.

    PDF  → knowledge-base RAG uploads  (UPLOAD_DIR)
    CSV / XLSX / XLS → datasets folder (DATASETS_DIR)
    """
    if not filename:
        return {"success": False, "error": "No filename provided."}

    llm = _get_llm()
    ext = filename.rsplit(".", 1)[-1].lower()

    if ext in ("csv", "xlsx", "xls"):
        return _run_tabular_summarization(llm, filename, query)

    if ext == "pdf":
        return _run_pdf_summarization(llm, filename, query)

    return {
        "success": False,
        "error": f"Unsupported file type '.{ext}'. Use PDF, CSV, or Excel files.",
    }