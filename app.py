import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

from config.settings import (
    UPLOAD_DIR,
    CHUNK_STORE_PATH,
    VECTOR_DB_PATH,
    DATASETS_DIR,
    GENERATION_MODEL_NAME,
)
from services.ingestion import process_pdf, _save_chunk_store, _build_documents
from services.retrieval import retrieve_docs
from services.generation import AnswerGenerator
from services.tavily_search import TavilySearch
from services.embeddings import get_embedding_model
from services.query_logging import begin_query_log, reset_query_log, write_query_log
from agents.data_visualization_agent import run_visualization_agent
from agents.data_analysis_agent import run_data_analysis_agent
from langchain_community.vectorstores import FAISS


def _visualization_text_output(result):
    if result.get("success"):
        return result.get("summary", "")
    return result.get("error", "")


def _analysis_text_output(result):
    if not result.get("success"):
        return result.get("error", "")

    parts = [
        result.get("headline", ""),
        result.get("narrative", ""),
        "\n".join(result.get("key_findings", [])),
        result.get("recommendation") or "",
    ]
    return "\n\n".join(part for part in parts if part)

# =========================
# App Initialization
# =========================
def create_app():
    app = Flask(__name__)

    # Ensure all data directories exist on startup
    os.makedirs(UPLOAD_DIR,    exist_ok=True)
    os.makedirs(DATASETS_DIR,  exist_ok=True)

    # Initialize services (singleton style)
    generator = AnswerGenerator()
    tavily    = TavilySearch()

    # =========================
    # Routes
    # =========================

    @app.route("/")
    def index():
        return render_template("index4.html")

    # ──────────────────────────────────────────────
    # RAG Document File Management
    # ──────────────────────────────────────────────

    @app.route("/files", methods=["GET"])
    def list_files():
        """List files indexed into RAG (data/uploads/)."""
        try:
            if not os.path.exists(UPLOAD_DIR):
                return jsonify({"files": []})
            files = [
                f for f in os.listdir(UPLOAD_DIR)
                if os.path.isfile(os.path.join(UPLOAD_DIR, f))
            ]
            return jsonify({"files": sorted(files)})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/upload", methods=["POST"])
    def upload_file():
        """Upload and index a document into RAG (data/uploads/)."""
        try:
            file = request.files.get("file")
            if not file or file.filename == "":
                return jsonify({"error": "No file provided"}), 400

            filename  = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_DIR, filename)
            file.save(file_path)

            result = process_pdf(file_path)

            return jsonify({
                "message":  "File processed",
                "filename": filename,
                "chunks":   result["chunks"],
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/remove-file", methods=["POST"])
    def remove_file():
        """Remove a RAG-indexed document and rebuild the FAISS index."""
        try:
            data     = request.json
            filename = data.get("filename")
            if not filename:
                return jsonify({"error": "Filename required"}), 400

            file_path = os.path.join(UPLOAD_DIR, filename)
            if os.path.exists(file_path):
                os.remove(file_path)

            if os.path.exists(CHUNK_STORE_PATH):
                import json
                with open(CHUNK_STORE_PATH, "r", encoding="utf-8") as f:
                    chunks = json.load(f)

                filtered_chunks = [
                    c for c in chunks
                    if c["metadata"].get("source") != filename
                ]
                _save_chunk_store(filtered_chunks)

                if filtered_chunks:
                    documents      = _build_documents(filtered_chunks)
                    embedding_model = get_embedding_model()
                    db = FAISS.from_documents(documents, embedding_model)
                    db.save_local(VECTOR_DB_PATH)
                else:
                    import shutil
                    shutil.rmtree(VECTOR_DB_PATH)
                    os.makedirs(VECTOR_DB_PATH, exist_ok=True)

            return jsonify({"message": f"{filename} removed successfully"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # ──────────────────────────────────────────────
    # Dataset File Management (for Agents)
    # These files are NOT indexed into RAG
    # ──────────────────────────────────────────────

    @app.route("/datasets", methods=["GET"])
    def list_datasets():
        """List dataset files available for agents (data/datasets/)."""
        try:
            if not os.path.exists(DATASETS_DIR):
                return jsonify({"files": []})
            files = [
                f for f in os.listdir(DATASETS_DIR)
                if os.path.isfile(os.path.join(DATASETS_DIR, f))
                and f.rsplit(".", 1)[-1].lower() in ("csv", "xlsx", "xls")
            ]
            return jsonify({"files": sorted(files)})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/upload-dataset", methods=["POST"])
    def upload_dataset():
        """
        Save a CSV or Excel file to data/datasets/.
        Does NOT run ingestion — file is only used by agents.
        """
        try:
            file = request.files.get("file")
            if not file or file.filename == "":
                return jsonify({"error": "No file provided"}), 400

            filename = secure_filename(file.filename)
            ext      = filename.rsplit(".", 1)[-1].lower()

            if ext not in ("csv", "xlsx", "xls"):
                return jsonify({"error": "Only CSV and Excel files are supported"}), 400

            file_path = os.path.join(DATASETS_DIR, filename)
            file.save(file_path)

            return jsonify({
                "message":  "Dataset saved",
                "filename": filename,
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/remove-dataset", methods=["POST"])
    def remove_dataset():
        """Delete a dataset file from data/datasets/."""
        try:
            data     = request.json
            filename = data.get("filename")
            if not filename:
                return jsonify({"error": "Filename required"}), 400

            file_path = os.path.join(DATASETS_DIR, secure_filename(filename))
            if os.path.exists(file_path):
                os.remove(file_path)
                return jsonify({"message": f"{filename} removed"})
            else:
                return jsonify({"error": "File not found"}), 404
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # ──────────────────────────────────────────────
    # Agent Endpoints
    # ──────────────────────────────────────────────

    @app.route("/agent/visualize", methods=["POST"])
    def agent_visualize():
        """
        Visualization agent — generates an interactive Plotly chart.

        POST body:
          {
            "message":  "create a histogram of age_band vs salary",
            "filename": "employees.xlsx"
          }

        Response:
          {
            "success":  true,
            "figure":   { ...Plotly figure dict... },
            "summary":  "This chart shows ...",
            "filename": "employees.xlsx",
            "rows":     500,
            "columns":  ["age_band", "salary", ...]
          }
        """
        try:
            data     = request.json or {}
            query    = data.get("message", "").strip()
            filename = data.get("filename", "").strip()

            if not query:
                return jsonify({"success": False, "error": "No query provided"}), 400
            if not filename:
                return jsonify({
                    "success": False,
                    "error": "No dataset filename provided. Please upload a dataset first using the sidebar."
                }), 400

            log_token = begin_query_log()
            result = run_visualization_agent(query=query, filename=filename)
            write_query_log(
                query=query,
                endpoint="/agent/visualize",
                use_case="data_visualization",
                response_text=_visualization_text_output(result),
                status="success" if result.get("success") else "error",
                model_name=GENERATION_MODEL_NAME,
                metadata={
                    "filename": filename,
                    "rows": result.get("rows"),
                    "columns": result.get("columns"),
                    "chart_type": (result.get("plan") or {}).get("chart", {}).get("type"),
                },
            )
            reset_query_log(log_token)
            log_token = None
            return jsonify(result)

        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500
        
    @app.route("/agent/analyze", methods=["POST"])          # ← NEW ROUTE
    def agent_analyze():
        """
        Data Analysis Agent — answers analytical questions about a dataset.
 
        POST body:
          { "message": "What is the average profit per region?", "filename": "sales.csv" }
 
        Response:
          {
            "success":        true,
            "headline":       "South leads profit with $142K avg.",
            "narrative":      "...",
            "key_findings":   ["Finding 1", "Finding 2", ...],
            "recommendation": "...",
            "stats_table":    [{"label": "...", "value": "...", "note": "..."}, ...],
            "primary_table":  {"label": "...", "columns": [...], "rows": [...]},
            "operations":     [...],
            "filename":       "sales.csv",
            "rows":           9994,
            "columns":        [...]
          }
        """
        try:
            data     = request.json or {}
            query    = data.get("message", "").strip()
            filename = data.get("filename", "").strip()
 
            if not query:
                return jsonify({"success": False, "error": "No query provided"}), 400
            if not filename:
                return jsonify({
                    "success": False,
                    "error": "No dataset filename provided. Please upload a dataset first using the sidebar."
                }), 400
 
            log_token = begin_query_log()
            result = run_data_analysis_agent(query=query, filename=filename)
            write_query_log(
                query=query,
                endpoint="/agent/analyze",
                use_case="data_analysis",
                response_text=_analysis_text_output(result),
                status="success" if result.get("success") else "error",
                model_name=GENERATION_MODEL_NAME,
                metadata={
                    "filename": filename,
                    "rows": result.get("rows"),
                    "columns": result.get("columns"),
                    "operations": result.get("operations", []),
                },
            )
            reset_query_log(log_token)
            log_token = None
            return jsonify(result)
 
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    # ──────────────────────────────────────────────
    # Main Chat Endpoint
    # ──────────────────────────────────────────────

    @app.route("/chat", methods=["POST"])
    def chat():
        log_token = None
        query = ""
        try:
            data            = request.json or {}
            query           = data.get("message", "").strip()
            rag_mode        = data.get("rag", False)
            web_search_mode = data.get("web_search", False)

            if not query:
                return jsonify({"response": "Empty query received.", "sources": []})

            log_token = begin_query_log()

            # CASE 1: Web Search Mode
            if web_search_mode and not rag_mode:
                web_result = tavily.search(query)
                context    = " ".join(web_result.get("answer", "").split())
                sources    = []
                if web_result.get("source"):
                    sources.append({"label": web_result["source"]})
                answer = generator.generate_web(query, context)
                write_query_log(
                    query=query,
                    endpoint="/chat",
                    use_case="web_search_chat",
                    response_text=answer,
                    model_name=GENERATION_MODEL_NAME,
                    metadata={"sources": sources},
                )
                reset_query_log(log_token)
                log_token = None
                return jsonify({"response": answer, "sources": sources})

            # CASE 2: RAG Mode
            if rag_mode:
                docs = retrieve_docs(query)
                if not docs:
                    answer = "No relevant documents found."
                    write_query_log(
                        query=query,
                        endpoint="/chat",
                        use_case="rag_chat",
                        response_text=answer,
                        status="no_results",
                        model_name=GENERATION_MODEL_NAME,
                        metadata={"sources": []},
                    )
                    reset_query_log(log_token)
                    log_token = None
                    return jsonify({"response": "No relevant documents found.", "sources": []})

                context_blocks = []
                sources        = []
                for index, doc in enumerate(docs, start=1):
                    source        = doc.metadata.get("source", "Unknown source")
                    page          = doc.metadata.get("page", "?")
                    rerank_score  = doc.metadata.get("rerank_score", 0)
                    context_blocks.append(
                        f"[Chunk {index} | {source} | page {page} | rerank {rerank_score}]\n{doc.page_content}"
                    )
                    sources.append({
                        "source":        source,
                        "page":          page,
                        "rerank_score":  rerank_score,
                        "hybrid_score":  doc.metadata.get("hybrid_score", 0),
                        "semantic_score": doc.metadata.get("semantic_score", 0),
                        "keyword_score": doc.metadata.get("keyword_score", 0),
                        "label":         f"{source} (p.{page})",
                    })

                context = "\n\n".join(context_blocks)
                answer  = generator.generate_rag(query, context)
                write_query_log(
                    query=query,
                    endpoint="/chat",
                    use_case="rag_chat",
                    response_text=answer,
                    model_name=GENERATION_MODEL_NAME,
                    metadata={"sources": sources},
                )
                reset_query_log(log_token)
                log_token = None
                return jsonify({"response": answer, "context_used": context, "sources": sources})

            # CASE 3: Direct LLM Mode
            answer = generator.generate_direct(query)
            write_query_log(
                query=query,
                endpoint="/chat",
                use_case="direct_chat",
                response_text=answer,
                model_name=GENERATION_MODEL_NAME,
                metadata={"sources": []},
            )
            reset_query_log(log_token)
            log_token = None
            return jsonify({"response": answer, "sources": []})

        except Exception as e:
            error_response = f"⚠ Error: {str(e)}"
            if log_token is not None:
                write_query_log(
                    query=query,
                    endpoint="/chat",
                    use_case="chat_error",
                    response_text=error_response,
                    status="error",
                    model_name=GENERATION_MODEL_NAME,
                    metadata={"error": str(e)},
                )
                reset_query_log(log_token)
                log_token = None
            return jsonify({"response": error_response, "sources": []}), 500

    return app


# =========================
# Run Server
# =========================
if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
