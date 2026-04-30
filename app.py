import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

from config.settings import UPLOAD_DIR, CHUNK_STORE_PATH, VECTOR_DB_PATH, DATASETS_DIR
from services.ingestion import process_pdf, _save_chunk_store, _build_documents
from services.retrieval import retrieve_docs
from services.generation import AnswerGenerator
from services.tavily_search import TavilySearch
from services.embeddings import get_embedding_model
from agents.data_visualization_agent import run_visualization_agent
from agents.data_analysis_agent import run_data_analysis_agent
from agents.Summarization_agent import run_summarization_agent
from langchain_community.vectorstores import FAISS

from config.settings import GENERATION_MODEL_NAME


# =========================
# App Initialization
# =========================
def create_app():
    app = Flask(__name__)

    os.makedirs(UPLOAD_DIR,   exist_ok=True)
    os.makedirs(DATASETS_DIR, exist_ok=True)

    _model_holder     = {"name": GENERATION_MODEL_NAME}
    _generator_holder = {"gen": AnswerGenerator()}
    tavily            = TavilySearch()

    # ──────────────────────────────────────────────
    # Homepage & App Routes
    # ──────────────────────────────────────────────

    @app.route("/")
    def home():
        """Serve the main Nexus AI platform homepage."""
        return render_template("home.html")

    @app.route("/lexis")
    def lexis():
        """Serve the Lexis RAG & Agentic chatbot application."""
        return render_template("lexis/index.html")

    # ──────────────────────────────────────────────
    # Model Management
    # ──────────────────────────────────────────────

    @app.route("/model", methods=["GET"])
    def get_model():
        full  = _model_holder["name"]
        short = full.split(":", 1)[-1]
        return jsonify({"model": short, "full": full})

    @app.route("/model", methods=["POST"])
    def set_model():
        try:
            data       = request.json or {}
            short_name = data.get("model", "").strip()
            if not short_name:
                return jsonify({"error": "model name required"}), 400

            full_name = f"groq:{short_name}"
            new_gen   = AnswerGenerator(model_name=full_name)

            _generator_holder["gen"] = new_gen
            _model_holder["name"]    = full_name

            return jsonify({"model": short_name, "full": full_name, "message": "Model switched"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # ──────────────────────────────────────────────
    # RAG Document File Management
    # ──────────────────────────────────────────────

    @app.route("/files", methods=["GET"])
    def list_files():
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
        try:
            file = request.files.get("file")
            if not file or file.filename == "":
                return jsonify({"error": "No file provided"}), 400

            filename  = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_DIR, filename)
            file.save(file_path)

            result = process_pdf(file_path)
            return jsonify({"message": "File processed", "filename": filename, "chunks": result["chunks"]})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/remove-file", methods=["POST"])
    def remove_file():
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

                filtered_chunks = [c for c in chunks if c["metadata"].get("source") != filename]
                _save_chunk_store(filtered_chunks)

                if filtered_chunks:
                    documents       = _build_documents(filtered_chunks)
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
    # Dataset File Management (Agents)
    # ──────────────────────────────────────────────

    @app.route("/datasets", methods=["GET"])
    def list_datasets():
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
            return jsonify({"message": "Dataset saved", "filename": filename})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/remove-dataset", methods=["POST"])
    def remove_dataset():
        try:
            data     = request.json
            filename = data.get("filename")
            if not filename:
                return jsonify({"error": "Filename required"}), 400

            file_path = os.path.join(DATASETS_DIR, secure_filename(filename))
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({"message": f"{filename} removed"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # ──────────────────────────────────────────────
    # Agent Endpoints
    # ──────────────────────────────────────────────

    @app.route("/agent/visualize", methods=["POST"])
    def agent_visualize():
        try:
            data     = request.json or {}
            query    = data.get("message", "").strip()
            filename = data.get("filename", "").strip()
            if not query:
                return jsonify({"success": False, "error": "No query provided"}), 400
            if not filename:
                return jsonify({"success": False, "error": "No dataset filename provided."}), 400
            result = run_visualization_agent(query=query, filename=filename)
            return jsonify(result)
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/agent/analyze", methods=["POST"])
    def agent_analyze():
        try:
            data     = request.json or {}
            query    = data.get("message", "").strip()
            filename = data.get("filename", "").strip()
            if not query:
                return jsonify({"success": False, "error": "No query provided"}), 400
            if not filename:
                return jsonify({"success": False, "error": "No dataset filename provided."}), 400
            result = run_data_analysis_agent(query=query, filename=filename)
            return jsonify(result)
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/agent/summarize", methods=["POST"])
    def agent_summarize():
        try:
            data     = request.json or {}
            query    = data.get("message", "").strip()
            filename = data.get("filename", "").strip()
            if not filename:
                return jsonify({"success": False, "error": "No filename provided."}), 400
            result = run_summarization_agent(query=query, filename=filename)
            return jsonify(result)
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    # ──────────────────────────────────────────────
    # Main Chat Endpoint
    # ──────────────────────────────────────────────

    @app.route("/chat", methods=["POST"])
    def chat():
        try:
            data            = request.json or {}
            query           = data.get("message", "").strip()
            rag_mode        = data.get("rag", False)
            web_search_mode = data.get("web_search", False)

            if not query:
                return jsonify({"response": "Empty query received.", "sources": []})

            if web_search_mode and not rag_mode:
                web_result = tavily.search(query)
                context    = " ".join(web_result.get("answer", "").split())
                sources    = []
                if web_result.get("source"):
                    sources.append({"label": web_result["source"]})
                answer = _generator_holder["gen"].generate_web(query, context)
                return jsonify({"response": answer, "sources": sources})

            if rag_mode:
                docs = retrieve_docs(query)
                if not docs:
                    return jsonify({"response": "No relevant documents found.", "sources": []})

                context_blocks = []
                sources        = []
                for index, doc in enumerate(docs, start=1):
                    source       = doc.metadata.get("source", "Unknown source")
                    page         = doc.metadata.get("page", "?")
                    rerank_score = doc.metadata.get("rerank_score", 0)
                    context_blocks.append(
                        f"[Chunk {index} | {source} | page {page} | rerank {rerank_score}]\n{doc.page_content}"
                    )
                    sources.append({
                        "source":         source,
                        "page":           page,
                        "rerank_score":   rerank_score,
                        "hybrid_score":   doc.metadata.get("hybrid_score", 0),
                        "semantic_score": doc.metadata.get("semantic_score", 0),
                        "keyword_score":  doc.metadata.get("keyword_score", 0),
                        "label":          f"{source} (p.{page})",
                    })

                context = "\n\n".join(context_blocks)
                answer  = _generator_holder["gen"].generate_rag(query, context)
                return jsonify({"response": answer, "context_used": context, "sources": sources})

            answer = _generator_holder["gen"].generate_direct(query)
            return jsonify({"response": answer, "sources": []})

        except Exception as e:
            return jsonify({"response": f"⚠ Error: {str(e)}", "sources": []}), 500

    return app


# =========================
# Run Server
# =========================
if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)