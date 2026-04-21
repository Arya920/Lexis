import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

from config.settings import UPLOAD_DIR
from services.ingestion import process_pdf
from services.retrieval import retrieve_docs
from services.generation import AnswerGenerator
from services.tavily_search import TavilySearch


# =========================
# App Initialization
# =========================
def create_app():
    app = Flask(__name__)

    # Initialize services (singleton style)
    generator = AnswerGenerator()
    tavily = TavilySearch()

    # =========================
    # Routes
    # =========================

    @app.route("/")
    def index():
        return render_template("index3.html")

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

    # =========================
    # Upload + Index
    # =========================
    @app.route("/upload", methods=["POST"])
    def upload_file():
        try:
            file = request.files.get("file")

            if not file or file.filename == "":
                return jsonify({"error": "No file provided"}), 400

            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_DIR, filename)

            file.save(file_path)

            result = process_pdf(file_path)

            return jsonify({
                "message": "File processed",
                "filename": filename,
                "chunks": result["chunks"]
            })

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # =========================
    # Chat Endpoint
    # =========================
    @app.route("/chat", methods=["POST"])
    def chat():
        try:
            data = request.json or {}

            query = data.get("message", "").strip()
            rag_mode = data.get("rag", False)
            web_search_mode = data.get("web_search", False)

            if not query:
                return jsonify({"response": "Empty query received.", "sources": []})

            # -------------------------
            # CASE 1: Web Search Mode
            # -------------------------
            if web_search_mode and not rag_mode:
                web_result = tavily.search(query)

                context = web_result.get("answer", "")
                context = " ".join(context.split())  # trim
                print("Web Context :",context)

                sources = []
                if web_result.get("source"):
                    sources.append({
                        "label": web_result["source"]
                    })

                answer = generator.generate_web(query, context)

                return jsonify({
                    "response": answer,
                    "sources": sources
                })

            # -------------------------
            # CASE 2: RAG Mode
            # -------------------------
            if rag_mode:
                docs = retrieve_docs(query)

                if not docs:
                    return jsonify({
                        "response": "No relevant documents found.",
                        "sources": []
                    })

                context_blocks = []
                sources = []

                for index, doc in enumerate(docs, start=1):
                    source = doc.metadata.get("source", "Unknown source")
                    page = doc.metadata.get("page", "?")
                    rerank_score = doc.metadata.get("rerank_score", 0)

                    context_blocks.append(
                        f"[Chunk {index} | {source} | page {page} | rerank {rerank_score}]\n{doc.page_content}"
                    )

                    sources.append({
                        "source": source,
                        "page": page,
                        "rerank_score": rerank_score,
                        "hybrid_score": doc.metadata.get("hybrid_score", 0),
                        "semantic_score": doc.metadata.get("semantic_score", 0),
                        "keyword_score": doc.metadata.get("keyword_score", 0),
                        "label": f"{source} (p.{page})"
                    })

                context = "\n\n".join(context_blocks)

                answer = generator.generate_rag(query, context)

                return jsonify({
                    "response": answer,
                    "context_used": context,
                    "sources": sources
                })

            # -------------------------
            # CASE 3: Direct LLM Mode
            # -------------------------
            answer = generator.generate_direct(query)

            return jsonify({
                "response": answer,
                "sources": []
            })

        except Exception as e:
            return jsonify({
                "response": f"⚠ Error: {str(e)}",
                "sources": []
            }), 500

    return app


# =========================
# Run Server
# =========================
if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)