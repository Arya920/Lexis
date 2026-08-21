import os
from flask import Flask, render_template

from config.settings import UPLOAD_DIR, DATASETS_DIR, GENERATION_MODEL_NAME
from services.generation import AnswerGenerator
from services.tavily_search import TavilySearch


# =========================
# App Initialization
# =========================
def create_app():
    app = Flask(__name__)

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(DATASETS_DIR, exist_ok=True)

    # ── Shared app-level state ──────────────────────────────────────────────
    # These are injected into each blueprint via app.config so that all apps
    # share one generator/search instance without re-initialising on import.
    app.config["MODEL_HOLDER"]     = {"name": GENERATION_MODEL_NAME}
    app.config["GENERATOR_HOLDER"] = {"gen": AnswerGenerator()}
    app.config["TAVILY"]           = TavilySearch()

    # ── Register blueprints ─────────────────────────────────────────────────
    # Each AI app lives in apps/<name>/routes.py and registers itself here.
    # To add a new app, import its blueprint and add one register_blueprint line.
    from apps.lexis.routes import lexis_bp
    app.register_blueprint(lexis_bp)

    # Future apps — as simple as:
    # from apps.nova.routes import nova_bp
    # app.register_blueprint(nova_bp)

    # ── Nexus homepage ──────────────────────────────────────────────────────
    @app.route("/")
    def home():
        """Serve the Nexus AI platform homepage."""
        return render_template("home.html")

    return app


# =========================
# Run Server
# =========================
if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)