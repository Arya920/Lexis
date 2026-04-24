# agents/visualization_agent.py
"""
Visualization Agent for Lexis
──────────────────────────────────────────────────────────────────
Flow:
  1. Receive query + filename from Flask route
  2. Load the dataset (CSV or Excel) from data/datasets/
  3. Build a schema summary (columns, dtypes, sample rows)
  4. Ask the LLM to produce a Plotly figure as valid JSON
  5. Validate & sanitize the JSON (no exec, no eval)
  6. Return the Plotly figure dict + a plain-English summary

The frontend renders the figure dict using Plotly.js.
──────────────────────────────────────────────────────────────────
"""

import os
import json
import re
import traceback

import pandas as pd
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────
DATASETS_DIR = os.path.join("data", "datasets")

# Re-use the same LLM used by AnswerGenerator
# Import from config so it's always in sync
try:
    from config.settings import GENERATION_MODEL_NAME
except ImportError:
    GENERATION_MODEL_NAME = "groq:llama-3.3-70b-versatile"

print(GENERATION_MODEL_NAME)

_MAX_SAMPLE_ROWS = 5      # rows shown to LLM for context
_MAX_UNIQUE_VALS = 20     # max unique values shown per column


# ── Dataset loader ─────────────────────────────────────────────────
def load_dataset(filename: str) -> pd.DataFrame:
    """Load CSV or Excel file from the datasets directory."""
    path = os.path.join(DATASETS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset '{filename}' not found in {DATASETS_DIR}/")

    ext = filename.rsplit(".", 1)[-1].lower()
    if ext == "csv":
        return pd.read_csv(path)
    elif ext in ("xlsx", "xls"):
        return pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file type: .{ext}  (only CSV and Excel are supported)")


# ── Schema builder ─────────────────────────────────────────────────
def build_schema_summary(df: pd.DataFrame) -> str:
    """
    Produce a compact, LLM-readable description of the dataframe:
      - shape
      - column names, dtypes
      - sample unique values for categorical columns
      - numeric range for numeric columns
      - first N sample rows as a markdown table
    """
    lines = []
    lines.append(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")

    lines.append("Columns:")
    for col in df.columns:
        dtype = str(df[col].dtype)
        n_null = int(df[col].isna().sum())
        if pd.api.types.is_numeric_dtype(df[col]):
            info = f"numeric | min={df[col].min():.4g}, max={df[col].max():.4g}, mean={df[col].mean():.4g}"
        else:
            uniq = df[col].dropna().unique()
            if len(uniq) <= _MAX_UNIQUE_VALS:
                info = f"categorical | unique values: {list(uniq[:_MAX_UNIQUE_VALS])}"
            else:
                info = f"categorical | {len(uniq)} unique values, e.g. {list(uniq[:5])}"
        lines.append(f"  • {col!r}  [{dtype}]  nulls={n_null}  — {info}")

    lines.append(f"\nFirst {_MAX_SAMPLE_ROWS} rows (markdown):")
    lines.append(df.head(_MAX_SAMPLE_ROWS).to_markdown(index=False))

    return "\n".join(lines)


# ── System prompt ──────────────────────────────────────────────────
_SYSTEM_PROMPT = """You are a data visualization expert.

You will receive:
  1. A dataset schema (columns, dtypes, sample rows)
  2. A user request describing a chart they want

Your job is to output a SINGLE valid JSON object that represents a Plotly figure.

The JSON must be a Plotly figure dict with two top-level keys:
  - "data"   → list of trace dicts (e.g. go.Bar, go.Scatter, go.Pie, go.Histogram, etc.)
  - "layout" → layout dict (title, xaxis, yaxis, etc.)

STRICT RULES:
1. Output ONLY the raw JSON — no markdown, no backticks, no explanation before or after.
2. Do NOT use Python code or executable code anywhere in your response.
3. Use exact column names from the schema — do not invent column names.
4. For aggregations (e.g., "average salary by age band"), compute the aggregation
   by producing the aggregated x and y arrays DIRECTLY in the JSON using the raw values
   that would result from that computation. Do NOT use formulas.
5. Choose the most appropriate chart type based on the request.
6. Always include a descriptive title in layout.title.text
7. Always include axis labels: layout.xaxis.title.text and layout.yaxis.title.text
   (skip yaxis label for pie charts)
8. Use a clean, professional color scheme.
9. The JSON must be parseable by json.loads() — no trailing commas, no comments.

Example valid output structure:
{
  "data": [
    {
      "type": "bar",
      "x": ["A", "B", "C"],
      "y": [10, 25, 15],
      "marker": {"color": "#818cf8"}
    }
  ],
  "layout": {
    "title": {"text": "My Chart"},
    "xaxis": {"title": {"text": "Category"}},
    "yaxis": {"title": {"text": "Value"}},
    "plot_bgcolor": "rgba(0,0,0,0)",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": "#f2f2f2"}
  }
}
"""


# ── LLM caller ────────────────────────────────────────────────────
def _call_llm(schema_summary: str, user_query: str, df: pd.DataFrame) -> dict:
    """
    Call the LLM with schema + query.
    Returns the parsed Plotly figure dict.
    """
    llm = init_chat_model(GENERATION_MODEL_NAME)

    user_message = f"""Dataset schema:
{schema_summary}

User request:
{user_query}

Remember: output ONLY raw JSON. No markdown, no explanation."""

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()

    # Strip any accidental markdown code fences the LLM added
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()

    try:
        figure_dict = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"LLM returned invalid JSON: {e}\n\nRaw response (first 500 chars):\n{raw[:500]}"
        )

    return figure_dict


# ── Figure validator ───────────────────────────────────────────────
def _validate_figure(figure_dict: dict) -> dict:
    """
    Basic sanity checks on the Plotly figure dict.
    Adds transparent background so it blends with the UI.
    """
    if not isinstance(figure_dict, dict):
        raise ValueError("Figure must be a JSON object (dict)")
    if "data" not in figure_dict:
        raise ValueError("Figure JSON missing required key: 'data'")
    if not isinstance(figure_dict["data"], list):
        raise ValueError("'data' must be a list of trace objects")

    # Ensure layout exists
    figure_dict.setdefault("layout", {})
    layout = figure_dict["layout"]

    # Transparent backgrounds so chart blends into the dark/light UI
    layout.setdefault("plot_bgcolor",  "rgba(0,0,0,0)")
    layout.setdefault("paper_bgcolor", "rgba(0,0,0,0)")
    layout.setdefault("font", {}).setdefault("color", "#f2f2f2")

    # Clean margins
    layout.setdefault("margin", {"t": 60, "r": 20, "b": 60, "l": 60})

    # Responsive
    figure_dict.setdefault("config", {
        "responsive": True,
        "displayModeBar": True,
        "modeBarButtonsToRemove": ["toImage"],
    })

    return figure_dict


# ── Summary generator ──────────────────────────────────────────────
def _generate_summary(user_query: str, figure_dict: dict, df: pd.DataFrame) -> str:
    """
    Generate a short plain-English summary of what the chart shows.
    Uses the LLM for a conversational 1-2 sentence interpretation.
    """
    llm = init_chat_model(GENERATION_MODEL_NAME)

    # Pull chart title if available
    title = (
        figure_dict.get("layout", {})
        .get("title", {})
        .get("text", "the chart")
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful data analyst. "
                "Write 1-2 concise, conversational sentences describing what the chart shows. "
                "Do not mention Plotly or technical details. "
                "Be insightful — mention the key trend or takeaway if obvious."
            ),
        },
        {
            "role": "user",
            "content": (
                f"The user asked: '{user_query}'\n"
                f"A chart titled '{title}' was generated from a dataset "
                f"with {df.shape[0]} rows and columns: {list(df.columns)}.\n"
                "Write a short summary of what this chart likely shows."
            ),
        },
    ]

    response = llm.invoke(messages)
    return response.content.strip()


# ── Main entry point ───────────────────────────────────────────────
def run_visualization_agent(query: str, filename: str) -> dict:
    """
    Main entry point called by the Flask route.

    Args:
        query    : User's natural language chart request
        filename : Dataset filename (must exist in data/datasets/)

    Returns dict with keys:
        success      : bool
        figure       : Plotly figure dict (for Plotly.js on frontend)
        summary      : Plain-English description of the chart
        filename     : Echo back the filename used
        error        : str (only present if success=False)
    """
    try:
        # 1. Load dataset
        df = load_dataset(filename)

        # 2. Build schema summary for LLM
        schema = build_schema_summary(df)

        # 3. Ask LLM to generate Plotly JSON
        raw_figure = _call_llm(schema, query, df)

        # 4. Validate and apply UI theme
        figure = _validate_figure(raw_figure)

        # 5. Generate a short text summary
        summary = _generate_summary(query, figure, df)

        return {
            "success":  True,
            "figure":   figure,
            "summary":  summary,
            "filename": filename,
            "rows":     df.shape[0],
            "columns":  list(df.columns),
        }

    except FileNotFoundError as e:
        return {"success": False, "error": str(e)}
    except ValueError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {
            "success": False,
            "error":   f"Unexpected error: {str(e)}",
            "detail":  traceback.format_exc(),
        }