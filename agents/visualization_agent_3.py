import os
import pandas as pd
from agents.viz_engine import build_schema, generate_plan, validate_plan, ExecutionEngine

DATASETS_DIR = os.path.join("data", "datasets")


# ─────────────────────────────────────────────
# DATA LOADER
# ─────────────────────────────────────────────
def load_dataset(filename: str) -> pd.DataFrame:
    path = os.path.join(DATASETS_DIR, filename)

    if filename.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    # ✅ CLEAN COLUMN NAMES (CRITICAL)
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    return df


# ─────────────────────────────────────────────
# MAIN AGENT FUNCTION
# ─────────────────────────────────────────────
def run_visualization_agent(query: str, filename: str):

    try:
        df = load_dataset(filename)

        schema = build_schema(df)

        plan = generate_plan(query, schema)
        validate_plan(plan, df)   # ✅ ADD THIS


        engine = ExecutionEngine(df)
        figure = engine.run(plan)

        return {
            "success": True,
            "figure": figure,
            "plan": plan,
            "rows": df.shape[0],
            "columns": list(df.columns),
            "filename": filename
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }