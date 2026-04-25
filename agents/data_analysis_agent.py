# agents/data_analysis_agent.py
"""
Data Analysis Agent for Lexis
══════════════════════════════════════════════════════════════════════
Architecture — Two-pass LLM + real pandas execution:

  PASS 1  (Plan)
    LLM receives the dataset schema and user query.
    It returns a JSON "analysis plan" — a list of named operations,
    each specifying which pandas method to call and on which columns.
    No code is exec'd from the LLM; we map operation names to
    whitelisted pandas calls.

  EXECUTE
    The backend runs the whitelisted pandas operations and collects
    the computed results (numbers, tables, ranked lists, etc.)

  PASS 2  (Interpret)
    LLM receives the user query + the actual computed results.
    It writes a rich, structured analytical response in plain English,
    with key findings, patterns and recommendations.

  RETURN
    {
      "success":   true,
      "narrative": "...",          # LLM's full analytical write-up
      "sections":  [...],          # structured sections for the UI card
      "stats_table": [...],        # optional summary table rows
      "filename":  "...",
      "rows": N, "columns": [...]
    }

Safe by design:
  - No eval(), no exec(), no arbitrary code from LLM
  - All operations are whitelisted pandas method calls
  - LLM only sees column names and schema — never raw data
══════════════════════════════════════════════════════════════════════
"""

import os
import json
import re
import math
import traceback
from typing import Any

import numpy as np
import pandas as pd
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from services.query_logging import record_llm_call

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────
DATASETS_DIR = os.path.join("data", "datasets")

try:
    from config.settings import GENERATION_MODEL_NAME
except ImportError:
    GENERATION_MODEL_NAME = "groq:llama-3.3-70b-versatile"

_MAX_SAMPLE_ROWS  = 6
_MAX_UNIQUE_VALS  = 25
_MAX_RESULT_ROWS  = 20   # cap table results sent to LLM


# ── Dataset loader (shared pattern with viz agent) ─────────────────
def load_dataset(filename: str) -> pd.DataFrame:
    path = os.path.join(DATASETS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset '{filename}' not found in {DATASETS_DIR}/")
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext == "csv":
        df = pd.read_csv(path)
    elif ext in ("xlsx", "xls"):
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file type: .{ext}")
    return df


# ── Schema builder ─────────────────────────────────────────────────
def build_schema_summary(df: pd.DataFrame) -> str:
    lines = [f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n", "Columns:"]
    for col in df.columns:
        dtype  = str(df[col].dtype)
        n_null = int(df[col].isna().sum())
        if pd.api.types.is_numeric_dtype(df[col]):
            info = (
                f"numeric | min={df[col].min():.4g}, max={df[col].max():.4g}, "
                f"mean={df[col].mean():.4g}, std={df[col].std():.4g}"
            )
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            info = f"datetime | range: {df[col].min()} → {df[col].max()}"
        else:
            uniq = df[col].dropna().unique()
            if len(uniq) <= _MAX_UNIQUE_VALS:
                info = f"categorical | unique values: {list(uniq[:_MAX_UNIQUE_VALS])}"
            else:
                info = f"categorical | {len(uniq)} unique values, e.g. {list(uniq[:6])}"
        lines.append(f"  • {col!r}  [{dtype}]  nulls={n_null}  — {info}")

    lines.append(f"\nSample rows ({_MAX_SAMPLE_ROWS} rows):")
    lines.append(df.head(_MAX_SAMPLE_ROWS).to_markdown(index=False))
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# PASS 1 — LLM Analysis Planner
# ══════════════════════════════════════════════════════════════════

_PLAN_SYSTEM_PROMPT = """You are a senior data analyst planning an analysis for a junior analyst who will execute it in pandas.

You will receive:
  1. A dataset schema (columns, dtypes, sample rows)
  2. A user's analytical question

Your job is to produce a JSON array of analysis "operations" — a structured plan of what to compute.

Each operation is an object with these fields:
  - "op_id"     : short unique snake_case identifier (e.g. "top_salary", "avg_profit_region")
  - "op_type"   : one of the whitelisted operations listed below
  - "label"     : human-readable description of what this computes (e.g. "Top 5 days by Sales")
  - "col"       : primary column name to operate on (must match schema exactly)
  - "group_by"  : column name to group by (null if not applicable)
  - "n"         : integer — for top_n / bottom_n operations (null otherwise)
  - "col2"      : secondary column for ratio operations (null if not applicable)

WHITELISTED op_types:
  "top_n"           — Top N rows by col (sorted desc), optionally grouped by group_by
  "bottom_n"        — Bottom N rows by col (sorted asc)
  "mean_by_group"   — Mean of col grouped by group_by
  "sum_by_group"    — Sum of col grouped by group_by
  "count_by_group"  — Count of rows grouped by group_by
  "std_by_group"    — Std deviation of col grouped by group_by (for volatility/stability)
  "ratio_by_group"  — Compute col/col2 ratio then mean by group_by (for margins)
  "distribution"    — Percentile summary of col (min, p25, median, p75, max, mean, std)
  "outliers"        — IQR-based outlier detection on col, with group_by column for labeling
  "correlation"     — Correlation matrix of all numeric columns
  "overall_summary" — Full descriptive statistics of all numeric columns
  "value_counts"    — Frequency count of categorical col
  "time_trend"      — Group col by group_by (date column), compute sum/mean of col

RULES:
1. Output ONLY a raw JSON array — no markdown, no backticks, no explanation.
2. Use EXACT column names from the schema.
3. Choose only the operations actually needed to answer the question. Do not add unnecessary operations.
4. Maximum 6 operations per plan.
5. For complex questions (e.g. "which segment should be prioritized for growth"), include
   multiple complementary operations (e.g. mean_by_group + std_by_group + sum_by_group).

Example output for "What is the average profit per region?":
[
  {
    "op_id": "avg_profit_region",
    "op_type": "mean_by_group",
    "label": "Average Profit by Region",
    "col": "Profit",
    "group_by": "Region",
    "n": null,
    "col2": null
  }
]
"""


def _plan_analysis(schema: str, query: str) -> list[dict]:
    """PASS 1 — Ask LLM to produce a structured analysis plan."""
    llm = init_chat_model(GENERATION_MODEL_NAME)
    messages = [
        {"role": "system", "content": _PLAN_SYSTEM_PROMPT},
        {"role": "user",   "content": f"Dataset schema:\n{schema}\n\nUser question:\n{query}\n\nOutput ONLY the raw JSON array."},
    ]
    response = llm.invoke(messages)
    raw = response.content.strip()
    record_llm_call(
        use_case="data_analysis_plan",
        output_text=raw,
        response=response,
        model_name=GENERATION_MODEL_NAME,
    )
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$",          "", raw)
    raw = raw.strip()

    try:
        plan = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM returned invalid JSON plan: {e}\nRaw: {raw[:400]}")

    if not isinstance(plan, list):
        raise ValueError("Plan must be a JSON array")
    return plan


# ══════════════════════════════════════════════════════════════════
# EXECUTE — Whitelisted pandas operations
# ══════════════════════════════════════════════════════════════════

def _safe_val(v: Any) -> Any:
    """Convert numpy/pandas scalars to JSON-safe Python types."""
    if isinstance(v, (np.integer,)):           return int(v)
    if isinstance(v, (np.floating,)):
        if math.isnan(v) or math.isinf(v):    return None
        return round(float(v), 4)
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):    return None
        return round(v, 4)
    if isinstance(v, (np.bool_,)):             return bool(v)
    if pd.isna(v):                             return None
    return v


def _df_to_records(df: pd.DataFrame, max_rows: int = _MAX_RESULT_ROWS) -> list[dict]:
    """Convert a dataframe to a list of JSON-safe dicts."""
    df = df.head(max_rows).copy()
    # Round numeric columns to 4dp
    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].apply(lambda x: round(float(x), 4) if pd.notna(x) and not math.isinf(float(x)) else None)
    records = df.to_dict(orient="records")
    return [{k: _safe_val(v) for k, v in row.items()} for row in records]


def _execute_operation(op: dict, df: pd.DataFrame) -> dict:
    """
    Execute one whitelisted operation and return a result dict.
    Returns: { op_id, label, op_type, result_type, data, columns, error? }
    """
    op_id   = op.get("op_id",   "unnamed")
    op_type = op.get("op_type", "")
    label   = op.get("label",   op_id)
    col     = op.get("col")
    group   = op.get("group_by")
    n       = op.get("n") or 10
    col2    = op.get("col2")

    base = {"op_id": op_id, "label": label, "op_type": op_type}

    try:
        # ── top_n ──────────────────────────────────────────────
        if op_type == "top_n":
            if group:
                result = (
                    df.groupby(group)[col]
                    .mean()
                    .reset_index()
                    .sort_values(col, ascending=False)
                    .head(n)
                )
            else:
                cols_keep = [c for c in [col, group] if c]
                result = df.nlargest(n, col)[list(set(df.columns) & set(cols_keep + [col]))]
                # Include a sensible label column if available
                label_candidates = [c for c in df.columns if df[c].dtype == object and c != col]
                if label_candidates and label_candidates[0] not in result.columns:
                    result = df.nlargest(n, col)[[label_candidates[0], col]]
            return {**base, "result_type": "table", "data": _df_to_records(result), "columns": list(result.columns)}

        # ── bottom_n ───────────────────────────────────────────
        elif op_type == "bottom_n":
            label_candidates = [c for c in df.columns if df[c].dtype == object and c != col]
            if label_candidates:
                result = df.nsmallest(n, col)[[label_candidates[0], col]]
            else:
                result = df.nsmallest(n, col)[[col]]
            return {**base, "result_type": "table", "data": _df_to_records(result), "columns": list(result.columns)}

        # ── mean_by_group ──────────────────────────────────────
        elif op_type == "mean_by_group":
            result = df.groupby(group)[col].mean().reset_index().sort_values(col, ascending=False)
            result[col] = result[col].round(4)
            return {**base, "result_type": "table", "data": _df_to_records(result), "columns": [group, col]}

        # ── sum_by_group ───────────────────────────────────────
        elif op_type == "sum_by_group":
            result = df.groupby(group)[col].sum().reset_index().sort_values(col, ascending=False)
            result[col] = result[col].round(4)
            return {**base, "result_type": "table", "data": _df_to_records(result), "columns": [group, col]}

        # ── count_by_group ─────────────────────────────────────
        elif op_type == "count_by_group":
            result = df.groupby(group).size().reset_index(name="count").sort_values("count", ascending=False)
            return {**base, "result_type": "table", "data": _df_to_records(result), "columns": [group, "count"]}

        # ── std_by_group ───────────────────────────────────────
        elif op_type == "std_by_group":
            result = df.groupby(group)[col].std().reset_index().sort_values(col, ascending=True)
            result.columns = [group, f"{col}_std"]
            result[f"{col}_std"] = result[f"{col}_std"].round(4)
            return {**base, "result_type": "table", "data": _df_to_records(result), "columns": list(result.columns)}

        # ── ratio_by_group ─────────────────────────────────────
        elif op_type == "ratio_by_group":
            if not col2:
                raise ValueError("ratio_by_group requires col2")
            ratio_col = f"{col}_margin"
            temp = df.copy()
            temp[ratio_col] = temp.apply(
                lambda r: (r[col] / r[col2]) if pd.notna(r[col2]) and r[col2] != 0 else None,
                axis=1
            )
            result = temp.groupby(group)[ratio_col].mean().reset_index().sort_values(ratio_col, ascending=False)
            result[ratio_col] = result[ratio_col].round(4)
            return {**base, "result_type": "table", "data": _df_to_records(result), "columns": [group, ratio_col]}

        # ── distribution ───────────────────────────────────────
        elif op_type == "distribution":
            s = df[col].dropna()
            dist = {
                "count": int(len(s)),
                "min":   _safe_val(s.min()),
                "p25":   _safe_val(s.quantile(0.25)),
                "median":_safe_val(s.median()),
                "p75":   _safe_val(s.quantile(0.75)),
                "max":   _safe_val(s.max()),
                "mean":  _safe_val(s.mean()),
                "std":   _safe_val(s.std()),
            }
            return {**base, "result_type": "scalar_dict", "data": dist, "columns": list(dist.keys())}

        # ── outliers ───────────────────────────────────────────
        elif op_type == "outliers":
            q1  = df[col].quantile(0.25)
            q3  = df[col].quantile(0.75)
            iqr = q3 - q1
            low  = q1 - 1.5 * iqr
            high = q3 + 1.5 * iqr
            outlier_df = df[(df[col] < low) | (df[col] > high)].copy()
            keep_cols  = [col]
            if group and group in df.columns:
                keep_cols = [group, col]
            # Try to add a label column
            label_candidates = [c for c in df.columns if df[c].dtype == object and c not in keep_cols]
            if label_candidates:
                keep_cols = [label_candidates[0]] + keep_cols
            outlier_df = outlier_df[keep_cols].sort_values(col, ascending=False).head(_MAX_RESULT_ROWS)
            summary = {
                "total_outliers": int(len(df[(df[col] < low) | (df[col] > high)])),
                "iqr_low_bound":  _safe_val(low),
                "iqr_high_bound": _safe_val(high),
                "q1": _safe_val(q1), "q3": _safe_val(q3), "iqr": _safe_val(iqr),
            }
            return {
                **base,
                "result_type": "outliers",
                "data":    _df_to_records(outlier_df),
                "columns": keep_cols,
                "summary": summary,
            }

        # ── correlation ────────────────────────────────────────
        elif op_type == "correlation":
            num_cols = df.select_dtypes(include="number").columns.tolist()
            corr = df[num_cols].corr().round(4)
            records = corr.reset_index().rename(columns={"index": "column"})
            return {**base, "result_type": "table", "data": _df_to_records(records, 30), "columns": list(records.columns)}

        # ── overall_summary ────────────────────────────────────
        elif op_type == "overall_summary":
            desc = df.describe(include="number").T.reset_index().rename(columns={"index": "column"})
            desc = desc.round(4)
            return {**base, "result_type": "table", "data": _df_to_records(desc, 30), "columns": list(desc.columns)}

        # ── value_counts ───────────────────────────────────────
        elif op_type == "value_counts":
            vc = df[col].value_counts().reset_index()
            vc.columns = [col, "count"]
            return {**base, "result_type": "table", "data": _df_to_records(vc), "columns": [col, "count"]}

        # ── time_trend ─────────────────────────────────────────
        elif op_type == "time_trend":
            if group not in df.columns:
                raise ValueError(f"time_trend: column '{group}' not found")
            temp = df.copy()
            temp[group] = pd.to_datetime(temp[group], errors="coerce")
            temp = temp.dropna(subset=[group])
            # Try monthly grouping first, fall back to daily
            try:
                temp["_period"] = temp[group].dt.to_period("M").astype(str)
            except Exception:
                temp["_period"] = temp[group].dt.strftime("%Y-%m-%d")
            result = (
                temp.groupby("_period")[col]
                .sum()
                .reset_index()
                .rename(columns={"_period": group})
                .sort_values(group)
            )
            result[col] = result[col].round(4)
            return {**base, "result_type": "table", "data": _df_to_records(result, 36), "columns": [group, col]}

        else:
            return {**base, "result_type": "error", "error": f"Unknown op_type: '{op_type}'"}

    except Exception as e:
        return {**base, "result_type": "error", "error": str(e), "detail": traceback.format_exc()}


def execute_plan(plan: list[dict], df: pd.DataFrame) -> list[dict]:
    """Execute all operations in the plan and return results."""
    return [_execute_operation(op, df) for op in plan]


# ══════════════════════════════════════════════════════════════════
# PASS 2 — LLM Interpreter
# ══════════════════════════════════════════════════════════════════

_INTERPRET_SYSTEM_PROMPT = """You are a senior business data analyst delivering insights to an executive audience.

You will receive:
  1. The user's analytical question
  2. Pre-computed results from a pandas analysis (actual numbers, tables, ranked lists)

Your job is to write a comprehensive, structured analytical response based ONLY on the provided results.

OUTPUT FORMAT — return a single JSON object with these keys:

{
  "headline":  "One crisp sentence summarising the single most important finding.",
  "narrative": "3-6 paragraph detailed analytical write-up. Be specific — cite actual numbers from the results. Explain patterns, causes, and business implications. Write like a McKinsey analyst, not a chatbot.",
  "key_findings": [
    "Bullet point 1 — specific finding with a number",
    "Bullet point 2 — specific finding with a number",
    "Bullet point 3 — specific finding with a number"
  ],
  "recommendation": "1-2 sentences of actionable recommendation based on the findings. If the question is purely factual (e.g. 'who has highest salary'), set this to null.",
  "stats_table": [
    {"label": "Metric name", "value": "formatted value", "note": "optional context"}
  ]
}

RULES:
1. Output ONLY raw JSON — no markdown, no backticks, no explanation outside the JSON.
2. Cite EXACT numbers from the computed results. Never invent numbers.
3. key_findings must have 3-6 items, each starting with a capital letter.
4. stats_table should capture the top 5-8 most important numeric findings as key-value pairs.
5. narrative must be substantive — minimum 100 words.
6. If a result contains an error, acknowledge it gracefully and work with the other results.
"""


def _interpret_results(query: str, results: list[dict], df: pd.DataFrame) -> dict:
    """PASS 2 — Ask LLM to interpret computed results into a structured analytical response."""
    llm = init_chat_model(GENERATION_MODEL_NAME)

    # Serialize results compactly for the LLM
    results_text = json.dumps(results, indent=2, default=str)

    # Trim if very long
    if len(results_text) > 8000:
        results_text = results_text[:8000] + "\n... [truncated for length]"

    user_message = (
        f"User question: {query}\n\n"
        f"Dataset: {df.shape[0]} rows, columns: {list(df.columns)}\n\n"
        f"Computed analysis results:\n{results_text}\n\n"
        "Write the structured analytical response as a JSON object. Output ONLY raw JSON."
    )

    messages = [
        {"role": "system", "content": _INTERPRET_SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()
    record_llm_call(
        use_case="data_analysis_interpretation",
        output_text=raw,
        response=response,
        model_name=GENERATION_MODEL_NAME,
    )
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$",          "", raw)
    raw = raw.strip()

    try:
        interpretation = json.loads(raw)
    except json.JSONDecodeError:
        # Graceful fallback if JSON is malformed
        interpretation = {
            "headline":       "Analysis complete.",
            "narrative":      raw[:2000],  # use raw text as narrative
            "key_findings":   [],
            "recommendation": None,
            "stats_table":    [],
        }

    return interpretation


# ══════════════════════════════════════════════════════════════════
# RESULT BUILDER — builds the final stats table the UI will render
# ══════════════════════════════════════════════════════════════════

def _build_primary_table(results: list[dict]) -> dict | None:
    """
    Pick the most relevant result table to surface in the UI card.
    Returns the first table-type result that has data.
    """
    for r in results:
        if r.get("result_type") in ("table", "outliers") and r.get("data"):
            return {
                "label":   r["label"],
                "columns": r["columns"],
                "rows":    r["data"][:15],  # cap at 15 rows in UI
            }
    return None


# ══════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════

def run_data_analysis_agent(query: str, filename: str) -> dict:
    """
    Main entry point called by the Flask route /agent/analyze.

    Args:
        query    : User's analytical question in natural language
        filename : Dataset filename (must exist in data/datasets/)

    Returns dict:
        success        : bool
        headline       : str — one-line finding
        narrative      : str — full analytical write-up
        key_findings   : list[str]
        recommendation : str | None
        stats_table    : list[{label, value, note}]
        primary_table  : {label, columns, rows} | None — best result table
        operations     : list — the operations that were executed
        filename, rows, columns
        error          : str (only on failure)
    """
    try:
        # 1. Load dataset
        df = load_dataset(filename)

        # 2. Schema for LLM
        schema = build_schema_summary(df)

        # 3. PASS 1 — get analysis plan
        plan = _plan_analysis(schema, query)

        # 4. EXECUTE — run whitelisted pandas ops
        results = execute_plan(plan, df)

        # 5. PASS 2 — interpret results
        interpretation = _interpret_results(query, results, df)

        # 6. Build primary display table
        primary_table = _build_primary_table(results)

        return {
            "success":        True,
            "headline":       interpretation.get("headline", ""),
            "narrative":      interpretation.get("narrative", ""),
            "key_findings":   interpretation.get("key_findings", []),
            "recommendation": interpretation.get("recommendation"),
            "stats_table":    interpretation.get("stats_table", []),
            "primary_table":  primary_table,
            "operations":     [
                {"op_id": r["op_id"], "label": r["label"], "status": "ok" if r.get("result_type") != "error" else "error"}
                for r in results
            ],
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
