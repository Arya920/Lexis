# agents/visualization_agent.py
"""
Production-Grade Visualization Agent
══════════════════════════════════════════════════════════════════════
Architecture — Two-pass LLM + deterministic execution:

  PASS 1  (Plan)
    LLM receives full dataset schema + user query.
    Returns a structured JSON plan:
      { "transforms": [...], "chart": {...} }
    No code is ever exec'd from LLM — all operations are whitelisted.

  VALIDATE
    Plan is validated against the actual DataFrame schema.
    Column names are tracked across transforms so post-groupby
    references are checked correctly. Retries LLM on failure.

  EXECUTE
    Deterministic pandas execution of each whitelisted operation.
    Null-safe and type-safe throughout.

  BUILD CHART
    chart spec → Plotly figure dict.
    13 chart types, consistent dark-UI theme.

Safe by design:
  - No eval(), no exec(), no arbitrary code from LLM
  - All operations are whitelisted pandas method calls
  - Column names validated at plan-time AND execute-time
  - Empty-dataframe guard after each transform

Replace visualization_agent_3.py + viz_engine.py with this file.
Update app.py import:
  from agents.visualization_agent import run_visualization_agent
══════════════════════════════════════════════════════════════════════
"""

import math
import os
import json
import re
import traceback
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from services.query_logging import record_llm_call

load_dotenv()

# ── Config ────────────────────────────────────────────────────────
DATASETS_DIR = os.path.join("data", "datasets")

try:
    from config.settings import GENERATION_MODEL_NAME
except ImportError:
    GENERATION_MODEL_NAME = "groq:llama-3.3-70b-versatile"

print("Available Model:",GENERATION_MODEL_NAME)

_MAX_SAMPLE_ROWS  = 5
_MAX_UNIQUE_VALS  = 30
_MAX_PLAN_RETRIES = 2   # how many times to retry LLM if plan validation fails

# Colour palette — vivid, dark-UI friendly
_PALETTE_CAT = [
    "#818cf8", "#34d399", "#fb923c", "#f472b6", "#60a5fa",
    "#facc15", "#a78bfa", "#4ade80", "#f87171", "#38bdf8",
    "#e879f9", "#2dd4bf", "#fbbf24", "#c084fc", "#86efac",
]


# ══════════════════════════════════════════════════════════════════
# SECTION 1 — DATASET LOADER
# ══════════════════════════════════════════════════════════════════

def load_dataset(filename: str) -> pd.DataFrame:
    """
    Load CSV / Excel from data/datasets/.
    Normalises column names and auto-detects datetime columns.
    """
    path = os.path.join(DATASETS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset '{filename}' not found in {DATASETS_DIR}/"
        )

    ext = filename.rsplit(".", 1)[-1].lower()
    if ext == "csv":
        df = pd.read_csv(path)
    elif ext in ("xlsx", "xls"):
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file type: .{ext}  (CSV and Excel only)")

    # ① Normalise column names: strip and collapse internal whitespace
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    # ② Auto-detect date-like object columns
    for col in df.columns:
        if df[col].dtype == object:
            if any(kw in col.lower() for kw in ("date", "time", "year", "month")):
                converted = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
                # Only keep if most rows parsed successfully
                if converted.notna().mean() > 0.7:
                    df[col] = converted

    return df


# ══════════════════════════════════════════════════════════════════
# SECTION 2 — SCHEMA BUILDER
# ══════════════════════════════════════════════════════════════════

def _col_tag(series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    return "categorical"


def build_schema(df: pd.DataFrame) -> str:
    """
    Produce a concise, LLM-readable schema: shape, per-column stats,
    and a sample of the first N rows as a markdown table.
    """
    lines = [
        f"Rows: {df.shape[0]}  |  Columns: {df.shape[1]}\n",
        "Column details:"
    ]

    for col in df.columns:
        tag   = _col_tag(df[col])
        dtype = str(df[col].dtype)
        nulls = int(df[col].isna().sum())

        if tag == "numeric":
            desc = (
                f"min={df[col].min():.4g}, max={df[col].max():.4g}, "
                f"mean={df[col].mean():.4g}, std={df[col].std():.4g}"
            )
        elif tag == "datetime":
            desc = f"range: {df[col].min()} → {df[col].max()}"
        else:
            uniq = df[col].dropna().unique()
            shown = list(uniq[:_MAX_UNIQUE_VALS])
            desc  = f"{len(uniq)} unique values, e.g.: {shown[:10]}"

        lines.append(
            f"  • {col!r}  [{dtype}|{tag}]  nulls={nulls}  — {desc}"
        )

    lines.append(f"\nFirst {_MAX_SAMPLE_ROWS} rows:")
    lines.append(df.head(_MAX_SAMPLE_ROWS).to_markdown(index=False))
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# SECTION 3 — LLM PLANNER
# ══════════════════════════════════════════════════════════════════

_PLANNER_SYSTEM_PROMPT = """You are a senior data analyst and visualization expert.

You receive a dataset schema and a user's chart request.
Produce a STRICT JSON execution plan — nothing else.

OUTPUT FORMAT — a single JSON object:
{
  "transforms": [ ...transform steps... ],
  "chart": { ...chart spec... }
}

════════ ALLOWED TRANSFORM STEPS ════════

1. filter
{ "step": "filter", "col": "col", "op": "==" | "!=" | ">" | ">=" | "<" | "<=" | "in" | "contains", "value": "val or [list]" }

2. drop_nulls
{ "step": "drop_nulls", "cols": ["col1", "col2"] }

3. extract_time
{ "step": "extract_time", "col": "date_col", "unit": "year" | "month" | "quarter" | "day_of_week", "new_col": "NewColName" }

4. bin_numeric
{ "step": "bin_numeric", "col": "numeric_col", "bins": 5, "new_col": "BinnedCol" }

5. groupby
{ "step": "groupby", "by": ["col1"], "agg": { "col2": "mean" | "sum" | "count" | "min" | "max" | "median" } }
NOTE: After groupby, available columns = by-columns + agg-columns ONLY.

6. groupby_multi
{ "step": "groupby_multi", "by": ["col1", "col2"], "agg": { "col3": "mean" | "sum" | "count" } }
NOTE: After groupby_multi, available columns = by-columns + agg-columns ONLY.

7. sort
{ "step": "sort", "by": "col", "order": "asc" | "desc" }

8. limit
{ "step": "limit", "n": integer }

9. compute_col
{ "step": "compute_col", "new_col": "NewCol", "formula": "ratio" | "pct_of_total", "col": "numerator_col", "col2": "denominator_col" }

10. pivot
{ "step": "pivot", "index": "row_col", "columns": "category_col", "values": "val_col", "aggfunc": "mean" | "sum" | "count" }

════════ CHART SPEC ════════

{
  "type": "bar" | "horizontal_bar" | "line" | "area" | "scatter" |
          "pie" | "donut" | "histogram" | "box" | "heatmap" |
          "grouped_bar" | "stacked_bar" | "funnel",
  "x": "col",              // required for all except pie/donut
  "y": "col",              // required for all except pie/donut/histogram
  "color": "col" | null,   // for multi-series / grouped / scatter
  "values": "col" | null,  // pie / donut only
  "names":  "col" | null,  // pie / donut only
  "title": "Descriptive Chart Title",
  "x_label": "label" | null,
  "y_label": "label" | null,
  "bins": integer | null   // histogram only; default 20
}

════════ STRICT RULES ════════
1. Output ONLY the raw JSON object — NO markdown, NO backticks, NO extra text.
2. ALL column names MUST EXACTLY match the schema (case-sensitive).
3. After groupby/groupby_multi: only the by-columns and agg-result columns exist.
   Do NOT reference original columns in subsequent steps or the chart spec.
4. PIE/DONUT: use "values" + "names" in chart spec, NOT "x"/"y".
5. HISTOGRAM: set "x" to the numeric column; omit "y".
6. BOX: "x" = optional category column, "y" = numeric column.
7. HEATMAP: use pivot step first, then set chart.x to the pivot row column.
8. GROUPED_BAR / STACKED_BAR: use groupby_multi → set chart.color to second group col.
9. TIME-BASED: always use extract_time BEFORE groupby.
10. TOP N: groupby → sort → limit → bar/horizontal_bar.
11. CATEGORY FREQUENCY: groupby with count agg → bar chart.
12. DISTRIBUTION of numeric: histogram (no groupby needed).
13. Do NOT add unnecessary transform steps.
14. After groupby, aggregated columns KEEP THEIR ORIGINAL NAMES.
    Example:
      { "agg": { "Sales": "sum" } }
    → resulting column is still "Sales", NOT "sum_Sales".

════════ FEW-SHOT EXAMPLES ════════

Query: "average salary by department"
Schema: 'Department' (categorical), 'Salary' (numeric)
→
{
  "transforms": [
    { "step": "groupby", "by": ["Department"], "agg": { "Salary": "mean" } },
    { "step": "sort", "by": "Salary", "order": "desc" }
  ],
  "chart": {
    "type": "bar", "x": "Department", "y": "Salary", "color": null,
    "title": "Average Salary by Department", "x_label": "Department", "y_label": "Avg Salary"
  }
}

Query: "monthly sales trend"
Schema: 'Order Date' (datetime), 'Sales' (numeric)
→
{
  "transforms": [
    { "step": "extract_time", "col": "Order Date", "unit": "month", "new_col": "Month" },
    { "step": "groupby", "by": ["Month"], "agg": { "Sales": "sum" } },
    { "step": "sort", "by": "Month", "order": "asc" }
  ],
  "chart": {
    "type": "line", "x": "Month", "y": "Sales",
    "title": "Monthly Sales Trend", "x_label": "Month", "y_label": "Total Sales"
  }
}

Query: "top 10 products by revenue"
Schema: 'Product Name' (categorical), 'Revenue' (numeric)
→
{
  "transforms": [
    { "step": "groupby", "by": ["Product Name"], "agg": { "Revenue": "sum" } },
    { "step": "sort", "by": "Revenue", "order": "desc" },
    { "step": "limit", "n": 10 }
  ],
  "chart": {
    "type": "horizontal_bar", "x": "Revenue", "y": "Product Name",
    "title": "Top 10 Products by Revenue", "x_label": "Revenue", "y_label": "Product"
  }
}

Query: "sales by region as pie chart"
Schema: 'Region' (categorical), 'Sales' (numeric)
→
{
  "transforms": [
    { "step": "groupby", "by": ["Region"], "agg": { "Sales": "sum" } }
  ],
  "chart": {
    "type": "pie", "values": "Sales", "names": "Region",
    "title": "Sales Distribution by Region"
  }
}

Query: "distribution of age"
Schema: 'Age' (numeric)
→
{
  "transforms": [],
  "chart": {
    "type": "histogram", "x": "Age", "bins": 20,
    "title": "Age Distribution", "x_label": "Age", "y_label": "Count"
  }
}

Query: "profit by segment and region (grouped bar)"
Schema: 'Segment' (categorical), 'Region' (categorical), 'Profit' (numeric)
→
{
  "transforms": [
    { "step": "groupby_multi", "by": ["Region", "Segment"], "agg": { "Profit": "sum" } }
  ],
  "chart": {
    "type": "grouped_bar", "x": "Region", "y": "Profit", "color": "Segment",
    "title": "Profit by Region and Segment", "x_label": "Region", "y_label": "Total Profit"
  }
}
"""


def _call_planner(schema: str, query: str, error_hint: str = "") -> dict:
    """
    Call LLM to generate the plan.
    error_hint is appended when retrying after a validation failure.
    """
    llm = init_chat_model(GENERATION_MODEL_NAME)

    user_content = f"Dataset schema:\n{schema}\n\nUser chart request:\n{query}"
    if error_hint:
        user_content += f"\n\n[Previous plan was rejected — fix this]: {error_hint}"
    user_content += "\n\nOutput ONLY the raw JSON plan."

    messages = [
        {"role": "system", "content": _PLANNER_SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()
    record_llm_call(
        use_case="data_visualization_plan",
        output_text=raw,
        response=response,
        model_name=GENERATION_MODEL_NAME,
    )
    print(f"[VizAgent] Raw LLM plan:\n{raw}\n")

    # Strip accidental markdown fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()

    try:
        plan = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM returned invalid JSON: {e}\nRaw:\n{raw[:600]}")

    if not isinstance(plan, dict):
        raise ValueError("Plan must be a JSON object with 'transforms' and 'chart' keys.")
    if "chart" not in plan:
        raise ValueError("Plan missing required 'chart' key.")

    plan.setdefault("transforms", [])
    return plan


# ══════════════════════════════════════════════════════════════════
# SECTION 4 — PLAN VALIDATOR
# ══════════════════════════════════════════════════════════════════

_ALLOWED_STEPS = {
    "filter", "drop_nulls", "extract_time", "bin_numeric",
    "groupby", "groupby_multi", "sort", "limit", "compute_col", "pivot",
}
_ALLOWED_CHART_TYPES = {
    "bar", "horizontal_bar", "line", "area", "scatter",
    "pie", "donut", "histogram", "box", "heatmap",
    "grouped_bar", "stacked_bar", "funnel",
}
_ALLOWED_AGGS = {"mean", "sum", "count", "min", "max", "median", "std"}
_ALLOWED_OPS  = {"==", "!=", ">", ">=", "<", "<=", "in", "contains"}


def validate_plan(plan: dict, df: pd.DataFrame) -> None:
    """
    Validate plan against the actual DataFrame.
    Tracks column availability across transforms so post-groupby
    references can be caught before execution.
    Raises ValueError with a clear message on any issue.
    """
    available = set(df.columns)

    def _need(col: str, ctx: str):
        if col not in available:
            raise ValueError(
                f"[{ctx}] Column '{col}' not available. "
                f"Available columns at this point: {sorted(available)}"
            )

    for i, step in enumerate(plan.get("transforms", [])):
        ctx   = f"transform[{i}]"
        stype = step.get("step")

        if stype not in _ALLOWED_STEPS:
            raise ValueError(f"[{ctx}] Unknown step type '{stype}'")

        if stype == "filter":
            _need(step["col"], ctx)
            if step.get("op") not in _ALLOWED_OPS:
                raise ValueError(f"[{ctx}] Unknown operator '{step.get('op')}'")

        elif stype == "drop_nulls":
            for c in step.get("cols", []):
                _need(c, ctx)

        elif stype == "extract_time":
            _need(step["col"], ctx)
            new_col = step.get("new_col")
            if new_col:
                available.add(new_col)

        elif stype == "bin_numeric":
            _need(step["col"], ctx)
            new_col = step.get("new_col")
            if new_col:
                available.add(new_col)

        elif stype in ("groupby", "groupby_multi"):
            by  = step.get("by", [])
            agg = step.get("agg", {})
            for c in by:
                _need(c, ctx)
            for c, fn in agg.items():
                _need(c, ctx)
                if fn not in _ALLOWED_AGGS:
                    raise ValueError(
                        f"[{ctx}] Unknown aggregation '{fn}' for column '{c}'. "
                        f"Allowed: {sorted(_ALLOWED_AGGS)}"
                    )
            # After groupby only by + agg result columns exist
            available = set(by) | set(agg.keys())

        elif stype == "sort":
            _need(step["by"], ctx)

        elif stype == "compute_col":
            _need(step["col"], ctx)
            if step.get("col2"):
                _need(step["col2"], ctx)
            available.add(step.get("new_col", "computed"))

        elif stype == "pivot":
            for k in ("index", "columns", "values"):
                _need(step[k], ctx)
            # After pivot, columns are dynamic — clear tracking
            available = set()  # can't know exactly; skip further checks

    # Validate chart spec
    chart = plan.get("chart", {})
    ctype = chart.get("type")
    if ctype not in _ALLOWED_CHART_TYPES:
        raise ValueError(
            f"[chart] Unknown chart type '{ctype}'. "
            f"Allowed: {sorted(_ALLOWED_CHART_TYPES)}"
        )

    if ctype in ("pie", "donut"):
        for k in ("values", "names"):
            v = chart.get(k)
            if v and v not in available:
                raise ValueError(
                    f"[chart.{k}] '{v}' not available. "
                    f"Available: {sorted(available)}"
                )
    elif ctype == "histogram":
        if chart.get("x") and chart["x"] not in available:
            raise ValueError(
                f"[chart.x] '{chart['x']}' not available. "
                f"Available: {sorted(available)}"
            )
    else:
        for k in ("x", "y"):
            v = chart.get(k)
            if v and v not in available:
                raise ValueError(
                    f"[chart.{k}] '{v}' not available. "
                    f"Available: {sorted(available)}"
                )
        c = chart.get("color")
        if c and c not in available:
            raise ValueError(
                f"[chart.color] '{c}' not available. "
                f"Available: {sorted(available)}"
            )


# ══════════════════════════════════════════════════════════════════
# SECTION 5 — EXECUTION ENGINE
# ══════════════════════════════════════════════════════════════════

def _clean_val(v: Any) -> Any:
    """Convert numpy/pandas scalar to a JSON-safe Python type."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 6)
    if isinstance(v, float):
        return None if (math.isnan(v) or math.isinf(v)) else round(v, 6)
    if isinstance(v, np.bool_):
        return bool(v)
    if pd.isna(v) if not isinstance(v, (list, dict, np.ndarray)) else False:
        return None
    return v


def _series_to_list(s: pd.Series) -> list:
    """Convert a pandas Series to a JSON-safe list."""
    return [_clean_val(v) for v in s]


class ExecutionEngine:
    """
    Deterministic, whitelisted pandas execution of the transform plan.
    Each step is validated at runtime for column existence and type compatibility.
    """

    def __init__(self, df: pd.DataFrame):
        self.original_df = df.copy()

    def run(self, transforms: List[dict]) -> pd.DataFrame:
        df = self.original_df.copy()

        for i, step in enumerate(transforms):
            stype = step.get("step")
            try:
                df = self._apply(df, step)
            except Exception as e:
                raise RuntimeError(
                    f"Transform step {i} ('{stype}') failed: {e}\n"
                    f"Available columns were: {list(df.columns)}"
                ) from e

            # Guard: if transforms empty the df, warn early
            if df.empty:
                raise RuntimeError(
                    f"Transform step {i} ('{stype}') produced an empty dataframe. "
                    "Your filter may be too strict, or the group yielded no rows."
                )

        return df

    # ── individual step handlers ───────────────────────────────────

    def _apply(self, df: pd.DataFrame, step: dict) -> pd.DataFrame:
        stype = step["step"]

        # ── filter ─────────────────────────────────────────────────
        if stype == "filter":
            col, op, val = step["col"], step["op"], step["value"]
            s = df[col]
            if op == "==":       df = df[s == val]
            elif op == "!=":     df = df[s != val]
            elif op == ">":      df = df[s > val]
            elif op == ">=":     df = df[s >= val]
            elif op == "<":      df = df[s < val]
            elif op == "<=":     df = df[s <= val]
            elif op == "in":
                vals = val if isinstance(val, list) else [val]
                df = df[s.isin(vals)]
            elif op == "contains":
                df = df[s.astype(str).str.contains(str(val), case=False, na=False)]
            return df.reset_index(drop=True)

        # ── drop_nulls ─────────────────────────────────────────────
        elif stype == "drop_nulls":
            cols = step.get("cols") or list(df.columns)
            # Only drop on columns that actually exist
            cols = [c for c in cols if c in df.columns]
            return df.dropna(subset=cols).reset_index(drop=True)

        # ── extract_time ────────────────────────────────────────────
        elif stype == "extract_time":
            col     = step["col"]
            unit    = step.get("unit", "month")
            new_col = step.get("new_col") or unit.title()

            series = pd.to_datetime(df[col], errors="coerce")
            if unit == "year":
                df[new_col] = series.dt.year.astype("Int64").astype(str)
            elif unit == "month":
                df[new_col] = series.dt.to_period("M").astype(str)
            elif unit == "quarter":
                df[new_col] = series.dt.to_period("Q").astype(str)
            elif unit == "day_of_week":
                df[new_col] = series.dt.day_name()
            else:
                df[new_col] = series.dt.to_period("M").astype(str)
            return df

        # ── bin_numeric ─────────────────────────────────────────────
        elif stype == "bin_numeric":
            col     = step["col"]
            bins    = step.get("bins", 5)
            labels  = step.get("labels") or None
            new_col = step.get("new_col") or f"{col}_bin"
            df[new_col] = (
                pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
                .astype(str)
            )
            return df

        # ── groupby / groupby_multi ─────────────────────────────────
        elif stype in ("groupby", "groupby_multi"):
            by  = step["by"]
            agg = step["agg"]

            # Separate count cols (need special handling) from others
            agg_dict   = {c: fn for c, fn in agg.items() if fn != "count"}
            count_cols = [c for c, fn in agg.items() if fn == "count"]

            if agg_dict:
                result = (
                    df.groupby(by, dropna=True)[list(agg_dict.keys())]
                    .agg(agg_dict)
                    .reset_index()
                )
            else:
                # Pure count
                result = (
                    df.groupby(by, dropna=True)
                    .size()
                    .reset_index(name=count_cols[0] if count_cols else "count")
                )
                return result

            # Add count columns
            if count_cols:
                size_df = (
                    df.groupby(by, dropna=True)
                    .size()
                    .reset_index(name="_tmp_count")
                )
                for c in count_cols:
                    result = result.merge(
                        size_df.rename(columns={"_tmp_count": c}),
                        on=by, how="left"
                    )
            return result

        # ── sort ────────────────────────────────────────────────────
        elif stype == "sort":
            return df.sort_values(
                by=step["by"],
                ascending=(step.get("order", "asc") == "asc")
            ).reset_index(drop=True)

        # ── limit ───────────────────────────────────────────────────
        elif stype == "limit":
            return df.head(int(step["n"])).reset_index(drop=True)

        # ── compute_col ─────────────────────────────────────────────
        elif stype == "compute_col":
            col     = step["col"]
            col2    = step.get("col2")
            new_col = step.get("new_col", "computed")
            formula = step.get("formula", "ratio")

            if formula == "ratio" and col2:
                df[new_col] = df.apply(
                    lambda r: (r[col] / r[col2])
                    if (pd.notna(r[col2]) and r[col2] != 0) else None,
                    axis=1,
                )
            elif formula == "pct_of_total":
                total = df[col].sum()
                df[new_col] = (df[col] / total * 100) if total != 0 else 0.0
            return df

        # ── pivot ───────────────────────────────────────────────────
        elif stype == "pivot":
            result = df.pivot_table(
                index=step["index"],
                columns=step["columns"],
                values=step["values"],
                aggfunc=step.get("aggfunc", "mean"),
            ).reset_index()
            # Flatten multi-level column names
            result.columns = [
                str(c).strip() if not isinstance(c, tuple) else " ".join(str(x) for x in c if x)
                for c in result.columns
            ]
            return result

        else:
            raise ValueError(f"Unknown step type '{stype}'")


# ══════════════════════════════════════════════════════════════════
# SECTION 6 — CHART BUILDER
# ══════════════════════════════════════════════════════════════════

_BASE_LAYOUT = {
    "plot_bgcolor":  "rgba(0,0,0,0)",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "font":  {"color": "#f2f2f2", "family": "Inter, system-ui, sans-serif"},
    "margin": {"t": 70, "r": 30, "b": 80, "l": 80},
    "legend": {"bgcolor": "rgba(0,0,0,0)", "borderwidth": 0},
    "hoverlabel": {"bgcolor": "#1e293b", "bordercolor": "#334155", "font": {"color": "#f8fafc"}},
}
_GRID_COLOR  = "rgba(255,255,255,0.08)"
_AXIS_STYLE  = {"gridcolor": _GRID_COLOR, "linecolor": "rgba(255,255,255,0.15)", "zerolinecolor": _GRID_COLOR}


def _make_layout(title: str, x_label: str = "", y_label: str = "", extra: dict = None) -> dict:
    layout = {**_BASE_LAYOUT, "title": {"text": title, "font": {"size": 18, "color": "#f8fafc"}}}
    if x_label:
        layout["xaxis"] = {**_AXIS_STYLE, "title": {"text": x_label}}
    if y_label:
        layout["yaxis"] = {**_AXIS_STYLE, "title": {"text": y_label}}
    if extra:
        layout.update(extra)
    return layout


def _resolve_col(df: pd.DataFrame, col: Optional[str]) -> Optional[str]:
    """Return col if it exists in df, else None."""
    return col if col and col in df.columns else None


def build_plotly_figure(df: pd.DataFrame, chart: dict) -> dict:
    """
    Build a Plotly figure dict from a transformed DataFrame + chart spec.
    Supports 13 chart types with a consistent dark-UI theme.
    """
    ctype   = chart.get("type", "bar")
    title   = chart.get("title", "Chart")
    x_label = chart.get("x_label") or chart.get("x", "")
    y_label = chart.get("y_label") or chart.get("y", "")

    x_col   = _resolve_col(df, chart.get("x"))
    y_col   = _resolve_col(df, chart.get("y"))
    c_col   = _resolve_col(df, chart.get("color"))
    v_col   = _resolve_col(df, chart.get("values"))
    n_col   = _resolve_col(df, chart.get("names"))

    data   = []
    layout = _make_layout(title, x_label, y_label)
    config = {
        "responsive": True,
        "displayModeBar": True,
        "modeBarButtonsToRemove": ["toImage"],
    }

    # ── bar / horizontal_bar ─────────────────────────────────────
    if ctype in ("bar", "horizontal_bar"):
        orientation = "h" if ctype == "horizontal_bar" else "v"

        if c_col:
            for i, grp in enumerate(df[c_col].dropna().unique()):
                sub = df[df[c_col] == grp]
                x_v = _series_to_list(sub[x_col if orientation == "v" else y_col])
                y_v = _series_to_list(sub[y_col if orientation == "v" else x_col])
                data.append({
                    "type": "bar", "name": str(grp),
                    "x": x_v, "y": y_v, "orientation": orientation,
                    "marker": {"color": _PALETTE_CAT[i % len(_PALETTE_CAT)]},
                })
            layout["barmode"] = "group"
        else:
            if not x_col or not y_col:
                raise ValueError(f"bar chart requires 'x' and 'y' columns. Got x={x_col}, y={y_col}")
            x_v = _series_to_list(df[x_col if orientation == "v" else y_col])
            y_v = _series_to_list(df[y_col if orientation == "v" else x_col])
            n   = len(x_v)
            colors = (_PALETTE_CAT * math.ceil(n / len(_PALETTE_CAT)))[:n]
            data.append({
                "type": "bar",
                "x": x_v, "y": y_v, "orientation": orientation,
                "marker": {"color": colors, "line": {"width": 0}},
                "hovertemplate": "%{x}<br>%{y}<extra></extra>",
            })

        if orientation == "v":
            layout.setdefault("xaxis", {}).update({**_AXIS_STYLE, "tickangle": -30, "automargin": True})
            layout.setdefault("yaxis", {}).update(_AXIS_STYLE)
        else:
            layout.setdefault("xaxis", {}).update(_AXIS_STYLE)
            layout.setdefault("yaxis", {}).update({**_AXIS_STYLE, "automargin": True})

    # ── grouped_bar ──────────────────────────────────────────────
    elif ctype == "grouped_bar":
        if not c_col:
            raise ValueError("grouped_bar requires 'color' column for grouping.")
        for i, grp in enumerate(df[c_col].dropna().unique()):
            sub = df[df[c_col] == grp]
            data.append({
                "type": "bar", "name": str(grp),
                "x": _series_to_list(sub[x_col]),
                "y": _series_to_list(sub[y_col]),
                "marker": {"color": _PALETTE_CAT[i % len(_PALETTE_CAT)]},
            })
        layout["barmode"] = "group"
        layout.setdefault("xaxis", {}).update({**_AXIS_STYLE, "tickangle": -30})
        layout.setdefault("yaxis", {}).update(_AXIS_STYLE)

    # ── stacked_bar ──────────────────────────────────────────────
    elif ctype == "stacked_bar":
        if not c_col:
            raise ValueError("stacked_bar requires 'color' column for stacking.")
        for i, grp in enumerate(df[c_col].dropna().unique()):
            sub = df[df[c_col] == grp]
            data.append({
                "type": "bar", "name": str(grp),
                "x": _series_to_list(sub[x_col]),
                "y": _series_to_list(sub[y_col]),
                "marker": {"color": _PALETTE_CAT[i % len(_PALETTE_CAT)]},
            })
        layout["barmode"] = "stack"
        layout.setdefault("xaxis", {}).update({**_AXIS_STYLE, "tickangle": -30})
        layout.setdefault("yaxis", {}).update(_AXIS_STYLE)

    # ── line ─────────────────────────────────────────────────────
    elif ctype == "line":
        if c_col:
            for i, grp in enumerate(df[c_col].dropna().unique()):
                sub = df[df[c_col] == grp]
                data.append({
                    "type": "scatter", "mode": "lines+markers",
                    "name": str(grp),
                    "x": _series_to_list(sub[x_col]),
                    "y": _series_to_list(sub[y_col]),
                    "line": {"color": _PALETTE_CAT[i % len(_PALETTE_CAT)], "width": 2},
                    "marker": {"size": 5},
                })
        else:
            data.append({
                "type": "scatter", "mode": "lines+markers",
                "x": _series_to_list(df[x_col]),
                "y": _series_to_list(df[y_col]),
                "line": {"color": _PALETTE_CAT[0], "width": 2},
                "marker": {"size": 5},
                "fill": "tozeroy",
                "fillcolor": "rgba(129,140,248,0.12)",
            })
        layout.setdefault("xaxis", {}).update({**_AXIS_STYLE, "tickangle": -30, "automargin": True})
        layout.setdefault("yaxis", {}).update(_AXIS_STYLE)

    # ── area ─────────────────────────────────────────────────────
    elif ctype == "area":
        data.append({
            "type": "scatter", "mode": "lines",
            "x": _series_to_list(df[x_col]),
            "y": _series_to_list(df[y_col]),
            "fill": "tozeroy",
            "line": {"color": _PALETTE_CAT[0], "width": 2},
            "fillcolor": "rgba(129,140,248,0.18)",
        })
        layout.setdefault("xaxis", {}).update({**_AXIS_STYLE, "tickangle": -30})
        layout.setdefault("yaxis", {}).update(_AXIS_STYLE)

    # ── scatter ──────────────────────────────────────────────────
    elif ctype == "scatter":
        if c_col:
            for i, grp in enumerate(df[c_col].dropna().unique()):
                sub = df[df[c_col] == grp]
                data.append({
                    "type": "scatter", "mode": "markers",
                    "name": str(grp),
                    "x": _series_to_list(sub[x_col]),
                    "y": _series_to_list(sub[y_col]),
                    "marker": {"color": _PALETTE_CAT[i % len(_PALETTE_CAT)], "size": 7, "opacity": 0.8},
                })
        else:
            data.append({
                "type": "scatter", "mode": "markers",
                "x": _series_to_list(df[x_col]),
                "y": _series_to_list(df[y_col]),
                "marker": {"color": _PALETTE_CAT[0], "size": 7, "opacity": 0.8},
            })
        layout.setdefault("xaxis", {}).update(_AXIS_STYLE)
        layout.setdefault("yaxis", {}).update(_AXIS_STYLE)

    # ── pie ──────────────────────────────────────────────────────
    elif ctype in ("pie", "donut"):
        if not v_col or not n_col:
            raise ValueError(
                f"pie/donut chart requires 'values' and 'names' columns. "
                f"Got values={v_col}, names={n_col}"
            )
        data.append({
            "type": "pie",
            "values": _series_to_list(df[v_col]),
            "labels": _series_to_list(df[n_col]),
            "hole": 0.4 if ctype == "donut" else 0,
            "marker": {"colors": _PALETTE_CAT},
            "textinfo": "label+percent",
            "hovertemplate": "%{label}<br>%{value:,.2f} (%{percent})<extra></extra>",
        })
        layout.pop("xaxis", None)
        layout.pop("yaxis", None)
        layout["margin"] = {"t": 70, "r": 30, "b": 30, "l": 30}

    # ── histogram ────────────────────────────────────────────────
    elif ctype == "histogram":
        if not x_col:
            raise ValueError("histogram requires 'x' column.")
        nbins = int(chart.get("bins") or 20)
        if c_col:
            for i, grp in enumerate(df[c_col].dropna().unique()):
                sub = df[df[c_col] == grp]
                data.append({
                    "type": "histogram", "name": str(grp),
                    "x": _series_to_list(sub[x_col]),
                    "nbinsx": nbins,
                    "marker": {"color": _PALETTE_CAT[i % len(_PALETTE_CAT)], "opacity": 0.75},
                })
            layout["barmode"] = "overlay"
        else:
            data.append({
                "type": "histogram",
                "x": _series_to_list(df[x_col]),
                "nbinsx": nbins,
                "marker": {"color": _PALETTE_CAT[0], "opacity": 0.85},
            })
        layout.setdefault("xaxis", {}).update(_AXIS_STYLE)
        layout["yaxis"] = {**_AXIS_STYLE, "title": {"text": "Count"}}

    # ── box ──────────────────────────────────────────────────────
    elif ctype == "box":
        if not y_col:
            raise ValueError("box chart requires 'y' column.")
        if x_col:
            for i, grp in enumerate(df[x_col].dropna().unique()):
                sub = df[df[x_col] == grp]
                data.append({
                    "type": "box", "name": str(grp),
                    "y": _series_to_list(sub[y_col]),
                    "marker": {"color": _PALETTE_CAT[i % len(_PALETTE_CAT)]},
                    "boxpoints": "outliers",
                })
        else:
            data.append({
                "type": "box",
                "y": _series_to_list(df[y_col]),
                "name": y_col,
                "marker": {"color": _PALETTE_CAT[0]},
                "boxpoints": "outliers",
            })
        layout.setdefault("xaxis", {}).update({**_AXIS_STYLE, "automargin": True})
        layout.setdefault("yaxis", {}).update(_AXIS_STYLE)

    # ── heatmap ──────────────────────────────────────────────────
    elif ctype == "heatmap":
        # Expects pivot step already ran; df has row_col + value columns
        row_col  = x_col or df.columns[0]
        val_cols = [c for c in df.columns if c != row_col]
        z = [
            [_clean_val(v) for v in row]
            for row in df[val_cols].values.tolist()
        ]
        data.append({
            "type": "heatmap",
            "x": val_cols,
            "y": _series_to_list(df[row_col]),
            "z": z,
            "colorscale": "Blues",
            "hoverongaps": False,
            "hovertemplate": "x=%{x}<br>y=%{y}<br>value=%{z:.2f}<extra></extra>",
        })
        layout.setdefault("xaxis", {}).update({**_AXIS_STYLE, "tickangle": -30, "automargin": True})
        layout.setdefault("yaxis", {}).update({**_AXIS_STYLE, "automargin": True})

    # ── funnel ───────────────────────────────────────────────────
    elif ctype == "funnel":
        if not x_col or not y_col:
            raise ValueError("funnel chart requires 'x' (values) and 'y' (labels) columns.")
        n = len(df)
        colors = (_PALETTE_CAT * math.ceil(n / len(_PALETTE_CAT)))[:n]
        data.append({
            "type": "funnel",
            "x": _series_to_list(df[x_col]),
            "y": _series_to_list(df[y_col]),
            "marker": {"color": colors},
            "textinfo": "value+percent initial",
        })
        layout.pop("yaxis", None)

    else:
        raise ValueError(f"Unsupported chart type: '{ctype}'")

    return {"data": data, "layout": layout, "config": config}


# ══════════════════════════════════════════════════════════════════
# SECTION 7 — SUMMARY GENERATOR
# ══════════════════════════════════════════════════════════════════

def _generate_summary(query: str, chart: dict, df: pd.DataFrame) -> str:
    """Generate a 1-2 sentence plain-English insight about the chart."""
    llm   = init_chat_model(GENERATION_MODEL_NAME)
    title = chart.get("title", "the chart")
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful data analyst. "
                "Write exactly 1-2 concise, insightful sentences describing what the chart shows. "
                "Mention the key trend or takeaway using actual values if visible. "
                "Do NOT mention Plotly, JSON, or any technical details."
            ),
        },
        {
            "role": "user",
            "content": (
                f"User asked: '{query}'\n"
                f"Chart title: '{title}'\n"
                f"Transformed dataset shape: {df.shape[0]} rows × {df.shape[1]} columns.\n"
                f"Columns: {list(df.columns)}\n"
                f"Top rows:\n{df.head(8).to_markdown(index=False)}\n\n"
                "Write a short, insightful summary of this chart."
            ),
        },
    ]
    response = llm.invoke(messages)
    summary = response.content.strip()
    record_llm_call(
        use_case="data_visualization_summary",
        output_text=summary,
        response=response,
        model_name=GENERATION_MODEL_NAME,
    )
    return summary


# ══════════════════════════════════════════════════════════════════
# SECTION 8 — MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════

def run_visualization_agent(query: str, filename: str) -> dict:
    """
    Main entry point called by Flask route  POST /agent/visualize

    Args:
        query    : Natural-language chart request (e.g. "bar chart of sales by region")
        filename : Dataset file name (must exist in data/datasets/)

    Returns dict:
        success  : bool
        figure   : Plotly figure dict  (for Plotly.js on the frontend)
        summary  : str                  (1-2 sentence insight)
        plan     : dict                 (the execution plan that was used)
        filename : str
        rows     : int                  (original dataset row count)
        columns  : list[str]            (original column names)
        error    : str  (only when success=False)
        detail   : str  (full traceback, only on unexpected errors)
    """
    try:
        # ── 1. Load + clean dataset ──────────────────────────────
        df = load_dataset(filename)

        # ── 2. Build schema for the LLM ─────────────────────────
        schema = build_schema(df)

        # ── 3. Generate + validate plan (with retries) ───────────
        plan       = None
        last_error = ""
        for attempt in range(_MAX_PLAN_RETRIES + 1):
            try:
                plan = _call_planner(schema, query, error_hint=last_error)
                validate_plan(plan, df)
                break   # plan is valid — stop retrying
            except ValueError as exc:
                last_error = str(exc)
                print(f"[VizAgent] Attempt {attempt + 1}/{_MAX_PLAN_RETRIES + 1} "
                      f"plan rejected: {last_error}")
                plan = None   # reset so we don't use a bad plan

        if plan is None:
            return {
                "success": False,
                "error": (
                    f"Could not produce a valid visualization plan after "
                    f"{_MAX_PLAN_RETRIES + 1} attempts. Last error: {last_error}"
                ),
            }

        # ── 4. Execute transforms deterministically ──────────────
        engine    = ExecutionEngine(df)
        result_df = engine.run(plan.get("transforms", []))

        if result_df.empty:
            return {
                "success": False,
                "error": (
                    "The transform pipeline produced an empty table. "
                    "Your filter may be too strict, or no data matches the criteria."
                ),
            }

        # ── 5. Build Plotly figure ───────────────────────────────
        figure = build_plotly_figure(result_df, plan["chart"])

        # ── 6. Generate insight ──────────────────────────────────
        summary = _generate_summary(query, plan["chart"], result_df)

        return {
            "success":  True,
            "figure":   figure,
            "summary":  summary,
            "plan":     plan,
            "filename": filename,
            "rows":     df.shape[0],
            "columns":  list(df.columns),
        }

    except FileNotFoundError as exc:
        return {"success": False, "error": str(exc)}
    except ValueError as exc:
        return {"success": False, "error": str(exc)}
    except RuntimeError as exc:
        return {"success": False, "error": str(exc)}
    except Exception as exc:
        return {
            "success": False,
            "error":   f"Unexpected error: {exc}",
            "detail":  traceback.format_exc(),
        }
