import json
import logging
import os
import threading
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

try:
    from config.settings import GENERATION_MODEL_NAME, LOG_DIR
except ImportError:
    GENERATION_MODEL_NAME = "unknown"
    LOG_DIR = os.path.join(os.getcwd(), "Logs")


_current_llm_calls: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "current_llm_calls",
    default=None,
)
_write_lock = threading.Lock()


def begin_query_log() -> Any:
    """Start collecting LLM-call details for the current request context."""
    return _current_llm_calls.set([])


def reset_query_log(token: Any) -> None:
    _current_llm_calls.reset(token)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _safe_json(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(k): _safe_json(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_safe_json(v) for v in value]
        return str(value)


def _extract_usage(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage_metadata", None)
    if usage:
        return _normalise_usage(dict(usage))

    metadata = getattr(response, "response_metadata", None) or {}
    token_usage = metadata.get("token_usage") or metadata.get("usage") or {}
    return _normalise_usage(token_usage)


def _normalise_usage(usage: dict[str, Any]) -> dict[str, Any]:
    input_tokens = (
        usage.get("input_tokens")
        or usage.get("prompt_tokens")
        or usage.get("promptTokenCount")
    )
    output_tokens = (
        usage.get("output_tokens")
        or usage.get("completion_tokens")
        or usage.get("completionTokenCount")
    )
    total_tokens = usage.get("total_tokens") or usage.get("totalTokenCount")

    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = int(input_tokens) + int(output_tokens)

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "raw": _safe_json(usage),
    }


def _extract_model(response: Any, fallback_model: str) -> str:
    metadata = getattr(response, "response_metadata", None) or {}
    return (
        metadata.get("model_name")
        or metadata.get("model")
        or metadata.get("model_id")
        or fallback_model
    )


def record_llm_call(
    *,
    use_case: str,
    output_text: str,
    response: Any | None = None,
    model_name: str = GENERATION_MODEL_NAME,
) -> None:
    """Record one model invocation inside the current user query."""
    calls = _current_llm_calls.get()
    if calls is None:
        return

    usage = _extract_usage(response) if response is not None else _normalise_usage({})
    calls.append({
        "timestamp_utc": _utc_now().isoformat(),
        "use_case": use_case,
        "model": _extract_model(response, model_name) if response is not None else model_name,
        "token_count": {
            "input": usage.get("input_tokens"),
            "output": usage.get("output_tokens"),
            "total": usage.get("total_tokens"),
        },
        "usage_metadata": usage.get("raw", {}),
        "exact_model_output": output_text,
    })


def _sum_known_tokens(calls: list[dict[str, Any]], token_type: str) -> int | None:
    values = [
        call.get("token_count", {}).get(token_type)
        for call in calls
        if call.get("token_count", {}).get(token_type) is not None
    ]
    if not values:
        return None
    return int(sum(int(value) for value in values))


def write_query_log(
    *,
    query: str,
    endpoint: str,
    use_case: str,
    response_text: str,
    status: str = "success",
    model_name: str = GENERATION_MODEL_NAME,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Append a professional, daily JSONL record for one user query."""
    os.makedirs(LOG_DIR, exist_ok=True)
    calls = _current_llm_calls.get() or []

    record = {
        "event_id": str(uuid.uuid4()),
        "timestamp_utc": _utc_now().isoformat(),
        "endpoint": endpoint,
        "use_case": use_case,
        "status": status,
        "model_requested": model_name,
        "user_query": query,
        "text_output": response_text,
        "token_count": {
            "input": _sum_known_tokens(calls, "input"),
            "output": _sum_known_tokens(calls, "output"),
            "total": _sum_known_tokens(calls, "total"),
        },
        "llm_calls": calls,
        "metadata": _safe_json(metadata or {}),
    }

    date_part = _utc_now().strftime("%Y-%m-%d")
    log_path = os.path.join(LOG_DIR, f"{date_part}.jsonl")

    try:
        with _write_lock:
            with open(log_path, "a", encoding="utf-8") as log_file:
                log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        logging.getLogger(__name__).exception("Failed to write query log")
