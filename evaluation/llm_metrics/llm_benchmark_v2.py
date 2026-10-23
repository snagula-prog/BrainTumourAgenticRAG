"""
LLM Benchmark v2
================

Purpose:
- Compare local Ollama models on the three roles needed by the agent:
    1. Query analysis
    2. Structured fact extraction
    3. Grounded synthesis
- Use strict JSON schemas instead of loose JSON mode.
- Disable thinking/reasoning where the Ollama API supports it.
- Cap generated tokens so models are compared fairly.
- Warm each model once and report both cold-load and warmed latency.
- Score correctness against gold labels; JSON parse success is NOT treated as quality.

Run from the repository root:
    python evaluation/llm_benchmark_v2.py
"""

from __future__ import annotations

import json
import math
import re
import statistics
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import requests


OLLAMA_URL = "http://localhost:11434"

MODELS = [
    "qwen3:4b-instruct",
    "granite4:3b",
    "phi4-mini",
    "qwen3.5:4b",
    "qwen3.5:9b",
    "gemma3:4b",
    "granite4:7b-a1b-h",
]

# Fixed inference settings for fairness.
NUM_CTX = 4096
TEMPERATURE = 0.0
SEED = 42
KEEP_ALIVE = "5m"

# Keep structured tasks small and bounded.
NUM_PREDICT = {
    "query_analysis": 192,
    "fact_extraction": 320,
    "grounded_synthesis": 224,
}


# ---------------------------------------------------------------------------
# JSON SCHEMAS
# ---------------------------------------------------------------------------

ANALYZER_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {
            "type": "string",
            "enum": [
                "evidence_qa",
                "fact_lookup",
                "extraction",
                "comparison",
                "quantitative",
                "visualization",
                "multi_step",
            ],
        },
        "paper_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
        "metrics": {
            "type": "array",
            "items": {"type": "string"},
        },
        "datasets": {
            "type": "array",
            "items": {"type": "string"},
        },
        "needs_comparison": {"type": "boolean"},
        "needs_visualization": {"type": "boolean"},
        "needs_aggregation": {"type": "boolean"},
    },
    "required": [
        "intent",
        "paper_ids",
        "metrics",
        "datasets",
        "needs_comparison",
        "needs_visualization",
        "needs_aggregation",
    ],
    "additionalProperties": False,
}


EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "entity": {"type": "string"},
                    "field": {"type": "string"},
                    "value": {"type": ["string", "number"]},
                    "unit": {"type": "string"},
                    "condition": {"type": "string"},
                    "quote": {"type": "string"},
                },
                "required": [
                    "entity",
                    "field",
                    "value",
                    "unit",
                    "condition",
                    "quote",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["facts"],
    "additionalProperties": False,
}


SYNTHESIS_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "support_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "claim_type": {
                        "type": "string",
                        "enum": [
                            "reported",
                            "derived",
                            "comparison",
                            "limitation",
                            "inference",
                        ],
                    },
                },
                "required": [
                    "text",
                    "support_ids",
                    "claim_type",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["claims"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# BENCHMARK CASES + GOLD
# ---------------------------------------------------------------------------

ANALYZER_CASES = [
    {
        "id": "A01",
        "query": (
            "What is the best reported accuracy among the models "
            "evaluated in paper_008?"
        ),
        "gold": {
            "intent": "quantitative",
            "paper_ids": ["paper_008"],
            "metrics": ["accuracy"],
            "datasets": [],
            "needs_comparison": False,
            "needs_visualization": False,
            "needs_aggregation": True,
        },
    },
    {
        "id": "A02",
        "query": (
            "Compare the reported sensitivity and specificity of "
            "paper_010 and paper_011."
        ),
        "gold": {
            "intent": "comparison",
            "paper_ids": ["paper_010", "paper_011"],
            "metrics": ["sensitivity", "specificity"],
            "datasets": [],
            "needs_comparison": True,
            "needs_visualization": False,
            "needs_aggregation": False,
        },
    },
    {
        "id": "A03",
        "query": (
            "Plot the reported accuracy of the top three comparable "
            "methods across the relevant papers."
        ),
        "gold": {
            "intent": "visualization",
            "paper_ids": [],
            "metrics": ["accuracy"],
            "datasets": [],
            "needs_comparison": True,
            "needs_visualization": True,
            "needs_aggregation": True,
        },
    },
    {
        "id": "A04",
        "query": (
            "What dataset was used by the authors, and what accuracy "
            "did they report?"
        ),
        "gold": {
            "intent": "fact_lookup",
            "paper_ids": [],
            "metrics": ["accuracy"],
            "datasets": [],
            "needs_comparison": False,
            "needs_visualization": False,
            "needs_aggregation": False,
        },
    },
    {
        "id": "A05",
        "query": (
            "Which paper reports the highest sensitivity, and what was "
            "the value?"
        ),
        "gold": {
            "intent": "quantitative",
            "paper_ids": [],
            "metrics": ["sensitivity"],
            "datasets": [],
            "needs_comparison": True,
            "needs_visualization": False,
            "needs_aggregation": True,
        },
    },
]


EXTRACTION_CASES = [
    {
        "id": "E01",
        "text": (
            "EfficientNet-B0 achieved a top-1 accuracy of 76.3% and "
            "top-5 accuracy of 93.2% on ImageNet using 5.3M parameters."
        ),
        "gold": [
            {
                "entity": "EfficientNet-B0",
                "field": "top-1 accuracy",
                "value": "76.3",
                "unit": "%",
                "condition": "on ImageNet",
            },
            {
                "entity": "EfficientNet-B0",
                "field": "top-5 accuracy",
                "value": "93.2",
                "unit": "%",
                "condition": "on ImageNet",
            },
            {
                "entity": "EfficientNet-B0",
                "field": "parameters",
                "value": "5.3",
                "unit": "M",
                "condition": "on ImageNet",
            },
        ],
    },
    {
        "id": "E02",
        "text": (
            "The proposed model achieved 97.8% sensitivity and 95.4% "
            "specificity on the test set."
        ),
        "gold": [
            {
                "entity": "proposed model",
                "field": "sensitivity",
                "value": "97.8",
                "unit": "%",
                "condition": "test set",
            },
            {
                "entity": "proposed model",
                "field": "specificity",
                "value": "95.4",
                "unit": "%",
                "condition": "test set",
            },
        ],
    },
    {
        "id": "E03",
        "text": (
            "The experiment used the BraTS dataset with 3,000 training "
            "images and 500 validation images."
        ),
        "gold": [
            {
                "entity": "BraTS dataset",
                "field": "dataset",
                "value": "BraTS",
                "unit": "",
                "condition": "",
            },
            {
                "entity": "BraTS dataset",
                "field": "training images",
                "value": "3000",
                "unit": "images",
                "condition": "",
            },
            {
                "entity": "BraTS dataset",
                "field": "validation images",
                "value": "500",
                "unit": "images",
                "condition": "",
            },
        ],
    },
    {
        "id": "E04",
        "text": (
            "The method obtained an accuracy of 92.4%, while sensitivity "
            "was 90.1% and specificity was 94.7%."
        ),
        "gold": [
            {
                "entity": "method",
                "field": "accuracy",
                "value": "92.4",
                "unit": "%",
                "condition": "",
            },
            {
                "entity": "method",
                "field": "sensitivity",
                "value": "90.1",
                "unit": "%",
                "condition": "",
            },
            {
                "entity": "method",
                "field": "specificity",
                "value": "94.7",
                "unit": "%",
                "condition": "",
            },
        ],
    },
    {
        "id": "E05",
        "text": (
            "The authors report that their main limitation is the small "
            "dataset size and the absence of external validation."
        ),
        "gold": [
            {
                "entity": "study limitation",
                "field": "dataset size",
                "value": "small",
                "unit": "",
                "condition": "",
            },
            {
                "entity": "study limitation",
                "field": "external validation",
                "value": "absence",
                "unit": "",
                "condition": "",
            },
        ],
    },
]


SYNTHESIS_CASES = [
    {
        "id": "S01",
        "query": "Which model performed best?",
        "evidence": [
            {"id": "F1", "text": "Model A achieved 91.2% accuracy."},
            {"id": "F2", "text": "Model B achieved 94.7% accuracy."},
            {"id": "F3", "text": "Model C achieved 92.1% accuracy."},
        ],
        "gold": {
            "required_terms": ["Model B"],
            "required_numbers": ["94.7"],
            "required_support_ids": ["F2"],
        },
    },
    {
        "id": "S02",
        "query": (
            "Compare the two methods and state which one had higher "
            "sensitivity."
        ),
        "evidence": [
            {"id": "F1", "text": "Method A reported sensitivity of 91.0%."},
            {"id": "F2", "text": "Method B reported sensitivity of 95.2%."},
        ],
        "gold": {
            "required_terms": ["Method B", "higher sensitivity"],
            "required_numbers": ["95.2", "91.0"],
            "required_support_ids": ["F1", "F2"],
        },
    },
    {
        "id": "S03",
        "query": (
            "State the reported result without adding information not "
            "contained in the evidence."
        ),
        "evidence": [
            {"id": "F1", "text": "The model achieved 88.4% accuracy on the test set."},
        ],
        "gold": {
            "required_terms": ["88.4", "accuracy"],
            "required_numbers": ["88.4"],
            "required_support_ids": ["F1"],
        },
    },
    {
        "id": "S04",
        "query": "Are the two results directly comparable?",
        "evidence": [
            {
                "id": "F1",
                "text": (
                    "Method A was evaluated on BraTS 2021 using the "
                    "validation split."
                ),
            },
            {
                "id": "F2",
                "text": (
                    "Method B was evaluated on BraTS 2020 using the "
                    "held-out test set."
                ),
            },
        ],
        "gold": {
            "required_terms": ["not directly comparable", "BraTS 2021", "BraTS 2020"],
            "required_numbers": [],
            "required_support_ids": ["F1", "F2"],
        },
    },
    {
        "id": "S05",
        "query": "What is the main limitation reported by the authors?",
        "evidence": [
            {
                "id": "F1",
                "text": "The study is limited by a small single-center dataset.",
            },
            {
                "id": "F2",
                "text": "External validation was not performed.",
            },
        ],
        "gold": {
            "required_terms": [
                "small single-center dataset",
                "external validation",
            ],
            "required_numbers": [],
            "required_support_ids": ["F1", "F2"],
        },
    },
]


# ---------------------------------------------------------------------------
# PROMPTS
# ---------------------------------------------------------------------------

ANALYZER_SYSTEM = f"""
You are the query-analysis component of a research-paper assistant.

Classify the user request and extract only information explicitly
present in the request.

Important:
- paper_ids: only explicit paper IDs such as paper_008.
- metrics: requested metric names such as accuracy, sensitivity.
- datasets: named dataset entities explicitly mentioned by the user.
  Do not put generic slot names like "dataset_name" here.
- Do not invent paper IDs.
- "best", "highest", "average", "rank", "top" imply aggregation.
- comparison means the user needs multiple entities/papers/methods
  compared.
- return ONLY the schema below.

JSON schema:
{json.dumps(ANALYZER_SCHEMA, separators=(",", ":"))}
"""


EXTRACTION_SYSTEM = f"""
You extract atomic research facts from the supplied passage.

Return ONLY facts explicitly supported by the passage.

Rules:
- One fact per entity/field/value.
- Preserve the relationship between entity and value.
- Keep units separate from numeric values.
- Put the relevant evaluation condition in condition.
- quote MUST be an exact substring of the supplied passage.
- Never invent facts.
- Never merge unrelated values.
- Return an empty facts list if nothing is stated.

JSON schema:
{json.dumps(EXTRACTION_SCHEMA, separators=(",", ":"))}
"""


SYNTHESIS_SYSTEM = f"""
You are the grounded synthesis component of a research-paper assistant.

Use ONLY the supplied evidence.

Rules:
- Every claim must have at least one valid support_id.
- Do not introduce facts, numbers, or comparisons absent from evidence.
- For comparisons, cite all evidence needed for the comparison.
- Keep claims concise.
- Do not manually create citation text outside support_ids.
- Return ONLY the schema below.

JSON schema:
{json.dumps(SYNTHESIS_SCHEMA, separators=(",", ":"))}
"""


# ---------------------------------------------------------------------------
# DATA CLASSES
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    model: str
    task: str
    case_id: str
    available: bool
    success: bool
    schema_valid: bool
    elapsed_seconds: float
    load_duration_seconds: float | None
    prompt_tokens: int | None
    output_tokens: int | None
    response_text: str
    parsed_json: Any | None
    score: float
    metrics: dict[str, float]
    error: str | None


# ---------------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------------

def norm_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).lower()
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w.%+-]+", " ", text)
    return text.strip()


def norm_number(value: Any) -> str:
    if value is None:
        return ""

    text = str(value).strip().replace(",", "")

    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return norm_text(text)

    try:
        number = float(match.group(0))
        if number.is_integer():
            return str(int(number))
        return f"{number:.10f}".rstrip("0").rstrip(".")
    except ValueError:
        return norm_text(text)


FIELD_ALIASES = {
    "parameters": {
        "parameter",
        "parameters",
        "number parameters",
        "number of parameters",
        "model parameters",
    },
    "top-1 accuracy": {
        "top 1 accuracy",
        "top 1 acc",
        "top1 accuracy",
    },
    "top-5 accuracy": {
        "top 5 accuracy",
        "top 5 acc",
        "top5 accuracy",
    },
    "training images": {
        "training images",
        "training image count",
        "training images count",
        "train images",
    },
    "validation images": {
        "validation images",
        "validation image count",
        "validation images count",
        "val images",
    },
    "dataset size": {
        "dataset size",
        "size",
    },
    "external validation": {
        "external validation",
    },
}


ENTITY_ALIASES = {
    "proposed model": {"proposed model", "model"},
    "method": {"method", "model", "proposed method"},
    "study limitation": {"study limitation", "main limitation", "limitation"},
    "BraTS dataset": {"brats dataset", "brats", "dataset"},
    "EfficientNet-B0": {"efficientnet b0", "efficientnet-b0"},
}


def canonical_field(field: Any) -> str:
    n = norm_text(field)
    for canonical, aliases in FIELD_ALIASES.items():
        if n in {norm_text(a) for a in aliases}:
            return canonical
    return n


def entity_match(pred: Any, gold: str) -> bool:
    p = norm_text(pred)
    aliases = ENTITY_ALIASES.get(gold, {gold})
    return p in {norm_text(a) for a in aliases}


def value_match(pred: Any, gold: str) -> bool:
    p = str(pred).strip()
    g = str(gold).strip()

    # Numeric values compare numerically.
    pn = norm_number(p)
    gn = norm_number(g)

    if pn and gn and re.fullmatch(r"-?\d+(?:\.\d+)?", pn):
        return pn == gn

    return norm_text(p) == norm_text(g)


def unit_match(pred: Any, gold: str) -> bool:
    p = norm_text(pred)
    g = norm_text(gold)

    aliases = {
        "%": {"%", "percent"},
        "M": {"m", "million", "millions"},
        "images": {"image", "images"},
    }

    if g in aliases:
        return p in {norm_text(x) for x in aliases[g]}

    return p == g


def condition_match(pred: Any, gold: str) -> bool:
    g = norm_text(gold)
    p = norm_text(pred)

    if not g:
        return True

    # Allow compact variants such as "on the test set" vs "test set".
    return g in p or p in g


def extract_numeric_tokens(text: str) -> set[str]:
    tokens = set()
    for m in re.findall(r"\d+(?:\.\d+)?", text):
        try:
            tokens.add(norm_number(m))
        except Exception:
            pass
    return tokens


def strip_code_fence(text: str) -> str:
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    return text.strip()


def parse_json(text: str) -> Any:
    return json.loads(strip_code_fence(text))


# ---------------------------------------------------------------------------
# SIMPLE STRICT-SCHEMA VALIDATION
# ---------------------------------------------------------------------------

def validate_analyzer(obj: Any) -> tuple[bool, str | None]:
    if not isinstance(obj, dict):
        return False, "root_not_object"

    required = {
        "intent",
        "paper_ids",
        "metrics",
        "datasets",
        "needs_comparison",
        "needs_visualization",
        "needs_aggregation",
    }

    if set(obj.keys()) != required:
        return False, "wrong_keys"

    if obj["intent"] not in ANALYZER_SCHEMA["properties"]["intent"]["enum"]:
        return False, "invalid_intent"

    for key in ["paper_ids", "metrics", "datasets"]:
        if not isinstance(obj[key], list) or not all(
            isinstance(x, str) for x in obj[key]
        ):
            return False, f"invalid_{key}"

    for key in [
        "needs_comparison",
        "needs_visualization",
        "needs_aggregation",
    ]:
        if not isinstance(obj[key], bool):
            return False, f"invalid_{key}"

    return True, None


def validate_extraction(obj: Any) -> tuple[bool, str | None]:
    if not isinstance(obj, dict) or set(obj.keys()) != {"facts"}:
        return False, "wrong_keys"

    if not isinstance(obj["facts"], list):
        return False, "facts_not_list"

    required = {
        "entity",
        "field",
        "value",
        "unit",
        "condition",
        "quote",
    }

    for fact in obj["facts"]:
        if not isinstance(fact, dict) or set(fact.keys()) != required:
            return False, "invalid_fact_keys"

        if not isinstance(fact["entity"], str):
            return False, "entity_not_string"
        if not isinstance(fact["field"], str):
            return False, "field_not_string"
        if not isinstance(fact["value"], (str, int, float)):
            return False, "value_invalid"
        for key in ["unit", "condition", "quote"]:
            if not isinstance(fact[key], str):
                return False, f"{key}_not_string"

    return True, None


def validate_synthesis(obj: Any) -> tuple[bool, str | None]:
    if not isinstance(obj, dict) or set(obj.keys()) != {"claims"}:
        return False, "wrong_keys"

    if not isinstance(obj["claims"], list):
        return False, "claims_not_list"

    valid_types = {
        "reported",
        "derived",
        "comparison",
        "limitation",
        "inference",
    }

    for claim in obj["claims"]:
        if not isinstance(claim, dict):
            return False, "invalid_claim"

        if set(claim.keys()) != {
            "text",
            "support_ids",
            "claim_type",
        }:
            return False, "invalid_claim_keys"

        if not isinstance(claim["text"], str):
            return False, "claim_text_invalid"

        if not isinstance(claim["support_ids"], list):
            return False, "support_ids_invalid"

        if not all(isinstance(x, str) for x in claim["support_ids"]):
            return False, "support_id_type_invalid"

        if claim["claim_type"] not in valid_types:
            return False, "claim_type_invalid"

    return True, None


# ---------------------------------------------------------------------------
# OLLAMA
# ---------------------------------------------------------------------------

def post_json(
    endpoint: str,
    payload: dict[str, Any],
    timeout: int = 180,
) -> tuple[dict[str, Any], float]:

    start = time.perf_counter()

    response = requests.post(
        f"{OLLAMA_URL}{endpoint}",
        json=payload,
        timeout=timeout,
    )

    elapsed = time.perf_counter() - start
    response.raise_for_status()

    return response.json(), elapsed


def model_available(model: str) -> tuple[bool, str | None]:
    try:
        data, _ = post_json(
            "/api/show",
            {"model": model},
            timeout=30,
        )

        # Successful /api/show is enough.
        if data:
            return True, None

        return False, "empty_show_response"

    except requests.HTTPError as exc:
        try:
            detail = exc.response.text[:500]
        except Exception:
            detail = str(exc)
        return False, f"show_failed: {detail}"

    except Exception as exc:
        return False, f"show_failed: {exc}"


def warm_model(model: str) -> float:
    """
    Load the model once before scoring cases.

    Ollama's API supports keep_alive and exposes load_duration in responses.
    We use an empty generate request so benchmark latency is mostly warmed
    inference rather than the one-time model load.
    """
    payload = {
        "model": model,
        "prompt": "",
        "stream": False,
        "keep_alive": KEEP_ALIVE,
        "options": {
            "num_ctx": NUM_CTX,
            "temperature": TEMPERATURE,
            "seed": SEED,
        },
    }

    data, wall_time = post_json(
        "/api/generate",
        payload,
        timeout=240,
    )

    # Prefer Ollama's own load duration if present.
    load_ns = data.get("load_duration")
    if isinstance(load_ns, (int, float)):
        return float(load_ns) / 1_000_000_000.0

    return wall_time


def ollama_chat(
    model: str,
    system_prompt: str,
    user_prompt: str,
    schema: dict[str, Any],
    task: str,
) -> tuple[str, dict[str, Any], float]:

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        "stream": False,
        "keep_alive": KEEP_ALIVE,
        "think": False,
        "format": schema,
        "options": {
            "num_ctx": NUM_CTX,
            "temperature": TEMPERATURE,
            "seed": SEED,
            "num_predict": NUM_PREDICT[task],
        },
    }

    try:
        return _ollama_chat_request(payload)

    except requests.HTTPError as exc:
        # Compatibility fallback for older Ollama servers that reject
        # the think field. Keep all other benchmark controls unchanged.
        if exc.response is not None and exc.response.status_code == 400:
            payload.pop("think", None)
            return _ollama_chat_request(payload)
        raise


def _ollama_chat_request(
    payload: dict[str, Any]
) -> tuple[str, dict[str, Any], float]:

    data, wall_time = post_json(
        "/api/chat",
        payload,
        timeout=240,
    )

    message = data.get("message") or {}
    text = message.get("content", "")

    if not isinstance(text, str):
        text = str(text)

    return text, data, wall_time


# ---------------------------------------------------------------------------
# SCORING: QUERY ANALYSIS
# ---------------------------------------------------------------------------

def score_analyzer(pred: Any, gold: dict[str, Any]) -> tuple[float, dict[str, float]]:
    valid, _ = validate_analyzer(pred)
    if not valid:
        return 0.0, {
            "intent_accuracy": 0.0,
            "paper_id_accuracy": 0.0,
            "metric_accuracy": 0.0,
            "dataset_accuracy": 0.0,
            "comparison_accuracy": 0.0,
            "visualization_accuracy": 0.0,
            "aggregation_accuracy": 0.0,
        }

    def set_score(a: list[Any], b: list[Any]) -> float:
        return float({norm_text(x) for x in a} == {norm_text(x) for x in b})

    metrics = {
        "intent_accuracy": float(pred["intent"] == gold["intent"]),
        "paper_id_accuracy": set_score(
            pred["paper_ids"], gold["paper_ids"]
        ),
        "metric_accuracy": set_score(
            pred["metrics"], gold["metrics"]
        ),
        "dataset_accuracy": set_score(
            pred["datasets"], gold["datasets"]
        ),
        "comparison_accuracy": float(
            pred["needs_comparison"] == gold["needs_comparison"]
        ),
        "visualization_accuracy": float(
            pred["needs_visualization"] == gold["needs_visualization"]
        ),
        "aggregation_accuracy": float(
            pred["needs_aggregation"] == gold["needs_aggregation"]
        ),
    }

    return statistics.mean(metrics.values()), metrics


# ---------------------------------------------------------------------------
# SCORING: FACT EXTRACTION
# ---------------------------------------------------------------------------

def match_fact(pred: dict[str, Any], gold: dict[str, Any]) -> bool:
    return (
        canonical_field(pred["field"]) == canonical_field(gold["field"])
        and entity_match(pred["entity"], gold["entity"])
        and value_match(pred["value"], gold["value"])
        and unit_match(pred["unit"], gold["unit"])
        and condition_match(pred["condition"], gold["condition"])
    )


def score_extraction(
    pred: Any,
    gold: list[dict[str, Any]],
    source_text: str,
) -> tuple[float, dict[str, float]]:

    valid, _ = validate_extraction(pred)

    if not valid:
        return 0.0, {
            "fact_precision": 0.0,
            "fact_recall": 0.0,
            "fact_f1": 0.0,
            "entity_binding": 0.0,
            "value_exactness": 0.0,
            "quote_fidelity": 0.0,
        }

    predicted = pred["facts"]

    used_gold: set[int] = set()
    matched: list[tuple[dict[str, Any], dict[str, Any]]] = []

    for pf in predicted:
        found = None

        for idx, gf in enumerate(gold):
            if idx in used_gold:
                continue

            if match_fact(pf, gf):
                found = idx
                break

        if found is not None:
            used_gold.add(found)
            matched.append((pf, gold[found]))

    precision = len(matched) / len(predicted) if predicted else 0.0
    recall = len(matched) / len(gold) if gold else 1.0

    if precision + recall:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    entity_binding = (
        sum(
            entity_match(pf["entity"], gf["entity"])
            and value_match(pf["value"], gf["value"])
            for pf, gf in matched
        ) / len(matched)
        if matched else 0.0
    )

    value_exactness = (
        sum(
            value_match(pf["value"], gf["value"])
            and unit_match(pf["unit"], gf["unit"])
            for pf, gf in matched
        ) / len(matched)
        if matched else 0.0
    )

    normalized_source = norm_text(source_text)

    quote_valid = 0
    for pf in predicted:
        quote = norm_text(pf["quote"])
        if quote and quote in normalized_source:
            quote_valid += 1

    quote_fidelity = (
        quote_valid / len(predicted)
        if predicted
        else 0.0
    )

    metrics = {
        "fact_precision": precision,
        "fact_recall": recall,
        "fact_f1": f1,
        "entity_binding": entity_binding,
        "value_exactness": value_exactness,
        "quote_fidelity": quote_fidelity,
    }

    return statistics.mean(metrics.values()), metrics


# ---------------------------------------------------------------------------
# SCORING: GROUNDED SYNTHESIS
# ---------------------------------------------------------------------------

def score_synthesis(
    pred: Any,
    case: dict[str, Any],
) -> tuple[float, dict[str, float]]:

    valid, _ = validate_synthesis(pred)

    if not valid:
        return 0.0, {
            "answer_key_match": 0.0,
            "support_recall": 0.0,
            "support_validity": 0.0,
            "claim_support_coverage": 0.0,
            "numeric_fidelity": 0.0,
        }

    claims = pred["claims"]
    gold = case["gold"]
    evidence = case["evidence"]

    evidence_ids = {item["id"] for item in evidence}
    evidence_text = " ".join(item["text"] for item in evidence)
    claim_text = " ".join(c["text"] for c in claims)

    normalized_claim = norm_text(claim_text)

    required_terms = [
        norm_text(x)
        for x in gold["required_terms"]
    ]

    required_numbers = {
        norm_number(x)
        for x in gold["required_numbers"]
    }

    answer_key = float(
        all(term in normalized_claim for term in required_terms)
        and all(
            number in extract_numeric_tokens(claim_text)
            for number in required_numbers
        )
    )

    predicted_support = set()

    valid_support_claims = 0

    for claim in claims:
        ids = claim["support_ids"]
        predicted_support.update(ids)

        if ids and all(x in evidence_ids for x in ids):
            valid_support_claims += 1

    required_support = set(gold["required_support_ids"])

    support_recall = (
        len(required_support & predicted_support) / len(required_support)
        if required_support
        else 1.0
    )

    total_support_ids = sum(
        len(c["support_ids"])
        for c in claims
    )

    valid_support_ids = sum(
        1
        for c in claims
        for support_id in c["support_ids"]
        if support_id in evidence_ids
    )

    support_validity = (
        valid_support_ids / total_support_ids
        if total_support_ids
        else 0.0
    )

    claim_support_coverage = (
        valid_support_claims / len(claims)
        if claims
        else 0.0
    )

    generated_numbers = extract_numeric_tokens(claim_text)
    source_numbers = extract_numeric_tokens(evidence_text)

    # No generated number should be absent from the evidence.
    numeric_fidelity = (
        len(generated_numbers & source_numbers)
        / len(generated_numbers)
        if generated_numbers
        else 1.0
    )

    metrics = {
        "answer_key_match": answer_key,
        "support_recall": support_recall,
        "support_validity": support_validity,
        "claim_support_coverage": claim_support_coverage,
        "numeric_fidelity": numeric_fidelity,
    }

    return statistics.mean(metrics.values()), metrics


# ---------------------------------------------------------------------------
# CASE EXECUTION
# ---------------------------------------------------------------------------

def run_analyzer_case(model: str, case: dict[str, Any]) -> CaseResult:
    try:
        text, raw, elapsed = ollama_chat(
            model,
            ANALYZER_SYSTEM,
            f"User query:\n{case['query']}",
            ANALYZER_SCHEMA,
            "query_analysis",
        )

        try:
            parsed = parse_json(text)
            schema_valid, schema_error = validate_analyzer(parsed)
        except Exception as exc:
            parsed = None
            schema_valid = False
            schema_error = f"json_parse_failed: {exc}"

        score, metrics = (
            score_analyzer(parsed, case["gold"])
            if parsed is not None
            else (0.0, {})
        )

        return CaseResult(
            model=model,
            task="query_analysis",
            case_id=case["id"],
            available=True,
            success=True,
            schema_valid=schema_valid,
            elapsed_seconds=elapsed,
            load_duration_seconds=None,
            prompt_tokens=raw.get("prompt_eval_count"),
            output_tokens=raw.get("eval_count"),
            response_text=text,
            parsed_json=parsed,
            score=score,
            metrics=metrics,
            error=schema_error,
        )

    except Exception as exc:
        return CaseResult(
            model=model,
            task="query_analysis",
            case_id=case["id"],
            available=True,
            success=False,
            schema_valid=False,
            elapsed_seconds=0.0,
            load_duration_seconds=None,
            prompt_tokens=None,
            output_tokens=None,
            response_text="",
            parsed_json=None,
            score=0.0,
            metrics={},
            error=str(exc),
        )


def run_extraction_case(model: str, case: dict[str, Any]) -> CaseResult:
    try:
        text, raw, elapsed = ollama_chat(
            model,
            EXTRACTION_SYSTEM,
            f"Passage:\n{case['text']}",
            EXTRACTION_SCHEMA,
            "fact_extraction",
        )

        try:
            parsed = parse_json(text)
            schema_valid, schema_error = validate_extraction(parsed)
        except Exception as exc:
            parsed = None
            schema_valid = False
            schema_error = f"json_parse_failed: {exc}"

        score, metrics = (
            score_extraction(
                parsed,
                case["gold"],
                case["text"],
            )
            if parsed is not None
            else (0.0, {})
        )

        return CaseResult(
            model=model,
            task="fact_extraction",
            case_id=case["id"],
            available=True,
            success=True,
            schema_valid=schema_valid,
            elapsed_seconds=elapsed,
            load_duration_seconds=None,
            prompt_tokens=raw.get("prompt_eval_count"),
            output_tokens=raw.get("eval_count"),
            response_text=text,
            parsed_json=parsed,
            score=score,
            metrics=metrics,
            error=schema_error,
        )

    except Exception as exc:
        return CaseResult(
            model=model,
            task="fact_extraction",
            case_id=case["id"],
            available=True,
            success=False,
            schema_valid=False,
            elapsed_seconds=0.0,
            load_duration_seconds=None,
            prompt_tokens=None,
            output_tokens=None,
            response_text="",
            parsed_json=None,
            score=0.0,
            metrics={},
            error=str(exc),
        )


def run_synthesis_case(model: str, case: dict[str, Any]) -> CaseResult:
    evidence = json.dumps(
        case["evidence"],
        ensure_ascii=False,
        indent=2,
    )

    prompt = (
        f"Question:\n{case['query']}\n\n"
        f"Evidence:\n{evidence}"
    )

    try:
        text, raw, elapsed = ollama_chat(
            model,
            SYNTHESIS_SYSTEM,
            prompt,
            SYNTHESIS_SCHEMA,
            "grounded_synthesis",
        )

        try:
            parsed = parse_json(text)
            schema_valid, schema_error = validate_synthesis(parsed)
        except Exception as exc:
            parsed = None
            schema_valid = False
            schema_error = f"json_parse_failed: {exc}"

        score, metrics = (
            score_synthesis(parsed, case)
            if parsed is not None
            else (0.0, {})
        )

        return CaseResult(
            model=model,
            task="grounded_synthesis",
            case_id=case["id"],
            available=True,
            success=True,
            schema_valid=schema_valid,
            elapsed_seconds=elapsed,
            load_duration_seconds=None,
            prompt_tokens=raw.get("prompt_eval_count"),
            output_tokens=raw.get("eval_count"),
            response_text=text,
            parsed_json=parsed,
            score=score,
            metrics=metrics,
            error=schema_error,
        )

    except Exception as exc:
        return CaseResult(
            model=model,
            task="grounded_synthesis",
            case_id=case["id"],
            available=True,
            success=False,
            schema_valid=False,
            elapsed_seconds=0.0,
            load_duration_seconds=None,
            prompt_tokens=None,
            output_tokens=None,
            response_text="",
            parsed_json=None,
            score=0.0,
            metrics={},
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# REPORTING
# ---------------------------------------------------------------------------

def summarize(
    results: list[CaseResult],
    load_times: dict[str, float],
    availability: dict[str, tuple[bool, str | None]],
) -> None:

    print("\n" + "=" * 110)
    print("LLM BENCHMARK V2 — SUMMARY")
    print("=" * 110)

    for model in MODELS:
        available, availability_error = availability[model]

        print(f"\n{model}")

        if not available:
            print(f"  STATUS: UNAVAILABLE")
            print(f"  ERROR : {availability_error}")
            continue

        print(
            f"  Cold load: {load_times.get(model, 0.0):.2f}s"
        )

        model_results = [
            r for r in results
            if r.model == model
        ]

        for task in [
            "query_analysis",
            "fact_extraction",
            "grounded_synthesis",
        ]:

            rows = [
                r for r in model_results
                if r.task == task
            ]

            if not rows:
                continue

            schema_rate = (
                sum(r.schema_valid for r in rows)
                / len(rows)
            )

            scores = [r.score for r in rows]

            latencies = [
                r.elapsed_seconds
                for r in rows
                if r.success
            ]

            output_tokens = [
                r.output_tokens
                for r in rows
                if r.output_tokens is not None
            ]

            med_latency = (
                statistics.median(latencies)
                if latencies else 0.0
            )

            med_tokens = (
                statistics.median(output_tokens)
                if output_tokens else 0.0
            )

            print(
                f"  {task:22} "
                f"score={statistics.mean(scores):.3f} "
                f"schema={schema_rate:.1%} "
                f"median={med_latency:.2f}s "
                f"output_tok={med_tokens:.0f}"
            )

    print("\n" + "-" * 110)
    print("ROLE RANKINGS")
    print("-" * 110)

    for task in [
        "query_analysis",
        "fact_extraction",
        "grounded_synthesis",
    ]:

        ranking = []

        for model in MODELS:
            if not availability[model][0]:
                continue

            rows = [
                r for r in results
                if r.model == model and r.task == task
            ]

            if not rows:
                continue

            score = statistics.mean(r.score for r in rows)

            latencies = [
                r.elapsed_seconds
                for r in rows
                if r.success
            ]

            median_latency = (
                statistics.median(latencies)
                if latencies
                else math.inf
            )

            ranking.append(
                (score, -median_latency, model)
            )

        ranking.sort(reverse=True)

        print(f"\n{task}")
        for idx, (score, neg_latency, model) in enumerate(
            ranking,
            start=1,
        ):
            print(
                f"  {idx}. {model:24} "
                f"score={score:.3f} "
                f"median_latency={-neg_latency:.2f}s"
            )


def main():

    output_dir = Path(
        "storage/evaluation/llm_benchmark"
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results: list[CaseResult] = []
    availability: dict[str, tuple[bool, str | None]] = {}
    load_times: dict[str, float] = {}

    total_cases = (
        len(ANALYZER_CASES)
        + len(EXTRACTION_CASES)
        + len(SYNTHESIS_CASES)
    )

    print("=" * 110)
    print("LOCAL LLM BAKE-OFF V2")
    print("=" * 110)

    print(f"Models:      {len(MODELS)}")
    print(f"Cases/model: {total_cases}")
    print(
        f"Total calls: {len(MODELS) * total_cases}"
    )
    print(
        f"num_ctx={NUM_CTX}, temperature={TEMPERATURE}, "
        f"seed={SEED}, keep_alive={KEEP_ALIVE}"
    )
    print(
        "Thinking disabled where supported; strict JSON Schema; "
        "bounded num_predict."
    )

    # -----------------------------------------------------------------------
    # Preflight
    # -----------------------------------------------------------------------

    print("\n--- Preflight ---")

    for model in MODELS:
        available, error = model_available(model)
        availability[model] = (available, error)

        if available:
            print(f"[OK]   {model}")
        else:
            print(f"[SKIP] {model}: {error}")

    # -----------------------------------------------------------------------
    # Benchmark
    # -----------------------------------------------------------------------

    for model in MODELS:

        if not availability[model][0]:
            continue

        print(f"\n>>> {model}")

        # One-time cold load.
        print("  warming model...")
        try:
            load_times[model] = warm_model(model)
            print(
                f"  cold load: "
                f"{load_times[model]:.2f}s"
            )
        except Exception as exc:
            availability[model] = (
                False,
                f"warmup_failed: {exc}",
            )
            print(f"  FAILED warmup: {exc}")
            continue

        # Query analysis
        for case in ANALYZER_CASES:
            print(
                f"  analysis     {case['id']}",
                flush=True,
            )
            results.append(
                run_analyzer_case(model, case)
            )

        # Fact extraction
        for case in EXTRACTION_CASES:
            print(
                f"  extraction   {case['id']}",
                flush=True,
            )
            results.append(
                run_extraction_case(model, case)
            )

        # Grounded synthesis
        for case in SYNTHESIS_CASES:
            print(
                f"  synthesis    {case['id']}",
                flush=True,
            )
            results.append(
                run_synthesis_case(model, case)
            )

    # -----------------------------------------------------------------------
    # Save raw results
    # -----------------------------------------------------------------------

    raw_path = (
        output_dir /
        "llm_benchmark_v2_raw.json"
    )

    with raw_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "config": {
                    "models": MODELS,
                    "num_ctx": NUM_CTX,
                    "temperature": TEMPERATURE,
                    "seed": SEED,
                    "keep_alive": KEEP_ALIVE,
                    "num_predict": NUM_PREDICT,
                },
                "availability": availability,
                "cold_load_seconds": load_times,
                "results": [asdict(r) for r in results],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # -----------------------------------------------------------------------
    # Save compact summary
    # -----------------------------------------------------------------------

    summary_rows = []

    for model in MODELS:

        available, _ = availability[model]

        if not available:
            summary_rows.append(
                {
                    "model": model,
                    "available": False,
                }
            )
            continue

        row: dict[str, Any] = {
            "model": model,
            "available": True,
            "cold_load_seconds": load_times.get(model),
        }

        for task in [
            "query_analysis",
            "fact_extraction",
            "grounded_synthesis",
        ]:

            task_rows = [
                r for r in results
                if r.model == model and r.task == task
            ]

            if not task_rows:
                continue

            row[f"{task}_score"] = statistics.mean(
                r.score for r in task_rows
            )

            row[f"{task}_schema_rate"] = (
                sum(r.schema_valid for r in task_rows)
                / len(task_rows)
            )

            success_latencies = [
                r.elapsed_seconds
                for r in task_rows
                if r.success
            ]

            row[f"{task}_median_latency"] = (
                statistics.median(success_latencies)
                if success_latencies
                else None
            )

            output_tokens = [
                r.output_tokens
                for r in task_rows
                if r.output_tokens is not None
            ]

            row[f"{task}_median_output_tokens"] = (
                statistics.median(output_tokens)
                if output_tokens
                else None
            )

        summary_rows.append(row)

    summary_path = (
        output_dir /
        "llm_benchmark_v2_summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary_rows,
            f,
            indent=2,
            ensure_ascii=False,
        )

    summarize(
        results,
        load_times,
        availability,
    )

    print("\n" + "=" * 110)
    print("OUTPUT")
    print("=" * 110)
    print(f"Raw results : {raw_path}")
    print(f"Summary     : {summary_path}")


if __name__ == "__main__":
    main()
