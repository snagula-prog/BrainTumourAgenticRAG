"""
LLM Benchmark V3 — Expanded, role-specific model bake-off.

Run from the repository root:
    python evaluation/llm_benchmark_v3.py

This is an expanded capability benchmark, not the final paper-corpus evaluation.
It uses gold-labelled controlled cases to compare:
  - query analysis
  - structured fact extraction
  - evidence-grounded synthesis

Inference roles use different candidate pools to avoid spending time testing
models on roles where V2 gave strong reasons to deprioritize them.

Outputs:
  storage/evaluation/llm_benchmark_v3/llm_benchmark_v3_raw.json
  storage/evaluation/llm_benchmark_v3/llm_benchmark_v3_summary.json
"""

from __future__ import annotations

import json
import re
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests


OLLAMA_URL = "http://localhost:11434"
NUM_CTX = 4096
TEMPERATURE = 0.0
SEED = 42
KEEP_ALIVE = "5m"
REQUEST_TIMEOUT = 240

ROLE_MODELS = {
    "query_analysis": [
        "qwen3:4b-instruct",
        "phi4-mini",
        "qwen3.5:4b",
        "granite4:3b",
        "gemma3:4b",
    ],
    "fact_extraction": [
        "qwen3:4b-instruct",
        "phi4-mini",
        "qwen3.5:9b",
        "gemma3:4b",
    ],
    "grounded_synthesis": [
        "qwen3:4b-instruct",
        "phi4-mini",
        "qwen3.5:4b",
        "qwen3.5:9b",
        "gemma3:4b",
    ],
}

NUM_PREDICT = {
    "query_analysis": 192,
    "fact_extraction": 384,
    "grounded_synthesis": 256,
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
                "evidence_qa", "fact_lookup", "extraction",
                "comparison", "quantitative", "visualization", "multi_step",
            ],
        },
        "paper_ids": {"type": "array", "items": {"type": "string"}},
        "metrics": {"type": "array", "items": {"type": "string"}},
        "datasets": {"type": "array", "items": {"type": "string"}},
        "needs_comparison": {"type": "boolean"},
        "needs_visualization": {"type": "boolean"},
        "needs_aggregation": {"type": "boolean"},
    },
    "required": [
        "intent", "paper_ids", "metrics", "datasets",
        "needs_comparison", "needs_visualization", "needs_aggregation",
    ],
    "additionalProperties": False,
}

FACT_SCHEMA = {
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
                    "entity", "field", "value", "unit", "condition", "quote",
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
                            "reported", "derived", "comparison",
                            "limitation", "inference",
                        ],
                    },
                },
                "required": ["text", "support_ids", "claim_type"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["claims"],
    "additionalProperties": False,
}


ANALYZER_SYSTEM = f"""
You analyze user requests for a research-paper assistant. Return only JSON.

Rules:
- Allowed intents: evidence_qa, fact_lookup, extraction, comparison,
  quantitative, visualization, multi_step.
- paper_ids: ONLY explicit IDs from the query (e.g. paper_008). Never invent IDs.
- metrics: only metrics explicitly requested (e.g. accuracy, sensitivity, AUC,
  parameters, FLOPs, latency).
- datasets: ONLY named datasets explicitly mentioned in the query. A generic
  phrase such as "what dataset" is not a named dataset.
- needs_comparison is true when the task compares papers, methods, models,
  or their results.
- needs_visualization is true when the user requests a graph, plot, chart, or
  other visual representation.
- needs_aggregation is true for best/highest/lowest/average/mean/rank/top-N
  or other aggregation/ranking operations.
- Ask for no implicit data and do not answer the research question.
Schema:
{json.dumps(ANALYZER_SCHEMA, separators=(",", ":"))}
"""

FACT_SYSTEM = f"""
Extract atomic facts explicitly stated in the supplied passage. Return only JSON.

Rules:
- Emit one fact per entity + field + value.
- Preserve entity-to-value relationships.
- Keep numeric value and unit separate (e.g. value "92.4", unit "%").
- Keep evaluation conditions such as dataset, split, hardware, or cross-validation
  in condition if explicitly given.
- quote must be a contiguous, exact excerpt from the supplied passage. It must
  include the fact's value.
- Do not invent values for omitted metrics. If a metric is explicitly stated as
  not reported, do not create a numeric fact for it.
- Do not merge separate limitations into one record.
Schema:
{json.dumps(FACT_SCHEMA, separators=(",", ":"))}
"""

SYNTHESIS_SYSTEM = f"""
Answer using only the supplied evidence items. Return only JSON.

Rules:
- Each factual claim must cite one or more evidence IDs in support_ids.
- Use only IDs supplied in the evidence.
- Do not invent numbers, datasets, methods, model names, conditions, or conclusions.
- If evidence does not answer the question, return claims=[] or explicitly state
  that the information is not reported / not specified / cannot be determined.
- For comparisons with different datasets or evaluation splits, state the
  comparability limitation instead of declaring a winner.
- Keep the response concise. No outside knowledge.
Schema:
{json.dumps(SYNTHESIS_SCHEMA, separators=(",", ":"))}
"""


# ---------------------------------------------------------------------------
# GOLD-LABELLED QUERY ANALYSIS CASES (16)
# ---------------------------------------------------------------------------

A = [
    {
        "id": "A01",
        "query": "What is the best reported accuracy among the models evaluated in paper_008?",
        "gold": dict(intent="quantitative", paper_ids=["paper_008"],
                     metrics=["accuracy"], datasets=[], needs_comparison=False,
                     needs_visualization=False, needs_aggregation=True),
    },
    {
        "id": "A02",
        "query": "Compare the reported sensitivity and specificity of paper_010 and paper_011.",
        "gold": dict(intent="comparison", paper_ids=["paper_010", "paper_011"],
                     metrics=["sensitivity", "specificity"], datasets=[],
                     needs_comparison=True, needs_visualization=False,
                     needs_aggregation=False),
    },
    {
        "id": "A03",
        "query": "Plot the reported accuracy of the top three comparable methods across the relevant papers.",
        "gold": dict(intent="visualization", paper_ids=[], metrics=["accuracy"],
                     datasets=[], needs_comparison=True, needs_visualization=True,
                     needs_aggregation=True),
    },
    {
        "id": "A04",
        "query": "What dataset was used by the authors, and what accuracy did they report?",
        "gold": dict(intent="fact_lookup", paper_ids=[], metrics=["accuracy"],
                     datasets=[], needs_comparison=False, needs_visualization=False,
                     needs_aggregation=False),
    },
    {
        "id": "A05",
        "query": "Which paper reports the highest sensitivity, and what was the value?",
        "gold": dict(intent="quantitative", paper_ids=[], metrics=["sensitivity"],
                     datasets=[], needs_comparison=True, needs_visualization=False,
                     needs_aggregation=True),
    },
    {
        "id": "A06",
        "query": "Summarize the main limitations discussed in paper_012.",
        "gold": dict(intent="evidence_qa", paper_ids=["paper_012"], metrics=[],
                     datasets=[], needs_comparison=False, needs_visualization=False,
                     needs_aggregation=False),
    },
    {
        "id": "A07",
        "query": "Extract the architecture, dataset, accuracy, and limitations from paper_004.",
        "gold": dict(intent="extraction", paper_ids=["paper_004"], metrics=["accuracy"],
                     datasets=[], needs_comparison=False, needs_visualization=False,
                     needs_aggregation=False),
    },
    {
        "id": "A08",
        "query": "Build a comparison table for paper_001 and paper_003 covering sensitivity and AUC.",
        "gold": dict(intent="comparison", paper_ids=["paper_001", "paper_003"],
                     metrics=["sensitivity", "AUC"], datasets=[],
                     needs_comparison=True, needs_visualization=False,
                     needs_aggregation=False),
    },
    {
        "id": "A09",
        "query": "Create a bar chart of sensitivity for paper_001 and paper_003.",
        "gold": dict(intent="visualization", paper_ids=["paper_001", "paper_003"],
                     metrics=["sensitivity"], datasets=[], needs_comparison=True,
                     needs_visualization=True, needs_aggregation=False),
    },
    {
        "id": "A10",
        "query": "Does paper_009 report sensitivity?",
        "gold": dict(intent="fact_lookup", paper_ids=["paper_009"],
                     metrics=["sensitivity"], datasets=[], needs_comparison=False,
                     needs_visualization=False, needs_aggregation=False),
    },
    {
        "id": "A11",
        "query": "Calculate mean accuracy across the three methods and rank them from highest to lowest.",
        "gold": dict(intent="quantitative", paper_ids=[], metrics=["accuracy"],
                     datasets=[], needs_comparison=True, needs_visualization=False,
                     needs_aggregation=True),
    },
    {
        "id": "A12",
        "query": "Compare the reported results on BraTS 2020 and BraTS 2021.",
        "gold": dict(intent="comparison", paper_ids=[],
                     metrics=[], datasets=["BraTS 2020", "BraTS 2021"],
                     needs_comparison=True, needs_visualization=False,
                     needs_aggregation=False),
    },
    {
        "id": "A13",
        "query": "Explain whether two papers are comparable when they use different dataset splits.",
        "gold": dict(intent="comparison", paper_ids=[], metrics=[], datasets=[],
                     needs_comparison=True, needs_visualization=False,
                     needs_aggregation=False),
    },
    {
        "id": "A14",
        "query": "List every model and its reported top-1 accuracy in paper_008.",
        "gold": dict(intent="extraction", paper_ids=["paper_008"],
                     metrics=["top-1 accuracy"], datasets=[],
                     needs_comparison=False, needs_visualization=False,
                     needs_aggregation=False),
    },
    {
        "id": "A15",
        "query": "Which study reports the lowest inference latency, and under what hardware condition?",
        "gold": dict(intent="quantitative", paper_ids=[], metrics=["inference latency"],
                     datasets=[], needs_comparison=True, needs_visualization=False,
                     needs_aggregation=True),
    },
    {
        "id": "A16",
        "query": "Graph parameter count against accuracy for MobileNetV2, EfficientNet-B0, and ResNet in paper_008.",
        "gold": dict(intent="visualization", paper_ids=["paper_008"],
                     metrics=["parameter count", "accuracy"], datasets=[],
                     needs_comparison=True, needs_visualization=True,
                     needs_aggregation=False),
    },
]


# ---------------------------------------------------------------------------
# GOLD-LABELLED EXTRACTION CASES (16)
# Gold records use canonical entity/field/value/unit; conditions are scored
# separately so verbose free-form wording doesn't corrupt core record F1.
# ---------------------------------------------------------------------------

E = [
    {
        "id": "E01",
        "text": "EfficientNet-B0 achieved a top-1 accuracy of 76.3% and top-5 accuracy of 93.2% on ImageNet using 5.3M parameters.",
        "gold": [
            ("EfficientNet-B0", "top-1 accuracy", "76.3", "%"),
            ("EfficientNet-B0", "top-5 accuracy", "93.2", "%"),
            ("EfficientNet-B0", "parameters", "5.3", "M"),
        ],
        "conditions": ["on ImageNet", "on ImageNet", "on ImageNet"],
    },
    {
        "id": "E02",
        "text": "The proposed model achieved 97.8% sensitivity and 95.4% specificity on the test set.",
        "gold": [
            ("proposed model", "sensitivity", "97.8", "%"),
            ("proposed model", "specificity", "95.4", "%"),
        ],
        "conditions": ["test set", "test set"],
    },
    {
        "id": "E03",
        "text": "The experiment used the BraTS dataset with 3,000 training images and 500 validation images.",
        "gold": [
            ("BraTS dataset", "dataset", "BraTS", ""),
            ("BraTS dataset", "training images", "3000", "images"),
            ("BraTS dataset", "validation images", "500", "images"),
        ],
        "conditions": ["", "", ""],
    },
    {
        "id": "E04",
        "text": "The method obtained an accuracy of 92.4%, while sensitivity was 90.1% and specificity was 94.7%.",
        "gold": [
            ("method", "accuracy", "92.4", "%"),
            ("method", "sensitivity", "90.1", "%"),
            ("method", "specificity", "94.7", "%"),
        ],
        "conditions": ["", "", ""],
    },
    {
        "id": "E05",
        "text": "The authors identify two limitations: the dataset is small, and external validation was not performed.",
        "gold": [
            ("study limitation", "dataset size", "small", ""),
            ("study limitation", "external validation", "absence", ""),
        ],
        "conditions": ["", ""],
    },
    {
        "id": "E06",
        "text": "Model X reached 91.2% accuracy on the validation split and 88.9% accuracy on the held-out test set.",
        "gold": [
            ("Model X", "accuracy", "91.2", "%"),
            ("Model X", "accuracy", "88.9", "%"),
        ],
        "conditions": ["validation split", "held-out test set"],
    },
    {
        "id": "E07",
        "text": "Model Y contains 11.2 million parameters and requires 1.7 GFLOPs per inference.",
        "gold": [
            ("Model Y", "parameters", "11.2", "M"),
            ("Model Y", "FLOPs", "1.7", "G"),
        ],
        "conditions": ["", "per inference"],
    },
    {
        "id": "E08",
        "text": "Under five-fold cross-validation, the proposed network achieved 93.6% accuracy, 91.4% precision, and 92.2% recall.",
        "gold": [
            ("proposed network", "accuracy", "93.6", "%"),
            ("proposed network", "precision", "91.4", "%"),
            ("proposed network", "recall", "92.2", "%"),
        ],
        "conditions": ["five-fold cross-validation"] * 3,
    },
    {
        "id": "E09",
        "text": "For BraTS 2021, the model obtained a Dice score of 91.2% for enhancing tumor and 89.4% for whole tumor.",
        "gold": [
            ("model", "Dice enhancing tumor", "91.2", "%"),
            ("model", "Dice whole tumor", "89.4", "%"),
        ],
        "conditions": ["BraTS 2021"] * 2,
    },
    {
        "id": "E10",
        "text": "The MRI dataset included 1,250 studies: 800 for training, 200 for validation, and 250 for testing.",
        "gold": [
            ("MRI dataset", "total studies", "1250", "studies"),
            ("MRI dataset", "training studies", "800", "studies"),
            ("MRI dataset", "validation studies", "200", "studies"),
            ("MRI dataset", "test studies", "250", "studies"),
        ],
        "conditions": ["", "", "", ""],
    },
    {
        "id": "E11",
        "text": "On the same test split, Model A achieved 89.1% accuracy, Model B achieved 92.4%, and Model C achieved 90.7%.",
        "gold": [
            ("Model A", "accuracy", "89.1", "%"),
            ("Model B", "accuracy", "92.4", "%"),
            ("Model C", "accuracy", "90.7", "%"),
        ],
        "conditions": ["same test split"] * 3,
    },
    {
        "id": "E12",
        "text": "On the held-out test set, the proposed system achieved an AUC of 0.981, sensitivity of 94.1%, and specificity of 96.2%.",
        "gold": [
            ("proposed system", "AUC", "0.981", ""),
            ("proposed system", "sensitivity", "94.1", "%"),
            ("proposed system", "specificity", "96.2", "%"),
        ],
        "conditions": ["held-out test set"] * 3,
    },
    {
        "id": "E13",
        "text": "Training took 3.5 hours, and average inference required 42 milliseconds per scan on an NVIDIA T4 GPU.",
        "gold": [
            ("model", "training time", "3.5", "hours"),
            ("model", "inference latency", "42", "ms"),
        ],
        "conditions": ["", "per scan on an NVIDIA T4 GPU"],
    },
    {
        "id": "E14",
        "text": "The authors list class imbalance and evaluation at a single center as the main study limitations.",
        "gold": [
            ("study limitation", "class imbalance", "present", ""),
            ("study limitation", "single-center evaluation", "present", ""),
        ],
        "conditions": ["", ""],
    },
    {
        "id": "E15",
        "text": "The baseline model achieved 89.4% accuracy; the proposed model achieved 92.8% accuracy on the same test set.",
        "gold": [
            ("baseline model", "accuracy", "89.4", "%"),
            ("proposed model", "accuracy", "92.8", "%"),
        ],
        "conditions": ["same test set", "same test set"],
    },
    {
        "id": "E16",
        "text": "The reported accuracy was 94.2%. Sensitivity, specificity, and AUC were not reported.",
        "gold": [
            ("study", "accuracy", "94.2", "%"),
        ],
        "conditions": [""],
    },
]


# ---------------------------------------------------------------------------
# GROUNDED SYNTHESIS CASES (16)
# required_anchor_groups are alternative acceptable phrases for each required
# semantic anchor. Scores remain heuristic and require manual semantic review.
# ---------------------------------------------------------------------------

S = [
    {
        "id": "S01",
        "query": "Which model achieved the highest accuracy?",
        "evidence": [
            {"id": "F1", "text": "Model A achieved 91.2% accuracy."},
            {"id": "F2", "text": "Model B achieved 94.7% accuracy."},
            {"id": "F3", "text": "Model C achieved 92.1% accuracy."},
        ],
        "anchor_groups": [["model b"], ["highest accuracy", "best accuracy", "performed best"]],
        "required_numbers": ["94.7"],
        "required_support": ["F2"],
        "expect_abstain": False,
    },
    {
        "id": "S02",
        "query": "Which method had higher sensitivity? Compare both values.",
        "evidence": [
            {"id": "F1", "text": "Method A reported sensitivity of 91.0%."},
            {"id": "F2", "text": "Method B reported sensitivity of 95.2%."},
        ],
        "anchor_groups": [["method b"], ["higher sensitivity", "greater sensitivity"]],
        "required_numbers": ["95.2", "91.0"],
        "required_support": ["F1", "F2"],
        "expect_abstain": False,
    },
    {
        "id": "S03",
        "query": "State the reported result.",
        "evidence": [
            {"id": "F1", "text": "The model achieved 88.4% accuracy on the test set."},
        ],
        "anchor_groups": [["88.4"], ["accuracy"], ["test set"]],
        "required_numbers": ["88.4"],
        "required_support": ["F1"],
        "expect_abstain": False,
    },
    {
        "id": "S04",
        "query": "Are these results directly comparable?",
        "evidence": [
            {"id": "F1", "text": "Method A was evaluated on BraTS 2021 using the validation split."},
            {"id": "F2", "text": "Method B was evaluated on BraTS 2020 using the held-out test set."},
        ],
        "anchor_groups": [["not directly comparable", "not comparable"], ["brats 2021"], ["brats 2020"]],
        "required_numbers": [],
        "required_support": ["F1", "F2"],
        "expect_abstain": False,
    },
    {
        "id": "S05",
        "query": "What limitations did the authors report?",
        "evidence": [
            {"id": "F1", "text": "The study is limited by a small single-center dataset."},
            {"id": "F2", "text": "External validation was not performed."},
        ],
        "anchor_groups": [["small single-center dataset", "small dataset"], ["external validation"]],
        "required_numbers": [],
        "required_support": ["F1", "F2"],
        "expect_abstain": False,
    },
    {
        "id": "S06",
        "query": "What specificity did the paper report?",
        "evidence": [
            {"id": "F1", "text": "The model achieved 91.4% accuracy on the validation set."},
        ],
        "anchor_groups": [],
        "required_numbers": [],
        "required_support": [],
        "expect_abstain": True,
    },
    {
        "id": "S07",
        "query": "Summarize both accuracy results without treating them as the same split.",
        "evidence": [
            {"id": "F1", "text": "Model X had 91.2% accuracy on the validation split."},
            {"id": "F2", "text": "Model X had 88.9% accuracy on the held-out test set."},
        ],
        "anchor_groups": [["91.2"], ["88.9"], ["validation"], ["test set"]],
        "required_numbers": ["91.2", "88.9"],
        "required_support": ["F1", "F2"],
        "expect_abstain": False,
    },
    {
        "id": "S08",
        "query": "Which model uses fewer parameters?",
        "evidence": [
            {"id": "F1", "text": "Model A contains 12.0 million parameters."},
            {"id": "F2", "text": "Model B contains 8.5 million parameters."},
        ],
        "anchor_groups": [["model b"], ["fewer parameters", "less parameters", "smaller parameter count"]],
        "required_numbers": ["8.5", "12.0"],
        "required_support": ["F1", "F2"],
        "expect_abstain": False,
    },
    {
        "id": "S09",
        "query": "What was the test-set accuracy?",
        "evidence": [
            {"id": "F1", "text": "The paper reports 92.1% accuracy on the validation set."},
        ],
        "anchor_groups": [],
        "required_numbers": [],
        "required_support": [],
        "expect_abstain": True,
    },
    {
        "id": "S10",
        "query": "Which limitations are stated in the evidence?",
        "evidence": [
            {"id": "F1", "text": "The dataset showed class imbalance."},
            {"id": "F2", "text": "The study was evaluated at a single center."},
        ],
        "anchor_groups": [["class imbalance"], ["single center", "single-center"]],
        "required_numbers": [],
        "required_support": ["F1", "F2"],
        "expect_abstain": False,
    },
    {
        "id": "S11",
        "query": "Compare the sensitivity values.",
        "evidence": [
            {"id": "F1", "text": "Method A reached 90.0% sensitivity."},
            {"id": "F2", "text": "Method B reached 93.5% sensitivity."},
        ],
        "anchor_groups": [["method b"], ["higher sensitivity", "greater sensitivity", "higher value"]],
        "required_numbers": ["90.0", "93.5"],
        "required_support": ["F1", "F2"],
        "expect_abstain": False,
    },
    {
        "id": "S12",
        "query": "Which model had the highest accuracy?",
        "evidence": [
            {"id": "F1", "text": "Model A reached 93.0% accuracy."},
            {"id": "F2", "text": "Model B reached 94.3% accuracy."},
            {"id": "F3", "text": "Model C reached 91.8% accuracy."},
        ],
        "anchor_groups": [["model b"], ["highest accuracy", "best accuracy", "performed best"]],
        "required_numbers": ["94.3"],
        "required_support": ["F2"],
        "expect_abstain": False,
    },
    {
        "id": "S13",
        "query": "Can we conclude Model B is better than Model A?",
        "evidence": [
            {"id": "F1", "text": "Model A achieved 93.1% accuracy on BraTS 2021 validation data."},
            {"id": "F2", "text": "Model B achieved 95.0% accuracy on BraTS 2020 test data."},
        ],
        "anchor_groups": [["not directly comparable", "cannot conclude", "cannot reliably conclude", "not a fair comparison"]],
        "required_numbers": [],
        "required_support": ["F1", "F2"],
        "expect_abstain": False,
    },
    {
        "id": "S14",
        "query": "Which backbone architecture did the authors use?",
        "evidence": [
            {"id": "F1", "text": "The proposed model achieved 91.0% accuracy."},
        ],
        "anchor_groups": [],
        "required_numbers": [],
        "required_support": [],
        "expect_abstain": True,
    },
    {
        "id": "S15",
        "query": "What overall accuracy did the model achieve?",
        "evidence": [
            {"id": "F1", "text": "The proposed model achieved 95.1% overall accuracy."},
            {"id": "F2", "text": "The model achieved 89.2% accuracy on a difficult subgroup."},
        ],
        "anchor_groups": [["95.1"], ["overall accuracy"]],
        "required_numbers": ["95.1"],
        "required_support": ["F1"],
        "expect_abstain": False,
    },
    {
        "id": "S16",
        "query": "Did Model C outperform Model A on accuracy?",
        "evidence": [
            {"id": "F1", "text": "Model A achieved 88.4% accuracy."},
            {"id": "F2", "text": "Model C achieved 90.1% accuracy."},
        ],
        "anchor_groups": [["model c"], ["outperform", "higher accuracy", "better accuracy"]],
        "required_numbers": ["90.1", "88.4"],
        "required_support": ["F1", "F2"],
        "expect_abstain": False,
    },
]


# ---------------------------------------------------------------------------
# NORMALIZATION / VALIDATION
# ---------------------------------------------------------------------------

def norm(value: Any) -> str:
    text = str(value if value is not None else "").lower()
    text = text.replace("–", "-").replace("—", "-").replace("‑", "-")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w.%+-]+", " ", text)
    return text.strip()


def number(value: Any) -> str:
    raw = str(value if value is not None else "").strip().replace(",", "")
    m = re.search(r"-?\d+(?:\.\d+)?", raw)
    if not m:
        return norm(raw)
    try:
        f = float(m.group())
        return str(int(f)) if f.is_integer() else f"{f:.10f}".rstrip("0").rstrip(".")
    except ValueError:
        return norm(raw)


def number_tokens(text: str) -> set[str]:
    return {number(x) for x in re.findall(r"-?\d+(?:\.\d+)?", text)}


FIELD_ALIASES = {
    "parameters": {"parameters", "parameter count", "number of parameters",
                   "number parameters", "model parameters", "parameters count"},
    "top-1 accuracy": {"top 1 accuracy", "top1 accuracy", "top 1 acc"},
    "training images": {"training images", "train images", "training image count"},
    "validation images": {"validation images", "val images", "validation image count"},
    "test images": {"test images", "testing images", "test image count"},
    "FLOPs": {"flops", "gflops", "floating point operations"},
    "AUC": {"auc", "area under the curve", "roc auc"},
    "inference latency": {"inference latency", "inference time", "inference speed"},
    "Dice enhancing tumor": {"dice enhancing tumor", "enhancing tumor dice", "dice score for enhancing tumor"},
    "Dice whole tumor": {"dice whole tumor", "whole tumor dice", "dice score for whole tumor"},
    "external validation": {"external validation"},
    "dataset size": {"dataset size", "size"},
    "class imbalance": {"class imbalance"},
    "single-center evaluation": {"single center evaluation", "single-center evaluation"},
}

ENTITY_ALIASES = {
    "proposed model": {"proposed model", "model", "proposed system", "proposed network"},
    "proposed system": {"proposed system", "proposed model", "model"},
    "proposed network": {"proposed network", "proposed model", "network", "model"},
    "model": {"model", "proposed model", "proposed system", "proposed network"},
    "study limitation": {"study limitation", "main limitation", "limitations", "limitation"},
    "BraTS dataset": {"brats", "brats dataset", "dataset", "experiment"},
    "MRI dataset": {"mri dataset", "dataset", "mri studies"},
    "method": {"method", "model", "proposed method", "proposed model"},
    "study": {"study", "paper", "model"},
}


def alias_set(mapping: dict[str, set[str]], key: str) -> set[str]:
    return {norm(x) for x in mapping.get(key, {key}) | {key}}


def same_entity(pred: Any, gold: str) -> bool:
    p = norm(pred)
    return p in alias_set(ENTITY_ALIASES, gold)


def same_field(pred: Any, gold: str) -> bool:
    p = norm(pred)
    g = norm(gold)
    if p == g:
        return True
    for canonical, aliases in FIELD_ALIASES.items():
        group = {norm(canonical)} | {norm(x) for x in aliases}
        if g in group:
            return p in group
    return False


def same_value(pred: Any, gold: str) -> bool:
    p, g = str(pred).strip(), str(gold).strip()
    pn, gn = number(p), number(g)
    if re.fullmatch(r"-?\d+(?:\.\d+)?", pn) and re.fullmatch(r"-?\d+(?:\.\d+)?", gn):
        return pn == gn
    return norm(p) == norm(g)


def same_unit(pred: Any, gold: str) -> bool:
    p, g = norm(pred), norm(gold)
    groups = [
        {"%", "percent", "percentage", "percentile"},
        {"m", "million", "millions", "m params"},
        {"g", "gflops", "g flops", "billion flops"},
        {"ms", "millisecond", "milliseconds", "millisecond per scan", "milliseconds per scan"},
        {"hour", "hours", "hr", "hrs"},
        {"image", "images"},
        {"study", "studies"},
    ]
    if p == g:
        return True
    return any(p in group and g in group for group in groups)


def validate_object(obj: Any, task: str) -> tuple[bool, str | None]:
    if not isinstance(obj, dict):
        return False, "root_not_object"

    if task == "query_analysis":
        expected = set(ANALYZER_SCHEMA["required"])
        if set(obj) != expected:
            return False, "wrong_keys"
        if obj["intent"] not in ANALYZER_SCHEMA["properties"]["intent"]["enum"]:
            return False, "invalid_intent"
        for k in ("paper_ids", "metrics", "datasets"):
            if not isinstance(obj[k], list) or not all(isinstance(x, str) for x in obj[k]):
                return False, f"invalid_{k}"
        for k in ("needs_comparison", "needs_visualization", "needs_aggregation"):
            if not isinstance(obj[k], bool):
                return False, f"invalid_{k}"
        return True, None

    if task == "fact_extraction":
        if set(obj) != {"facts"} or not isinstance(obj["facts"], list):
            return False, "invalid_facts_root"
        required = {"entity", "field", "value", "unit", "condition", "quote"}
        for item in obj["facts"]:
            if not isinstance(item, dict) or set(item) != required:
                return False, "invalid_fact_keys"
            if not all(isinstance(item[k], str) for k in ("entity", "field", "unit", "condition", "quote")):
                return False, "invalid_fact_string"
            if not isinstance(item["value"], (str, int, float)):
                return False, "invalid_fact_value"
        return True, None

    if task == "grounded_synthesis":
        if set(obj) != {"claims"} or not isinstance(obj["claims"], list):
            return False, "invalid_claims_root"
        required = {"text", "support_ids", "claim_type"}
        types = {"reported", "derived", "comparison", "limitation", "inference"}
        for claim in obj["claims"]:
            if not isinstance(claim, dict) or set(claim) != required:
                return False, "invalid_claim_keys"
            if not isinstance(claim["text"], str):
                return False, "invalid_claim_text"
            if not isinstance(claim["support_ids"], list) or not all(
                isinstance(x, str) for x in claim["support_ids"]
            ):
                return False, "invalid_support_ids"
            if claim["claim_type"] not in types:
                return False, "invalid_claim_type"
        return True, None

    return False, "unknown_task"


# ---------------------------------------------------------------------------
# SCORERS
# ---------------------------------------------------------------------------

def canonical_metric(value: Any) -> str:
    n = norm(value)
    aliases = {
        "parameter count": "parameters",
        "number of parameters": "parameters",
        "model parameters": "parameters",
        "top 1 accuracy": "top-1 accuracy",
        "top1 accuracy": "top-1 accuracy",
        "inference time": "inference latency",
        "latency": "inference latency",
        "roc auc": "auc",
        "area under the curve": "auc",
    }
    return aliases.get(n, n)


def score_analysis(pred: Any, gold: dict[str, Any]) -> tuple[float, dict[str, float]]:
    valid, _ = validate_object(pred, "query_analysis")
    if not valid:
        return 0.0, {"schema_valid": 0.0}

    def set_equal(a: list[str], b: list[str]) -> float:
        return float({norm(x) for x in a} == {norm(x) for x in b})

    scores = {
        "intent_accuracy": float(pred["intent"] == gold["intent"]),
        "paper_id_exact": set_equal(pred["paper_ids"], gold["paper_ids"]),
        "metric_exact": float(
            {canonical_metric(x) for x in pred["metrics"]}
            == {canonical_metric(x) for x in gold["metrics"]}
        ),
        "dataset_exact": set_equal(pred["datasets"], gold["datasets"]),
        "comparison_accuracy": float(pred["needs_comparison"] == gold["needs_comparison"]),
        "visualization_accuracy": float(pred["needs_visualization"] == gold["needs_visualization"]),
        "aggregation_accuracy": float(pred["needs_aggregation"] == gold["needs_aggregation"]),
    }
    return statistics.mean(scores.values()), scores


def match_fact(pred: dict[str, Any], gold_fact: tuple[str, str, str, str]) -> bool:
    entity, field, value, unit = gold_fact
    return (
        same_entity(pred["entity"], entity)
        and same_field(pred["field"], field)
        and same_value(pred["value"], value)
        and same_unit(pred["unit"], unit)
    )


def score_extraction(pred: Any, case: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    valid, err = validate_object(pred, "fact_extraction")
    if not valid:
        return 0.0, {"schema_valid": False, "schema_error": err,
                     "core_precision": 0.0, "core_recall": 0.0,
                     "core_f1": 0.0, "verified_precision": 0.0,
                     "verified_recall": 0.0, "verified_f1": 0.0,
                     "quote_fidelity": 0.0, "value_in_quote_rate": 0.0}

    facts = pred["facts"]
    gold = case["gold"]
    source = norm(case["text"])
    used_gold: set[int] = set()
    matched_pairs = []
    for pi, pf in enumerate(facts):
        for gi, gf in enumerate(gold):
            if gi in used_gold:
                continue
            if match_fact(pf, gf):
                used_gold.add(gi)
                matched_pairs.append((pi, gi, pf, gf))
                break

    core_correct = len(matched_pairs)
    precision = core_correct / len(facts) if facts else (1.0 if not gold else 0.0)
    recall = core_correct / len(gold) if gold else (1.0 if not facts else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    grounded_pairs = []
    quote_valid_predictions = 0
    value_in_quote_predictions = 0
    for pi, fact in enumerate(facts):
        q = norm(fact["quote"])
        quote_ok = bool(q and q in source)
        value_txt = str(fact["value"]).strip()
        value_norm = number(value_txt)
        in_quote = bool(q) and (
            norm(value_txt) in q
            or (re.fullmatch(r"-?\d+(?:\.\d+)?", value_norm)
                and value_norm in number_tokens(fact["quote"]))
        )
        if quote_ok:
            quote_valid_predictions += 1
        if in_quote:
            value_in_quote_predictions += 1
        if quote_ok and in_quote and any(pair[0] == pi for pair in matched_pairs):
            grounded_pairs.append(pi)

    verified = len(grounded_pairs)
    vprecision = verified / len(facts) if facts else (1.0 if not gold else 0.0)
    vrecall = verified / len(gold) if gold else (1.0 if not facts else 0.0)
    vf1 = 2 * vprecision * vrecall / (vprecision + vrecall) if vprecision + vrecall else 0.0
    quote_fidelity = quote_valid_predictions / len(facts) if facts else 1.0
    value_in_quote = value_in_quote_predictions / len(facts) if facts else 1.0

    metrics = {
        "schema_valid": True,
        "gold_count": len(gold),
        "prediction_count": len(facts),
        "correct_records": core_correct,
        "verified_records": verified,
        "core_precision": precision,
        "core_recall": recall,
        "core_f1": f1,
        "verified_precision": vprecision,
        "verified_recall": vrecall,
        "verified_f1": vf1,
        "quote_fidelity": quote_fidelity,
        "value_in_quote_rate": value_in_quote,
        "missed_gold_count": len(gold) - core_correct,
        "extra_prediction_count": len(facts) - core_correct,
        "pairs": [
            {"prediction_index": pi, "gold_index": gi,
             "predicted": pf, "gold": {
                 "entity": gf[0], "field": gf[1], "value": gf[2], "unit": gf[3]
             }}
            for pi, gi, pf, gf in matched_pairs
        ],
    }
    # Core and verification have equal weighting in the role score.
    score = statistics.mean([f1, vf1, quote_fidelity, value_in_quote])
    return score, metrics


ABSTENTION_PHRASES = [
    "not reported", "not specified", "cannot determine",
    "can't determine", "insufficient evidence", "not provided",
    "does not state", "no information",
]


def score_synthesis(pred: Any, case: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    valid, err = validate_object(pred, "grounded_synthesis")
    if not valid:
        return 0.0, {"schema_valid": False, "schema_error": err,
                     "strict_pass": False, "answer_key_match": 0.0,
                     "support_recall": 0.0, "support_id_validity": 0.0,
                     "numeric_fidelity": 0.0}

    claims = pred["claims"]
    claim_text = " ".join(c["text"] for c in claims)
    nclaim = norm(claim_text)
    evidence = case["evidence"]
    evidence_ids = {item["id"] for item in evidence}
    evidence_text = " ".join(item["text"] for item in evidence)
    allowed_numbers = number_tokens(evidence_text)
    generated_numbers = number_tokens(claim_text)
    extra_numbers = generated_numbers - allowed_numbers
    numeric_fidelity = (
        1.0 if not generated_numbers else
        len(generated_numbers & allowed_numbers) / len(generated_numbers)
    )

    all_support_ids = [
        support for claim in claims for support in claim["support_ids"]
    ]
    valid_support_count = sum(x in evidence_ids for x in all_support_ids)
    support_id_validity = (
        valid_support_count / len(all_support_ids) if all_support_ids else
        (1.0 if case["expect_abstain"] and not claims else 0.0)
    )
    claims_with_support = sum(
        bool(c["support_ids"]) and all(x in evidence_ids for x in c["support_ids"])
        for c in claims
    )
    claim_support_coverage = (
        claims_with_support / len(claims) if claims else
        (1.0 if case["expect_abstain"] else 0.0)
    )

    expected_support = set(case["required_support"])
    predicted_support = set(all_support_ids)
    support_recall = (
        len(expected_support & predicted_support) / len(expected_support)
        if expected_support else
        (1.0 if case["expect_abstain"] and not claims else 0.0)
    )
    support_id_precision = (
        len(expected_support & predicted_support) / len(predicted_support)
        if predicted_support else
        (1.0 if not expected_support else 0.0)
    )

    anchor_results = []
    for group in case["anchor_groups"]:
        anchor_results.append(any(norm(anchor) in nclaim for anchor in group))
    anchors_match = (
        sum(anchor_results) / len(anchor_results) if anchor_results else
        (1.0 if case["expect_abstain"] else 0.0)
    )

    missing_numbers = [
        n for n in case["required_numbers"] if number(n) not in generated_numbers
    ]
    number_recall = (
        (len(case["required_numbers"]) - len(missing_numbers))
        / len(case["required_numbers"])
        if case["required_numbers"] else 1.0
    )

    if case["expect_abstain"]:
        empty_answer = len(claims) == 0
        explicit_abstention = any(x in nclaim for x in ABSTENTION_PHRASES)
        abstention_correct = empty_answer or explicit_abstention
        # Do not allow unsupported numeric details in an abstention.
        strict = (
            abstention_correct
            and numeric_fidelity == 1.0
            and support_id_validity == 1.0
        )
        answer_key_match = float(abstention_correct)
    else:
        strict = (
            all(anchor_results)
            and not missing_numbers
            and support_recall == 1.0
            and support_id_validity == 1.0
            and claim_support_coverage == 1.0
            and numeric_fidelity == 1.0
        )
        answer_key_match = float(all(anchor_results) and not missing_numbers)

    metrics = {
        "schema_valid": True,
        "expect_abstain": case["expect_abstain"],
        "answer_key_match": answer_key_match,
        "anchor_results": anchor_results,
        "required_number_recall": number_recall,
        "missing_required_numbers": missing_numbers,
        "support_recall": support_recall,
        "support_id_precision": support_id_precision,
        "support_id_validity": support_id_validity,
        "claim_support_coverage": claim_support_coverage,
        "numeric_fidelity": numeric_fidelity,
        "extra_numeric_tokens": sorted(extra_numbers),
        "strict_pass": bool(strict),
        "note": "Heuristic; support IDs are not semantic entailment proof.",
    }

    score = statistics.mean([
        answer_key_match, support_recall, support_id_validity,
        claim_support_coverage, numeric_fidelity,
    ])
    return score, metrics


# ---------------------------------------------------------------------------
# OLLAMA REQUESTS
# ---------------------------------------------------------------------------

def post(path: str, payload: dict[str, Any], timeout: int = REQUEST_TIMEOUT):
    start = time.perf_counter()
    response = requests.post(
        f"{OLLAMA_URL}{path}",
        json=payload,
        timeout=timeout,
    )
    elapsed = time.perf_counter() - start
    response.raise_for_status()
    return response.json(), elapsed


def check_model(model: str) -> tuple[bool, str | None]:
    try:
        data, _ = post("/api/show", {"model": model}, timeout=30)
        return bool(data), None if data else "empty /api/show response"
    except Exception as exc:
        return False, str(exc)


def warm_model(model: str) -> float:
    data, elapsed = post(
        "/api/generate",
        {
            "model": model,
            "prompt": "",
            "stream": False,
            "keep_alive": KEEP_ALIVE,
            "options": {
                "num_ctx": NUM_CTX,
                "temperature": TEMPERATURE,
                "seed": SEED,
                "num_predict": 1,
            },
        },
        timeout=REQUEST_TIMEOUT,
    )
    return data.get("load_duration", 0) / 1_000_000_000 or elapsed


def request_llm(model: str, task: str, system_prompt: str, user_prompt: str, schema: dict):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
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
        data, elapsed = post("/api/chat", payload)
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 400:
            payload.pop("think", None)
            data, elapsed = post("/api/chat", payload)
        else:
            raise

    message = data.get("message") or {}
    text = message.get("content", "")
    parsed = None
    parse_error = None
    try:
        raw = text.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        parsed = json.loads(raw)
    except Exception as exc:
        parse_error = f"json_parse_failed: {exc}"

    return text, parsed, data, elapsed, parse_error


@dataclass
class Result:
    model: str
    task: str
    case_id: str
    inference_ok: bool
    schema_valid: bool
    elapsed_seconds: float
    prompt_tokens: int | None
    output_tokens: int | None
    score: float
    metrics: dict[str, Any]
    response_text: str
    error: str | None


def failure_result(model: str, task: str, case_id: str, exc: Exception) -> Result:
    return Result(
        model=model, task=task, case_id=case_id, inference_ok=False,
        schema_valid=False, elapsed_seconds=0.0, prompt_tokens=None,
        output_tokens=None, score=0.0, metrics={}, response_text="",
        error=str(exc),
    )


def run_case(model: str, task: str, case: dict[str, Any]) -> Result:
    try:
        if task == "query_analysis":
            system, schema = ANALYZER_SYSTEM, ANALYZER_SCHEMA
            user_prompt = f"User query:\n{case['query']}"
            scorer = lambda obj: score_analysis(obj, case["gold"])
        elif task == "fact_extraction":
            system, schema = FACT_SYSTEM, FACT_SCHEMA
            user_prompt = f"Passage:\n{case['text']}"
            scorer = lambda obj: score_extraction(obj, case)
        else:
            system, schema = SYNTHESIS_SYSTEM, SYNTHESIS_SCHEMA
            user_prompt = (
                f"Question:\n{case['query']}\n\nEvidence:\n"
                + json.dumps(case["evidence"], ensure_ascii=False, indent=2)
            )
            scorer = lambda obj: score_synthesis(obj, case)

        text, parsed, raw, elapsed, parse_error = request_llm(
            model, task, system, user_prompt, schema
        )
        schema_valid, schema_error = validate_object(parsed, task) if parsed is not None else (
            False, parse_error
        )
        score, metrics = scorer(parsed) if parsed is not None else (0.0, {})
        err = schema_error if schema_error else parse_error

        return Result(
            model=model,
            task=task,
            case_id=case["id"],
            inference_ok=True,
            schema_valid=schema_valid,
            elapsed_seconds=elapsed,
            prompt_tokens=raw.get("prompt_eval_count"),
            output_tokens=raw.get("eval_count"),
            score=score,
            metrics=metrics,
            response_text=text,
            error=err,
        )
    except Exception as exc:
        return failure_result(model, task, case["id"], exc)


# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------

def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def main():
    output_dir = Path("storage/evaluation/llm_benchmark_v3")
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = {
        "query_analysis": A,
        "fact_extraction": E,
        "grounded_synthesis": S,
    }
    all_models = sorted({m for values in ROLE_MODELS.values() for m in values})
    availability = {}
    cold_load = {}
    results: list[Result] = []

    total_calls = sum(
        len(candidates[task]) * len(models)
        for task, models in ROLE_MODELS.items()
    )

    print("=" * 100)
    print("LLM BENCHMARK V3 — EXPANDED")
    print("=" * 100)
    print("Cases/task:", {k: len(v) for k, v in candidates.items()})
    print("Candidates/task:", {k: len(v) for k, v in ROLE_MODELS.items()})
    print("Maximum inference calls:", total_calls)
    print(f"Settings: ctx={NUM_CTX}, temp={TEMPERATURE}, seed={SEED}, keep_alive={KEEP_ALIVE}")
    print("Strict schema; bounded output; thinking disabled where supported.")
    print("Please keep the Ollama server running for the duration of the run.")

    print("\n--- Preflight ---")
    for model in all_models:
        ok, error = check_model(model)
        availability[model] = {"available": ok, "error": error}
        print(("[OK]   " if ok else "[SKIP] ") + model + (f" — {error}" if error else ""))

    for task, cases in candidates.items():
        print(f"\n{'#' * 80}\nROLE: {task}\n{'#' * 80}")
        for model in ROLE_MODELS[task]:
            if not availability[model]["available"]:
                continue

            print(f"\n>>> Warming {model}")
            try:
                cold_load[model] = warm_model(model)
                print(f"    cold-load estimate: {cold_load[model]:.2f}s")
            except Exception as exc:
                availability[model] = {"available": False, "error": f"warmup_failed: {exc}"}
                print(f"    skipped after warmup failure: {exc}")
                continue

            for idx, case in enumerate(cases, start=1):
                print(f"[{idx:02d}/{len(cases):02d}] {model} {task} {case['id']}", flush=True)
                result = run_case(model, task, case)
                results.append(result)
                if not result.inference_ok:
                    print(f"    ERROR: {result.error}")
                else:
                    print(
                        f"    score={result.score:.3f} "
                        f"schema={result.schema_valid} "
                        f"latency={result.elapsed_seconds:.2f}s"
                    )

    # Per-role summary
    summary_rows = []
    for task, models in ROLE_MODELS.items():
        for model in models:
            rows = [r for r in results if r.task == task and r.model == model]
            if not availability[model]["available"]:
                summary_rows.append({
                    "task": task, "model": model, "available": False,
                    "availability_error": availability[model]["error"],
                })
                continue
            if not rows:
                continue

            schema_rate = sum(r.schema_valid for r in rows) / len(rows)
            inference_rate = sum(r.inference_ok for r in rows) / len(rows)
            successful = [r for r in rows if r.inference_ok]
            latencies = [r.elapsed_seconds for r in successful]
            output_tokens = [r.output_tokens for r in successful if r.output_tokens is not None]

            row = {
                "task": task,
                "model": model,
                "available": True,
                "case_count": len(rows),
                "inference_success_rate": inference_rate,
                "schema_valid_rate": schema_rate,
                "mean_task_score": mean([r.score for r in rows]),
                "median_latency_seconds": statistics.median(latencies) if latencies else None,
                "median_output_tokens": statistics.median(output_tokens) if output_tokens else None,
                "cold_load_seconds": cold_load.get(model),
            }

            if task == "fact_extraction":
                row.update({
                    "core_macro_case_f1": mean([r.metrics.get("core_f1", 0.0) for r in rows]),
                    "verified_macro_case_f1": mean([r.metrics.get("verified_f1", 0.0) for r in rows]),
                    "mean_quote_fidelity": mean([r.metrics.get("quote_fidelity", 0.0) for r in rows]),
                    "mean_value_in_quote_rate": mean([r.metrics.get("value_in_quote_rate", 0.0) for r in rows]),
                    "mean_record_precision": mean([r.metrics.get("core_precision", 0.0) for r in rows]),
                    "mean_record_recall": mean([r.metrics.get("core_recall", 0.0) for r in rows]),
                })
            elif task == "grounded_synthesis":
                row.update({
                    "key_answer_rate": mean([r.metrics.get("answer_key_match", 0.0) for r in rows]),
                    "strict_heuristic_pass_rate": mean([float(r.metrics.get("strict_pass", False)) for r in rows]),
                    "support_recall": mean([r.metrics.get("support_recall", 0.0) for r in rows]),
                    "support_id_validity": mean([r.metrics.get("support_id_validity", 0.0) for r in rows]),
                    "numeric_fidelity": mean([r.metrics.get("numeric_fidelity", 0.0) for r in rows]),
                })

            summary_rows.append(row)

    raw_path = output_dir / "llm_benchmark_v3_raw.json"
    summary_path = output_dir / "llm_benchmark_v3_summary.json"

    raw_payload = {
        "version": 3,
        "purpose": "Expanded role-specific model bake-off; controlled gold-labelled cases.",
        "config": {
            "role_models": ROLE_MODELS,
            "num_ctx": NUM_CTX,
            "temperature": TEMPERATURE,
            "seed": SEED,
            "keep_alive": KEEP_ALIVE,
            "num_predict": NUM_PREDICT,
            "task_case_counts": {k: len(v) for k, v in candidates.items()},
        },
        "availability": availability,
        "cold_load_seconds": cold_load,
        "results": [asdict(r) for r in results],
    }

    raw_path.write_text(json.dumps(raw_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_path.write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 100)
    print("ROLE SUMMARY")
    print("=" * 100)
    for task in ROLE_MODELS:
        print(f"\n{task}")
        rows = [x for x in summary_rows if x.get("task") == task and x.get("available")]
        rows.sort(
            key=lambda x: (
                x.get("mean_task_score", 0.0),
                -(x.get("median_latency_seconds") or 999999.0),
            ),
            reverse=True,
        )
        for rank, row in enumerate(rows, start=1):
            print(
                f"{rank}. {row['model']:22} "
                f"score={row.get('mean_task_score', 0):.3f} "
                f"schema={row.get('schema_valid_rate', 0):.1%} "
                f"median={row.get('median_latency_seconds', 0):.2f}s"
            )

    print("\nSaved:")
    print(raw_path)
    print(summary_path)
    print("\nCaution: 16 cases/role is still a pilot. Review gold labels and manually audit synthesis before claiming accuracy.")


if __name__ == "__main__":
    main()
