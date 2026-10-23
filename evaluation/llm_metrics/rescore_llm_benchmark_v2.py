"""Offline, corrected scorer for llm_benchmark_v2_raw.json.

Does not call Ollama and does not run inference. It reuses the benchmark's
query-analysis gold labels/cases, but independently scores extraction records
so condition wording no longer invalidates otherwise correct facts.

Run from repository root:
  python evaluation/rescore_llm_benchmark_v2.py
Or specify paths:
  python evaluation/rescore_llm_benchmark_v2.py --input path\raw.json --output-dir path\rescored
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

# Reuse the exact case text and gold labels; do not duplicate benchmark data.
try:
    from llm_benchmark_v2 import (
        ANALYZER_CASES,
        EXTRACTION_CASES,
        SYNTHESIS_CASES,
        score_analyzer,
        validate_analyzer,
        validate_extraction,
        validate_synthesis,
    )
except ImportError:
    # Also supports importing when this script is executed from the repo root.
    from evaluation.llm_metrics.llm_benchmark_v2 import (
        ANALYZER_CASES,
        EXTRACTION_CASES,
        SYNTHESIS_CASES,
        score_analyzer,
        validate_analyzer,
        validate_extraction,
        validate_synthesis,
    )


TASKS = ("query_analysis", "fact_extraction", "grounded_synthesis")


def norm_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.lower().replace("–", "-").replace("—", "-")
    text = text.replace("‑", "-")
    # Treat hyphenated words (single-center) and spaced words (single center)
    # as equivalent, while keeping hyphens in paper IDs and negative numbers.
    text = re.sub(r"(?<=[a-z])-(?=[a-z])", " ", text)
    text = re.sub(r"[^\w.%+-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def norm_number(value: Any) -> str:
    if value is None:
        return ""
    match = re.search(r"-?\d+(?:,\d{3})*(?:\.\d+)?", str(value))
    if not match:
        return ""
    try:
        number = float(match.group(0).replace(",", ""))
        return str(int(number)) if number.is_integer() else f"{number:.10f}".rstrip("0").rstrip(".")
    except ValueError:
        return ""


def text_has(text: Any, phrase: Any) -> bool:
    return norm_text(phrase) in norm_text(text) if norm_text(phrase) else False


FIELD_ALIASES = {
    "top-1 accuracy": {"top 1 accuracy", "top1 accuracy", "top 1 acc", "top 1 accuracy"},
    "top-5 accuracy": {"top 5 accuracy", "top5 accuracy", "top 5 acc"},
    "parameters": {"parameter", "parameters", "number of parameters", "number parameters", "parameter count", "model parameters"},
    "dataset": {"dataset", "dataset name", "dataset used", "data set"},
    "training images": {"training images", "number of training images", "number training images", "training image count", "training images count", "train images", "training_images_count"},
    "validation images": {"validation images", "number of validation images", "number validation images", "validation image count", "validation images count", "val images", "validation_images_count"},
    "dataset size": {"dataset size", "size"},
    "external validation": {"external validation"},
    "accuracy": {"accuracy", "acc"},
    "sensitivity": {"sensitivity", "sens"},
    "specificity": {"specificity", "spec"},
}

ENTITY_ALIASES = {
    "efficientnet-b0": {"efficientnet-b0", "efficientnet b0"},
    "proposed model": {"proposed model", "model", "proposed method"},
    "method": {"method", "proposed method"},
    "brats dataset": {"brats dataset", "brats", "the brats dataset"},
    "study limitation": {"study limitation", "main limitation", "limitation"},
}


def canonical_field(field: Any, entity: Any = "") -> str:
    f = norm_text(field)
    e = norm_text(entity)
    for canonical, aliases in FIELD_ALIASES.items():
        if f in {norm_text(alias) for alias in aliases}:
            return canonical
    # Context-dependent repair for the common output "entity=training images,
    # field=count". This does not accept a generic "performance metric" as
    # sensitivity/specificity, because that would hide an entity-field swap.
    if f in {"count", "number", "quantity"}:
        if "training image" in e:
            return "training images"
        if "validation image" in e:
            return "validation images"
    return f


def canonical_entity(entity: Any) -> str:
    e = norm_text(entity)
    for canonical, aliases in ENTITY_ALIASES.items():
        if e in {norm_text(alias) for alias in aliases}:
            return canonical
    # These alternatives are limited to the split-count facts, handled with
    # field-specific logic in entity_match().
    if e in {"experiment", "the experiment", "study", "dataset split", "brats dataset split"}:
        return e
    return e


def entity_match(pred_entity: Any, gold_entity: str, field: str) -> bool:
    p = canonical_entity(pred_entity)
    g = canonical_entity(gold_entity)
    f = canonical_field(field)
    if p == g:
        return True
    if f in {"training images", "validation images"}:
        acceptable = {"brats dataset", "experiment", "study", "dataset split", "brats dataset split"}
        return p in acceptable and g in acceptable
    # For the dataset-name fact itself, "experiment" is the sentence subject
    # and is acceptable only when the extracted field is dataset.
    if f == "dataset" and p == "experiment" and g == "brats dataset":
        return True
    return False


def canonical_value(value: Any, field: str) -> str:
    raw = norm_text(value)
    f = canonical_field(field)
    n = norm_number(value)
    if n:
        return f"num:{n}"
    if f == "dataset":
        raw = re.sub(r"\b(dataset|data set)\b", "", raw).strip()
    if f == "external validation":
        if any(x in raw for x in ["absence", "absent", "not performed", "not conducted", "not done", "no external validation"]):
            return "absence"
    if f == "dataset size" and "small" in raw:
        return "small"
    return raw


def value_match(pred_value: Any, gold_value: Any, field: str) -> bool:
    pv = canonical_value(pred_value, field)
    gv = canonical_value(gold_value, field)
    if pv == gv:
        return True
    # Narrow textual aliases; used for phrases such as "absence of external
    # validation" vs gold value "absence" and not generic fuzzy matching.
    if gv == "absence" and field == "external validation" and "absence" in pv:
        return True
    if gv == "small" and field == "dataset size" and "small" in pv:
        return True
    if field == "dataset" and pv == "brats" and gv == "brats":
        return True
    return False


def expected_unit(gold: dict[str, Any]) -> str:
    return norm_text(gold.get("unit", ""))


def predicted_semantic_unit(pred: dict[str, Any]) -> str:
    """Unit conveyed either in the unit field or explicitly in value.

    Unit-field formatting is scored separately, so a model that puts '%' in
    value but leaves unit blank is semantically understood but not normalized.
    """
    unit = norm_text(pred.get("unit", ""))
    val = str(pred.get("value", ""))
    if unit:
        if unit in {"percent", "percentage"}:
            return "%"
        if unit in {"million", "millions"}:
            return "m"
        if unit in {"image"}:
            return "images"
        return unit
    if "%" in val:
        return "%"
    if re.search(r"\b\d+(?:\.\d+)?\s*m\b", val, re.I):
        return "m"
    return ""


def unit_match(pred_unit: Any, gold_unit: Any) -> bool:
    p = norm_text(pred_unit)
    g = norm_text(gold_unit)
    aliases = {"%": {"%", "percent", "percentage"}, "m": {"m", "million", "millions"}, "images": {"image", "images"}}
    if g in aliases:
        return p in aliases[g]
    return p == g


def quote_is_grounded(quote: Any, source: str) -> bool:
    q = norm_text(quote)
    s = norm_text(source)
    return bool(q) and q in s


def value_in_quote(pred: dict[str, Any]) -> bool:
    quote = norm_text(pred.get("quote", ""))
    value = str(pred.get("value", ""))
    number = norm_number(value)
    if number:
        return number in {norm_number(x) for x in re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", quote)}
    return bool(norm_text(value)) and norm_text(value) in quote


def pair_score(pred: dict[str, Any], gold: dict[str, Any]) -> float:
    pf = canonical_field(pred.get("field", ""), pred.get("entity", ""))
    gf = canonical_field(gold.get("field", ""), gold.get("entity", ""))
    field_ok = pf == gf
    val_ok = value_match(pred.get("value", ""), gold.get("value", ""), gf)
    ent_ok = entity_match(pred.get("entity", ""), gold.get("entity", ""), gf)
    unit_ok = unit_match(pred.get("unit", ""), gold.get("unit", ""))
    # Prefer matching the fact field/value; then entity and unit. Assignment is
    # one-to-one, so a merged prediction cannot satisfy multiple gold facts.
    return (8 if field_ok else 0) + (7 if val_ok else 0) + (2 if ent_ok else 0) + (1 if unit_ok else 0)


def best_pairs(preds: list[dict[str, Any]], golds: list[dict[str, Any]]) -> list[tuple[int, int]]:
    candidates = []
    for i, p in enumerate(preds):
        for j, g in enumerate(golds):
            score = pair_score(p, g)
            pf = canonical_field(p.get("field", ""), p.get("entity", ""))
            gf = canonical_field(g.get("field", ""), g.get("entity", ""))
            val_ok = value_match(p.get("value", ""), g.get("value", ""), gf)
            # Only pair things when at least field or value offers evidence of
            # being the same fact. Entity/unit alone are not enough.
            if score > 0 and (pf == gf or val_ok):
                candidates.append((score, i, j))
    candidates.sort(reverse=True)
    used_p, used_g, pairs = set(), set(), []
    for _, i, j in candidates:
        if i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
            pairs.append((i, j))
    return pairs


def score_extraction_corrected(parsed: Any, golds: list[dict[str, Any]], source: str) -> dict[str, Any]:
    schema_valid, schema_error = validate_extraction(parsed)
    if not schema_valid:
        return {
            "schema_valid": False,
            "schema_error": schema_error,
            "gold_fact_count": len(golds),
            "predicted_fact_count": None if not isinstance(parsed, dict) else len(parsed.get("facts", [])) if isinstance(parsed.get("facts"), list) else None,
            "correct_record_count": 0,
            "verified_record_count": 0,
            "record_precision": 0.0,
            "record_recall": 0.0,
            "record_f1": 0.0,
            "verified_record_precision": 0.0,
            "verified_record_recall": 0.0,
            "verified_record_f1": 0.0,
            "entity_accuracy_matched": 0.0,
            "field_accuracy_matched": 0.0,
            "value_exactness_matched": 0.0,
            "unit_field_accuracy_matched": 0.0,
            "semantic_unit_accuracy_matched": 0.0,
            "condition_accuracy_assessed": None,
            "quote_fidelity": 0.0,
            "value_in_quote_rate": 0.0,
            "matched_pair_count": 0,
            "unmatched_prediction_count": None,
            "missed_gold_count": len(golds),
        }

    preds = parsed["facts"]
    pairs = best_pairs(preds, golds)
    pair_by_pred = {i: j for i, j in pairs}
    exact_core = 0
    verified_core = 0
    entity_scores, field_scores, value_scores = [], [], []
    unit_field_scores, semantic_unit_scores = [], []
    condition_scores = []

    for i, j in pairs:
        p, g = preds[i], golds[j]
        gf = canonical_field(g.get("field", ""), g.get("entity", ""))
        field_ok = canonical_field(p.get("field", ""), p.get("entity", "")) == gf
        entity_ok = entity_match(p.get("entity", ""), g.get("entity", ""), gf)
        value_ok = value_match(p.get("value", ""), g.get("value", ""), gf)
        unit_field_ok = unit_match(p.get("unit", ""), g.get("unit", ""))
        semantic_unit_ok = unit_match(predicted_semantic_unit(p), g.get("unit", ""))

        entity_scores.append(float(entity_ok))
        field_scores.append(float(field_ok))
        value_scores.append(float(value_ok))
        unit_field_scores.append(float(unit_field_ok))
        semantic_unit_scores.append(float(semantic_unit_ok))

        # Main record correctness is based on the semantic core; free-form
        # condition wording is deliberately scored separately.
        core_correct = field_ok and entity_ok and value_ok and semantic_unit_ok
        quote_ok = quote_is_grounded(p.get("quote", ""), source)
        value_quote_ok = value_in_quote(p)
        if core_correct:
            exact_core += 1
        if core_correct and quote_ok and value_quote_ok:
            verified_core += 1

        expected_condition = norm_text(g.get("condition", ""))
        if expected_condition:
            condition_scores.append(float(text_has(p.get("condition", ""), expected_condition)))

    predicted_count = len(preds)
    gold_count = len(golds)
    precision = exact_core / predicted_count if predicted_count else (1.0 if gold_count == 0 else 0.0)
    recall = exact_core / gold_count if gold_count else (1.0 if predicted_count == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    verified_precision = verified_core / predicted_count if predicted_count else (1.0 if gold_count == 0 else 0.0)
    verified_recall = verified_core / gold_count if gold_count else (1.0 if predicted_count == 0 else 0.0)
    verified_f1 = 2 * verified_precision * verified_recall / (verified_precision + verified_recall) if verified_precision + verified_recall else 0.0

    quote_fidelity = (
        sum(quote_is_grounded(p.get("quote", ""), source) for p in preds) / predicted_count
        if predicted_count else 0.0
    )
    value_in_quote_rate = (
        sum(value_in_quote(p) for p in preds) / predicted_count
        if predicted_count else 0.0
    )

    return {
        "schema_valid": True,
        "schema_error": None,
        "gold_fact_count": gold_count,
        "predicted_fact_count": predicted_count,
        "correct_record_count": exact_core,
        "verified_record_count": verified_core,
        "record_precision": precision,
        "record_recall": recall,
        "record_f1": f1,
        "verified_record_precision": verified_precision,
        "verified_record_recall": verified_recall,
        "verified_record_f1": verified_f1,
        "entity_accuracy_matched": statistics.mean(entity_scores) if entity_scores else 0.0,
        "field_accuracy_matched": statistics.mean(field_scores) if field_scores else 0.0,
        "value_exactness_matched": statistics.mean(value_scores) if value_scores else 0.0,
        "unit_field_accuracy_matched": statistics.mean(unit_field_scores) if unit_field_scores else 0.0,
        "semantic_unit_accuracy_matched": statistics.mean(semantic_unit_scores) if semantic_unit_scores else 0.0,
        "condition_accuracy_assessed": statistics.mean(condition_scores) if condition_scores else None,
        "quote_fidelity": quote_fidelity,
        "value_in_quote_rate": value_in_quote_rate,
        "matched_pair_count": len(pairs),
        "unmatched_prediction_count": predicted_count - len(pairs),
        "missed_gold_count": gold_count - len(pairs),
        "pairs": [
            {
                "prediction_index": i,
                "gold_index": j,
                "predicted": preds[i],
                "gold": golds[j],
                "field_match": canonical_field(preds[i].get("field", ""), preds[i].get("entity", "")) == canonical_field(golds[j].get("field", ""), golds[j].get("entity", "")),
                "entity_match": entity_match(preds[i].get("entity", ""), golds[j].get("entity", ""), canonical_field(golds[j].get("field", ""), golds[j].get("entity", ""))),
                "value_match": value_match(preds[i].get("value", ""), golds[j].get("value", ""), canonical_field(golds[j].get("field", ""), golds[j].get("entity", ""))),
                "unit_field_match": unit_match(preds[i].get("unit", ""), golds[j].get("unit", "")),
                "semantic_unit_match": unit_match(predicted_semantic_unit(preds[i]), golds[j].get("unit", "")),
                "condition_match": (text_has(preds[i].get("condition", ""), golds[j].get("condition", "")) if norm_text(golds[j].get("condition", "")) else None),
                "quote_grounded": quote_is_grounded(preds[i].get("quote", ""), source),
                "value_in_quote": value_in_quote(preds[i]),
                "core_record_correct": (
                    canonical_field(preds[i].get("field", ""), preds[i].get("entity", "")) == canonical_field(golds[j].get("field", ""), golds[j].get("entity", ""))
                    and entity_match(preds[i].get("entity", ""), golds[j].get("entity", ""), canonical_field(golds[j].get("field", ""), golds[j].get("entity", "")))
                    and value_match(preds[i].get("value", ""), golds[j].get("value", ""), canonical_field(golds[j].get("field", ""), golds[j].get("entity", "")))
                    and unit_match(predicted_semantic_unit(preds[i]), golds[j].get("unit", ""))
                ),
            }
            for i, j in pairs
        ],
    }


def extract_numbers(text: str) -> set[str]:
    vals = set()
    for token in re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text):
        num = norm_number(token)
        if num:
            vals.add(num)
    return vals


SYNTHESIS_ANCHORS = {
    "S01": [["model b"]],
    "S02": [["method b"], ["higher sensitivity", "higher sensivity"]],
    "S03": [["accuracy"]],
    "S04": [["not directly comparable", "not comparable", "cannot be compared", "cannot directly compare", "cannot fairly compare", "not a fair comparison"]],
    "S05": [["small single center dataset", "small dataset", "small single center"], ["external validation"]],
}


def score_synthesis_corrected(parsed: Any, case: dict[str, Any], evidence: list[dict[str, str]]) -> dict[str, Any]:
    schema_valid, schema_error = validate_synthesis(parsed)
    if not schema_valid:
        return {
            "schema_valid": False,
            "schema_error": schema_error,
            "answer_key_match": 0.0,
            "support_recall": 0.0,
            "support_id_precision": 0.0,
            "support_id_validity": 0.0,
            "claim_support_coverage": 0.0,
            "numeric_fidelity": 0.0,
            "strict_task_pass": 0.0,
        }

    claims = parsed["claims"]
    combined = norm_text(" ".join(c.get("text", "") for c in claims))
    nums_in_answer = extract_numbers(combined)
    nums_in_evidence = extract_numbers(" ".join(e.get("text", "") for e in evidence))

    anchors = SYNTHESIS_ANCHORS.get(case["id"], [])
    anchor_results = [any(norm_text(variant) in combined for variant in group) for group in anchors]
    required_numbers = {norm_number(x) for x in case["gold"]["required_numbers"]}
    answer_key_match = float(all(anchor_results) and required_numbers.issubset(nums_in_answer))

    valid_ids = {x["id"] for x in evidence}
    predicted_ids = [sid for claim in claims for sid in claim.get("support_ids", [])]
    predicted_id_set = set(predicted_ids)
    required_ids = set(case["gold"]["required_support_ids"])
    support_recall = len(required_ids & predicted_id_set) / len(required_ids) if required_ids else 1.0
    support_id_precision = sum(sid in valid_ids for sid in predicted_ids) / len(predicted_ids) if predicted_ids else 0.0
    support_id_validity = float(bool(claims) and all(claim.get("support_ids") for claim in claims) and all(sid in valid_ids for sid in predicted_ids))
    claim_support_coverage = sum(bool(c.get("support_ids")) for c in claims) / len(claims) if claims else 0.0
    numeric_fidelity = len(nums_in_answer & nums_in_evidence) / len(nums_in_answer) if nums_in_answer else 1.0

    # This is a strict *heuristic task pass*, not a semantic entailment proof.
    # Valid support IDs only prove references resolve, not that each claim is
    # entailed by those sources; semantic groundedness still needs spot audit.
    strict_pass = float(
        answer_key_match == 1.0
        and support_recall == 1.0
        and support_id_validity == 1.0
        and numeric_fidelity == 1.0
        and claim_support_coverage == 1.0
    )

    return {
        "schema_valid": True,
        "schema_error": None,
        "answer_key_match": answer_key_match,
        "required_anchor_results": anchor_results,
        "support_recall": support_recall,
        "support_id_precision": support_id_precision,
        "support_id_validity": support_id_validity,
        "claim_support_coverage": claim_support_coverage,
        "numeric_fidelity": numeric_fidelity,
        "strict_task_pass": strict_pass,
        "note": "Heuristic: valid support IDs do not prove semantic entailment.",
    }


def lookup_case_map(cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {case["id"]: case for case in cases}


def mean_or_none(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("storage/evaluation/llm_benchmark/llm_benchmark_v2_raw.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("storage/evaluation/llm_benchmark/rescored_v2"),
    )
    args = parser.parse_args()

    raw = json.loads(args.input.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    analysis_cases = lookup_case_map(ANALYZER_CASES)
    extraction_cases = lookup_case_map(EXTRACTION_CASES)
    synthesis_cases = lookup_case_map(SYNTHESIS_CASES)

    rescored_cases = []
    extraction_stats: dict[str, list[dict[str, Any]]] = defaultdict(list)
    synthesis_stats: dict[str, list[dict[str, Any]]] = defaultdict(list)
    analysis_stats: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in raw["results"]:
        model = row["model"]
        task = row["task"]
        cid = row["case_id"]
        parsed = row.get("parsed_json")

        base = {
            "model": model,
            "task": task,
            "case_id": cid,
            "elapsed_seconds": row.get("elapsed_seconds", 0),
            "schema_valid_original": row.get("schema_valid"),
            "error_original": row.get("error"),
        }

        if task == "query_analysis":
            case = analysis_cases[cid]
            valid, schema_error = validate_analyzer(parsed)
            if parsed is not None:
                score, metrics = score_analyzer(parsed, case["gold"])
            else:
                score, metrics = 0.0, {}
            result = {
                **base,
                "schema_valid_corrected": valid,
                "schema_error_corrected": schema_error,
                "score": score,
                "metrics": metrics,
            }
            analysis_stats[model].append(result)

        elif task == "fact_extraction":
            case = extraction_cases[cid]
            result_metrics = score_extraction_corrected(
                parsed,
                case["gold"],
                case["text"],
            )
            result = {**base, **result_metrics}
            extraction_stats[model].append(result)

        elif task == "grounded_synthesis":
            case = synthesis_cases[cid]
            result_metrics = score_synthesis_corrected(
                parsed,
                case,
                case["evidence"],
            )
            result = {**base, **result_metrics}
            synthesis_stats[model].append(result)

        else:
            continue

        rescored_cases.append(result)

    availability = raw.get("availability", {})
    models = raw.get("config", {}).get("models", sorted(availability))
    summary = []

    for model in models:
        available_info = availability.get(model, [False, "not listed"])
        is_available = bool(available_info[0]) if isinstance(available_info, list) else bool(available_info)
        row_summary: dict[str, Any] = {
            "model": model,
            "available": is_available,
            "availability_error": (available_info[1] if isinstance(available_info, list) and len(available_info) > 1 else None),
            "cold_load_seconds": raw.get("cold_load_seconds", {}).get(model),
        }

        arows = analysis_stats.get(model, [])
        if arows:
            row_summary["query_analysis_score"] = mean_or_none([x["score"] for x in arows])
            row_summary["query_analysis_schema_rate"] = mean_or_none([float(x["schema_valid_corrected"]) for x in arows])
            row_summary["query_analysis_median_latency"] = statistics.median([x["elapsed_seconds"] for x in arows if x["elapsed_seconds"] > 0]) if any(x["elapsed_seconds"] > 0 for x in arows) else None

        erows = extraction_stats.get(model, [])
        if erows:
            predicted_total = sum(x.get("predicted_fact_count") or 0 for x in erows)
            gold_total = sum(x.get("gold_fact_count") or 0 for x in erows)
            correct_total = sum(x.get("correct_record_count") or 0 for x in erows)
            verified_total = sum(x.get("verified_record_count") or 0 for x in erows)
            p = correct_total / predicted_total if predicted_total else 0.0
            r = correct_total / gold_total if gold_total else 0.0
            vp = verified_total / predicted_total if predicted_total else 0.0
            vr = verified_total / gold_total if gold_total else 0.0
            row_summary["fact_extraction_micro_precision"] = p
            row_summary["fact_extraction_micro_recall"] = r
            row_summary["fact_extraction_micro_f1"] = 2*p*r/(p+r) if p+r else 0.0
            row_summary["fact_extraction_verified_micro_precision"] = vp
            row_summary["fact_extraction_verified_micro_recall"] = vr
            row_summary["fact_extraction_verified_micro_f1"] = 2*vp*vr/(vp+vr) if vp+vr else 0.0
            row_summary["fact_extraction_macro_record_f1"] = mean_or_none([x["record_f1"] for x in erows])
            row_summary["fact_extraction_macro_verified_record_f1"] = mean_or_none([x["verified_record_f1"] for x in erows])
            for metric in [
                "entity_accuracy_matched",
                "field_accuracy_matched",
                "value_exactness_matched",
                "unit_field_accuracy_matched",
                "semantic_unit_accuracy_matched",
                "condition_accuracy_assessed",
                "quote_fidelity",
                "value_in_quote_rate",
            ]:
                vals = [x[metric] for x in erows if isinstance(x.get(metric), (int, float))]
                row_summary[f"fact_extraction_{metric}"] = mean_or_none(vals)
            row_summary["fact_extraction_schema_rate"] = mean_or_none([float(x["schema_valid"]) for x in erows])
            row_summary["fact_extraction_median_latency"] = statistics.median([x["elapsed_seconds"] for x in erows if x["elapsed_seconds"] > 0]) if any(x["elapsed_seconds"] > 0 for x in erows) else None

        srows = synthesis_stats.get(model, [])
        if srows:
            row_summary["grounded_synthesis_key_answer_rate"] = mean_or_none([x["answer_key_match"] for x in srows])
            row_summary["grounded_synthesis_strict_pass_rate"] = mean_or_none([x["strict_task_pass"] for x in srows])
            for metric in ["support_recall", "support_id_precision", "support_id_validity", "claim_support_coverage", "numeric_fidelity"]:
                row_summary[f"grounded_synthesis_{metric}"] = mean_or_none([x[metric] for x in srows])
            row_summary["grounded_synthesis_schema_rate"] = mean_or_none([float(x["schema_valid"]) for x in srows])
            row_summary["grounded_synthesis_median_latency"] = statistics.median([x["elapsed_seconds"] for x in srows if x["elapsed_seconds"] > 0]) if any(x["elapsed_seconds"] > 0 for x in srows) else None

        summary.append(row_summary)

    detailed_path = args.output_dir / "llm_benchmark_v2_rescored_cases.json"
    summary_path = args.output_dir / "llm_benchmark_v2_rescored_summary.json"
    report_path = args.output_dir / "llm_benchmark_v2_rescored_report.md"

    detailed_path.write_text(json.dumps(rescored_cases, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# LLM Benchmark V2 — Corrected Offline Rescore",
        "",
        "No model inference was run for this report; saved V2 outputs were rescored offline.",
        "",
        "## Summary",
        "",
        "Query analysis retains the V2 gold-label scoring. Extraction now reports record precision/recall/F1 separately from entity, field, value, unit, condition and quote dimensions. Condition wording no longer determines whether the core fact record is correct.",
        "",
        "Synthesis reports required-answer-key coverage, required evidence coverage, support-ID validity and numeric fidelity separately. `strict_pass_rate` is a heuristic task pass, not a proof of semantic entailment.",
        "",
        "| Model | Query score | Fact core micro-F1 | Fact verified micro-F1 | Fact macro-F1 | Quote fidelity | Synthesis key-answer rate | Synthesis strict pass | Median latency (analysis / extraction / synthesis) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    for r in summary:
        if not r["available"]:
            lines.append(f"| {r['model']} | — | — | — | — | — | — | — | unavailable: {r.get('availability_error') or 'not available'} |")
            continue
        lat = " / ".join(
            "—" if r.get(k) is None else f"{r[k]:.2f}s"
            for k in ["query_analysis_median_latency", "fact_extraction_median_latency", "grounded_synthesis_median_latency"]
        )
        lines.append(
            "| {model} | {q} | {fm} | {vfm} | {fa} | {quote} | {key} | {strict} | {lat} |".format(
                model=r["model"],
                q="—" if r.get("query_analysis_score") is None else f"{r['query_analysis_score']:.3f}",
                fm="—" if r.get("fact_extraction_micro_f1") is None else f"{r['fact_extraction_micro_f1']:.3f}",
                vfm="—" if r.get("fact_extraction_verified_micro_f1") is None else f"{r['fact_extraction_verified_micro_f1']:.3f}",
                fa="—" if r.get("fact_extraction_macro_record_f1") is None else f"{r['fact_extraction_macro_record_f1']:.3f}",
                quote="—" if r.get("fact_extraction_quote_fidelity") is None else f"{r['fact_extraction_quote_fidelity']:.3f}",
                key="—" if r.get("grounded_synthesis_key_answer_rate") is None else f"{r['grounded_synthesis_key_answer_rate']:.3f}",
                strict="—" if r.get("grounded_synthesis_strict_pass_rate") is None else f"{r['grounded_synthesis_strict_pass_rate']:.3f}",
                lat=lat,
            )
        )

    lines += [
        "",
        "## Interpretation safeguards",
        "",
        "- Only five cases per task were used; treat all rankings as pilot evidence, not a statistically reliable final model selection.",
        "- Core record F1 requires a matched entity, field, value and semantically represented unit. Verified record F1 additionally requires a source-grounded quote containing the extracted value. Free-form condition mismatch is reported separately and does not erase a correct core fact.",
        "- Quote fidelity is normalized substring containment (case/punctuation/whitespace tolerant), not byte-for-byte equality.",
        "- Support-ID validity only checks that each support ID exists in the provided evidence. It cannot establish semantic entailment; manually review synthesis claims, especially extra claims.",
        "- Query-analysis gold annotations are unchanged and may need a human review before the next formal benchmark.",
        "",
        "## Detailed output files",
        "",
        f"- Per-case details: `{detailed_path.name}`",
        f"- Machine-readable summary: `{summary_path.name}`",
        "",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("Rescore complete (offline; no model calls).")
    print(f"Detailed cases: {detailed_path}")
    print(f"Summary JSON : {summary_path}")
    print(f"Report       : {report_path}")
    header = "| Model | Query score | Fact core micro-F1 | Fact verified micro-F1 | Fact macro-F1 | Quote fidelity | Synthesis key-answer rate | Synthesis strict pass | Median latency (analysis / extraction / synthesis) |"
    print("\n" + lines[lines.index(header)])
    for line in lines[lines.index(header) + 2:]:
        if line.startswith("|"):
            print(line)


if __name__ == "__main__":
    main()
