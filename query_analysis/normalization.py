"""Deterministic query cue detection, normalization, and validation."""
from __future__ import annotations

import re
from typing import Mapping, Collection

from .schemas import (
    EntityMention,
    QueryAnalysis,
    QueryAnalysisDraft,
    QueryAmbiguity,
    QueryValidationError,
)

PROMPT_VERSION = "query-analysis-v1"

# Canonical names are intentionally conservative; this is a small starter registry,
# not a claim that every field in the research corpus is covered.
DEFAULT_FIELD_ALIASES: dict[str, str] = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "sensitivity": "sensitivity",
    "specificity": "specificity",
    "f1": "f1-score",
    "f1 score": "f1-score",
    "f1-score": "f1-score",
    "dice": "dice score",
    "dice score": "dice score",
    "iou": "IoU",
    "intersection over union": "IoU",
    "auc": "AUC",
    "roc auc": "AUC",
    "roc-auc": "AUC",
    "area under the curve": "AUC",
    "parameter count": "parameters",
    "number of parameters": "parameters",
    "model parameters": "parameters",
    "parameters": "parameters",
    "flops": "FLOPs",
    "top1 accuracy": "top-1 accuracy",
    "top 1 accuracy": "top-1 accuracy",
    "top-1 accuracy": "top-1 accuracy",
    "inference time": "inference latency",
    "latency": "inference latency",
    "inference latency": "inference latency",
    "training time": "training time",
    "memory usage": "memory usage",
}


def _key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _contains_phrase(text: str, phrase: str) -> bool:
    """Phrase match with punctuation/whitespace variants and word boundaries."""
    parts = [re.escape(part) for part in re.split(r"[^a-z0-9]+", phrase.casefold()) if part]
    if not parts:
        return False
    pattern = r"(?<![a-z0-9])" + r"[\s_\-/]+".join(parts) + r"(?![a-z0-9])"
    return re.search(pattern, text.casefold()) is not None


def extract_explicit_paper_ids(query: str) -> list[str]:
    """Return only IDs actually written in the user's query, canonicalized to paper_NNN."""
    found: list[str] = []
    for match in re.finditer(r"\bpaper[_\s-]*(\d{1,6})\b", query, flags=re.IGNORECASE):
        paper_id = f"paper_{int(match.group(1)):03d}"
        if paper_id not in found:
            found.append(paper_id)
    return found


def _explicit_field_mentions(query: str, aliases: Mapping[str, str]) -> list[str]:
    # Match longer aliases first, then de-duplicate canonical values in query order.
    matched: list[tuple[int, str, str]] = []
    for alias, canonical in aliases.items():
        if _contains_phrase(query, alias):
            idx = _key(query).find(_key(alias))
            matched.append((idx if idx >= 0 else 10**9, alias, canonical))
    matched.sort(key=lambda item: (item[0], -len(item[1])))
    result: list[str] = []
    for _, _, canonical in matched:
        if canonical not in result:
            result.append(canonical)
    return result


def _query_cues(query: str) -> dict[str, bool]:
    q = query.casefold()
    return {
        "comparison": any(x in q for x in ("compare", "comparison", "versus", " vs ", "difference between", "differences between")),
        "visualization": any(x in q for x in ("plot", "chart", "graph", "visualize", "visualise", "visualization", "visualisation")),
        "table": "table" in q or "tabulate" in q,
        "aggregation": any(x in q for x in ("average", "mean", "median", "sum", "aggregate", "highest", "lowest", "best", "worst", "rank", "ranking", "top-", "top ")),
        "multi_hop": any(x in q for x in ("step by step", "multi-hop", "multi hop", "trace how", "first determine", "then explain", "across stages")),
        "corpus_wide": any(x in q for x in ("all papers", "every paper", "across the corpus", "entire corpus", "whole corpus", "all studies", "all articles")),
        "artifact_ref": bool(re.search(r"\b(fig(?:ure)?|table|equation|eq\.?|algorithm)\s*(?:no\.?\s*)?\d+[a-z]?\b", q)),
        "clarification": any(x in q for x in ("which one", "that paper", "this model", "the previous paper")),
        "unsupported": False,
    }


def normalize_and_validate(
    draft: QueryAnalysisDraft,
    *,
    query: str,
    model_name: str,
    valid_paper_ids: Collection[str] | None = None,
    field_aliases: Mapping[str, str] | None = None,
    valid_fields: Collection[str] | None = None,
    entity_aliases: Mapping[str, Mapping[str, str]] | None = None,
    prompt_version: str = PROMPT_VERSION,
) -> QueryAnalysis:
    """Reconcile model output with deterministic cues and trusted registries.

    ``valid_paper_ids`` and ``valid_fields`` should be supplied when the repository's
    authoritative registries are available. Without them, this layer still enforces
    that paper IDs came from the user query, but cannot establish corpus membership.
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")

    validators: list[str] = ["pydantic_schema", "explicit_scope_guard", "cue_reconciliation", "alias_normalization"]
    explicit_ids = extract_explicit_paper_ids(query)
    if valid_paper_ids is not None:
        known = set(valid_paper_ids)
        unknown_explicit = [paper_id for paper_id in explicit_ids if paper_id not in known]
        if unknown_explicit:
            raise QueryValidationError(
                "Unknown explicit paper ID(s): " + ", ".join(unknown_explicit)
            )
        validators.append("paper_registry")

    # An LLM is not allowed to invent a paper ID or broaden explicit user scope.
    if len(explicit_ids) == 1:
        scope_mode = "single_paper"
    elif len(explicit_ids) > 1:
        scope_mode = "named_papers"
    else:
        scope_mode = draft.scope.mode
        if scope_mode in {"single_paper", "named_papers"}:
            # Titles/name resolution require a real registry; IDs alone cannot be inferred.
            scope_mode = "filtered_corpus"
        cues = _query_cues(query)
        if cues["corpus_wide"]:
            scope_mode = "whole_corpus"
        elif scope_mode == "whole_corpus" and not cues["corpus_wide"]:
            # Avoid a model silently broadening scope when the user did not ask for it.
            scope_mode = "filtered_corpus"
    scope_ids = explicit_ids

    aliases = {_key(alias): canonical for alias, canonical in DEFAULT_FIELD_ALIASES.items()}
    if field_aliases:
        aliases.update({_key(alias): canonical for alias, canonical in field_aliases.items()})

    canonical_fields: list[str] = []
    for field in draft.fields_requested:
        clean = field.strip()
        if not clean:
            continue
        canonical = aliases.get(_key(clean), clean)
        if valid_fields is not None and canonical not in set(valid_fields):
            raise QueryValidationError(f"Unknown requested field: {clean!r} (normalized to {canonical!r})")
        if canonical not in canonical_fields:
            canonical_fields.append(canonical)
    # Explicit metric mentions are reliably recoverable even if the model omitted them.
    for field in _explicit_field_mentions(query, aliases):
        if valid_fields is not None and field not in set(valid_fields):
            raise QueryValidationError(f"Unknown explicitly requested field: {field!r}")
        if field not in canonical_fields:
            canonical_fields.append(field)
    if valid_fields is not None:
        validators.append("field_registry")

    # Entity canonical names are trusted only when an explicit alias registry resolves them.
    entity_aliases = entity_aliases or {}
    normalized_entities: list[EntityMention] = []
    seen_entities: set[tuple[str, str]] = set()
    for entity in draft.entities:
        surface = entity.surface.strip()
        entity_type = str(entity.type)
        key = _key(surface)
        if not surface or (entity_type, key) in seen_entities:
            continue
        seen_entities.add((entity_type, key))
        type_aliases = entity_aliases.get(entity_type, {})
        canonical = None
        for alias, target in type_aliases.items():
            if _key(alias) == key:
                canonical = target
                break
        normalized_entities.append(
            EntityMention(surface=surface, type=entity.type, canonical=canonical, confidence=entity.confidence)
        )
    if entity_aliases:
        validators.append("entity_alias_registry")

    cues = _query_cues(query)
    # Rule-based cues override model flags; generated flags are never authoritative.
    needs_comparison = cues["comparison"] or draft.task_type == "comparison"
    needs_aggregation = cues["aggregation"] or draft.task_type in {"aggregation", "superlative_rank"}
    needs_visualization = cues["visualization"] or draft.task_type == "chart_request"
    multi_doc = len(scope_ids) > 1 or scope_mode in {"named_papers", "filtered_corpus", "whole_corpus"}
    structured_facts = draft.needs.structured_facts or draft.task_type in {
        "factual_lookup", "extraction", "comparison", "aggregation", "superlative_rank",
        "table_request", "chart_request", "artifact_lookup", "corpus_meta",
    }
    artifact_access = cues["artifact_ref"] or draft.task_type == "artifact_lookup"
    iterative_hint = cues["multi_hop"] or draft.task_type == "multi_hop"

    task_type = draft.task_type
    output_format = draft.output.format
    chart_type = draft.output.chart_type
    if cues["visualization"]:
        task_type, output_format = "chart_request", "chart"
    elif cues["table"]:
        task_type, output_format = "table_request", "table"
    elif cues["multi_hop"]:
        task_type = "multi_hop"
    elif cues["aggregation"] and task_type not in {"chart_request", "table_request"}:
        task_type = "superlative_rank" if any(x in query.casefold() for x in ("best", "worst", "highest", "lowest", "rank", "top-", "top ")) else "aggregation"
    elif cues["comparison"] and task_type not in {"chart_request", "table_request"}:
        task_type = "comparison"

    ambiguity = QueryAmbiguity(**_model_to_dict(draft.ambiguity))
    if cues["clarification"] and not ambiguity.is_ambiguous:
        ambiguity.is_ambiguous = True
        ambiguity.reasons = [*ambiguity.reasons, "The query uses a reference that may depend on conversation context."]
    if ambiguity.is_ambiguous and not (ambiguity.clarifying_question or "").strip():
        ambiguity.clarifying_question = "Could you clarify which paper, model, or metric you want me to focus on?"

    # Keep only filters that have a literal anchor in the user query.
    safe_filters = {
        key.strip(): value.strip()
        for key, value in draft.scope.filters.items()
        if key.strip() and value.strip() and _contains_phrase(query, value)
    }

    payload = _model_to_dict(draft)
    payload.update(
        task_type=task_type,
        scope={"mode": scope_mode, "paper_ids": scope_ids, "filters": safe_filters},
        entities=[_model_to_dict(item) for item in normalized_entities],
        fields_requested=canonical_fields,
        constraints=_normalize_constraints(draft.constraints, query, entity_aliases),
        output={"format": output_format, "chart_type": chart_type if output_format == "chart" else None},
        needs={
            "structured_facts": structured_facts,
            "comparison": needs_comparison,
            "aggregation": needs_aggregation,
            "visualization": needs_visualization,
            "multi_doc": multi_doc,
            "iterative_retrieval_hint": iterative_hint,
            "artifact_access": artifact_access,
        },
        ambiguity=_model_to_dict(ambiguity),
        analyzer_meta={"model": model_name, "prompt_version": prompt_version, "validators_applied": validators},
    )
    return QueryAnalysis(**payload)


def _normalize_constraints(constraints, query: str, entity_aliases: Mapping[str, Mapping[str, str]]) -> dict[str, str | None]:
    dataset = constraints.dataset
    if dataset:
        dataset_map = {_key(alias): canonical for alias, canonical in entity_aliases.get("dataset", {}).items()}
        dataset = dataset_map.get(_key(dataset), dataset if _contains_phrase(query, dataset) else None)
    # Constraint values not literally present in the query are treated as model guesses.
    output: dict[str, str | None] = {}
    for field in ("split", "metric_variant", "condition"):
        value = getattr(constraints, field)
        output[field] = value if value and _contains_phrase(query, value) else None
    output["dataset"] = dataset
    return output


def _model_to_dict(model):
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()
