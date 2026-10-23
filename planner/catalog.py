"""Static operation contracts. This catalog is the planner's allowlist."""
from __future__ import annotations
from dataclasses import dataclass
from typing import FrozenSet
from .schemas import Operation

@dataclass(frozen=True)
class OperationContract:
    requires: FrozenSet[str]
    requires_any: FrozenSet[str]
    produces: FrozenSet[str]


CONTRACTS: dict[Operation, OperationContract] = {
    "search_evidence": OperationContract(frozenset({"query", "scope"}), frozenset(), frozenset({"evidence"})),
    "lookup_facts": OperationContract(frozenset({"analysis", "evidence"}), frozenset(), frozenset({"facts", "facts_diagnostics"})),
    "gap_fill": OperationContract(frozenset({"analysis", "scope"}), frozenset(), frozenset({"facts", "evidence"})),
    "check_document_metrics": OperationContract(frozenset({"analysis", "scope", "evidence"}), frozenset(), frozenset({"document_metric_availability", "evidence"})),
    "decompose": OperationContract(frozenset({"query"}), frozenset(), frozenset({"subquestions"})),
    "merge_evidence": OperationContract(frozenset({"evidence", "subquestions"}), frozenset(), frozenset({"evidence"})),
    "resolve_entities": OperationContract(frozenset({"analysis"}), frozenset(), frozenset({"resolved_entities"})),
    "normalize_facts": OperationContract(frozenset(), frozenset({"facts"}), frozenset({"normalized_facts", "chart_data"})),
    "compare": OperationContract(frozenset({"normalized_facts", "resolved_entities"}), frozenset(), frozenset({"comparison", "chart_data", "derived_values"})),
    "validate_derived_values": OperationContract(frozenset({"derived_values"}), frozenset(), frozenset({"validated_derived_values"})),
    "aggregate": OperationContract(frozenset({"normalized_facts"}), frozenset(), frozenset({"aggregate", "chart_data", "derived_values"})),
    "rank": OperationContract(frozenset({"normalized_facts"}), frozenset(), frozenset({"ranking", "chart_data", "derived_values"})),
    "render_table": OperationContract(frozenset(), frozenset({"facts", "normalized_facts", "comparison", "aggregate", "ranking"}), frozenset({"table_artifact"})),
    "validate_chart_data": OperationContract(frozenset({"chart_data"}), frozenset(), frozenset({"validated_chart_data"})),
    "build_chart_spec": OperationContract(frozenset({"validated_chart_data"}), frozenset(), frozenset({"chart_spec"})),
    "render_chart": OperationContract(frozenset({"chart_spec"}), frozenset(), frozenset({"chart_artifact"})),
    "synthesize": OperationContract(frozenset(), frozenset({"evidence", "facts", "normalized_facts", "comparison", "aggregate", "ranking", "artifact_description", "themes", "chart_artifact", "table_artifact"}), frozenset({"answer_draft"})),
    "verify": OperationContract(frozenset({"answer_draft"}), frozenset(), frozenset({"answer_bundle"})),
    "search_metadata": OperationContract(frozenset({"scope"}), frozenset(), frozenset({"metadata_result"})),
    "render_metadata_response": OperationContract(frozenset({"metadata_result"}), frozenset(), frozenset({"answer_bundle"})),
    "find_artifact": OperationContract(frozenset({"analysis", "scope"}), frozenset(), frozenset({"artifact_ref"})),
    "inspect_artifact": OperationContract(frozenset({"artifact_ref"}), frozenset(), frozenset({"artifact_description"})),
    "group_themes": OperationContract(frozenset({"evidence"}), frozenset(), frozenset({"themes"})),
    "count_themes": OperationContract(frozenset({"themes"}), frozenset(), frozenset({"aggregate", "chart_data", "derived_values"})),
    "ask_clarification": OperationContract(frozenset({"analysis"}), frozenset(), frozenset({"answer_bundle"})),
    "decline": OperationContract(frozenset({"analysis"}), frozenset(), frozenset({"answer_bundle"})),
}

NUMERIC_OPERATIONS = frozenset({"compare", "aggregate", "rank", "count_themes"})
TERMINAL_OPERATIONS = frozenset({"verify", "render_metadata_response", "ask_clarification", "decline"})
EXTERNAL_INPUTS = frozenset({"query", "analysis", "scope"})
