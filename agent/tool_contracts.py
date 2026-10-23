"""Runtime tool contracts derived from the planner's operation catalog."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from planner.catalog import CONTRACTS
from planner.schemas import Operation

ResourceKind = Literal["none", "retrieval", "llm"]


@dataclass(frozen=True)
class ToolContract:
    operation: Operation
    requires: frozenset[str]
    requires_any: frozenset[str]
    produces: frozenset[str]
    resource_kind: ResourceKind


_RETRIEVAL_OPERATIONS = frozenset({
    "search_evidence", "gap_fill", "check_document_metrics", "search_metadata", "find_artifact", "inspect_artifact",
})
_LLM_OPERATIONS = frozenset({
    "lookup_facts", "decompose", "group_themes", "synthesize",
})


def _resource_kind(operation: str) -> ResourceKind:
    if operation in _RETRIEVAL_OPERATIONS:
        return "retrieval"
    if operation in _LLM_OPERATIONS:
        return "llm"
    # Verification and arithmetic/normalization are deterministic in v1.
    return "none"


TOOL_CONTRACTS: dict[Operation, ToolContract] = {
    operation: ToolContract(
        operation=operation,
        requires=contract.requires,
        requires_any=contract.requires_any,
        produces=contract.produces,
        resource_kind=_resource_kind(operation),
    )
    for operation, contract in CONTRACTS.items()
}
