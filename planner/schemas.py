"""Typed contracts for deterministic planning."""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field

PlanTemplate = Literal[
    "T_LOOKUP", "T_FACT_LOOKUP", "T_MULTI_HOP", "T_COMPARE",
    "T_AGGREGATE", "T_RANK", "T_TABLE", "T_CHART", "T_SYNTH",
    "T_ARTIFACT", "T_META", "T_CLARIFY", "T_DECLINE",
]

Operation = Literal[
    "search_evidence", "lookup_facts", "gap_fill", "check_document_metrics", "decompose",
    "merge_evidence", "resolve_entities", "normalize_facts", "compare",
    "validate_derived_values", "aggregate", "rank", "render_table",
    "validate_chart_data", "build_chart_spec", "render_chart", "synthesize",
    "verify", "search_metadata", "render_metadata_response", "find_artifact",
    "inspect_artifact", "group_themes", "count_themes", "ask_clarification",
    "decline",
]


class StrictModel(BaseModel):
    class Config:
        extra = "forbid"
        validate_assignment = True


class PlanStep(StrictModel):
    step_id: str = Field(min_length=1)
    operation: Operation
    depends_on: list[str] = Field(default_factory=list)
    # These are copied from the static operation catalog, never supplied by an LLM.
    requires: list[str] = Field(default_factory=list)
    requires_any: list[str] = Field(default_factory=list)
    produces: list[str] = Field(default_factory=list)


class PlanBudget(StrictModel):
    max_steps: int = Field(ge=1, le=32)
    max_retrieval_calls: int = Field(ge=0, le=20)
    max_llm_calls: int = Field(ge=0, le=20)
    max_wall_time_seconds: int = Field(ge=1, le=900)


class Plan(StrictModel):
    plan_id: str = Field(min_length=1)
    template_id: PlanTemplate
    scope_mode: Literal["single_paper", "named_papers", "filtered_corpus", "whole_corpus"]
    paper_ids: list[str] = Field(default_factory=list)
    filters: dict[str, str] = Field(default_factory=dict)
    steps: list[PlanStep] = Field(min_items=1)
    budget: PlanBudget
    selection_reason: str = Field(min_length=1)
    required_validators: list[str] = Field(default_factory=list)
