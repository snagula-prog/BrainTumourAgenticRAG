"""Typed, validated schemas for the research assistant's query-analysis stage.

The schema follows Agentic_RAG_Architecture_v1.md §7.2. Pydantic v1/v2 compatible.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

TaskType = Literal[
    "factual_lookup",
    "passage_retrieval",
    "multi_hop",
    "comparison",
    "extraction",
    "aggregation",
    "superlative_rank",
    "table_request",
    "chart_request",
    "corpus_synthesis",
    "artifact_lookup",
    "corpus_meta",
    "clarification_needed",
    "unsupported",
]
ScopeMode = Literal["single_paper", "named_papers", "filtered_corpus", "whole_corpus"]
EntityType = Literal["model", "dataset", "metric", "method", "section", "artifact"]
OutputFormat = Literal["prose", "table", "chart", "json"]
ChartType = Literal[
    "bar", "line", "scatter", "box", "histogram", "heatmap", "radar", "other"
]


class StrictModel(BaseModel):
    """Reject unexpected keys instead of silently accepting schema drift."""

    class Config:
        extra = "forbid"
        validate_assignment = True


class ScopeSpec(StrictModel):
    mode: ScopeMode
    paper_ids: list[str] = Field(default_factory=list)
    # Flat key/value filters only in v1; arbitrary nested execution payloads are disallowed.
    filters: dict[str, str] = Field(default_factory=dict)


class EntityMention(StrictModel):
    surface: str = Field(min_length=1)
    type: EntityType
    canonical: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)


class QueryConstraints(StrictModel):
    dataset: str | None = None
    split: str | None = None
    metric_variant: str | None = None
    condition: str | None = None


class OutputSpec(StrictModel):
    format: OutputFormat
    chart_type: ChartType | None = None


class QueryNeeds(StrictModel):
    structured_facts: bool
    comparison: bool
    aggregation: bool
    visualization: bool
    multi_doc: bool
    iterative_retrieval_hint: bool
    artifact_access: bool


class QueryAmbiguity(StrictModel):
    is_ambiguous: bool
    reasons: list[str] = Field(default_factory=list)
    clarifying_question: str | None = None


class AnalyzerMeta(StrictModel):
    model: str
    prompt_version: str
    validators_applied: list[str] = Field(default_factory=list)


class QueryAnalysisDraft(StrictModel):
    """Structured model proposal before deterministic normalization and validation."""

    task_type: TaskType
    scope: ScopeSpec
    entities: list[EntityMention] = Field(default_factory=list)
    fields_requested: list[str] = Field(default_factory=list)
    constraints: QueryConstraints
    output: OutputSpec
    needs: QueryNeeds
    ambiguity: QueryAmbiguity
    confidence: float = Field(ge=0.0, le=1.0)


class QueryAnalysis(QueryAnalysisDraft):
    """Validated query representation consumed by the deterministic planner."""

    analyzer_meta: AnalyzerMeta


class QueryAnalysisError(RuntimeError):
    """Base error for query analysis failures."""


class QueryValidationError(QueryAnalysisError):
    """A semantically invalid model proposal or explicit scope was detected."""
