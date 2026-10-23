"""Deterministic template planner for validated QueryAnalysis objects."""
from __future__ import annotations

import hashlib
import json
from typing import Iterable

from query_analysis.schemas import QueryAnalysis

from .catalog import CONTRACTS
from .schemas import Plan, PlanBudget, PlanStep, PlanTemplate
from .validator import PlanValidator


class DeterministicPlanner:
    """Select a fixed template and build a validated, sequential plan."""

    def __init__(self, validator: PlanValidator | None = None) -> None:
        self.validator = validator or PlanValidator()

    def plan(self, analysis: QueryAnalysis) -> Plan:
        template, operations, reason = self._select_template(analysis)
        # Dense retrieval is still the normal evidence path. For an explicitly
        # named paper + requested metric, add one exhaustive indexed-text check so
        # absence is never inferred from similarity top-k alone.
        scope = analysis.scope
        explicitly_scoped = (
            scope.mode in {"single_paper", "named_papers"}
            and bool(scope.paper_ids)
            and bool(analysis.fields_requested)
            and template not in {"T_META", "T_ARTIFACT", "T_CLARIFY", "T_DECLINE"}
        )
        if explicitly_scoped and "check_document_metrics" not in operations:
            insertion_after = "gap_fill" if "gap_fill" in operations else "search_evidence"
            if insertion_after in operations:
                operations.insert(operations.index(insertion_after) + 1, "check_document_metrics")
        steps = self._build_steps(operations)
        budget = self._budget_for(template, len(steps))
        scope = analysis.scope
        identity = {
            "template": template,
            "scope_mode": scope.mode,
            "paper_ids": list(scope.paper_ids),
            "filters": dict(scope.filters),
            "operations": [s.operation for s in steps],
        }
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:12]
        required_validators = ["plan_contracts"]
        if any(step.operation in {"compare", "aggregate", "rank", "count_themes"} for step in steps):
            required_validators.append("derived_value_validation")
        if any(step.operation == "render_chart" for step in steps):
            required_validators.append("chart_data_validation")

        plan = Plan(
            plan_id=f"plan_{digest}",
            template_id=template,
            scope_mode=scope.mode,
            paper_ids=list(scope.paper_ids),
            filters=dict(scope.filters),
            steps=steps,
            budget=budget,
            selection_reason=reason,
            required_validators=required_validators,
        )
        return self.validator.validate(plan)

    @staticmethod
    def _select_template(analysis: QueryAnalysis) -> tuple[PlanTemplate, list[str], str]:
        task = analysis.task_type
        needs = analysis.needs
        ambiguity = analysis.ambiguity
        if task == "unsupported":
            return "T_DECLINE", ["decline"], "unsupported_task_type"
        if task == "clarification_needed" or (ambiguity.is_ambiguous and ambiguity.clarifying_question):
            return "T_CLARIFY", ["ask_clarification"], "blocking_ambiguity"
        if task == "corpus_meta":
            return "T_META", ["search_metadata", "render_metadata_response"], "corpus_metadata_request"
        if task == "artifact_lookup" or needs.artifact_access:
            return "T_ARTIFACT", ["find_artifact", "inspect_artifact", "synthesize", "verify"], "artifact_lookup_request"
        if task == "corpus_synthesis":
            return "T_SYNTH", ["search_evidence", "gap_fill", "lookup_facts", "group_themes", "count_themes", "validate_derived_values", "synthesize", "verify"], "corpus_synthesis_request"
        # Chart requests take precedence, but retain requested comparison/aggregation operations.
        if task == "chart_request" or analysis.output.format == "chart" or needs.visualization:
            ops: list[str] = []
            if needs.multi_doc or task == "multi_hop":
                ops.extend(["decompose", "search_evidence", "merge_evidence"])
            ops.extend(["resolve_entities"])
            if "search_evidence" not in ops:
                ops.append("search_evidence")
            ops.extend(["gap_fill", "lookup_facts", "normalize_facts"])
            if task == "comparison" or needs.comparison:
                ops.extend(["compare", "validate_derived_values"])
            elif task == "aggregation" or needs.aggregation:
                ops.extend(["aggregate", "validate_derived_values"])
            elif task == "superlative_rank":
                ops.extend(["rank", "validate_derived_values"])
            ops.extend(["validate_chart_data", "build_chart_spec", "render_chart", "synthesize", "verify"])
            return "T_CHART", ops, "visualization_requested"
        if task == "multi_hop" or needs.iterative_retrieval_hint:
            return "T_MULTI_HOP", ["decompose", "search_evidence", "merge_evidence", "synthesize", "verify"], "multi_step_retrieval_required"
        if task == "superlative_rank":
            ops = ["search_evidence", "gap_fill", "lookup_facts", "normalize_facts", "rank", "validate_derived_values"]
            if analysis.output.format == "table":
                ops.append("render_table")
            ops.extend(["synthesize", "verify"])
            return "T_RANK", ops, "ranking_request"
        if task == "comparison" or needs.comparison:
            ops = ["resolve_entities", "search_evidence", "gap_fill", "lookup_facts", "normalize_facts", "compare", "validate_derived_values"]
            if analysis.output.format == "table":
                ops.append("render_table")
            ops.extend(["synthesize", "verify"])
            return "T_COMPARE", ops, "comparison_requested"
        if task == "aggregation" or needs.aggregation:
            ops = ["search_evidence", "gap_fill", "lookup_facts", "normalize_facts", "aggregate", "validate_derived_values"]
            if analysis.output.format == "table":
                ops.append("render_table")
            ops.extend(["synthesize", "verify"])
            return "T_AGGREGATE", ops, "aggregation_requested"
        if task == "table_request" or analysis.output.format == "table":
            return "T_TABLE", ["search_evidence", "gap_fill", "lookup_facts", "normalize_facts", "render_table", "synthesize", "verify"], "table_output_requested"
        if task == "extraction" or needs.structured_facts:
            return "T_FACT_LOOKUP", ["search_evidence", "gap_fill", "lookup_facts", "synthesize", "verify"], "structured_fact_request"
        return "T_LOOKUP", ["search_evidence", "synthesize", "verify"], "default_evidence_lookup"

    @staticmethod
    def _build_steps(operations: Iterable[str]) -> list[PlanStep]:
        steps: list[PlanStep] = []
        previous_id: str | None = None
        for index, operation in enumerate(operations, start=1):
            contract = CONTRACTS[operation]  # allowlist: unknown operations fail closed
            step_id = f"s{index:02d}_{operation}"
            steps.append(PlanStep(
                step_id=step_id,
                operation=operation,
                depends_on=[] if previous_id is None else [previous_id],
                requires=sorted(contract.requires),
                requires_any=sorted(contract.requires_any),
                produces=sorted(contract.produces),
            ))
            previous_id = step_id
        return steps

    @staticmethod
    def _budget_for(template: PlanTemplate, step_count: int) -> PlanBudget:
        retrieval = {
            "T_LOOKUP": 3, "T_FACT_LOOKUP": 3, "T_MULTI_HOP": 5,
            "T_COMPARE": 5, "T_AGGREGATE": 5, "T_RANK": 5,
            "T_TABLE": 4, "T_CHART": 6, "T_SYNTH": 5,
            "T_ARTIFACT": 2, "T_META": 1, "T_CLARIFY": 0, "T_DECLINE": 0,
        }[template]
        llm = {
            "T_LOOKUP": 2, "T_FACT_LOOKUP": 2, "T_MULTI_HOP": 5,
            "T_COMPARE": 2, "T_AGGREGATE": 2, "T_RANK": 2,
            "T_TABLE": 2, "T_CHART": 3, "T_SYNTH": 4,
            "T_ARTIFACT": 2, "T_META": 0, "T_CLARIFY": 0, "T_DECLINE": 0,
        }[template]
        wall_time = 120 if template in {"T_MULTI_HOP", "T_CHART", "T_SYNTH"} else 90
        return PlanBudget(
            max_steps=max(step_count, 1),
            max_retrieval_calls=retrieval,
            max_llm_calls=llm,
            max_wall_time_seconds=wall_time,
        )
