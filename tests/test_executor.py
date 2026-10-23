from __future__ import annotations

import unittest

from query_analysis.schemas import QueryAnalysis
from planner import DeterministicPlanner, Plan, PlanBudget, PlanStep
from planner.catalog import CONTRACTS
from agent.executor import Executor
from agent.tool_contracts import TOOL_CONTRACTS


def make_analysis(*, task_type="comparison", paper_ids=None):
    paper_ids = list(paper_ids or ["paper_010", "paper_011"])
    mode = "named_papers" if paper_ids else "whole_corpus"
    return QueryAnalysis(**{
        "task_type": task_type,
        "scope": {"mode": mode, "paper_ids": paper_ids, "filters": {}},
        "entities": [
            {"surface": "paper_010", "type": "artifact", "canonical": None, "confidence": 1.0},
            {"surface": "paper_011", "type": "artifact", "canonical": None, "confidence": 1.0},
        ] if paper_ids else [],
        "fields_requested": ["sensitivity", "specificity"],
        "constraints": {"dataset": None, "split": None, "metric_variant": None, "condition": None},
        "output": {"format": "prose", "chart_type": None},
        "needs": {
            "structured_facts": True, "comparison": True, "aggregation": False,
            "visualization": False, "multi_doc": True,
            "iterative_retrieval_hint": False, "artifact_access": False,
        },
        "ambiguity": {"is_ambiguous": False, "reasons": [], "clarifying_question": None},
        "confidence": 0.95,
        "analyzer_meta": {"model": "qwen3:4b-instruct", "prompt_version": "query-analysis-v1", "validators_applied": []},
    })


def make_comparison_handlers(calls):
    def search_evidence(ctx):
        calls.append("search_evidence")
        return {"evidence": [{"evidence_id": "ev_initial_1", "text": "initial retrieval result"}]}

    def resolve_entities(ctx):
        calls.append("resolve_entities")
        return {"resolved_entities": []}

    def lookup_facts(ctx):
        calls.append("lookup_facts")
        # Empty result is valid: gap_fill must still execute.
        return {"facts": [], "facts_diagnostics": {}}

    def gap_fill(ctx):
        calls.append("gap_fill")
        # Gap fill deliberately runs before fact extraction and must accept absent facts.
        assert not ctx.has_artifact("facts")
        return {"facts": [], "evidence": [{"evidence_id": "ev_gap_1", "text": "targeted retrieval result"}]}

    def normalize_facts(ctx):
        calls.append("normalize_facts")
        return {"normalized_facts": [], "chart_data": []}

    def compare(ctx):
        calls.append("compare")
        return {"comparison": {"status": "insufficient_data"}, "chart_data": [], "derived_values": []}

    def validate_derived_values(ctx):
        calls.append("validate_derived_values")
        return {"validated_derived_values": []}

    def synthesize(ctx):
        calls.append("synthesize")
        return {"answer_draft": {"text": "Insufficient comparable metrics were found.", "evidence_ids": ["ev_gap_1"]}}

    def verify(ctx):
        calls.append("verify")
        return {"answer_bundle": {"answer": "Insufficient comparable metrics were found.", "verified": True}}

    return {
        "search_evidence": search_evidence,
        "resolve_entities": resolve_entities,
        "lookup_facts": lookup_facts,
        "gap_fill": gap_fill,
        "normalize_facts": normalize_facts,
        "compare": compare,
        "validate_derived_values": validate_derived_values,
        "synthesize": synthesize,
        "verify": verify,
    }


class ExecutorTests(unittest.TestCase):
    def test_gap_fill_contract_does_not_require_existing_facts(self):
        contract = CONTRACTS["gap_fill"]
        self.assertEqual(contract.requires, frozenset({"analysis", "scope"}))
        self.assertEqual(contract.requires_any, frozenset())
        self.assertEqual(contract.produces, frozenset({"facts", "evidence"}))
        runtime_contract = TOOL_CONTRACTS["gap_fill"]
        self.assertEqual(runtime_contract.requires_any, frozenset())

    def test_empty_fact_result_does_not_skip_gap_fill(self):
        analysis = make_analysis()
        plan = DeterministicPlanner().plan(analysis)
        calls = []
        executor = Executor(make_comparison_handlers(calls))

        result = executor.execute(
            plan,
            query="Compare sensitivity and specificity in paper_010 and paper_011",
            analysis=analysis,
        )

        self.assertTrue(result.succeeded, result.error)
        self.assertEqual(calls.count("gap_fill"), 1)
        self.assertIn("evidence", result.artifacts)
        self.assertEqual(result.retrieval_calls_used, 2)
        self.assertEqual(result.llm_calls_used, 2)

    def test_gap_fill_can_run_when_facts_artifact_is_absent(self):
        analysis = make_analysis(task_type="factual_lookup", paper_ids=["paper_010"])
        operations = ["gap_fill", "synthesize", "verify"]
        steps = []
        previous = None
        for i, operation in enumerate(operations, start=1):
            contract = CONTRACTS[operation]
            step_id = f"s{i:02d}_{operation}"
            steps.append(PlanStep(
                step_id=step_id,
                operation=operation,
                depends_on=[] if previous is None else [previous],
                requires=sorted(contract.requires),
                requires_any=sorted(contract.requires_any),
                produces=sorted(contract.produces),
            ))
            previous = step_id
        plan = Plan(
            plan_id="test-gap-fill-without-facts",
            template_id="T_LOOKUP",
            scope_mode=analysis.scope.mode,
            paper_ids=list(analysis.scope.paper_ids),
            filters=dict(analysis.scope.filters),
            steps=steps,
            budget=PlanBudget(max_steps=3, max_retrieval_calls=1, max_llm_calls=2, max_wall_time_seconds=15),
            selection_reason="unit_test",
            required_validators=["plan_contracts"],
        )
        seen_fact_presence = []

        def gap(ctx):
            seen_fact_presence.append(ctx.has_artifact("facts"))
            return {"facts": [{"field": "sensitivity", "value": 0.91}], "evidence": [{"evidence_id": "ev1"}]}

        result = Executor({
            "gap_fill": gap,
            "synthesize": lambda ctx: {"answer_draft": {"text": "value found", "evidence_ids": ["ev1"]}},
            "verify": lambda ctx: {"answer_bundle": {"answer": "value found", "verified": True}},
        }).execute(plan, query="Find sensitivity in paper_010", analysis=analysis)

        self.assertTrue(result.succeeded, result.error)
        self.assertEqual(seen_fact_presence, [False])
        self.assertEqual(result.artifacts["facts"][0]["value"], 0.91)

    def test_missing_handler_fails_before_any_handler_runs(self):
        analysis = make_analysis()
        plan = DeterministicPlanner().plan(analysis)
        calls = []
        result = Executor({"resolve_entities": lambda ctx: calls.append("called") or {"resolved_entities": []}}).execute(
            plan,
            query="Compare sensitivity and specificity in paper_010 and paper_011",
            analysis=analysis,
        )
        self.assertEqual(result.status, "failed")
        self.assertEqual(calls, [])
        self.assertIn("No handler registered", result.error)
        self.assertEqual(result.step_records, [])

    def test_output_contract_mismatch_stops_execution(self):
        analysis = make_analysis()
        plan = DeterministicPlanner().plan(analysis)
        calls = []
        handlers = make_comparison_handlers(calls)
        handlers["gap_fill"] = lambda ctx: {"facts": []}  # evidence is contractually required
        result = Executor(handlers).execute(
            plan,
            query="Compare sensitivity and specificity in paper_010 and paper_011",
            analysis=analysis,
        )
        self.assertEqual(result.status, "failed")
        self.assertIn("Output contract mismatch", result.error)
        self.assertNotIn("normalize_facts", calls)

    def test_retrieval_budget_is_enforced(self):
        analysis = make_analysis()
        plan = DeterministicPlanner().plan(analysis)
        plan.budget.max_retrieval_calls = 1
        calls = []
        result = Executor(make_comparison_handlers(calls)).execute(
            plan,
            query="Compare sensitivity and specificity in paper_010 and paper_011",
            analysis=analysis,
        )
        self.assertEqual(result.status, "failed")
        self.assertIn("Retrieval-call budget exhausted", result.error)
        self.assertEqual(result.retrieval_calls_used, 1)
        self.assertNotIn("gap_fill", calls)

    def test_scope_mismatch_is_rejected(self):
        analysis = make_analysis()
        plan = DeterministicPlanner().plan(analysis)
        other_analysis = make_analysis(paper_ids=["paper_010"])
        with self.assertRaisesRegex(ValueError, "scope_mode|paper_ids|filters"):
            Executor(make_comparison_handlers([])).execute(
                plan,
                query="Compare sensitivity and specificity in paper_010 and paper_011",
                analysis=other_analysis,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
