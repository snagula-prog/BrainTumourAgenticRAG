from __future__ import annotations
import unittest
from query_analysis.schemas import QueryAnalysis, AnalyzerMeta
from planner import DeterministicPlanner, Plan, PlanValidationError, PlanValidator


def make_analysis(**overrides):
    data = {
        "task_type": "factual_lookup",
        "scope": {"mode": "whole_corpus", "paper_ids": [], "filters": {}},
        "entities": [],
        "fields_requested": ["accuracy"],
        "constraints": {"dataset": None, "split": None, "metric_variant": None, "condition": None},
        "output": {"format": "prose", "chart_type": None},
        "needs": {"structured_facts": False, "comparison": False, "aggregation": False,
                  "visualization": False, "multi_doc": False, "iterative_retrieval_hint": False,
                  "artifact_access": False},
        "ambiguity": {"is_ambiguous": False, "reasons": [], "clarifying_question": None},
        "confidence": 0.9,
        "analyzer_meta": {"model": "qwen3:4b-instruct", "prompt_version": "query-analysis-v1", "validators_applied": []},
    }
    for key, value in overrides.items():
        data[key] = value
    return QueryAnalysis(**data)


class PlannerTests(unittest.TestCase):
    def setUp(self):
        self.planner = DeterministicPlanner()

    def test_default_lookup_is_fixed_template(self):
        plan = self.planner.plan(make_analysis())
        self.assertEqual(plan.template_id, "T_LOOKUP")
        self.assertEqual([s.operation for s in plan.steps], ["search_evidence", "synthesize", "verify"])
        self.assertEqual(plan.steps[1].depends_on, [plan.steps[0].step_id])

    def test_comparison_uses_deterministic_analytics_and_validator(self):
        needs = {"structured_facts": True, "comparison": True, "aggregation": False,
                 "visualization": False, "multi_doc": True, "iterative_retrieval_hint": False,
                 "artifact_access": False}
        output = {"format": "table", "chart_type": None}
        plan = self.planner.plan(make_analysis(task_type="comparison", needs=needs, output=output))
        ops = [s.operation for s in plan.steps]
        self.assertEqual(plan.template_id, "T_COMPARE")
        self.assertLess(ops.index("compare"), ops.index("validate_derived_values"))
        self.assertIn("render_table", ops)
        self.assertEqual(ops[-1], "verify")

    def test_chart_validates_data_and_chart_spec_before_rendering(self):
        needs = {"structured_facts": True, "comparison": True, "aggregation": False,
                 "visualization": True, "multi_doc": False, "iterative_retrieval_hint": False,
                 "artifact_access": False}
        plan = self.planner.plan(make_analysis(task_type="chart_request", output={"format": "chart", "chart_type": "bar"}, needs=needs))
        ops = [s.operation for s in plan.steps]
        self.assertEqual(plan.template_id, "T_CHART")
        self.assertLess(ops.index("validate_chart_data"), ops.index("build_chart_spec"))
        self.assertLess(ops.index("build_chart_spec"), ops.index("render_chart"))
        self.assertIn("chart_data_validation", plan.required_validators)

    def test_ambiguity_and_unsupported_requests_stop_before_tools(self):
        clarify = make_analysis(task_type="clarification_needed", ambiguity={"is_ambiguous": True, "reasons": ["scope"], "clarifying_question": "Which papers?"})
        decline = make_analysis(task_type="unsupported")
        self.assertEqual([s.operation for s in self.planner.plan(clarify).steps], ["ask_clarification"])
        self.assertEqual([s.operation for s in self.planner.plan(decline).steps], ["decline"])

    def test_corpus_metadata_has_no_retrieval_or_synthesis(self):
        plan = self.planner.plan(make_analysis(task_type="corpus_meta"))
        self.assertEqual(plan.template_id, "T_META")
        self.assertEqual([s.operation for s in plan.steps], ["search_metadata", "render_metadata_response"])

    def test_validator_rejects_a_broken_plan(self):
        good = self.planner.plan(make_analysis(task_type="aggregation", needs={
            "structured_facts": True, "comparison": False, "aggregation": True,
            "visualization": False, "multi_doc": False, "iterative_retrieval_hint": False,
            "artifact_access": False,
        }))
        bad = good.model_copy(deep=True) if hasattr(good, "model_copy") else good.copy(deep=True)
        bad.steps = bad.steps[:-2] + bad.steps[-1:]
        with self.assertRaises(PlanValidationError):
            PlanValidator().validate(bad)


if __name__ == "__main__":
    unittest.main()
