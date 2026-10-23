from __future__ import annotations

import unittest
from types import SimpleNamespace

from agent.executor import Executor
from agent.runtime import ResearchAgent
from agent.runtime_adapters import (
    AnswerClaim,
    FactCandidate,
    FactExtractionOutput,
    SynthesisOutput,
    build_runtime_handlers,
)
from planner import DeterministicPlanner
from query_analysis.schemas import QueryAnalysis


class FakeAnalyzer:
    def __init__(self, analysis):
        self.analysis = analysis

    def analyze(self, query):
        return self.analysis


class FakeRetrievalTool:
    def __init__(self):
        self.calls = []
        self.records = [
            {
                "paper_id": "paper_010",
                "paper_title": "Paper 010",
                "chunk_id": "chunk010",
                "page": 2,
                "text": "The evaluated model achieved sensitivity of 91% on the test set.",
            },
            {
                "paper_id": "paper_011",
                "paper_title": "Paper 011",
                "chunk_id": "chunk011",
                "page": 4,
                "text": "The proposed approach achieved sensitivity of 94% on the test set.",
            },
        ]

    def execute(self, arguments):
        self.calls.append(dict(arguments))
        wanted = arguments.get("paper_id")
        records = self.records if not wanted else [r for r in self.records if r["paper_id"] == wanted]
        return {"query": arguments["query"], "results": records, "timing": {}, "diagnostics": {}}


class GapFillRetrievalTool(FakeRetrievalTool):
    def execute(self, arguments):
        self.calls.append(dict(arguments))
        if len(self.calls) == 1:
            records = self.records
        else:
            records = [
                {"paper_id": "paper_010", "paper_title": "Paper 010", "chunk_id": "spec010", "page": 3,
                 "text": "The model's specificity was 88% on the test set."},
                {"paper_id": "paper_011", "paper_title": "Paper 011", "chunk_id": "spec011", "page": 5,
                 "text": "The model's specificity was 90% on the test set."},
            ]
        wanted = arguments.get("paper_id")
        if wanted:
            records = [record for record in records if record["paper_id"] == wanted]
        return {"query": arguments["query"], "results": records, "timing": {}, "diagnostics": {}}


class FakeModel:
    def complete_structured(self, prompt, *, schema, system_prompt=None):
        if schema is FactExtractionOutput:
            data = FactExtractionOutput(facts=[
                FactCandidate(
                    paper_id="paper_010", field="sensitivity", value="91%", unit="%",
                    evidence_id="ev_chunk010", quote="sensitivity of 91%",
                ),
                FactCandidate(
                    paper_id="paper_011", field="sensitivity", value="94%", unit="%",
                    evidence_id="ev_chunk011", quote="sensitivity of 94%",
                ),
                # Must be dropped: the quote isn't supported by the evidence.
                FactCandidate(
                    paper_id="paper_010", field="specificity", value="99%", unit="%",
                    evidence_id="ev_chunk010", quote="specificity was 99%",
                ),
            ])
        elif schema is SynthesisOutput:
            data = SynthesisOutput(claims=[
                AnswerClaim(text="Paper 010 reported sensitivity of 91%.", evidence_ids=["ev_chunk010"]),
                AnswerClaim(text="Paper 011 reported sensitivity of 94%.", evidence_ids=["ev_chunk011"]),
            ], caveats=[])
        else:
            raise AssertionError(f"Unexpected schema: {schema.__name__}")
        return data, SimpleNamespace(model="fake-qwen3:4b-instruct", elapsed_seconds=0.01)


class BadNumberModel(FakeModel):
    def complete_structured(self, prompt, *, schema, system_prompt=None):
        if schema is SynthesisOutput:
            data = SynthesisOutput(claims=[
                AnswerClaim(text="Paper 010 reported sensitivity of 99%.", evidence_ids=["ev_chunk010"]),
                AnswerClaim(text="Paper 011 reported sensitivity of 94%.", evidence_ids=["ev_chunk011"]),
            ], caveats=[])
            return data, SimpleNamespace(model="fake-qwen3:4b-instruct", elapsed_seconds=0.01)
        return super().complete_structured(prompt, schema=schema, system_prompt=system_prompt)


class RuntimeIntegrationTests(unittest.TestCase):
    @staticmethod
    def make_analysis():
        return QueryAnalysis(**{
            "task_type": "comparison",
            "scope": {"mode": "named_papers", "paper_ids": ["paper_010", "paper_011"], "filters": {}},
            "entities": [],
            "fields_requested": ["sensitivity"],
            "constraints": {"dataset": None, "split": "test", "metric_variant": None, "condition": None},
            "output": {"format": "prose", "chart_type": None},
            "needs": {"structured_facts": True, "comparison": True, "aggregation": False,
                      "visualization": False, "multi_doc": True, "iterative_retrieval_hint": False,
                      "artifact_access": False},
            "ambiguity": {"is_ambiguous": False, "reasons": [], "clarifying_question": None},
            "confidence": 0.95,
            "analyzer_meta": {"model": "qwen3:4b-instruct", "prompt_version": "query-analysis-v1", "validators_applied": []},
        })

    def test_end_to_end_comparison_uses_existing_retrieval_tool_contract(self):
        analysis = self.make_analysis()
        retrieval = FakeRetrievalTool()
        model = FakeModel()
        planner = DeterministicPlanner()
        executor = Executor(build_runtime_handlers(retrieval, model))
        agent = ResearchAgent(FakeAnalyzer(analysis), planner, executor)

        result = agent.run("Compare sensitivity in paper_010 and paper_011 on the test set")

        self.assertEqual(result.plan.template_id, "T_COMPARE")
        self.assertTrue(result.execution.succeeded, result.execution.error)
        operations = [step.operation for step in result.plan.steps]
        self.assertLess(operations.index("search_evidence"), operations.index("gap_fill"))
        self.assertLess(operations.index("gap_fill"), operations.index("lookup_facts"))
        self.assertEqual(len(result.execution.artifacts["facts"]), 2)
        self.assertEqual(result.execution.artifacts["comparison"]["status"], "completed")
        self.assertAlmostEqual(result.execution.artifacts["derived_values"][0]["value"], -3.0)
        self.assertTrue(result.answer_bundle["verified"])
        self.assertIn("91%", result.answer_bundle["answer"])
        self.assertIn("94%", result.answer_bundle["answer"])
        self.assertGreaterEqual(len(retrieval.calls), 1)

    def test_verifier_rejects_unsupported_numeric_claim(self):
        analysis = self.make_analysis()
        retrieval = FakeRetrievalTool()
        agent = ResearchAgent(
            FakeAnalyzer(analysis), DeterministicPlanner(),
            Executor(build_runtime_handlers(retrieval, BadNumberModel())),
        )

        result = agent.run("Compare sensitivity in paper_010 and paper_011 on the test set")

        self.assertTrue(result.execution.succeeded, result.execution.error)
        bundle = result.answer_bundle
        self.assertFalse(bundle["verified"])
        self.assertNotIn("99%", bundle["answer"])
        self.assertTrue(any(item["reason"] == "numeric_value_not_found_in_support" for item in bundle["verification"]["rejected_claims"]))

    def test_gap_fill_retrieves_when_requested_field_is_missing(self):
        analysis = self.make_analysis()
        analysis.fields_requested = ["sensitivity", "specificity"]
        retrieval = GapFillRetrievalTool()
        model = FakeModel()
        agent = ResearchAgent(
            FakeAnalyzer(analysis), DeterministicPlanner(),
            Executor(build_runtime_handlers(retrieval, model)),
        )

        result = agent.run("Compare sensitivity and specificity in paper_010 and paper_011 on the test set")

        self.assertTrue(result.execution.succeeded, result.execution.error)
        self.assertEqual(len(retrieval.calls), 4)
        self.assertIn("specificity", retrieval.calls[1]["query"].casefold())
        self.assertEqual(
            {item["paper_id"] for item in result.execution.artifacts["evidence"]},
            {"paper_010", "paper_011"},
        )
        self.assertTrue(any("specificity" in item["text"].casefold() for item in result.execution.artifacts["evidence"]))

    def test_search_results_are_filtered_to_named_paper_scope(self):
        analysis = self.make_analysis()
        retrieval = FakeRetrievalTool()
        retrieval.records.append({
            "paper_id": "paper_001", "chunk_id": "out_of_scope",
            "text": "Sensitivity was 100% in an unrelated paper.",
        })
        handlers = build_runtime_handlers(retrieval, FakeModel())
        plan = DeterministicPlanner().plan(analysis)
        result = Executor(handlers).execute(
            plan, query="Compare sensitivity in paper_010 and paper_011", analysis=analysis
        )
        self.assertTrue(result.succeeded, result.error)
        found_ids = {item["paper_id"] for item in result.artifacts["evidence"]}
        self.assertEqual(found_ids, {"paper_010", "paper_011"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
