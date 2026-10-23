from __future__ import annotations

import re
import unittest
from typing import Any

from agent.tools.document_metric_check import DocumentMetricCheckTool
from agent.tools.retrieval import RetrievalTool
from agent.runtime_adapters import build_runtime_handlers
from planner import DeterministicPlanner


def field_matcher(text: str, field: str) -> bool:
    return re.search(rf"\b{re.escape(field)}\b", text, re.IGNORECASE) is not None


def pattern_factory(field: str) -> re.Pattern[str] | None:
    return re.compile(
        rf"\b{re.escape(field)}\b\s*(?:is|was|of|:|=)\s*"
        r"(?P<number>[-+]?(?:\d+(?:\.\d*)?|\.\d+))"
        r"(?P<unit>\s*%)?",
        re.IGNORECASE,
    )


def terms_factory(field: str) -> list[str]:
    return [field]


class FakeMetricScanRetrieval:
    def __init__(self, segments: list[dict[str, Any]], coverage_status: str) -> None:
        self.segments = segments
        self.coverage_status = coverage_status

    def get_indexed_segments(self, paper_id: str) -> dict[str, Any]:
        return {
            "paper_id": paper_id,
            "segments": self.segments,
            "coverage": {
                "status": self.coverage_status,
                "expected_segment_count": len(self.segments) if self.coverage_status == "complete_indexed_coverage" else len(self.segments) + 1,
                "segments_scanned": len(self.segments),
                "unique_parent_chunks_scanned": len({item["chunk_id"] for item in self.segments}),
                "issues": [] if self.coverage_status == "complete_indexed_coverage" else ["fixture intentionally incomplete"],
            },
        }


class DocumentMetricCheckTests(unittest.TestCase):
    def make_checker(self, retrieval: Any) -> DocumentMetricCheckTool:
        return DocumentMetricCheckTool(
            retrieval,
            field_matcher=field_matcher,
            metric_pattern_factory=pattern_factory,
            field_terms_factory=terms_factory,
            paper_title_resolver=lambda paper_id: {"paper_010": "Threshold Segmentation", "paper_011": "Optimized U-Net"}.get(paper_id),
        )

    def test_complete_scan_finds_explicit_numeric_metric_values(self) -> None:
        text = "Sensitivity is 93.55 Specificity is 69.79 Accuracy is 85.32"
        segment = {
            "segment_id": "seg_010_1",
            "paper_id": "paper_010",
            "chunk_id": "chunk_21",
            "segment_index": 0,
            "segment_count": 1,
            "source_sha256": "sha-010",
            "text": text,
            "metadata": {"source_metadata": {"page": 2}},
        }
        result = self.make_checker(FakeMetricScanRetrieval([segment], "complete_indexed_coverage")).execute({
            "paper_ids": ["paper_010"],
            "fields": ["sensitivity", "specificity"],
        })
        availability = result["document_metric_availability"]
        fields = availability["paper_results"][0]["fields"]
        self.assertEqual(fields["sensitivity"]["status"], "numeric_value_evidence_found")
        self.assertEqual(fields["specificity"]["status"], "numeric_value_evidence_found")
        self.assertEqual(len(result["evidence"]), 1)
        self.assertIn("ev_indexseg_", result["evidence"][0]["evidence_id"])
        self.assertEqual(result["evidence"][0]["page"], 2)
        self.assertTrue(fields["sensitivity"]["matched_evidence_ids"])

    def test_complete_scan_reports_only_indexed_text_absence(self) -> None:
        segment = {
            "segment_id": "seg_011_1",
            "paper_id": "paper_011",
            "chunk_id": "chunk_1",
            "segment_index": 0,
            "segment_count": 1,
            "source_sha256": "sha-011",
            "text": "The baseline U-Net achieved a mean Dice score of 0.9130.",
            "metadata": {},
        }
        result = self.make_checker(FakeMetricScanRetrieval([segment], "complete_indexed_coverage")).execute({
            "paper_ids": ["paper_011"],
            "fields": ["sensitivity", "specificity"],
        })
        paper = result["document_metric_availability"]["paper_results"][0]
        self.assertEqual(paper["fields"]["sensitivity"]["status"], "not_found_in_indexed_text")
        self.assertEqual(paper["fields"]["specificity"]["status"], "not_found_in_indexed_text")
        self.assertEqual(result["evidence"], [])
        self.assertEqual(result["document_metric_availability"]["scope"], "indexed_chroma_segments_only")

    def test_incomplete_scan_never_claims_metric_absence(self) -> None:
        segment = {
            "segment_id": "seg_011_1",
            "paper_id": "paper_011",
            "chunk_id": "chunk_1",
            "segment_index": 0,
            "segment_count": 2,
            "source_sha256": "sha-011",
            "text": "The model reports Dice scores.",
            "metadata": {},
        }
        result = self.make_checker(FakeMetricScanRetrieval([segment], "incomplete_indexed_coverage")).execute({
            "paper_ids": ["paper_011"],
            "fields": ["sensitivity"],
        })
        field = result["document_metric_availability"]["paper_results"][0]["fields"]["sensitivity"]
        self.assertEqual(field["status"], "unknown_incomplete_index")
        self.assertNotEqual(field["status"], "not_found_in_indexed_text")

    def test_complete_scan_matches_decimal_across_parent_segments(self) -> None:
        segments = [
            {"segment_id": "s1", "paper_id": "paper_010", "chunk_id": "c", "segment_index": 0, "segment_count": 2, "source_sha256": "x", "text": "Sensitivity is 93", "metadata": {}},
            {"segment_id": "s2", "paper_id": "paper_010", "chunk_id": "c", "segment_index": 1, "segment_count": 2, "source_sha256": "x", "text": ".55%", "metadata": {}},
        ]
        result = self.make_checker(FakeMetricScanRetrieval(segments, "complete_indexed_coverage")).execute({"paper_ids": ["paper_010"], "fields": ["sensitivity"]})
        self.assertEqual(result["document_metric_availability"]["paper_results"][0]["fields"]["sensitivity"]["status"], "numeric_value_evidence_found")
        self.assertIn("s1", result["evidence"][0]["source_segment_ids"])


class FakeCollection:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records
        self.calls: list[dict[str, Any]] = []

    def get(self, *, where: dict[str, Any], limit: int, offset: int, include: list[str]) -> dict[str, Any]:
        self.calls.append({"where": where, "limit": limit, "offset": offset, "include": include})
        filtered = [item for item in self.records if item["metadata"].get("paper_id") == where["paper_id"]]
        page = filtered[offset:offset + limit]
        return {
            "ids": [item["id"] for item in page],
            "documents": [item["document"] for item in page] if "documents" in include else None,
            "metadatas": [item["metadata"] for item in page] if "metadatas" in include else None,
        }


class FakeDense:
    def __init__(self, collection: FakeCollection, expected_count: int) -> None:
        self.collection = collection
        self._paper_segment_counts = {"paper_010": expected_count}
        self.paper_ids = ["paper_010"]


class FakeResearchRetriever:
    def __init__(self, dense: FakeDense) -> None:
        self.dense_retriever = dense


class RetrievalInventoryTests(unittest.TestCase):
    @staticmethod
    def metadata(chunk_id: str, index: int, count: int) -> dict[str, Any]:
        return {
            "paper_id": "paper_010",
            "chunk_id": chunk_id,
            "segment_id": f"{chunk_id}_seg_{index}",
            "segment_index": index,
            "segment_count": count,
            "source_sha256": "chunks-sha-010",
            "source_metadata_json": '{"page": 2}',
        }

    def test_reads_all_filtered_pages_without_query_embeddings(self) -> None:
        records = [
            {"id": "a0", "document": "Sensitivity is", "metadata": self.metadata("a", 0, 2)},
            {"id": "a1", "document": "93.55 Specificity is 69.79", "metadata": self.metadata("a", 1, 2)},
            {"id": "b0", "document": "Dice score is 0.91", "metadata": self.metadata("b", 0, 1)},
        ]
        collection = FakeCollection(records)
        tool = RetrievalTool(FakeResearchRetriever(FakeDense(collection, expected_count=3)))
        scan = tool.get_indexed_segments("paper_010", page_size=2)
        self.assertEqual(scan["coverage"]["status"], "complete_indexed_coverage")
        self.assertEqual(scan["coverage"]["segments_scanned"], 3)
        self.assertEqual(scan["coverage"]["unique_parent_chunks_scanned"], 2)
        self.assertEqual([call["offset"] for call in collection.calls], [0, 2])
        self.assertTrue(all(call["where"] == {"paper_id": "paper_010"} for call in collection.calls))

    def test_inventory_mismatch_marks_scan_incomplete(self) -> None:
        records = [
            {"id": "a0", "document": "Sensitivity is", "metadata": self.metadata("a", 0, 2)},
            {"id": "b0", "document": "Dice score is 0.91", "metadata": self.metadata("b", 0, 1)},
        ]
        collection = FakeCollection(records)
        tool = RetrievalTool(FakeResearchRetriever(FakeDense(collection, expected_count=3)))
        scan = tool.get_indexed_segments("paper_010", page_size=2)
        self.assertEqual(scan["coverage"]["status"], "incomplete_indexed_coverage")
        self.assertTrue(scan["coverage"]["issues"])


class FakeRuntimeScope:
    def __init__(self, paper_ids: list[str]) -> None:
        self.paper_ids = paper_ids
        self.mode = "named_papers" if len(paper_ids) > 1 else "single_paper"


class FakeRuntimeAnalysis:
    def __init__(self, fields_requested: list[str]) -> None:
        self.fields_requested = fields_requested


class FakeRuntimeContext:
    def __init__(self, *, paper_ids: list[str], fields: list[str], artifacts: dict[str, Any] | None = None) -> None:
        self.query = "Compare requested metrics"
        self.scope = FakeRuntimeScope(paper_ids)
        self.analysis = FakeRuntimeAnalysis(fields)
        self._artifacts = dict(artifacts or {})

    def get_artifact(self, name: str, default: Any = None) -> Any:
        return self._artifacts.get(name, default)


class FakeSynthesisModel:
    pass


class RuntimeAvailabilityVerificationTests(unittest.TestCase):
    @staticmethod
    def segment(text: str) -> dict[str, Any]:
        return {
            "segment_id": "seg_011_1",
            "paper_id": "paper_011",
            "chunk_id": "chunk_1",
            "segment_index": 0,
            "segment_count": 1,
            "source_sha256": "sha-011",
            "text": text,
            "metadata": {},
        }

    def handlers_for(self, coverage_status: str) -> dict[str, Any]:
        class Retrieval:
            def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
                return {"query": arguments.get("query", ""), "results": [], "timing": {}, "diagnostics": {}}

            def get_indexed_segments(self, paper_id: str) -> dict[str, Any]:
                segment = RuntimeAvailabilityVerificationTests.segment("The baseline U-Net reports a mean Dice score of 0.9130.")
                return {
                    "paper_id": paper_id,
                    "segments": [segment],
                    "coverage": {
                        "status": coverage_status,
                        "expected_segment_count": 1 if coverage_status == "complete_indexed_coverage" else 2,
                        "segments_scanned": 1,
                        "unique_parent_chunks_scanned": 1,
                        "issues": [] if coverage_status == "complete_indexed_coverage" else ["fixture incomplete"],
                    },
                }

        return build_runtime_handlers(Retrieval(), FakeSynthesisModel())

    def test_verifier_accepts_absence_only_with_complete_coverage_and_availability_id(self) -> None:
        handlers = self.handlers_for("complete_indexed_coverage")
        context = FakeRuntimeContext(paper_ids=["paper_011"], fields=["sensitivity"])
        checked = handlers["check_document_metrics"](context)
        context._artifacts.update(checked)
        availability_id = checked["document_metric_availability"]["paper_results"][0]["fields"]["sensitivity"]["availability_id"]
        context._artifacts["answer_draft"] = {
            "claims": [{
                "text": "No explicit numeric sensitivity value was found in the indexed text for paper_011.",
                "availability_ids": [availability_id],
                "evidence_ids": [],
                "derived_value_ids": [],
            }],
            "caveats": [],
        }
        verified = handlers["verify"](context)["answer_bundle"]
        self.assertTrue(verified["verified"])
        self.assertEqual(verified["status"], "verified")
        self.assertEqual(verified["availability_ids"], [availability_id])

    def test_document_scan_is_skipped_when_fact_lookup_resolved_every_cell(self) -> None:
        handlers = self.handlers_for("complete_indexed_coverage")
        context = FakeRuntimeContext(paper_ids=["paper_011"], fields=["sensitivity"], artifacts={
            "facts": [{"paper_id": "paper_011", "field": "sensitivity", "value": "91", "evidence_id": "ev1"}],
        })
        checked = handlers["check_document_metrics"](context)
        self.assertEqual(checked["document_metric_availability"]["status"], "not_checked")
        self.assertEqual(checked["document_metric_availability"]["paper_results"], [])

    def test_verifier_rejects_absence_when_index_coverage_is_incomplete(self) -> None:
        handlers = self.handlers_for("incomplete_indexed_coverage")
        context = FakeRuntimeContext(paper_ids=["paper_011"], fields=["sensitivity"])
        checked = handlers["check_document_metrics"](context)
        context._artifacts.update(checked)
        availability_id = checked["document_metric_availability"]["paper_results"][0]["fields"]["sensitivity"]["availability_id"]
        context._artifacts["answer_draft"] = {
            "claims": [{
                "text": "No explicit numeric sensitivity value was found in the indexed text for paper_011.",
                "availability_ids": [availability_id],
                "evidence_ids": [],
                "derived_value_ids": [],
            }],
            "caveats": [],
        }
        verified = handlers["verify"](context)["answer_bundle"]
        self.assertFalse(verified["verified"])
        self.assertEqual(verified["verification"]["rejected_claims"][0]["reason"], "metric_absence_not_supported_by_complete_index_scan")

    def test_verifier_rejects_unqualified_negative_even_if_scan_is_complete(self) -> None:
        handlers = self.handlers_for("complete_indexed_coverage")
        context = FakeRuntimeContext(paper_ids=["paper_011"], fields=["sensitivity"])
        checked = handlers["check_document_metrics"](context)
        context._artifacts.update(checked)
        availability_id = checked["document_metric_availability"]["paper_results"][0]["fields"]["sensitivity"]["availability_id"]
        context._artifacts["answer_draft"] = {
            "claims": [{
                "text": "Paper_011 does not report sensitivity.",
                "availability_ids": [availability_id],
                "evidence_ids": [],
                "derived_value_ids": [],
            }],
            "caveats": [],
        }
        verified = handlers["verify"](context)["answer_bundle"]
        self.assertFalse(verified["verified"])


    def test_verifier_rejects_unrequested_metric_in_absence_claim(self) -> None:
        handlers = self.handlers_for("complete_indexed_coverage")
        context = FakeRuntimeContext(paper_ids=["paper_011"], fields=["sensitivity"])
        checked = handlers["check_document_metrics"](context)
        context._artifacts.update(checked)
        availability_id = checked["document_metric_availability"]["paper_results"][0]["fields"]["sensitivity"]["availability_id"]
        context._artifacts["answer_draft"] = {
            "claims": [{
                "text": "No explicit numeric sensitivity or specificity value was found in the indexed text for paper_011.",
                "availability_ids": [availability_id],
                "evidence_ids": [],
                "derived_value_ids": [],
            }],
            "caveats": [],
        }
        verified = handlers["verify"](context)["answer_bundle"]
        self.assertFalse(verified["verified"])
        self.assertEqual(verified["verification"]["rejected_claims"][0]["reason"], "metric_absence_contains_unrequested_field")


class FakePlannerNeeds:
    multi_doc = False
    iterative_retrieval_hint = False
    visualization = False
    comparison = True
    aggregation = False
    structured_facts = True
    artifact_access = False


class FakePlannerAmbiguity:
    is_ambiguous = False
    clarifying_question = None


class FakePlannerScope:
    def __init__(self, mode: str, paper_ids: list[str]) -> None:
        self.mode = mode
        self.paper_ids = paper_ids
        self.filters: dict[str, str] = {}


class FakePlannerOutput:
    format = "prose"


class FakePlannerAnalysis:
    task_type = "comparison"
    needs = FakePlannerNeeds()
    ambiguity = FakePlannerAmbiguity()
    output = FakePlannerOutput()
    fields_requested = ["sensitivity", "specificity"]

    def __init__(self, mode: str, paper_ids: list[str]) -> None:
        self.scope = FakePlannerScope(mode, paper_ids)


class PlannerMetricCheckTests(unittest.TestCase):
    def test_named_paper_metric_comparison_inserts_document_scan(self) -> None:
        plan = DeterministicPlanner().plan(FakePlannerAnalysis("named_papers", ["paper_010", "paper_011"]))
        operations = [step.operation for step in plan.steps]
        self.assertIn("check_document_metrics", operations)
        self.assertGreater(operations.index("check_document_metrics"), operations.index("lookup_facts"))
        self.assertIn("document_metric_availability", plan.steps[operations.index("check_document_metrics")].produces)

    def test_unscoped_comparison_does_not_run_document_scan(self) -> None:
        plan = DeterministicPlanner().plan(FakePlannerAnalysis("whole_corpus", []))
        self.assertNotIn("check_document_metrics", [step.operation for step in plan.steps])



if __name__ == "__main__":
    unittest.main()
