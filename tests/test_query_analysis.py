from __future__ import annotations

import unittest
from dataclasses import dataclass

from query_analysis.analyzer import QueryAnalyzer
from query_analysis.normalization import extract_explicit_paper_ids, normalize_and_validate
from query_analysis.schemas import (
    QueryAnalysisDraft,
    QueryValidationError,
)


@dataclass(frozen=True)
class FakeResponse:
    model: str = "qwen3:4b-instruct"


class FakeChatModel:
    def __init__(self, draft: QueryAnalysisDraft):
        self.draft = draft
        self.last_prompt = None
        self.last_schema = None

    def complete(self, prompt, *, system_prompt=None):
        raise AssertionError("QueryAnalyzer should call complete_structured")

    def complete_structured(self, prompt, *, schema, system_prompt=None):
        self.last_prompt = prompt
        self.last_schema = schema
        return self.draft, FakeResponse()


def make_draft(**overrides):
    base = {
        "task_type": "factual_lookup",
        "scope": {"mode": "whole_corpus", "paper_ids": ["paper_999"], "filters": {"dataset": "BraTS 2021"}},
        "entities": [
            {"surface": "U-Net", "type": "model", "canonical": "hallucinated canonical", "confidence": 0.9},
            {"surface": "U-Net", "type": "model", "canonical": "another guess", "confidence": 0.8},
            {"surface": "BraTS 2021", "type": "dataset", "canonical": None, "confidence": 0.7},
        ],
        "fields_requested": ["ROC-AUC", "number of parameters"],
        "constraints": {"dataset": "BraTS 2021", "split": "test set", "metric_variant": None, "condition": "five-fold cross validation"},
        "output": {"format": "prose", "chart_type": None},
        "needs": {"structured_facts": False, "comparison": False, "aggregation": False, "visualization": False, "multi_doc": False, "iterative_retrieval_hint": False, "artifact_access": False},
        "ambiguity": {"is_ambiguous": False, "reasons": [], "clarifying_question": None},
        "confidence": 0.9,
    }
    base.update(overrides)
    return QueryAnalysisDraft(**base)


class QueryAnalysisTests(unittest.TestCase):
    def test_explicit_paper_ids_only_and_normalized(self):
        self.assertEqual(extract_explicit_paper_ids("Compare PAPER-8 and paper_010 please"), ["paper_008", "paper_010"])

    def test_deterministic_scope_and_cues_override_model_guesses(self):
        query = "Plot accuracy and ROC-AUC for paper_008 versus paper_010 on BraTS 2021 test set"
        result = normalize_and_validate(
            make_draft(), query=query, model_name="qwen3:4b-instruct",
            valid_paper_ids={"paper_008", "paper_010"},
            entity_aliases={"dataset": {"BraTS 2021": "BraTS 2021"}, "model": {"u-net": "U-Net"}},
        )
        self.assertEqual(result.scope.mode, "named_papers")
        self.assertEqual(result.scope.paper_ids, ["paper_008", "paper_010"])
        self.assertEqual(result.task_type, "chart_request")
        self.assertEqual(result.output.format, "chart")
        self.assertTrue(result.needs.comparison)
        self.assertTrue(result.needs.visualization)
        self.assertEqual(result.fields_requested, ["AUC", "parameters", "accuracy"])
        self.assertEqual(len(result.entities), 2)
        self.assertEqual(result.entities[0].canonical, "U-Net")
        self.assertEqual(result.entities[1].canonical, "BraTS 2021")
        self.assertEqual(result.constraints.dataset, "BraTS 2021")
        self.assertEqual(result.constraints.split, "test set")
        self.assertNotIn("paper_999", result.scope.paper_ids)
        self.assertIsNone(result.constraints.condition)  # Not present in the query.

    def test_unknown_explicit_paper_fails_when_registry_is_provided(self):
        with self.assertRaises(QueryValidationError):
            normalize_and_validate(
                make_draft(), query="What accuracy is in paper_999?", model_name="qwen3:4b-instruct",
                valid_paper_ids={"paper_008"},
            )

    def test_scope_does_not_expand_to_whole_corpus_without_user_cue(self):
        result = normalize_and_validate(
            make_draft(), query="What is the accuracy for U-Net?", model_name="qwen3:4b-instruct"
        )
        self.assertEqual(result.scope.mode, "filtered_corpus")
        self.assertEqual(result.scope.paper_ids, [])
        self.assertEqual(result.scope.filters, {})

    def test_analyzer_uses_structured_interface_and_attaches_metadata(self):
        draft = make_draft(scope={"mode": "filtered_corpus", "paper_ids": [], "filters": {}})
        model = FakeChatModel(draft)
        result = QueryAnalyzer(model).analyze("Get accuracy from paper_008")
        self.assertIs(model.last_schema, QueryAnalysisDraft)
        self.assertIn('"Get accuracy from paper_008"', model.last_prompt)
        self.assertEqual(result.analyzer_meta.model, "qwen3:4b-instruct")
        self.assertEqual(result.scope.paper_ids, ["paper_008"])

    def test_model_needs_flags_do_not_override_deterministic_rules(self):
        draft = make_draft(
            task_type="factual_lookup",
            scope={"mode": "filtered_corpus", "paper_ids": [], "filters": {}},
            fields_requested=["ROC-AUC"],
            needs={"structured_facts": False, "comparison": True, "aggregation": True, "visualization": True, "multi_doc": False, "iterative_retrieval_hint": True, "artifact_access": True},
        )
        result = normalize_and_validate(draft, query="What is the ROC-AUC?", model_name="qwen3:4b-instruct")
        self.assertEqual(result.fields_requested, ["AUC"])
        self.assertFalse(result.needs.comparison)
        self.assertFalse(result.needs.aggregation)
        self.assertFalse(result.needs.visualization)
        self.assertFalse(result.needs.iterative_retrieval_hint)
        self.assertFalse(result.needs.artifact_access)

    def test_empty_query_is_rejected(self):
        with self.assertRaises(ValueError):
            QueryAnalyzer(FakeChatModel(make_draft())).analyze("   ")


if __name__ == "__main__":
    unittest.main()
