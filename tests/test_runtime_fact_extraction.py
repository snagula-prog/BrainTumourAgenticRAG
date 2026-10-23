from __future__ import annotations

import unittest

from agent.runtime_adapters import (
    _deterministic_incomplete_comparison_draft,
    _extract_explicit_metric_facts,
    _reconstruct_parent_evidence,
    _select_balanced_evidence,
)


class RuntimeFactExtractionTests(unittest.TestCase):
    def test_extracts_explicitly_labelled_values_with_exact_quotes(self):
        source = {
            "paper_id": "paper_010",
            "evidence_id": "ev10",
            "page": 2,
            "text": "Sensitivity is 93.55 Specificity is 69.79 Accuracy is 85.32.",
        }
        facts = _extract_explicit_metric_facts([source], ["sensitivity", "specificity"])
        by_field = {fact["field"]: fact for fact in facts}
        self.assertEqual(set(by_field), {"sensitivity", "specificity"})
        self.assertEqual(by_field["sensitivity"]["value"], "93.55")
        self.assertEqual(by_field["specificity"]["value"], "69.79")
        self.assertEqual(by_field["sensitivity"]["quote"], "Sensitivity is 93.55")
        self.assertEqual(by_field["specificity"]["quote"], "Specificity is 69.79")
        self.assertIsNone(by_field["sensitivity"]["unit"])

    def test_preserves_percent_only_when_source_explicitly_contains_it(self):
        source = {
            "paper_id": "paper_010",
            "evidence_id": "ev10",
            "text": "Sensitivity: 93.55% and specificity was 69.79.",
        }
        facts = _extract_explicit_metric_facts([source], ["sensitivity", "specificity"])
        by_field = {fact["field"]: fact for fact in facts}
        self.assertEqual(by_field["sensitivity"]["unit"], "%")
        self.assertIsNone(by_field["specificity"]["unit"])

    def test_does_not_infer_a_number_from_unlabelled_prose(self):
        source = {
            "paper_id": "paper_010",
            "evidence_id": "ev10",
            "text": "Sensitivity improved substantially. Specificity was not discussed.",
        }
        facts = _extract_explicit_metric_facts([source], ["sensitivity", "specificity"])
        self.assertEqual(facts, [])

    def test_balances_context_across_named_papers(self):
        evidence = [
            {"paper_id": "paper_010", "evidence_id": f"a{i}", "text": str(i)}
            for i in range(10)
        ] + [
            {"paper_id": "paper_011", "evidence_id": f"b{i}", "text": str(i)}
            for i in range(10)
        ]
        selected = _select_balanced_evidence(evidence, ["paper_010", "paper_011"], limit=12)
        counts = {}
        for item in selected:
            counts[item["paper_id"]] = counts.get(item["paper_id"], 0) + 1
        self.assertEqual(counts, {"paper_010": 6, "paper_011": 6})

    def test_reconstructs_decimal_across_complete_segments_without_prefix_fact(self):
        evidence = [
            {"paper_id": "paper_010", "evidence_id": "a", "chunk_id": "c", "segment_index": 0, "segment_count": 2, "text": "Sensitivity is 93"},
            {"paper_id": "paper_010", "evidence_id": "b", "chunk_id": "c", "segment_index": 1, "segment_count": 2, "text": ".55%"},
        ]
        reconstructed = _reconstruct_parent_evidence(evidence)
        facts = _extract_explicit_metric_facts(reconstructed, ["sensitivity"])
        self.assertEqual([fact["value"] for fact in facts], ["93.55"])
        self.assertEqual(reconstructed[0]["source_evidence_ids"], ["a", "b"])

    def test_incomplete_comparison_keeps_supported_facts_and_uses_coverage_ids(self):
        facts = [
            {"paper_id": "paper_010", "field": "sensitivity", "value": "93.55", "unit": None, "evidence_id": "ev10"},
            {"paper_id": "paper_010", "field": "specificity", "value": "69.79", "unit": None, "evidence_id": "ev10"},
        ]
        availability = {
            "paper_results": [{
                "paper_id": "paper_011",
                "fields": {
                    "sensitivity": {"status": "not_found_in_indexed_text", "coverage_status": "complete_indexed_coverage", "availability_id": "av_sens"},
                    "specificity": {"status": "not_found_in_indexed_text", "coverage_status": "complete_indexed_coverage", "availability_id": "av_spec"},
                },
            }],
        }
        draft = _deterministic_incomplete_comparison_draft(
            facts, availability, {"status": "insufficient_comparable_data", "paper_ids": ["paper_010", "paper_011"]}
        )
        self.assertEqual(draft["model"], "deterministic_incomplete_comparison")
        self.assertEqual(len(draft["claims"]), 3)
        summary = draft["claims"][-1]
        self.assertIn("not a complete two-paper numeric comparison", summary["text"])
        self.assertEqual(summary["evidence_ids"], ["ev10"])
        self.assertEqual(summary["availability_ids"], ["av_sens", "av_spec"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
