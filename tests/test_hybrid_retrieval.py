from __future__ import annotations

import unittest

from retrieval.hybrid_retriever import BM25Index, HybridRetriever, scientific_tokenize


class FakeDense:
    profile = type("Profile", (), {"alias": "minilm"})()

    def query_with_stats(self, query, *, top_k=10, paper_id=None):
        ranked = [
            {"paper_id": "paper_001", "chunk_id": "alpha", "text": "Dense alpha", "metadata": {}, "score": 0.9, "distance": 0.1},
            {"paper_id": "paper_001", "chunk_id": "gamma", "text": "Dense gamma", "metadata": {}, "score": 0.8, "distance": 0.2},
        ]
        if paper_id:
            ranked = [item for item in ranked if item["paper_id"] == paper_id]
        return {
            "results": ranked[:top_k],
            "timing": {"query_encoding_seconds": 0.01, "retrieval_seconds": 0.002},
            "diagnostics": {"scope_segment_count": len(ranked)},
        }


class HybridRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.docs = [
            {"paper_id": "paper_001", "chunk_id": "alpha", "text": "TP53 mutation in glioma and SVM-RFE feature selection", "metadata": {}},
            {"paper_id": "paper_001", "chunk_id": "beta", "text": "Exact TP53 protein code appears in this report", "metadata": {}},
            {"paper_id": "paper_001", "chunk_id": "gamma", "text": "Dense-only semantic passage on MRI imaging", "metadata": {}},
            {"paper_id": "paper_002", "chunk_id": "delta", "text": "TP53 expression in another cohort", "metadata": {}},
        ]

    def test_tokenizer_preserves_compounds_and_components(self):
        tokens = scientific_tokenize("SVM-RFE and T1-weighted MRI; TP53")
        self.assertIn("svm-rfe", tokens)
        self.assertIn("svm", tokens)
        self.assertIn("rfe", tokens)
        self.assertIn("t1-weighted", tokens)
        self.assertIn("tp53", tokens)

    def test_bm25_finds_exact_technical_identifier(self):
        index = BM25Index(self.docs)
        ranked = index.search("TP53", top_k=3, paper_id="paper_001")
        self.assertTrue(ranked)
        self.assertEqual(ranked[0]["chunk_id"], "beta")
        self.assertTrue(all(item["paper_id"] == "paper_001" for item in ranked))

    def test_bm25_returns_no_results_when_no_terms_match(self):
        index = BM25Index(self.docs)
        self.assertEqual(index.search("unmatchable_xyz", top_k=5), [])

    def test_hybrid_fuses_and_returns_unique_parent_chunks(self):
        hybrid = HybridRetriever(
            "minilm", dense_retriever=FakeDense(), documents=self.docs,
            candidate_k=4, rrf_k=60,
        )
        results = hybrid.query("TP53 MRI", top_k=4, paper_id="paper_001")
        keys = [(item["paper_id"], item["chunk_id"]) for item in results]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertTrue(all(paper == "paper_001" for paper, _ in keys))
        self.assertIn("alpha", [item["chunk_id"] for item in results])
        self.assertTrue(all("rrf_score" in item for item in results))

    def test_rrf_gives_combined_document_two_channel_contributions(self):
        hybrid = HybridRetriever(
            "minilm", dense_retriever=FakeDense(), documents=self.docs,
            candidate_k=4, rrf_k=60,
        )
        results = hybrid.query("TP53", top_k=4)
        alpha = next(item for item in results if item["chunk_id"] == "alpha")
        self.assertEqual(alpha["retrieval_channels"], ["dense", "bm25"])
        self.assertIsNotNone(alpha["dense_rank"])
        self.assertIsNotNone(alpha["bm25_rank"])
        self.assertAlmostEqual(alpha["rrf_score"], 1 / (60 + alpha["dense_rank"]) + 1 / (60 + alpha["bm25_rank"]))


if __name__ == "__main__":
    unittest.main()
