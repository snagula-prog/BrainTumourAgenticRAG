from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any

from retrieval.research_retriever import ResearchRetriever


class FakeDenseRetriever:
    def __init__(self, results: list[dict[str, Any]]) -> None:
        self.profile = SimpleNamespace(alias="fake")
        self._results = results
        self.calls: list[dict[str, Any]] = []

    def query_with_stats(
        self, query: str, *, top_k: int = 10, paper_id: str | None = None
    ) -> dict[str, Any]:
        self.calls.append({"query": query, "top_k": top_k, "paper_id": paper_id})
        results = [
            dict(item) for item in self._results
            if paper_id is None or item["paper_id"] == paper_id
        ][:top_k]
        return {
            "results": results,
            "timing": {"query_encoding_seconds": 0.01, "retrieval_seconds": 0.02},
            "diagnostics": {"fake_dense": True},
        }


class FakeReranker:
    model_alias = "fake-reranker"

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.calls: list[dict[str, Any]] = []

    def rerank(
        self, query: str, documents: list[dict[str, Any]], *, top_k: int = 10
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        self.calls.append({"query": query, "ids": [d["chunk_id"] for d in documents], "top_k": top_k})
        ranked = []
        for initial_rank, document in enumerate(documents, start=1):
            item = dict(document)
            item["retrieval_rank"] = initial_rank
            item["retrieval_score"] = item.get("score")
            item["reranker_score"] = self.scores[item["chunk_id"]]
            item["score"] = item["reranker_score"]
            ranked.append(item)
        ranked.sort(key=lambda item: (-item["reranker_score"], item["retrieval_rank"]))
        ranked = ranked[:top_k]
        for rank, item in enumerate(ranked, start=1):
            item["rank"] = rank
        return ranked, {"documents_scored": len(documents), "passage_windows_scored": len(documents)}


def make_documents() -> list[dict[str, Any]]:
    return [
        {"paper_id": "p1", "chunk_id": "c1", "text": "ResNet50 reports accuracy on MRI tumor classification.", "metadata": {"page": 4}, "score": 0.8},
        {"paper_id": "p1", "chunk_id": "c2", "text": "The study uses a transfer learning CNN.", "metadata": {"page": 5}, "score": 0.7},
        {"paper_id": "p2", "chunk_id": "c3", "text": "A separate paper reports Dice score for segmentation.", "metadata": {"page": 7}, "score": 0.6},
    ]


class ResearchRetrieverTests(unittest.TestCase):
    def test_dense_without_reranker_preserves_score_and_sets_rank(self) -> None:
        dense = FakeDenseRetriever(make_documents())
        retriever = ResearchRetriever(strategy="dense", dense_retriever=dense)
        results = retriever.query("ResNet50", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["chunk_id"], "c1")
        self.assertEqual(results[0]["rank"], 1)
        self.assertEqual(results[0]["score"], 0.8)
        self.assertEqual(dense.calls[0]["top_k"], 1)

    def test_dense_reranking_reorders_candidate_pool_and_keeps_provenance(self) -> None:
        dense = FakeDenseRetriever(make_documents())
        reranker = FakeReranker({"c1": 0.2, "c2": 0.9, "c3": 0.1})
        retriever = ResearchRetriever(
            strategy="dense", dense_retriever=dense, reranker=reranker, candidate_k=3
        )
        response = retriever.query_with_stats("transfer learning", top_k=1)
        self.assertEqual(dense.calls[0]["top_k"], 3)
        self.assertEqual(reranker.calls[0]["top_k"], 1)
        self.assertEqual(reranker.calls[0]["ids"], ["c1", "c2", "c3"])
        self.assertEqual(response["results"][0]["chunk_id"], "c2")
        self.assertEqual(response["results"][0]["metadata"], {"page": 5})
        self.assertEqual(response["results"][0]["retrieval_rank"], 2)
        self.assertTrue(response["diagnostics"]["reranking_enabled"])
        self.assertGreaterEqual(response["timing"]["total_seconds"], 0)

    def test_hybrid_uses_bm25_and_dense_then_reranks(self) -> None:
        dense = FakeDenseRetriever(make_documents())
        reranker = FakeReranker({"c1": 0.9, "c2": 0.1, "c3": 0.2})
        retriever = ResearchRetriever(
            strategy="hybrid",
            dense_retriever=dense,
            documents=make_documents(),
            reranker=reranker,
            candidate_k=3,
        )
        response = retriever.query_with_stats("ResNet50 accuracy", top_k=2)
        self.assertEqual(dense.calls[0]["top_k"], 3)
        self.assertEqual(reranker.calls[0]["top_k"], 2)
        self.assertEqual([r["chunk_id"] for r in response["results"]][0], "c1")
        self.assertEqual(response["diagnostics"]["strategy"], "hybrid")
        self.assertEqual(response["diagnostics"]["candidate_count"], 3)
        self.assertIn("bm25", response["results"][0]["retrieval_channels"])

    def test_paper_filter_is_preserved(self) -> None:
        dense = FakeDenseRetriever(make_documents())
        retriever = ResearchRetriever(
            strategy="hybrid",
            dense_retriever=dense,
            documents=make_documents(),
        )
        results = retriever.query("accuracy", paper_id="p2", top_k=5)
        self.assertTrue(results)
        self.assertTrue(all(item["paper_id"] == "p2" for item in results))

    def test_invalid_inputs_are_rejected(self) -> None:
        dense = FakeDenseRetriever(make_documents())
        with self.assertRaises(ValueError):
            ResearchRetriever(strategy="unknown", dense_retriever=dense)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            ResearchRetriever(strategy="dense", dense_retriever=dense, candidate_k=0)
        retriever = ResearchRetriever(strategy="dense", dense_retriever=dense)
        with self.assertRaises(ValueError):
            retriever.query("  ")
        with self.assertRaises(ValueError):
            retriever.query("query", top_k=0)


if __name__ == "__main__":
    unittest.main()
