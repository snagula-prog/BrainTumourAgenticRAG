from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Any, Literal


RetrievalStrategy = Literal["dense", "hybrid"]


class ResearchRetriever:
    """Configurable dense/hybrid retriever with optional cross-encoder reranking.

    The existing dense and hybrid retrievers remain responsible for candidate
    generation. When enabled, the cross-encoder only reorders the retrieved
    candidate pool; it cannot recover chunks absent from that pool.

    Args:
        model_alias: Embedding profile used by DenseChromaRetriever.
        strategy: ``"dense"`` or ``"hybrid"`` (dense + BM25/RRF).
        candidate_k: Candidate count per channel for hybrid retrieval, and the
            candidate pool size for dense retrieval when reranking is enabled.
        rrf_k: RRF constant, used only with hybrid retrieval.
        reranker_alias: Optional cross-encoder profile. ``None`` disables
            reranking and avoids loading a reranker model.
        reranker_device: Device passed to CrossEncoderReranker if enabled.
        reranker_batch_size: Pair-scoring batch size for the cross-encoder.
        dense_retriever/documents/reranker: Dependency-injection seams for tests
            and controlled experiments. Normal application code can omit them.
    """

    def __init__(
        self,
        model_alias: str = "bge",
        *,
        strategy: RetrievalStrategy = "hybrid",
        candidate_k: int = 50,
        rrf_k: int = 60,
        reranker_alias: str | None = None,
        reranker_device: str = "cpu",
        reranker_batch_size: int = 8,
        dense_retriever: Any | None = None,
        documents: Iterable[dict[str, Any]] | None = None,
        reranker: Any | None = None,
    ) -> None:
        if not isinstance(strategy, str) or strategy.strip().lower() not in {"dense", "hybrid"}:
            raise ValueError("strategy must be either 'dense' or 'hybrid'")
        strategy = strategy.strip().lower()
        self.strategy: RetrievalStrategy = strategy  # type: ignore[assignment]
        self.model_alias = model_alias
        self.candidate_k = self._positive_int(candidate_k, "candidate_k")
        self.rrf_k = self._positive_int(rrf_k, "rrf_k")

        if reranker_alias is not None and (
            not isinstance(reranker_alias, str) or not reranker_alias.strip()
        ):
            raise ValueError("reranker_alias must be a non-empty string or None")
        if not isinstance(reranker_device, str) or not reranker_device.strip():
            raise ValueError("reranker_device must be a non-empty string")
        self.reranker_alias = reranker_alias.strip() if reranker_alias else None

        if dense_retriever is None:
            from retrieval.chroma_retriever import DenseChromaRetriever

            dense_retriever = DenseChromaRetriever(model_alias=model_alias)
        self.dense_retriever = dense_retriever

        if strategy == "dense":
            self.base_retriever = dense_retriever
        else:
            from retrieval.hybrid_retriever import HybridRetriever

            self.base_retriever = HybridRetriever(
                model_alias=model_alias,
                dense_retriever=dense_retriever,
                documents=documents,
                candidate_k=self.candidate_k,
                rrf_k=self.rrf_k,
            )

        if reranker is not None:
            self.reranker = reranker
            if self.reranker_alias is None:
                self.reranker_alias = getattr(reranker, "model_alias", "injected")
        elif self.reranker_alias is not None:
            from reranking.cross_encoder import CrossEncoderReranker

            self.reranker = CrossEncoderReranker(
                model_alias=self.reranker_alias,
                device=reranker_device,
                batch_size=reranker_batch_size,
            )
        else:
            self.reranker = None

    @staticmethod
    def _positive_int(value: int, name: str) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
        return value

    def query_with_stats(
        self,
        query: str,
        *,
        top_k: int = 10,
        paper_id: str | None = None,
        candidate_k: int | None = None,
    ) -> dict[str, Any]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string")
        top_k = self._positive_int(top_k, "top_k")
        candidates = self.candidate_k if candidate_k is None else self._positive_int(
            candidate_k, "candidate_k"
        )
        if paper_id is not None and (
            not isinstance(paper_id, str) or not paper_id.strip()
        ):
            raise ValueError("paper_id must be a non-empty string when supplied")

        started = time.perf_counter()
        reranking_enabled = self.reranker is not None
        pool_size = max(candidates, top_k) if reranking_enabled else candidates
        retrieval_top_k = pool_size if reranking_enabled else top_k

        if self.strategy == "hybrid":
            base_response = self.base_retriever.query_with_stats(
                query,
                top_k=retrieval_top_k,
                paper_id=paper_id,
                candidate_k=pool_size,
            )
        else:
            base_response = self.base_retriever.query_with_stats(
                query,
                top_k=retrieval_top_k,
                paper_id=paper_id,
            )

        candidates_found = [dict(item) for item in base_response.get("results", [])]
        retrieval_wall_seconds = time.perf_counter() - started
        reranking_seconds = 0.0
        reranker_diagnostics: dict[str, Any] = {
            "documents_scored": 0,
            "passage_windows_scored": 0,
        }

        if self.reranker is not None and candidates_found:
            rerank_started = time.perf_counter()
            results, reranker_diagnostics = self.reranker.rerank(
                query, candidates_found, top_k=top_k
            )
            reranking_seconds = time.perf_counter() - rerank_started
            results = [dict(item) for item in results]
        else:
            results = candidates_found[:top_k]
            for rank, item in enumerate(results, start=1):
                item["rank"] = rank

        base_timing = base_response.get("timing", {})
        if not isinstance(base_timing, dict):
            base_timing = {}
        base_diagnostics = base_response.get("diagnostics", {})
        if not isinstance(base_diagnostics, dict):
            base_diagnostics = {}

        return {
            "results": results,
            "timing": {
                **base_timing,
                "base_wall_seconds": round(retrieval_wall_seconds, 6),
                "reranking_seconds": round(reranking_seconds, 6),
                "total_seconds": round(time.perf_counter() - started, 6),
            },
            "diagnostics": {
                "strategy": self.strategy,
                "reranking_enabled": reranking_enabled,
                "reranker_alias": self.reranker_alias,
                "candidate_pool_k": pool_size if reranking_enabled else None,
                "candidate_count": len(candidates_found),
                "returned_count": len(results),
                "base": base_diagnostics,
                "reranker": reranker_diagnostics,
            },
        }

    def query(
        self,
        query: str,
        *,
        top_k: int = 10,
        paper_id: str | None = None,
        candidate_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return ranked parent chunks, preserving source/citation metadata."""
        return self.query_with_stats(
            query,
            top_k=top_k,
            paper_id=paper_id,
            candidate_k=candidate_k,
        )["results"]
