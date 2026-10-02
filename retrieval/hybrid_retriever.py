from __future__ import annotations

import math
import re
import time
import unicodedata
from collections import Counter, defaultdict
from typing import Any, Iterable


_TOKEN_PATTERN = re.compile(r"(?u)\w+(?:[./:+-]\w+)*")
_SPLIT_COMPOUND = re.compile(r"[._/+:-]+")


def scientific_tokenize(text: str) -> list[str]:
    """Tokenize scientific text while retaining codes and compound terms.

    Full forms such as ``svm-rfe`` and ``t1-weighted`` are retained, and their
    components are also emitted so exact compound and component queries match.
    No stemming is applied because identifiers and model names are meaningful.
    """
    if not isinstance(text, str):
        raise TypeError("BM25 input must be a string")
    normalized = unicodedata.normalize("NFKC", text).casefold()
    tokens: list[str] = []
    for match in _TOKEN_PATTERN.finditer(normalized):
        token = match.group(0).strip("._/+:-")
        if not token:
            continue
        tokens.append(token)
        parts = [part for part in _SPLIT_COMPOUND.split(token) if part]
        if len(parts) > 1:
            tokens.extend(parts)
    return tokens


class BM25Index:
    """Dependency-free BM25 index over original parent chunks."""

    def __init__(
        self,
        documents: Iterable[dict[str, Any]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not math.isfinite(k1) or k1 <= 0:
            raise ValueError("BM25 k1 must be a positive finite number")
        if not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("BM25 b must be between 0 and 1")
        self.k1 = float(k1)
        self.b = float(b)
        self.documents = [dict(item) for item in documents]
        self.tokens: list[list[str]] = []
        self.term_frequencies: list[Counter[str]] = []
        self.document_frequencies: Counter[str] = Counter()
        self.lengths: list[int] = []
        self._paper_indexes: dict[str, list[int]] = defaultdict(list)

        seen: set[tuple[str, str]] = set()
        for index, item in enumerate(self.documents):
            paper_id = item.get("paper_id")
            chunk_id = item.get("chunk_id")
            text = item.get("text")
            if not isinstance(paper_id, str) or not paper_id.strip():
                raise ValueError(f"BM25 document {index} has invalid paper_id")
            if not isinstance(chunk_id, str) or not chunk_id.strip():
                raise ValueError(f"BM25 document {index} has invalid chunk_id")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"BM25 document {index} has empty text")
            key = (paper_id, chunk_id)
            if key in seen:
                raise ValueError(f"Duplicate BM25 parent chunk: {key!r}")
            seen.add(key)

            tokens = scientific_tokenize(text)
            frequencies = Counter(tokens)
            self.tokens.append(tokens)
            self.term_frequencies.append(frequencies)
            self.lengths.append(len(tokens))
            self.document_frequencies.update(frequencies.keys())
            self._paper_indexes[paper_id].append(index)

        self.document_count = len(self.documents)
        self.average_length = (
            sum(self.lengths) / self.document_count if self.document_count else 0.0
        )
        self.idf = {
            term: math.log1p(
                (self.document_count - frequency + 0.5) / (frequency + 0.5)
            )
            for term, frequency in self.document_frequencies.items()
        }

    def search(
        self,
        query: str,
        *,
        top_k: int = 50,
        paper_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("BM25 query must be a non-empty string")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("BM25 top_k must be a positive integer")
        if paper_id is not None and (not isinstance(paper_id, str) or not paper_id.strip()):
            raise ValueError("paper_id must be a non-empty string when supplied")

        query_tokens = scientific_tokenize(query)
        if not query_tokens:
            return []
        doc_indexes = (
            self._paper_indexes.get(paper_id, []) if paper_id is not None
            else range(self.document_count)
        )
        query_frequency = Counter(query_tokens)
        ranked: list[tuple[float, int]] = []
        average_length = self.average_length or 1.0

        for index in doc_indexes:
            frequencies = self.term_frequencies[index]
            length = self.lengths[index]
            normalization = self.k1 * (
                1.0 - self.b + self.b * length / average_length
            )
            score = 0.0
            for term, query_count in query_frequency.items():
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                numerator = frequency * (self.k1 + 1.0)
                denominator = frequency + normalization
                score += query_count * self.idf.get(term, 0.0) * numerator / denominator
            if score > 0.0 and math.isfinite(score):
                ranked.append((score, index))

        ranked.sort(
            key=lambda pair: (
                -pair[0],
                self.documents[pair[1]]["paper_id"],
                self.documents[pair[1]]["chunk_id"],
            )
        )
        output: list[dict[str, Any]] = []
        for score, index in ranked[:top_k]:
            item = dict(self.documents[index])
            item["bm25_score"] = float(score)
            output.append(item)
        return output


class HybridRetriever:
    """Fuse dense ChromaDB and sparse BM25 parent-chunk rankings with RRF.

    Dense search remains segment-based in ChromaDB and is aggregated to parent
    chunks by DenseChromaRetriever. BM25 operates once per original parent chunk.
    RRF fuses on (paper_id, chunk_id), avoiding extra votes for long segmented
    passages. The BM25 index is built in memory from the indexed corpus on init.
    """

    def __init__(
        self,
        model_alias: str = "bge",
        *,
        dense_retriever: Any | None = None,
        documents: Iterable[dict[str, Any]] | None = None,
        candidate_k: int = 50,
        rrf_k: int = 60,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
    ) -> None:
        if not isinstance(candidate_k, int) or isinstance(candidate_k, bool) or candidate_k < 1:
            raise ValueError("candidate_k must be a positive integer")
        if not isinstance(rrf_k, int) or isinstance(rrf_k, bool) or rrf_k < 1:
            raise ValueError("rrf_k must be a positive integer")
        if dense_retriever is None:
            from retrieval.chroma_retriever import DenseChromaRetriever
            dense_retriever = DenseChromaRetriever(model_alias=model_alias)
        self.dense_retriever = dense_retriever
        self.profile = getattr(dense_retriever, "profile", None)
        self.model_alias = getattr(self.profile, "alias", model_alias)
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self.documents = list(documents) if documents is not None else self._load_indexed_parent_chunks()
        self.bm25 = BM25Index(self.documents, k1=bm25_k1, b=bm25_b)

    def _load_indexed_parent_chunks(self) -> list[dict[str, Any]]:
        collection = getattr(self.dense_retriever, "collection", None)
        resolver = getattr(self.dense_retriever, "_resolve_parent_chunk", None)
        if collection is None or not callable(resolver):
            raise ValueError(
                "Provide documents=... or a DenseChromaRetriever with a collection"
            )

        unique: dict[tuple[str, str], dict[str, Any]] = {}
        offset = 0
        page_size = 1000
        while True:
            batch = collection.get(limit=page_size, offset=offset, include=["metadatas"])
            ids = batch.get("ids", []) or []
            metadatas = batch.get("metadatas", []) or []
            if len(ids) != len(metadatas):
                raise RuntimeError("Chroma get returned misaligned metadata")
            for segment_id, metadata in zip(ids, metadatas, strict=True):
                if not isinstance(metadata, dict):
                    raise RuntimeError(f"Missing metadata for segment {segment_id!r}")
                paper_id, chunk_id = metadata.get("paper_id"), metadata.get("chunk_id")
                if not isinstance(paper_id, str) or not isinstance(chunk_id, str):
                    raise RuntimeError(f"Segment {segment_id!r} lacks parent identifiers")
                key = (paper_id, chunk_id)
                source_hash = metadata.get("source_sha256")
                previous = unique.get(key)
                if previous is not None:
                    if previous.get("source_sha256") != source_hash:
                        raise RuntimeError(f"Conflicting source hashes for parent chunk {key!r}")
                    continue
                unique[key] = {
                    "paper_id": paper_id,
                    "chunk_id": chunk_id,
                    "source_sha256": source_hash,
                }
            offset += len(ids)
            if len(ids) < page_size:
                break

        documents: list[dict[str, Any]] = []
        for key in sorted(unique):
            document = resolver(dict(unique[key]))
            if not isinstance(document.get("text"), str) or not document["text"].strip():
                raise RuntimeError(f"Parent chunk {key!r} has no source text")
            documents.append(document)
        if not documents:
            raise RuntimeError("Cannot build BM25: indexed corpus contains no parent chunks")
        return documents

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
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        candidates = self.candidate_k if candidate_k is None else candidate_k
        if not isinstance(candidates, int) or isinstance(candidates, bool) or candidates < 1:
            raise ValueError("candidate_k must be a positive integer")
        if paper_id is not None and (not isinstance(paper_id, str) or not paper_id.strip()):
            raise ValueError("paper_id must be a non-empty string when supplied")

        dense_response = self.dense_retriever.query_with_stats(
            query, top_k=candidates, paper_id=paper_id
        )
        dense_results = dense_response["results"]

        bm25_started = time.perf_counter()
        sparse_results = self.bm25.search(query, top_k=candidates, paper_id=paper_id)
        bm25_seconds = time.perf_counter() - bm25_started

        fusion_started = time.perf_counter()
        fused: dict[tuple[str, str], dict[str, Any]] = {}
        for rank, item in enumerate(dense_results, start=1):
            key = (item["paper_id"], item["chunk_id"])
            result = dict(item)
            dense_similarity = result.get("score")
            result["dense_similarity"] = dense_similarity
            result["dense_rank"] = rank
            result["bm25_rank"] = None
            result["bm25_score"] = None
            result["retrieval_channels"] = ["dense"]
            result["rrf_score"] = 1.0 / (self.rrf_k + rank)
            fused[key] = result

        for rank, item in enumerate(sparse_results, start=1):
            key = (item["paper_id"], item["chunk_id"])
            result = fused.get(key)
            if result is None:
                result = dict(item)
                result["dense_similarity"] = None
                result["dense_rank"] = None
                result["bm25_score"] = None
                result["retrieval_channels"] = []
                result["rrf_score"] = 0.0
                fused[key] = result
            result["bm25_rank"] = rank
            result["bm25_score"] = item["bm25_score"]
            result["retrieval_channels"].append("bm25")
            result["rrf_score"] += 1.0 / (self.rrf_k + rank)

        ranked = sorted(
            fused.values(),
            key=lambda item: (
                -item["rrf_score"],
                item["dense_rank"] if item["dense_rank"] is not None else math.inf,
                item["bm25_rank"] if item["bm25_rank"] is not None else math.inf,
                item["paper_id"],
                item["chunk_id"],
            ),
        )[:top_k]
        for item in ranked:
            item["score"] = item["rrf_score"]

        fusion_seconds = time.perf_counter() - fusion_started
        dense_timing = dense_response.get("timing", {})
        dense_retrieval_seconds = float(dense_timing.get("retrieval_seconds", 0.0))
        return {
            "results": ranked,
            "timing": {
                "query_encoding_seconds": float(dense_timing.get("query_encoding_seconds", 0.0)),
                "dense_retrieval_seconds": round(dense_retrieval_seconds, 6),
                "bm25_seconds": round(bm25_seconds, 6),
                "fusion_seconds": round(fusion_seconds, 6),
                "retrieval_seconds": round(dense_retrieval_seconds + bm25_seconds + fusion_seconds, 6),
            },
            "diagnostics": {
                "candidate_k": candidates,
                "rrf_k": self.rrf_k,
                "dense_candidate_count": len(dense_results),
                "bm25_candidate_count": len(sparse_results),
                "fused_candidate_count": len(fused),
                "unique_chunks_in_bm25_corpus": len(self.documents),
                "dense_diagnostics": dense_response.get("diagnostics", {}),
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
        return self.query_with_stats(
            query, top_k=top_k, paper_id=paper_id, candidate_k=candidate_k
        )["results"]
