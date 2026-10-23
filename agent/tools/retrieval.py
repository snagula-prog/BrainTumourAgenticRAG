from __future__ import annotations

import json
from typing import Any

from retrieval.research_retriever import ResearchRetriever


class RetrievalTool:
    """Expose ResearchRetriever as a validated agent tool."""

    name = "retrieve_papers"
    description = (
        "Retrieve relevant passages from research papers. "
        "Supports dense or hybrid retrieval and optional reranking. "
        "Can restrict retrieval to a specific paper."
    )

    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The question or search query.",
            },
            "top_k": {
                "type": "integer",
                "minimum": 1,
                "description": "Number of ranked passages to return.",
            },
            "paper_id": {
                "type": "string",
                "minLength": 1,
                "description": "Optional paper ID to restrict retrieval.",
            },
            "candidate_k": {
                "type": "integer",
                "minimum": 1,
                "description": "Optional candidate pool size.",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(self, retriever: ResearchRetriever) -> None:
        self._retriever = retriever

    @staticmethod
    def _positive_int(value: Any, name: str) -> int:
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
        return value

    def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be an object")

        allowed = {"query", "top_k", "paper_id", "candidate_k"}
        unknown = set(arguments) - allowed
        if unknown:
            raise ValueError(
                f"Unexpected arguments: {', '.join(sorted(unknown))}"
            )

        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        top_k = self._positive_int(
            arguments.get("top_k", 10), "top_k"
        )

        paper_id = arguments.get("paper_id")
        if paper_id is not None:
            if not isinstance(paper_id, str) or not paper_id.strip():
                raise ValueError(
                    "paper_id must be a non-empty string when supplied"
                )
            paper_id = paper_id.strip()

        candidate_k = arguments.get("candidate_k")
        if candidate_k is not None:
            candidate_k = self._positive_int(
                candidate_k, "candidate_k"
            )

        response = self._retriever.query_with_stats(
            query=query.strip(),
            top_k=top_k,
            paper_id=paper_id,
            candidate_k=candidate_k,
        )

        if not isinstance(response, dict) or "results" not in response:
            raise RuntimeError("Retriever returned an invalid response")

        return {
            "query": query.strip(),
            "results": response["results"],
            "timing": response.get("timing", {}),
            "diagnostics": response.get("diagnostics", {}),
        }

    def get_indexed_segments(
        self, paper_id: str, *, page_size: int = 1000
    ) -> dict[str, Any]:
        """Exhaustively read stored Chroma text segments for one paper.

        Unlike ``execute()``, this method does not embed a query or rank by similarity.
        It reads the existing collection via a paper_id metadata filter and reports
        complete coverage only when the collection's known segment inventory and
        per-parent segment metadata agree with the scan.
        """
        if not isinstance(paper_id, str) or not paper_id.strip():
            raise ValueError("paper_id must be a non-empty string")
        paper_id = paper_id.strip()
        if type(page_size) is not int or page_size < 1:
            raise ValueError("page_size must be a positive integer")

        dense = getattr(self._retriever, "dense_retriever", None)
        collection = getattr(dense, "collection", None)
        if collection is None or not callable(getattr(collection, "get", None)):
            return {
                "paper_id": paper_id,
                "segments": [],
                "coverage": {
                    "status": "not_checked",
                    "expected_segment_count": None,
                    "segments_scanned": 0,
                    "unique_parent_chunks_scanned": 0,
                    "issues": ["Existing Chroma collection is unavailable through the retriever."],
                },
            }

        counts = getattr(dense, "_paper_segment_counts", None)
        expected_count = counts.get(paper_id) if isinstance(counts, dict) else None
        known_papers = getattr(dense, "paper_ids", None)
        if isinstance(known_papers, (list, tuple, set)) and paper_id not in known_papers:
            return {
                "paper_id": paper_id,
                "segments": [],
                "coverage": {
                    "status": "not_indexed",
                    "expected_segment_count": 0,
                    "segments_scanned": 0,
                    "unique_parent_chunks_scanned": 0,
                    "issues": [f"No indexed Chroma segments are registered for {paper_id!r}."],
                },
            }

        issues: list[str] = []
        segments: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        offset = 0
        while True:
            batch = collection.get(
                where={"paper_id": paper_id},
                limit=page_size,
                offset=offset,
                include=["documents", "metadatas"],
            )
            ids = batch.get("ids", []) or []
            documents = batch.get("documents", []) or []
            metadatas = batch.get("metadatas", []) or []
            if not (len(ids) == len(documents) == len(metadatas)):
                issues.append(
                    f"Chroma returned misaligned arrays at offset {offset} "
                    f"(ids={len(ids)}, documents={len(documents)}, metadatas={len(metadatas)})."
                )
                break
            if not ids:
                break

            for raw_id, text, metadata in zip(ids, documents, metadatas, strict=True):
                segment_id = str(raw_id)
                if segment_id in seen_ids:
                    issues.append(f"Duplicate Chroma segment ID encountered: {segment_id}.")
                    continue
                seen_ids.add(segment_id)
                if not isinstance(metadata, dict):
                    issues.append(f"Segment {segment_id} has missing/invalid metadata.")
                    continue
                if metadata.get("paper_id") != paper_id:
                    issues.append(f"Filtered scan returned segment {segment_id} for a different paper.")
                    continue
                chunk_id = metadata.get("chunk_id")
                segment_index = metadata.get("segment_index")
                segment_count = metadata.get("segment_count")
                source_hash = metadata.get("source_sha256")
                if not isinstance(chunk_id, str) or not chunk_id.strip():
                    issues.append(f"Segment {segment_id} has no chunk_id.")
                    continue
                if type(segment_index) is not int or type(segment_count) is not int or segment_count < 1:
                    issues.append(f"Segment {segment_id} lacks valid segment_index/segment_count metadata.")
                    continue
                if not isinstance(source_hash, str) or not source_hash.strip():
                    issues.append(f"Segment {segment_id} has no source_sha256 metadata.")
                    continue
                if not isinstance(text, str) or not text.strip():
                    issues.append(f"Segment {segment_id} has empty stored text.")
                    continue
                source_metadata: dict[str, Any] = {}
                raw_source_metadata = metadata.get("source_metadata_json")
                if isinstance(raw_source_metadata, str):
                    try:
                        decoded = json.loads(raw_source_metadata)
                        if isinstance(decoded, dict):
                            source_metadata = decoded
                    except json.JSONDecodeError:
                        issues.append(f"Segment {segment_id} has malformed source_metadata_json.")
                segments.append({
                    "segment_id": segment_id,
                    "paper_id": paper_id,
                    "chunk_id": chunk_id,
                    "segment_index": segment_index,
                    "segment_count": segment_count,
                    "source_sha256": source_hash,
                    "text": text,
                    "metadata": {
                        **metadata,
                        "source_metadata": source_metadata,
                    },
                })

            offset += len(ids)
            if len(ids) < page_size:
                break
            # A guard against a pathological adapter which ignores pagination.
            if offset > (expected_count if isinstance(expected_count, int) else 0) + page_size * 2:
                issues.append("Chroma pagination exceeded the expected inventory; scan aborted.")
                break

        if expected_count is None:
            issues.append("Retriever did not expose its expected per-paper segment count.")
        elif len(seen_ids) != expected_count:
            issues.append(
                f"Segment inventory mismatch: expected {expected_count}, scanned {len(seen_ids)} unique IDs."
            )

        groups: dict[str, list[dict[str, Any]]] = {}
        for segment in segments:
            groups.setdefault(segment["chunk_id"], []).append(segment)
        for chunk_id, group in groups.items():
            declared_counts = {item["segment_count"] for item in group}
            if len(declared_counts) != 1:
                issues.append(f"Parent chunk {chunk_id} has conflicting segment_count values.")
                continue
            declared_count = next(iter(declared_counts))
            indices = [item["segment_index"] for item in group]
            if len(indices) != len(set(indices)):
                issues.append(f"Parent chunk {chunk_id} has duplicate segment indices.")
            zero_based = list(range(declared_count))
            one_based = list(range(1, declared_count + 1))
            if len(group) != declared_count or sorted(indices) not in (zero_based, one_based):
                issues.append(
                    f"Parent chunk {chunk_id} is incomplete: declared {declared_count} segments, "
                    f"scanned {len(group)} with indices {sorted(indices)}."
                )
            if len({item["source_sha256"] for item in group}) != 1:
                issues.append(f"Parent chunk {chunk_id} has conflicting source hashes.")

        if not segments:
            issues.append("No valid text segments were returned for the paper.")
        # Duplicate issue messages add no value but order is useful for diagnostics.
        issues = list(dict.fromkeys(issues))
        complete = not issues and expected_count == len(seen_ids)
        return {
            "paper_id": paper_id,
            "segments": sorted(segments, key=lambda item: (
                item["chunk_id"], item["segment_index"], item["segment_id"]
            )),
            "coverage": {
                "status": "complete_indexed_coverage" if complete else "incomplete_indexed_coverage",
                "expected_segment_count": expected_count,
                "segments_scanned": len(seen_ids),
                "unique_parent_chunks_scanned": len(groups),
                "issues": issues,
            },
        }

