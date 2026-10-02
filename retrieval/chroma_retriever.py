from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from config.settings import settings
from embeddings.encoder import EmbeddingEncoder
from embeddings.model_registry import get_model_profile


ROOT = Path(__file__).resolve().parents[1]


class DenseChromaRetriever:
    """Dense ChromaDB retriever returning unique parent chunks.

    Segment-level cosine distances are converted to similarities and
    aggregated to the original chunk with max similarity. Searches expand
    adaptively until they contain top_k unique parent chunks or exhaust the
    selected corpus.
    """

    def __init__(
        self,
        model_alias: str = "bge",
        *,
        encoder: EmbeddingEncoder | None = None,
        client: Any | None = None,
        collection: Any | None = None,
        persist_dir: str | None = None,
        initial_overfetch: int = 4,
    ) -> None:
        if initial_overfetch < 1:
            raise ValueError("initial_overfetch must be at least 1")
        self.profile = get_model_profile(model_alias)
        self.encoder = encoder or EmbeddingEncoder(
            model_name=self.profile.model_name,
            device=settings.embedding_device,
            batch_size=settings.embedding_batch_size,
            normalize=settings.embedding_normalize,
            revision=self.profile.revision,
            query_instruction=self.profile.query_instruction,
        )
        if self.encoder.model_name != self.profile.model_name:
            raise ValueError(
                f"Encoder model {self.encoder.model_name!r} does not match "
                f"profile {self.profile.model_name!r}"
            )
        if client is None and collection is None:
            try:
                import chromadb
            except ImportError as exc:
                raise RuntimeError(
                    "ChromaDB is required. Install the project's chromadb dependency."
                ) from exc
            root = Path(persist_dir or settings.chroma_persist_dir)
            if not root.is_absolute():
                root = (ROOT / root).resolve()
            client = chromadb.PersistentClient(path=str(root))
        self.client = client
        if collection is None:
            if self.client is None:
                raise ValueError("Provide a ChromaDB client or collection")
            try:
                collection = self.client.get_collection(self.profile.collection_name)
            except Exception as exc:
                # Convert a missing collection into a useful next step while
                # preserving non-not-found Chroma errors in the message.
                raise RuntimeError(
                    f"Collection {self.profile.collection_name!r} is unavailable. "
                    f"Run `python -m retrieval.indexing --model {self.profile.alias}` first."
                ) from exc
        self.collection = collection
        self.initial_overfetch = initial_overfetch
        self._validate_collection()
        self._paper_segment_counts = self._read_paper_segment_counts()
        self._collection_count = int(self.collection.count())
        self._parent_chunk_cache: dict[str, dict[str, dict[str, Any]]] = {}
        self._parent_source_hashes: dict[str, str] = {}
        if self._collection_count < 1:
            raise RuntimeError(f"Chroma collection {self.profile.collection_name!r} is empty")

    @property
    def segment_count(self) -> int:
        return self._collection_count

    @property
    def paper_ids(self) -> list[str]:
        return sorted(self._paper_segment_counts)

    def _validate_collection(self) -> None:
        metadata = getattr(self.collection, "metadata", None) or {}
        space = metadata.get("hnsw:space")
        if space != "cosine":
            raise ValueError(
                f"Collection must use cosine distance; found {space!r}. "
                "Rebuild it with retrieval.indexing."
            )
        stored_alias = metadata.get("model_alias")
        stored_name = metadata.get("model_name")
        stored_dimension = metadata.get("dimension")
        if stored_alias != self.profile.alias:
            raise ValueError(
                f"Collection model_alias {stored_alias!r} does not match "
                f"requested profile {self.profile.alias!r}"
            )
        if stored_name != self.profile.model_name:
            raise ValueError(
                f"Collection model {stored_name!r} does not match "
                f"profile model {self.profile.model_name!r}"
            )
        if stored_dimension != self.encoder.dimension:
            raise ValueError(
                f"Collection dimension {stored_dimension!r} does not match "
                f"encoder dimension {self.encoder.dimension}"
            )
        expected_revision = getattr(self.encoder, "model_revision", None)
        stored_revision = metadata.get("resolved_revision")
        if expected_revision and stored_revision and expected_revision != stored_revision:
            raise ValueError(
                "Collection model revision does not match the loaded encoder: "
                f"{stored_revision!r} != {expected_revision!r}"
            )

    def _read_paper_segment_counts(self) -> Counter[str]:
        counts: Counter[str] = Counter()
        offset = 0
        # Read metadata in pages to avoid asking Chroma for all large payloads.
        page_size = 1000
        while True:
            batch = self.collection.get(
                limit=page_size,
                offset=offset,
                include=["metadatas"],
            )
            ids = batch.get("ids", [])
            metadatas = batch.get("metadatas", []) or []
            for metadata in metadatas:
                if isinstance(metadata, dict) and isinstance(metadata.get("paper_id"), str):
                    counts[metadata["paper_id"]] += 1
            offset += len(ids)
            if len(ids) < page_size:
                break
        return counts

    def _load_parent_chunks(self, paper_id: str) -> dict[str, dict[str, Any]]:
        if paper_id in self._parent_chunk_cache:
            return self._parent_chunk_cache[paper_id]
        if Path(paper_id).name != paper_id or not paper_id.strip():
            raise ValueError(f"Invalid paper_id in retrieved metadata: {paper_id!r}")
        chunk_dir = Path(settings.chunks_dir)
        if not chunk_dir.is_absolute():
            chunk_dir = (ROOT / chunk_dir).resolve()
        source = chunk_dir / f"{paper_id}_chunks.json"
        if not source.is_file():
            raise FileNotFoundError(
                f"Original chunk file is missing for {paper_id}: {source}"
            )
        try:
            raw = source.read_bytes()
            source_hash = hashlib.sha256(raw).hexdigest()
            chunks = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid chunk JSON for {paper_id}: {exc}") from exc
        if not isinstance(chunks, list):
            raise ValueError(f"Chunk file for {paper_id} must contain a JSON list")
        lookup: dict[str, dict[str, Any]] = {}
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            chunk_id = chunk.get("chunk_id")
            if isinstance(chunk_id, str) and chunk.get("paper_id") == paper_id:
                if chunk_id in lookup:
                    raise ValueError(f"Duplicate parent chunk ID {chunk_id!r} in {source}")
                lookup[chunk_id] = chunk
        self._parent_chunk_cache[paper_id] = lookup
        self._parent_source_hashes[paper_id] = source_hash
        return lookup

    def _resolve_parent_chunk(self, result: dict[str, Any]) -> dict[str, Any]:
        parent = self._load_parent_chunks(result["paper_id"]).get(result["chunk_id"])
        if parent is None:
            raise RuntimeError(
                f"Indexed chunk {result['chunk_id']!r} is missing from the current "
                "chunk source. Re-embed and re-index the affected paper."
            )
        expected_source_hash = result.get("source_sha256")
        actual_source_hash = self._parent_source_hashes.get(result["paper_id"])
        if expected_source_hash and actual_source_hash != expected_source_hash:
            raise RuntimeError(
                f"Chunk source for {result['paper_id']} changed after indexing. "
                "Re-embed and re-index the affected paper."
            )
        text = parent.get("text")
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError(f"Parent chunk {result['chunk_id']!r} has empty text")
        result["text"] = text
        result["metadata"] = parent.get("metadata", {}) if isinstance(parent.get("metadata", {}), dict) else {}
        result["chunk_type"] = parent.get("chunk_type", result.get("chunk_type"))
        result["order_index"] = parent.get("order_index", result.get("order_index"))
        return result

    @staticmethod
    def _decode_source_metadata(raw: Any) -> dict[str, Any]:
        if not isinstance(raw, str):
            return {}
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    def query_with_stats(
        self,
        query: str,
        *,
        top_k: int = 10,
        paper_id: str | None = None,
    ) -> dict[str, Any]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if paper_id is not None:
            if not isinstance(paper_id, str) or not paper_id.strip():
                raise ValueError("paper_id must be a non-empty string when supplied")
            available = self._paper_segment_counts.get(paper_id, 0)
            if available == 0:
                raise ValueError(f"No indexed segments found for paper_id {paper_id!r}")
        else:
            available = self._collection_count

        encode_started = time.perf_counter()
        query_vector = self.encoder.encode_queries([query])[0]
        encode_seconds = time.perf_counter() - encode_started
        if query_vector.shape != (self.encoder.dimension,) or not np.isfinite(query_vector).all():
            raise ValueError("Query encoder returned an invalid vector")
        norm = float(np.linalg.norm(query_vector))
        if not math.isfinite(norm) or norm == 0.0:
            raise ValueError("Query embedding has zero or invalid norm")

        requested = min(available, max(top_k, top_k * self.initial_overfetch))
        where = {"paper_id": paper_id} if paper_id is not None else None
        best: dict[tuple[str, str], dict[str, Any]] = {}
        total_returned = 0
        retrieval_started = time.perf_counter()

        while True:
            kwargs: dict[str, Any] = {
                "query_embeddings": [query_vector.astype(np.float32).tolist()],
                "n_results": requested,
                "include": ["metadatas", "documents", "distances"],
            }
            if where is not None:
                kwargs["where"] = where
            response = self.collection.query(**kwargs)
            ids = (response.get("ids") or [[]])[0] or []
            metadatas = (response.get("metadatas") or [[]])[0] or []
            documents = (response.get("documents") or [[]])[0] or []
            distances = (response.get("distances") or [[]])[0] or []
            total_returned = len(ids)
            if not (len(ids) == len(metadatas) == len(documents) == len(distances)):
                raise RuntimeError("Chroma query returned misaligned result arrays")

            for segment_id, metadata, document, distance in zip(
                ids, metadatas, documents, distances, strict=True
            ):
                if not isinstance(metadata, dict):
                    raise RuntimeError(f"Missing metadata for segment {segment_id!r}")
                result_paper = metadata.get("paper_id")
                chunk_id = metadata.get("chunk_id")
                if not isinstance(result_paper, str) or not isinstance(chunk_id, str):
                    raise RuntimeError(f"Segment {segment_id!r} lacks paper_id/chunk_id metadata")
                if paper_id is not None and result_paper != paper_id:
                    continue
                distance_value = float(distance)
                if not math.isfinite(distance_value):
                    raise RuntimeError(f"Invalid cosine distance for segment {segment_id!r}")
                similarity = max(-1.0, min(1.0, 1.0 - distance_value))
                key = (result_paper, chunk_id)
                previous = best.get(key)
                if previous is None or similarity > previous["score"]:
                    best[key] = {
                        "paper_id": result_paper,
                        "chunk_id": chunk_id,
                        "score": similarity,
                        "distance": distance_value,
                        "text": document or "",
                        "metadata": self._decode_source_metadata(
                            metadata.get("source_metadata_json")
                        ),
                        "chunk_type": metadata.get("chunk_type"),
                        "order_index": metadata.get("order_index"),
                        "match_segment_id": segment_id,
                        "source_sha256": metadata.get("source_sha256"),
                        "match_segment_index": metadata.get("segment_index"),
                        "segment_count": metadata.get("segment_count"),
                    }

            # Enough unique parent chunks are present. Since segment results
            # are ordered by distance, the leading unique parents are ranked.
            if len(best) >= top_k or requested >= available or total_returned < requested:
                break
            requested = min(available, max(requested + 1, requested * 2))

        ranked = sorted(
            best.values(),
            key=lambda item: (-item["score"], item["paper_id"], item["chunk_id"]),
        )[:top_k]
        ranked = [self._resolve_parent_chunk(item) for item in ranked]
        retrieval_seconds = time.perf_counter() - retrieval_started
        return {
            "results": ranked,
            "timing": {
                "query_encoding_seconds": round(encode_seconds, 6),
                "retrieval_seconds": round(retrieval_seconds, 6),
            },
            "diagnostics": {
                "requested_segments": requested,
                "returned_segments": total_returned,
                "unique_chunks_returned": len(best),
                "scope_segment_count": available,
            },
        }

    def query(
        self,
        query: str,
        *,
        top_k: int = 10,
        paper_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return ranked unique parent chunks (best segment similarity first)."""
        return self.query_with_stats(query, top_k=top_k, paper_id=paper_id)["results"]
