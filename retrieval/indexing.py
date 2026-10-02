from __future__ import annotations

import argparse
import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from config.settings import settings
from embeddings.model_registry import (
    accepted_model_aliases,
    get_model_profile,
    model_embeddings_dir,
)


ROOT = Path(__file__).resolve().parents[1]
BATCH_SIZE = 100


def project_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_embedding_artifacts(model_alias: str) -> dict[str, Any]:
    profile = get_model_profile(model_alias)
    directory = model_embeddings_dir(project_path(settings.embeddings_dir), profile)
    manifests = sorted(directory.glob("*_embeddings.json"))
    if not manifests:
        raise FileNotFoundError(
            f"No manifests for model {model_alias!r} found in {directory}. "
            f"Run `python -m embeddings.pipeline --model {model_alias}` first."
        )

    segments: list[dict[str, Any]] = []
    paper_ids: list[str] = []
    dimension: int | None = None
    resolved_revision: str | None = None
    manifest_fingerprints = []
    seen_segment_ids: set[str] = set()

    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("schema_version") != 2:
            raise ValueError(f"{manifest_path.name}: expected schema_version 2")
        paper_id = manifest.get("paper_id")
        if not isinstance(paper_id, str) or not paper_id:
            raise ValueError(f"{manifest_path.name}: invalid paper_id")
        chunks_dir = project_path(settings.chunks_dir)
        source_chunks = chunks_dir / f"{paper_id}_chunks.json"
        if not source_chunks.is_file():
            raise FileNotFoundError(f"Current chunk source is missing: {source_chunks}")
        source_info = manifest.get("source", {})
        expected_source_hash = source_info.get("sha256") if isinstance(source_info, dict) else None
        if not isinstance(expected_source_hash, str) or sha256_file(source_chunks) != expected_source_hash:
            raise ValueError(
                f"{paper_id}: chunk source no longer matches the embeddings manifest. "
                "Re-embed this paper before indexing."
            )
        declared_alias = manifest.get("model_alias")
        if declared_alias is not None and declared_alias != profile.alias:
            raise ValueError(
                f"{manifest_path.name}: model_alias {declared_alias!r} does not "
                f"match requested {profile.alias!r}"
            )
        embedding = manifest.get("embedding")
        selection = manifest.get("selection")
        vectors_info = manifest.get("vectors")
        records = manifest.get("records")
        if not all(isinstance(value, dict) for value in (embedding, selection, vectors_info)):
            raise ValueError(f"{manifest_path.name}: invalid manifest structure")
        if not isinstance(records, list):
            raise ValueError(f"{manifest_path.name}: records must be a list")
        if embedding.get("model_name") != profile.model_name:
            raise ValueError(
                f"{manifest_path.name}: model {embedding.get('model_name')!r} does not "
                f"match profile {profile.model_name!r}"
            )
        current_dimension = embedding.get("dimension")
        if not isinstance(current_dimension, int) or isinstance(current_dimension, bool) or current_dimension < 1:
            raise ValueError(f"{manifest_path.name}: invalid embedding dimension")
        if profile.expected_dimension is not None and current_dimension != profile.expected_dimension:
            raise ValueError(
                f"{manifest_path.name}: expected dimension {profile.expected_dimension}, "
                f"got {current_dimension}"
            )
        if dimension is None:
            dimension = current_dimension
            resolved_revision = embedding.get("resolved_revision")
        elif current_dimension != dimension:
            raise ValueError("Embedding manifests have inconsistent dimensions")
        elif embedding.get("resolved_revision") != resolved_revision:
            raise ValueError("Embedding manifests have inconsistent model revisions")

        vector_name = vectors_info.get("file")
        if not isinstance(vector_name, str) or Path(vector_name).name != vector_name:
            raise ValueError(f"{manifest_path.name}: invalid vectors.file")
        vector_path = (manifest_path.parent / vector_name).resolve()
        if vector_path.parent != manifest_path.parent.resolve() or not vector_path.is_file():
            raise FileNotFoundError(f"Invalid or missing vector file: {vector_path}")
        actual_hash = sha256_file(vector_path)
        if actual_hash != vectors_info.get("sha256"):
            raise ValueError(f"{manifest_path.name}: vector checksum mismatch")

        with np.load(vector_path, allow_pickle=False) as stored:
            vectors = stored["embeddings"]
            segment_ids = stored["segment_ids"].tolist()
            chunk_ids = stored["chunk_ids"].tolist()
        expected_count = selection.get("embedded_segment_count")
        if not isinstance(expected_count, int) or expected_count < 0:
            raise ValueError(f"{manifest_path.name}: invalid embedded_segment_count")
        if vectors.shape != (expected_count, current_dimension):
            raise ValueError(
                f"{manifest_path.name}: vector shape {vectors.shape} does not match "
                f"expected {(expected_count, current_dimension)}"
            )
        if len(records) != expected_count:
            raise ValueError(f"{manifest_path.name}: record count mismatch")
        if segment_ids != [record.get("segment_id") for record in records]:
            raise ValueError(f"{manifest_path.name}: segment IDs mismatch")
        if chunk_ids != [record.get("chunk_id") for record in records]:
            raise ValueError(f"{manifest_path.name}: chunk IDs mismatch")
        if not np.isfinite(vectors).all():
            raise ValueError(f"{manifest_path.name}: non-finite embeddings")

        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(f"{manifest_path.name}: records[{index}] is not an object")
            segment_id = record.get("segment_id")
            chunk_id = record.get("chunk_id")
            if not isinstance(segment_id, str) or not segment_id:
                raise ValueError(f"{manifest_path.name}: invalid segment_id at {index}")
            if segment_id in seen_segment_ids:
                raise ValueError(f"Duplicate segment_id across manifests: {segment_id}")
            seen_segment_ids.add(segment_id)
            if record.get("paper_id") != paper_id:
                raise ValueError(f"{segment_id}: inconsistent paper_id")
            if not isinstance(chunk_id, str) or not chunk_id:
                raise ValueError(f"{segment_id}: invalid chunk_id")
            text = record.get("text")
            if not isinstance(text, str):
                raise ValueError(f"{segment_id}: missing text")
            metadata = record.get("metadata", {})
            if not isinstance(metadata, dict):
                raise ValueError(f"{segment_id}: metadata must be an object")
            segments.append({
                "record": record,
                "vector": vectors[index].astype(np.float32, copy=False),
                "source_sha256": expected_source_hash,
            })
        paper_ids.append(paper_id)
        manifest_fingerprints.append({
            "paper_id": paper_id,
            "manifest_sha256": sha256_file(manifest_path),
            "vectors_sha256": actual_hash,
        })

    if dimension is None or not segments:
        raise ValueError(f"No segments found for model {model_alias!r}")
    fingerprint = sha256_json(manifest_fingerprints)
    return {
        "profile": profile,
        "directory": directory,
        "paper_ids": sorted(paper_ids),
        "dimension": dimension,
        "resolved_revision": resolved_revision,
        "segments": segments,
        "fingerprint": fingerprint,
        "manifest_fingerprints": manifest_fingerprints,
    }


def _chroma_metadata(record: dict[str, Any], model_alias: str, source_sha256: str) -> dict[str, str | int | float | bool]:
    """Flatten source metadata to Chroma-supported scalar values."""
    metadata: dict[str, str | int | float | bool] = {
        "model_alias": model_alias,
        "paper_id": record["paper_id"],
        "chunk_id": record["chunk_id"],
        "segment_id": record["segment_id"],
        "source_sha256": source_sha256,
        "source_metadata_json": json.dumps(
            record.get("metadata", {}), sort_keys=True, ensure_ascii=False
        ),
    }
    scalar_keys = (
        "chunk_type", "order_index", "source_index", "segment_index",
        "segment_count", "token_start", "token_end", "token_count", "text_sha256",
    )
    for key in scalar_keys:
        value = record.get(key)
        if isinstance(value, (str, int, float, bool)):
            metadata[key] = value
    return metadata


def index_model(model_alias: str, *, batch_size: int = BATCH_SIZE) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    artifacts = load_embedding_artifacts(model_alias)
    profile = artifacts["profile"]
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError("ChromaDB is required. Install the project's chromadb dependency.") from exc

    persist_path = project_path(settings.chroma_persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_path))
    metadata = {
        "hnsw:space": "cosine",
        "model_alias": profile.alias,
        "model_name": profile.model_name,
        "resolved_revision": artifacts["resolved_revision"] or "",
        "dimension": artifacts["dimension"],
    }

    created = False
    try:
        collection = client.get_collection(profile.collection_name)
    except Exception:
        collection = client.create_collection(
            name=profile.collection_name,
            metadata=metadata,
        )
        created = True

    existing_metadata = getattr(collection, "metadata", None) or {}
    if existing_metadata.get("hnsw:space") != "cosine":
        raise ValueError(
            f"Existing collection {profile.collection_name!r} is not cosine-indexed. "
            "Use a new collection name or remove this derived collection before rebuilding."
        )
    if existing_metadata.get("model_alias") != profile.alias:
        raise ValueError("Existing Chroma collection belongs to a different or unknown model alias")
    if existing_metadata.get("model_name") != profile.model_name:
        raise ValueError("Existing Chroma collection belongs to a different or unknown embedding model")
    if existing_metadata.get("dimension") != artifacts["dimension"]:
        raise ValueError("Existing Chroma collection dimension differs from current embeddings")
    expected_revision = artifacts["resolved_revision"] or ""
    if existing_metadata.get("resolved_revision", "") != expected_revision:
        raise ValueError(
            "Existing Chroma collection model revision differs from current embeddings. "
            "Use a new collection name or remove this derived collection before rebuilding."
        )

    started = time.perf_counter()
    expected_ids = [item["record"]["segment_id"] for item in artifacts["segments"]]
    expected_set = set(expected_ids)
    upserted = 0
    for offset in range(0, len(artifacts["segments"]), batch_size):
        batch = artifacts["segments"][offset:offset + batch_size]
        collection.upsert(
            ids=[item["record"]["segment_id"] for item in batch],
            embeddings=[item["vector"].tolist() for item in batch],
            documents=[item["record"]["text"] for item in batch],
            metadatas=[
                _chroma_metadata(item["record"], profile.alias, item["source_sha256"])
                for item in batch
            ],
        )
        upserted += len(batch)

    existing = collection.get(include=["metadatas"])
    stale_ids = [segment_id for segment_id in existing.get("ids", []) if segment_id not in expected_set]
    for offset in range(0, len(stale_ids), batch_size):
        collection.delete(ids=stale_ids[offset:offset + batch_size])

    actual_count = int(collection.count())
    if actual_count != len(expected_ids):
        raise RuntimeError(
            f"Indexed segment count mismatch: expected {len(expected_ids)}, got {actual_count}"
        )

    return {
        "model_alias": profile.alias,
        "model_name": profile.model_name,
        "collection_name": profile.collection_name,
        "persist_dir": str(persist_path),
        "paper_ids": artifacts["paper_ids"],
        "paper_count": len(artifacts["paper_ids"]),
        "segment_count": actual_count,
        "chunk_count": len({
            (item["record"]["paper_id"], item["record"]["chunk_id"])
            for item in artifacts["segments"]
        }),
        "upserted_segments": upserted,
        "deleted_stale_segments": len(stale_ids),
        "corpus_fingerprint": artifacts["fingerprint"],
        "created_collection": created,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Index embedding artifacts in ChromaDB")
    parser.add_argument("--model", choices=accepted_model_aliases(), default="bge")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    run_id = f"chroma_index_{args.model}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    started_at = utc_now()
    try:
        result = index_model(args.model, batch_size=args.batch_size)
        run_record = {
            "schema_version": 1,
            "run_id": run_id,
            "phase": "chroma_indexing",
            "status": "completed",
            "started_at": started_at,
            "finished_at": utc_now(),
            **result,
        }
        print(f"[INDEX:{args.model}] {result}")
    except Exception as exc:
        run_record = {
            "schema_version": 1,
            "run_id": run_id,
            "phase": "chroma_indexing",
            "status": "failed",
            "started_at": started_at,
            "finished_at": utc_now(),
            "model_alias": args.model,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        print(f"[INDEX:{args.model}] FAILED: {type(exc).__name__}: {exc}")

    runs_dir = project_path(settings.evaluation_dir) / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    (runs_dir / f"{run_id}.json").write_text(
        json.dumps(run_record, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return 0 if run_record["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
