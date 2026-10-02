from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from config.settings import settings
from embeddings.encoder import EmbeddingEncoder
from embeddings.model_registry import (
    accepted_model_aliases,
    get_model_profile,
    model_embeddings_dir,
)


ROOT = Path(__file__).resolve().parents[1]


def project_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


CHUNKS_DIR = project_path(settings.chunks_dir)
EMBEDDINGS_ROOT = project_path(settings.embeddings_dir)
RUNS_DIR = project_path(settings.evaluation_dir) / "runs"
INCLUDED_TYPES = set(settings.embedding_include_chunk_types)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary = Path(file.name)
            json.dump(data, file, indent=2, ensure_ascii=False)
            file.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def write_npz_atomic(
    path: Path,
    embeddings: np.ndarray,
    segment_ids: list[str],
    chunk_ids: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as file:
            np.savez_compressed(
                file,
                embeddings=embeddings.astype(np.float32, copy=False),
                segment_ids=np.asarray(segment_ids, dtype=str),
                chunk_ids=np.asarray(chunk_ids, dtype=str),
            )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def get_model_config(encoder: EmbeddingEncoder, overlap_tokens: int) -> dict[str, Any]:
    # Keep the legacy BGE fingerprint fields stable so existing BGE artifacts
    # remain current and are not needlessly regenerated.
    config = {
        "model_name": encoder.model_name,
        "requested_revision": encoder.requested_revision,
        "resolved_revision": encoder.model_revision,
        "dimension": encoder.dimension,
        "max_seq_length": encoder.max_seq_length,
        "batch_size": encoder.batch_size,
        "device": encoder.device,
        "normalize": encoder.normalize,
        "windowing": {
            "strategy": "overlapping_token_windows",
            "overlap_tokens": overlap_tokens,
        },
        "included_chunk_types": sorted(INCLUDED_TYPES),
    }
    # Existing BGE-small and MiniLM manifests omit this query-only field.
    # Preserve both baselines' fingerprints; record it for new profiles.
    if encoder.model_name not in {
        "BAAI/bge-small-en-v1.5",
        "sentence-transformers/all-MiniLM-L6-v2",
    }:
        config["query_instruction"] = encoder.query_instruction
    return config


def get_fingerprint(source_hash: str, model_config: dict[str, Any]) -> str:
    payload = {"source_sha256": source_hash, "model_config": model_config}
    encoded = json.dumps(
        payload, sort_keys=True, ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_chunks(chunks: list[Any], paper_id: str) -> None:
    if not isinstance(chunks, list):
        raise ValueError("Chunk file must contain a JSON list")
    seen_ids: set[str] = set()
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            raise ValueError(f"Chunk {index} is not a JSON object")
        chunk_id = chunk.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError(f"Chunk {index} has no valid chunk_id")
        if chunk_id in seen_ids:
            raise ValueError(f"Duplicate chunk_id: {chunk_id}")
        seen_ids.add(chunk_id)
        if chunk.get("paper_id") != paper_id:
            raise ValueError(
                f"{chunk_id}: expected paper_id {paper_id!r}, "
                f"got {chunk.get('paper_id')!r}"
            )


def is_current(manifest_path: Path, vector_path: Path, fingerprint: str) -> bool:
    if not manifest_path.is_file() or not vector_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("input_fingerprint") != fingerprint:
            return False
        if manifest.get("vectors", {}).get("sha256") != sha256_file(vector_path):
            return False
        records = manifest["records"]
        expected = manifest["selection"]["embedded_segment_count"]
        dimension = manifest["embedding"]["dimension"]
        with np.load(vector_path, allow_pickle=False) as stored:
            vectors = stored["embeddings"]
            segment_ids = stored["segment_ids"].tolist()
            chunk_ids = stored["chunk_ids"].tolist()
        if vectors.shape != (expected, dimension) or not np.isfinite(vectors).all():
            return False
        if segment_ids != [record["segment_id"] for record in records]:
            return False
        if chunk_ids != [record["chunk_id"] for record in records]:
            return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def embed_paper(
    paper_id: str,
    encoder: EmbeddingEncoder,
    output_dir: Path,
    *,
    model_alias: str,
    force: bool = False,
    overlap_tokens: int = 64,
) -> dict[str, Any]:
    if Path(paper_id).name != paper_id or not paper_id.strip():
        raise ValueError(f"Invalid paper_id: {paper_id!r}")
    source = CHUNKS_DIR / f"{paper_id}_chunks.json"
    if not source.is_file():
        raise FileNotFoundError(f"Chunk file not found: {source}")

    vector_path = output_dir / f"{paper_id}_embeddings.npz"
    manifest_path = output_dir / f"{paper_id}_embeddings.json"
    started = time.perf_counter()
    source_hash = sha256_file(source)
    model_config = get_model_config(encoder, overlap_tokens)
    fingerprint = get_fingerprint(source_hash, model_config)

    if not force and is_current(manifest_path, vector_path, fingerprint):
        print(f"[EMBED:{model_alias}] {paper_id}: unchanged, skipped")
        return {
            "paper_id": paper_id,
            "status": "skipped",
            "embedded_count": 0,
            "embedded_chunk_count": 0,
            "reason": "outputs are current",
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }

    chunks = json.loads(source.read_text(encoding="utf-8"))
    validate_chunks(chunks, paper_id)
    selected: list[tuple[int, dict[str, Any]]] = []
    excluded_counts: Counter[str] = Counter()
    for source_index, chunk in enumerate(chunks):
        chunk_type = chunk.get("chunk_type", "unknown")
        if chunk_type not in INCLUDED_TYPES:
            excluded_counts[chunk_type] += 1
            continue
        text = chunk.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{chunk['chunk_id']}: included chunk has empty text")
        if not isinstance(chunk.get("metadata", {}), dict):
            raise ValueError(f"{chunk['chunk_id']}: metadata must be an object")
        selected.append((source_index, chunk))
    if not selected:
        raise ValueError(f"{paper_id}: no eligible chunks found")

    windows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    oversized_chunk_count = 0
    for source_index, chunk in selected:
        chunk_windows = encoder.split_passage(
            chunk["text"], overlap_tokens=overlap_tokens
        )
        if len(chunk_windows) > 1:
            oversized_chunk_count += 1
        for segment_index, window in enumerate(chunk_windows):
            segment_id = f"{chunk['chunk_id']}::segment_{segment_index:04d}"
            windows.append(window)
            records.append({
                "segment_id": segment_id,
                "chunk_id": chunk["chunk_id"],
                "paper_id": paper_id,
                "chunk_type": chunk["chunk_type"],
                "order_index": chunk.get("order_index"),
                "source_index": source_index,
                "segment_index": segment_index,
                "segment_count": len(chunk_windows),
                "token_start": window["token_start"],
                "token_end": window["token_end"],
                "token_count": window["token_count"],
                "text": window["text"],
                "metadata": chunk.get("metadata", {}),
                "text_sha256": sha256_text(window["text"]),
            })

    vectors = encoder.encode_token_windows(windows)
    if len(vectors) != len(records):
        raise RuntimeError("Vector and segment counts do not match")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_npz_atomic(
        vector_path,
        vectors,
        [record["segment_id"] for record in records],
        [record["chunk_id"] for record in records],
    )
    manifest = {
        "schema_version": 2,
        "model_alias": model_alias,
        "paper_id": paper_id,
        "created_at": utc_now(),
        "input_fingerprint": fingerprint,
        "source": {
            "file": str(source.relative_to(ROOT)),
            "sha256": source_hash,
            "chunk_count": len(chunks),
        },
        "embedding": model_config,
        "selection": {
            "included_chunk_types": sorted(INCLUDED_TYPES),
            "selected_chunk_count": len(selected),
            "embedded_segment_count": len(records),
            "oversized_chunk_count": oversized_chunk_count,
            "excluded_counts": dict(excluded_counts),
        },
        "vectors": {
            "file": vector_path.name,
            "sha256": sha256_file(vector_path),
            "dtype": "float32",
            "shape": list(vectors.shape),
            "normalized": encoder.normalize,
        },
        "records": records,
        "execution": {
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        },
    }
    write_json_atomic(manifest_path, manifest)
    print(
        f"[EMBED:{model_alias}] {paper_id}: {len(selected)} chunks -> "
        f"{len(records)} vectors ({encoder.dimension} dimensions)"
    )
    return {
        "paper_id": paper_id,
        "status": "embedded",
        "embedded_count": len(records),
        "embedded_chunk_count": len(selected),
        "oversized_chunk_count": oversized_chunk_count,
        "vector_file": str(vector_path.relative_to(ROOT)),
        "manifest_file": str(manifest_path.relative_to(ROOT)),
        "elapsed_seconds": manifest["execution"]["elapsed_seconds"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate local research-paper embeddings")
    parser.add_argument(
        "--model",
        choices=accepted_model_aliases(),
        default="bge",
        help="Embedding model profile (default: bge)",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--paper-id")
    selection.add_argument("--file", type=Path)
    parser.add_argument("--force", action="store_true", help="Regenerate current outputs")
    parser.add_argument("--batch-size", type=int, help="Override the model profile batch size")
    parser.add_argument(
        "--overlap-tokens",
        type=int,
        default=settings.embedding_overlap_tokens,
        help="Overlap between token windows (default: 64)",
    )
    args = parser.parse_args()

    profile = get_model_profile(args.model)
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    output_dir = model_embeddings_dir(EMBEDDINGS_ROOT, profile)
    if args.file:
        source = args.file.resolve()
        if source.parent != CHUNKS_DIR:
            parser.error("--file must be inside storage/chunks/")
        if not source.name.endswith("_chunks.json"):
            parser.error("--file must end with _chunks.json")
        paper_ids = [source.name.removesuffix("_chunks.json")]
    elif args.paper_id:
        paper_ids = [args.paper_id]
    else:
        paper_ids = sorted(path.name.removesuffix("_chunks.json") for path in CHUNKS_DIR.glob("*_chunks.json"))
    if not paper_ids:
        parser.error(f"No chunk files found in {CHUNKS_DIR}")

    encoder = EmbeddingEncoder(
        model_name=profile.model_name,
        device=settings.embedding_device,
        batch_size=args.batch_size or profile.batch_size or settings.embedding_batch_size,
        normalize=settings.embedding_normalize,
        revision=profile.revision,
        query_instruction=profile.query_instruction,
    )
    if profile.expected_dimension is not None and encoder.dimension != profile.expected_dimension:
        raise RuntimeError(
            f"{profile.alias}: expected {profile.expected_dimension}-dimensional vectors, "
            f"but loaded model returned {encoder.dimension}"
        )

    run_id = (
        f"embedding_{profile.alias}_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_"
        + uuid.uuid4().hex[:8]
    )
    started_at = utc_now()
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for paper_id in paper_ids:
        try:
            results.append(embed_paper(
                paper_id,
                encoder,
                output_dir,
                model_alias=profile.alias,
                force=args.force,
                overlap_tokens=args.overlap_tokens,
            ))
        except Exception as exc:
            print(f"[EMBED:{profile.alias}] {paper_id}: FAILED: {exc}")
            errors.append({
                "paper_id": paper_id,
                "error_type": type(exc).__name__,
                "message": str(exc),
            })
            results.append({"paper_id": paper_id, "status": "failed"})

    run_record = {
        "schema_version": 2,
        "run_id": run_id,
        "phase": "embeddings",
        "model_alias": profile.alias,
        "started_at": started_at,
        "finished_at": utc_now(),
        "paper_ids": paper_ids,
        "config": get_model_config(encoder, args.overlap_tokens),
        "results": results,
        "metrics": {
            "papers_total": len(paper_ids),
            "papers_embedded": sum(result["status"] == "embedded" for result in results),
            "papers_skipped": sum(result["status"] == "skipped" for result in results),
            "papers_failed": len(errors),
            "vectors_generated": sum(result.get("embedded_count", 0) for result in results),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        },
        "errors": errors,
    }
    write_json_atomic(RUNS_DIR / f"{run_id}.json", run_record)
    print(f"\n[EMBED:{profile.alias}] Artifacts: {output_dir}")
    print(f"[EMBED:{profile.alias}] Run record: {RUNS_DIR / (run_id + '.json')}")
    print(f"[EMBED:{profile.alias}] Summary: {run_record['metrics']}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
