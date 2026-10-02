
from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CHUNKS_DIR = ROOT / "storage" / "chunks"
EMBEDDINGS_DIR = ROOT / "storage" / "embeddings"
OUTPUT_DIR = ROOT / "storage" / "evaluation" / "retrieval_metrics"

OUTPUT_PATH = OUTPUT_DIR / "chunk_catalog.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_catalog() -> dict:
    catalog = []
    errors = []
    papers = []
    model_configs = []

    for chunk_path in sorted(CHUNKS_DIR.glob("paper_*_chunks.json")):
        paper_id = chunk_path.name.removesuffix("_chunks.json")
        manifest_path = (
            EMBEDDINGS_DIR / f"{paper_id}_embeddings.json"
        )

        if not manifest_path.is_file():
            errors.append(f"{paper_id}: missing embedding manifest")
            continue

        vector_path = None
        try:
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )

            vector_path = EMBEDDINGS_DIR / manifest["vectors"]["file"]

            if not vector_path.is_file():
                raise FileNotFoundError(
                    f"Missing vectors: {vector_path}"
                )

            source_hash = sha256_file(chunk_path)
            if source_hash != manifest["source"]["sha256"]:
                raise ValueError("Chunk source hash mismatch; embeddings stale")

            vector_hash = sha256_file(vector_path)
            if vector_hash != manifest["vectors"]["sha256"]:
                raise ValueError("Vector file checksum mismatch")

            with np.load(vector_path, allow_pickle=False) as data:
                vectors = data["embeddings"]
                segment_ids = data["segment_ids"].tolist()
                vector_chunk_ids = data["chunk_ids"].tolist()

            records = manifest["records"]
            dimension = manifest["embedding"]["dimension"]

            if vectors.shape != (len(records), dimension):
                raise ValueError(
                    f"Vector shape mismatch: {vectors.shape}"
                )

            if not np.isfinite(vectors).all():
                raise ValueError("Non-finite vector values")

            if segment_ids != [
                record["segment_id"] for record in records
            ]:
                raise ValueError("Segment ID ordering mismatch")

            if vector_chunk_ids != [
                record["chunk_id"] for record in records
            ]:
                raise ValueError("Chunk ID ordering mismatch")

            model_configs.append(manifest["embedding"])

            chunks = json.loads(
                chunk_path.read_text(encoding="utf-8")
            )
            if not isinstance(chunks, list):
                raise ValueError("Chunk file must contain a list")

            eligible = set(
                manifest["embedding"]["included_chunk_types"]
            )
            segments_by_chunk = Counter(
                record["chunk_id"] for record in records
            )

            paper_catalog = []
            seen_ids = set()

            for source_index, chunk in enumerate(chunks):
                if chunk.get("chunk_type") not in eligible:
                    continue

                chunk_id = chunk.get("chunk_id")
                if not isinstance(chunk_id, str) or not chunk_id:
                    raise ValueError(
                        f"Invalid chunk ID at index {source_index}"
                    )

                if chunk_id in seen_ids:
                    raise ValueError(f"Duplicate chunk ID: {chunk_id}")
                seen_ids.add(chunk_id)

                if chunk_id not in segments_by_chunk:
                    raise ValueError(
                        f"Chunk has no embedded segments: {chunk_id}"
                    )

                item = {
                    "chunk_id": chunk_id,
                    "paper_id": paper_id,
                    "chunk_type": chunk["chunk_type"],
                    "order_index": chunk.get("order_index"),
                    "text": chunk["text"],
                    "metadata": chunk.get("metadata", {}),
                    "segment_count": segments_by_chunk[chunk_id],
                }
                paper_catalog.append(item)

            embedded_ids = set(segments_by_chunk)
            selected_ids = {item["chunk_id"] for item in paper_catalog}

            if embedded_ids != selected_ids:
                extra = sorted(embedded_ids - selected_ids)
                if extra:
                    raise ValueError(
                        f"Vectors reference unknown chunks: {extra[:5]}"
                    )

            catalog.extend(paper_catalog)
            papers.append({
                "paper_id": paper_id,
                "chunk_count": len(chunks),
                "eligible_chunk_count": len(paper_catalog),
                "segment_count": len(records),
                "dimension": dimension,
            })

            print(
                f"{paper_id}: {len(paper_catalog)} chunks, "
                f"{len(records)} vector segments"
            )

        except Exception as exc:
            errors.append(f"{paper_id}: {type(exc).__name__}: {exc}")

    if errors:
        raise RuntimeError(
            "Benchmark preparation failed:\n- "
            + "\n- ".join(errors)
        )

    if len(model_configs) != len(papers):
        raise RuntimeError("Some papers are missing model configuration")

    if model_configs:
        reference = json.dumps(model_configs[0], sort_keys=True)
        if any(
            json.dumps(config, sort_keys=True) != reference
            for config in model_configs[1:]
        ):
            raise RuntimeError(
                "Papers use different embedding configurations. "
                "Regenerate them consistently before evaluation."
            )

    result = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Manual retrieval benchmark preparation",
        "embedding_config": model_configs[0] if model_configs else None,
        "summary": {
            "paper_count": len(papers),
            "chunk_count": len(catalog),
            "segment_count": sum(
                paper["segment_count"] for paper in papers
            ),
            "chunk_types": dict(Counter(
                item["chunk_type"] for item in catalog
            )),
        },
        "papers": papers,
        "chunks": catalog,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return result


if __name__ == "__main__":
    result = prepare_catalog()
    print("\nBenchmark corpus prepared.")
    print(json.dumps(result["summary"], indent=2))
    print(f"\nSaved to: {OUTPUT_PATH}")