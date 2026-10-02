from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import chromadb

from config.settings import settings

ROOT = Path(__file__).resolve().parents[1]
RETIRED = {
    "gte-modernbert": {
        "directory": "gte-modernbert-base",
        "collection": "research_papers_gte_modernbert_base",
    },
    "qwen3-0.6b": {
        "directory": "qwen3-embedding-0.6b",
        "collection": "research_papers_qwen3_embedding_0_6b",
    },
}


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def inspect_retired() -> dict[str, Any]:
    embeddings_root = resolve_project_path(settings.embeddings_dir)
    persist_dir = resolve_project_path(settings.chroma_persist_dir)
    client = chromadb.PersistentClient(path=str(persist_dir))
    collections = {item.name for item in client.list_collections()}
    items = []
    for alias, config in RETIRED.items():
        artifact_dir = (embeddings_root / config["directory"]).resolve()
        if embeddings_root.resolve() not in artifact_dir.parents:
            raise ValueError(f"Unsafe artifact path: {artifact_dir}")
        items.append({
            "alias": alias,
            "embedding_directory": str(artifact_dir),
            "embedding_directory_exists": artifact_dir.is_dir(),
            "collection": config["collection"],
            "collection_exists": config["collection"] in collections,
        })
    return {"embeddings_root": str(embeddings_root), "chroma_dir": str(persist_dir), "items": items}


def retire(*, delete: bool = False) -> dict[str, Any]:
    report = inspect_retired()
    if not delete:
        report["status"] = "dry_run"
        return report
    embeddings_root = resolve_project_path(settings.embeddings_dir)
    persist_dir = resolve_project_path(settings.chroma_persist_dir)
    client = chromadb.PersistentClient(path=str(persist_dir))
    deleted = []
    for item in report["items"]:
        if item["collection_exists"]:
            client.delete_collection(item["collection"])
            deleted.append({"type": "collection", "name": item["collection"]})
        artifact_dir = Path(item["embedding_directory"]).resolve()
        if item["embedding_directory_exists"]:
            if embeddings_root.resolve() not in artifact_dir.parents:
                raise ValueError(f"Refusing to delete outside embeddings root: {artifact_dir}")
            shutil.rmtree(artifact_dir)
            deleted.append({"type": "directory", "name": str(artifact_dir)})
    report["status"] = "deleted"
    report["deleted"] = deleted
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect or remove retired GTE/Qwen retrieval artifacts")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true", help="Inspect only (default)")
    group.add_argument("--delete", action="store_true", help="Delete only the named GTE/Qwen collections and embedding directories")
    args = parser.parse_args()
    report = retire(delete=args.delete)
    import json
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
