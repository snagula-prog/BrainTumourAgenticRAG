from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from config.settings import settings
from embeddings.model_registry import get_model_profile, model_embeddings_dir

ROOT = Path(__file__).resolve().parents[1]


def project_path(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_file_name(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or Path(value).name != value:
        raise ValueError(f"Invalid {label}: {value!r}")
    return value


def migrate_legacy_bge(*, move: bool = False, dry_run: bool = False) -> dict[str, Any]:
    """Copy/move legacy flat BGE-small artifacts into the model subdirectory.

    The source files are deleted only when --move is requested and all copied
    targets have been verified by SHA-256. Any conflicting destination fails
    closed and leaves original files untouched.
    """
    profile = get_model_profile("bge")
    root = project_path(settings.embeddings_dir)
    destination = model_embeddings_dir(root, profile)
    manifests = sorted(root.glob("*_embeddings.json"))
    vectors = sorted(root.glob("*_embeddings.npz"))

    if not manifests and not vectors:
        return {
            "status": "nothing_to_migrate",
            "source_dir": str(root),
            "destination_dir": str(destination),
            "papers": 0,
            "files": 0,
        }

    manifest_by_vector: dict[str, Path] = {}
    items: list[tuple[Path, Path]] = []
    for manifest_path in manifests:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read {manifest_path.name}: {exc}") from exc
        if not isinstance(manifest, dict):
            raise ValueError(f"{manifest_path.name}: manifest must be a JSON object")
        paper_id = manifest.get("paper_id")
        if not isinstance(paper_id, str) or not paper_id.strip() or Path(paper_id).name != paper_id:
            raise ValueError(f"{manifest_path.name}: invalid paper_id")
        embedding = manifest.get("embedding")
        if not isinstance(embedding, dict) or embedding.get("model_name") != profile.model_name:
            raise ValueError(
                f"{manifest_path.name}: only {profile.model_name!r} artifacts can be "
                "migrated from the legacy root directory"
            )
        alias = manifest.get("model_alias")
        if alias not in (None, "bge", "bge-small", "bge-small-en-v1.5"):
            raise ValueError(f"{manifest_path.name}: unexpected model_alias {alias!r}")
        vectors_info = manifest.get("vectors")
        if not isinstance(vectors_info, dict):
            raise ValueError(f"{manifest_path.name}: missing vectors metadata")
        vector_name = _safe_file_name(vectors_info.get("file"), label="vectors.file")
        vector_path = root / vector_name
        if not vector_path.is_file():
            raise FileNotFoundError(f"Vector file missing for {manifest_path.name}: {vector_path}")
        expected_hash = vectors_info.get("sha256")
        actual_hash = sha256_file(vector_path)
        if expected_hash and expected_hash != actual_hash:
            raise ValueError(f"{vector_path.name}: checksum does not match manifest")
        manifest_by_vector[vector_name] = manifest_path
        items.extend(((manifest_path, destination / manifest_path.name), (vector_path, destination / vector_name)))

    orphan_vectors = [path for path in vectors if path.name not in manifest_by_vector]
    if orphan_vectors:
        raise ValueError(
            "Found legacy vector files without matching manifests: "
            + ", ".join(path.name for path in orphan_vectors)
        )

    # Preflight every destination before writing anything.
    for source, target in items:
        if target.exists() and sha256_file(source) != sha256_file(target):
            raise FileExistsError(
                f"Refusing to overwrite different existing artifact: {target}"
            )

    if dry_run:
        return {
            "status": "dry_run",
            "source_dir": str(root),
            "destination_dir": str(destination),
            "papers": len(manifests),
            "files": len(items),
            "move_requested": move,
            "files_to_migrate": [target.name for _, target in items],
        }

    destination.mkdir(parents=True, exist_ok=True)
    for source, target in items:
        if not target.exists():
            shutil.copy2(source, target)
        if sha256_file(source) != sha256_file(target):
            raise IOError(f"Post-copy checksum verification failed: {target}")

    if move:
        for source, target in items:
            # All destinations have been validated before the first deletion.
            if sha256_file(source) != sha256_file(target):
                raise IOError(f"Refusing to remove source after a checksum mismatch: {source}")
        for source, _ in items:
            source.unlink()

    return {
        "status": "moved" if move else "copied",
        "source_dir": str(root),
        "destination_dir": str(destination),
        "papers": len(manifests),
        "files": len(items),
        "source_files_removed": len(items) if move else 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Migrate legacy flat BGE-small embedding artifacts into their model folder"
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--move", action="store_true", help="Remove legacy originals after verified copy")
    action.add_argument("--copy", action="store_true", help="Copy while retaining legacy originals (default)")
    parser.add_argument("--dry-run", action="store_true", help="Show the migration plan without writing")
    args = parser.parse_args()
    try:
        result = migrate_legacy_bge(move=args.move, dry_run=args.dry_run)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(f"[MIGRATE] FAILED: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
