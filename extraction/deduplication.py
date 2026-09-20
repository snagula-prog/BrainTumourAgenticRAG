"""
Deduplication and paper registry management.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from config.settings import settings


REGISTRY_FILE = (
    Path(settings.registry_dir)
    / "papers.json"
)


def compute_file_hash(
    file_path: Path,
) -> str:
    """Compute SHA-256 hash of a PDF."""

    sha256 = hashlib.sha256()

    with open(
        file_path,
        "rb",
    ) as file:

        for chunk in iter(
            lambda: file.read(8192),
            b"",
        ):
            sha256.update(chunk)

    return sha256.hexdigest()


def load_registry() -> dict:
    """Load the persistent registry."""

    if not REGISTRY_FILE.exists():
        return {}

    with open(
        REGISTRY_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def save_registry(
    registry: dict,
) -> None:

    REGISTRY_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        REGISTRY_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            registry,
            file,
            indent=2,
            ensure_ascii=False,
        )


def find_by_hash(
    registry: dict,
    file_hash: str,
) -> str | None:

    for paper_id, record in (
        registry.items()
    ):
        if (
            record.get("file_hash")
            == file_hash
        ):
            return paper_id

    return None


def next_paper_id(
    registry: dict,
) -> str:

    existing_numbers = [
        int(
            paper_id.split("_")[1]
        )
        for paper_id in registry.keys()
        if (
            paper_id.startswith(
                "paper_"
            )
            and
            paper_id.split("_")[1].isdigit()
        )
    ]

    next_number = (
        max(
            existing_numbers,
            default=0,
        )
        + 1
    )

    return (
        f"paper_{next_number:03d}"
    )


def register_new_paper(
    file_path: Path,
    file_hash: str,
) -> dict:

    registry = load_registry()

    paper_id = next_paper_id(
        registry
    )

    record = {
        "paper_id": paper_id,
        "filename": file_path.name,
        "file_hash": file_hash,
        "ingestion_date": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "status": "hashed",
        "title": None,
        "authors": None,
        "year": None,
        "num_pages": None,
        "num_chunks": None,
    }

    registry[paper_id] = record

    save_registry(
        registry
    )

    return record