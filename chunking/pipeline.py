# chunking/pipeline.py

from __future__ import annotations

import json
from pathlib import Path

from config.settings import settings
from chunking.text_chunker import (
    ScientificTextChunker,
)


CHUNKS_DIR = Path(
    "storage/chunks"
)


def chunk_paper(
    *,
    cleaned_path: Path,
    output_path: Path,
) -> None:

    with cleaned_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        cleaned = json.load(
            file
        )

    paper_id = cleaned[
        "paper_id"
    ]

    chunker = ScientificTextChunker(
        embedding_model=(
            settings.embedding_model
        ),
        max_tokens=450,
        overlap_tokens=60,
    )

    chunks = chunker.chunk(
        cleaned
    )

    output = {
        "paper_id": paper_id,
        "filename": cleaned.get(
            "filename"
        ),
        "title": cleaned.get(
            "title"
        ),
        "authors": cleaned.get(
            "authors",
            [],
        ),
        "year": cleaned.get(
            "year"
        ),
        "abstract": cleaned.get(
            "abstract",
            "",
        ),
        "num_chunks": len(
            chunks
        ),
        "chunks": chunks,

        # Keep structured information available
        # to the future visualizer.
        "references": cleaned.get(
            "references",
            [],
        ),
        "docling_document": cleaned.get(
            "docling_document"
        ),
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"[CHUNKED] {paper_id} -> "
        f"{len(chunks)} chunks"
    )


def chunk_all_papers() -> None:

    CHUNKS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    cleaned_dir = Path(
        settings.cleaned_dir
    )

    files = sorted(
        cleaned_dir.glob(
            "paper_*_clean.json"
        )
    )

    if not files:

        print(
            "No cleaned papers found."
        )

        return

    print(
        f"[INFO] Found "
        f"{len(files)} cleaned paper(s)"
    )

    success = 0
    failed = 0

    for cleaned_path in files:

        try:

            with cleaned_path.open(
                "r",
                encoding="utf-8",
            ) as file:

                cleaned = json.load(
                    file
                )

            paper_id = cleaned[
                "paper_id"
            ]

            output_path = (
                CHUNKS_DIR
                / f"{paper_id}_chunks.json"
            )

            chunk_paper(
                cleaned_path=cleaned_path,
                output_path=output_path,
            )

            success += 1

        except Exception as exc:

            print(
                f"[FAILED] "
                f"{cleaned_path.name}: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            failed += 1

    print()
    print("=" * 60)
    print("CHUNKING COMPLETE")
    print("=" * 60)
    print(f"Successful : {success}")
    print(f"Failed     : {failed}")
    print(f"Output     : {CHUNKS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    chunk_all_papers()