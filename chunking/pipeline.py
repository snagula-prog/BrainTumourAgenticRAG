# chunking/pipeline.py
import argparse
import json
from pathlib import Path

from chunking.text_chunker import SectionAwareChunker


ROOT = Path(__file__).resolve().parents[1]
CLEANED_DIR = ROOT / "storage" / "cleaned"
CHUNKS_DIR = ROOT / "storage" / "chunks"


def chunk_paper(paper_id: str) -> Path:
    source = CLEANED_DIR / f"{paper_id}_clean.json"

    if not source.exists():
        raise FileNotFoundError(
            f"Cleaned artifact not found: {source}"
        )

    with source.open("r", encoding="utf-8") as f:
        cleaned_doc = json.load(f)

    chunker = SectionAwareChunker(max_words=300)
    chunks = chunker.chunk(
        paper_id,
        cleaned_doc,
    )

    CHUNKS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination = (
        CHUNKS_DIR
        / f"{paper_id}_chunks.json"
    )

    with destination.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            chunks,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"[CHUNK] Input : {source}")
    print(f"[CHUNK] Output: {destination}")
    print(f"[CHUNK] Total Chunks Generated: {len(chunks)}")

    # Simple distribution stats
    types = {}
    for c in chunks:
        types[c["chunk_type"]] = types.get(c["chunk_type"], 0) + 1
    print(f"[CHUNK] Distribution: {types}")

    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 8 section-aware chunking"
    )

    selection = parser.add_mutually_exclusive_group()

    selection.add_argument(
        "--paper-id",
        help="Chunk one paper by paper_id.",
    )

    selection.add_argument(
        "--file",
        type=Path,
        help="Chunk one specific cleaned JSON file.",
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # --paper-id
    # ---------------------------------------------------------

    if args.paper_id:
        chunk_paper(args.paper_id)
        return

    # ---------------------------------------------------------
    # --file
    # ---------------------------------------------------------

    if args.file:
        if not args.file.exists():
            parser.error(
                f"Cleaned artifact not found: {args.file}"
            )

        paper_id = args.file.stem

        # Ensure the supplied file corresponds to the expected
        # cleaned artifact location.
        cleaned_path = CLEANED_DIR / f"{paper_id}.json"

        if args.file.resolve() != cleaned_path.resolve():
            parser.error(
                "--file must point to a file in "
                "storage/cleaned/"
            )

        chunk_paper(paper_id)
        return

    # ---------------------------------------------------------
    # NO ARGUMENTS = ALL PAPERS
    # ---------------------------------------------------------

    cleaned_files = sorted(
        CLEANED_DIR.glob("paper_*.json")
    )

    if not cleaned_files:
        print(
            f"[CHUNK] No cleaned papers found in "
            f"{CLEANED_DIR}"
        )
        return

    print(
        f"[CHUNK] Processing "
        f"{len(cleaned_files)} papers..."
    )

    success = 0
    failed = 0

    for source in cleaned_files:
        paper_id = source.stem.removesuffix("_clean")

        try:
            chunk_paper(paper_id)
            success += 1

        except Exception as exc:
            print(
                f"[CHUNK] {paper_id} -> FAILED: "
                f"{exc}"
            )
            failed += 1

    print()
    print(
        f"[CHUNK] Complete: "
        f"{success} succeeded, "
        f"{failed} failed"
    )



if __name__ == "__main__":
    main()