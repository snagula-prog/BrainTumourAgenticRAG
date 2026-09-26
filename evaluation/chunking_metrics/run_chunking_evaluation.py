from __future__ import annotations

import argparse
import json
from pathlib import Path

from .chunking_metrics import ChunkingEvaluator


ROOT = Path(__file__).resolve().parents[2]

CLEANED_DIR = ROOT / "storage" / "cleaned"
CHUNKS_DIR = ROOT / "storage" / "chunks"
OUTPUT_DIR = (
    ROOT
    / "storage"
    / "evaluation"
    / "chunking_metrics"
)


def evaluate_paper(
    paper_id: str,
    max_words: int = 300,
) -> dict:

    cleaned_path = CLEANED_DIR / f"{paper_id}_clean.json"

    chunks_path = (
        CHUNKS_DIR
        / f"{paper_id}_chunks.json"
    )

    if not cleaned_path.exists():
        raise FileNotFoundError(
            f"Missing cleaned document: {cleaned_path}"
        )

    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Missing chunks file: {chunks_path}"
        )

    canonical = json.loads(
        cleaned_path.read_text(
            encoding="utf-8"
        )
    )

    chunks = json.loads(
        chunks_path.read_text(
            encoding="utf-8"
        )
    )

    evaluator = ChunkingEvaluator()

    result = evaluator.evaluate(
        paper_id=paper_id,
        canonical=canonical,
        chunks=chunks,
        max_words=max_words,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / f"{paper_id}.json"
    )

    output_path.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        f"\nPaper: {paper_id}"
    )

    print(
        f"Integrity score: "
        f"{result['integrity_score']}"
    )

    print(
        f"Status: "
        f"{result['status']}"
    )

    print(
        "\nMetrics:"
    )

    for name, value in result[
        "metrics"
    ].items():
        print(
            f"  {name}: {value}"
        )

    print(
        "\nIssues:"
    )

    if result["issues"]:
        for issue in result[
            "issues"
        ]:
            print(
                f"  - {issue}"
            )
    else:
        print(
            "  none"
        )

    print(
        f"\nSaved: {output_path}"
    )

    return result


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate chunking integrity"
        )
    )

    parser.add_argument(
        "--paper-id",
    )

    parser.add_argument(
        "--max-words",
        type=int,
        default=300,
    )

    args = parser.parse_args()

    if args.paper_id:
        evaluate_paper(
            args.paper_id,
            args.max_words,
        )
        return

    paper_ids = sorted(
        path.stem
        for path in CLEANED_DIR.glob(
            "*.json"
        )
    )

    for paper_id in paper_ids:
        evaluate_paper(
            paper_id,
            args.max_words,
        )


if __name__ == "__main__":
    main()