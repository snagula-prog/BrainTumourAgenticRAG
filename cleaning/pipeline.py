# cleaning/pipeline.py
"""Phase 6 cleaning pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from cleaning.normalizer import CanonicalNormalizer
from evaluation.cleaning_metrics.cleaning_metrics import CleaningEvaluator


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DIR = ROOT / "storage" / "canonical"
CLEANED_DIR = ROOT / "storage" / "cleaned"
EVAL_DIR = ROOT / "storage" / "evaluation" / "cleaning_metrics"


def clean_paper(paper_id: str) -> Path:
    source = CANONICAL_DIR / f"{paper_id}.json"

    if not source.exists():
        raise FileNotFoundError(f"Canonical artifact not found: {source}")

    input_hash = sha256_file(source)
    destination = CLEANED_DIR / f"{paper_id}_clean.json"

    with source.open("r", encoding="utf-8") as f:
        canonical = json.load(f)

    normalizer = CanonicalNormalizer()
    cleaned, report = normalizer.clean(
        canonical,
        input_file_hash=input_hash,
    )

    CLEANED_DIR.mkdir(parents=True, exist_ok=True)

    with destination.open("w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)

    evaluator = CleaningEvaluator()
    eval_metrics = evaluator.evaluate(canonical, cleaned)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    eval_destination = EVAL_DIR / f"{paper_id}_cleaning.json"
    with eval_destination.open("w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, indent=2, ensure_ascii=False)

    print(f"[CLEAN] Input : {source}")
    print(f"[CLEAN] Output: {destination}")
    print(f"[CLEAN] Eval  : {eval_destination}")
    print(f"[CLEAN] Version: {report['version']}")
    print(f"[CLEAN] Stats: {report['stats']}")

    return destination


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 6 canonical cleaning"
    )

    selection = parser.add_mutually_exclusive_group()

    selection.add_argument(
        "--file",
        type=Path,
        help="Clean one specific canonical JSON file.",
    )

    selection.add_argument(
        "--paper-id",
        help="Clean one paper by paper_id.",
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # --file
    # ---------------------------------------------------------

    if args.file:
        input_path = args.file

        if not input_path.exists():
            parser.error(
                f"Canonical artifact not found: {input_path}"
            )

        paper_id = input_path.stem

        # clean_paper() uses storage/canonical as its source,
        # so make sure the supplied file belongs to that location.
        canonical_path = CANONICAL_DIR / f"{paper_id}.json"

        if input_path.resolve() != canonical_path.resolve():
            parser.error(
                "--file must point to a file in "
                "storage/canonical/"
            )

        clean_paper(paper_id)
        return

    # ---------------------------------------------------------
    # --paper-id
    # ---------------------------------------------------------

    if args.paper_id:
        clean_paper(args.paper_id)
        return

    # ---------------------------------------------------------
    # NO ARGUMENTS = ALL PAPERS
    # ---------------------------------------------------------

    canonical_files = sorted(
        CANONICAL_DIR.glob("paper_*.json")
    )

    if not canonical_files:
        print(
            f"[CLEAN] No papers found in "
            f"{CANONICAL_DIR}"
        )
        return

    print(
        f"[CLEAN] Processing "
        f"{len(canonical_files)} papers..."
    )

    success = 0
    failed = 0

    for source in canonical_files:
        paper_id = source.stem

        try:
            clean_paper(paper_id)
            success += 1

        except Exception as exc:
            print(
                f"[CLEAN] {paper_id} -> FAILED: "
                f"{exc}"
            )
            failed += 1

    print()
    print(
        f"[CLEAN] Complete: "
        f"{success} succeeded, "
        f"{failed} failed"
    )


if __name__ == "__main__":
    main()