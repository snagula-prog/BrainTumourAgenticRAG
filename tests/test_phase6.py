# tests/test_phase6.py
import hashlib
import json
from pathlib import Path


CANONICAL_DIR = Path("storage/canonical")
CLEANED_DIR = Path("storage/cleaned")
EVAL_DIR = Path("storage/evaluation/cleaning_metrics")

PAPERS = ["paper_001", "paper_002"]

EXACT_STRUCTURAL_FIELDS = [
    "sections",
    "tables",
    "figures",
    "formulas",
    "references",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_paper(paper_id: str):
    canonical_path = CANONICAL_DIR / f"{paper_id}.json"
    cleaned_path = CLEANED_DIR / f"{paper_id}.json"
    eval_path = EVAL_DIR / f"{paper_id}_cleaning.json"

    assert canonical_path.exists(), f"Missing canonical file: {canonical_path}"
    assert cleaned_path.exists(), f"Missing cleaned file: {cleaned_path}"
    assert eval_path.exists(), f"Missing evaluation metric file: {eval_path}"

    canonical = load_json(canonical_path)
    cleaned = load_json(cleaned_path)
    eval_metrics = load_json(eval_path)

    print(f"\n[{paper_id}]")

    # ---------------------------------------------------------
    # 1. Exact structural entity preservation
    # ---------------------------------------------------------
    for field in EXACT_STRUCTURAL_FIELDS:
        canonical_count = len(canonical.get(field, []))
        cleaned_count = len(cleaned.get(field, []))

        print(
            f"{field:12}: "
            f"{canonical_count} -> {cleaned_count}"
        )

        assert canonical_count == cleaned_count, (
            f"{paper_id}: {field} count changed unexpectedly"
        )

    # ---------------------------------------------------------
    # 2. Paragraph retention accounting for noise removal
    # ---------------------------------------------------------
    report = cleaned.get("cleaning")
    assert isinstance(report, dict), f"{paper_id}: missing cleaning report"
    assert report.get("version") == "phase6-normalizer-v1"

    stats = report.get("stats", {})
    removed_empty = stats.get("removed_empty_items", 0)
    duplicates_removed = stats.get("duplicate_paragraphs_removed", 0)

    canonical_paras = len(canonical.get("paragraphs", []))
    cleaned_paras = len(cleaned.get("paragraphs", []))
    expected_paras = canonical_paras - removed_empty - duplicates_removed

    print(f"paragraphs  : {canonical_paras} -> {cleaned_paras} (expected: {expected_paras})")
    assert cleaned_paras == expected_paras, (
        f"{paper_id}: paragraph mismatch. Canonical={canonical_paras}, "
        f"Cleaned={cleaned_paras}, Expected={expected_paras}"
    )

    # ---------------------------------------------------------
    # 3. Input hash verification
    # ---------------------------------------------------------
    expected_hash = sha256_file(canonical_path)
    recorded_hash = report.get("input_file_hash")

    print(f"input hash  : {recorded_hash}")
    assert recorded_hash == expected_hash, f"{paper_id}: input hash mismatch"

    # ---------------------------------------------------------
    # 4. Numeric safety
    # ---------------------------------------------------------
    numeric_failures = stats.get("numeric_guard_failures", 0)
    print(f"numeric guard failures: {numeric_failures}")
    assert numeric_failures == 0, f"{paper_id}: numeric guard failures detected"

    # ---------------------------------------------------------
    # 5. Evaluation metrics integrity check
    # ---------------------------------------------------------
    assert eval_metrics.get("evaluation_type") == "cleaning_integrity"
    assert "metrics" in eval_metrics
    assert eval_metrics["metrics"].get("numeric_token_retention") == 1.0

    print("PASS")


def main():
    for paper_id in PAPERS:
        validate_paper(paper_id)

    print("\nPhase 6 validation: PASS")


if __name__ == "__main__":
    main()