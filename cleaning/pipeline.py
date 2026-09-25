"""Phase 6 cleaning pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import hashlib

from cleaning.normalizer import CanonicalNormalizer


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DIR = ROOT / "storage" / "canonical"
CLEANED_DIR = ROOT / "storage" / "cleaned"


def clean_paper(paper_id: str) -> Path:
    source = CANONICAL_DIR / f"{paper_id}.json"
    
    input_hash = sha256_file(source)
    
    destination = CLEANED_DIR / f"{paper_id}.json"

    if not source.exists():
        raise FileNotFoundError(f"Canonical artifact not found: {source}")

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

    print(f"[CLEAN] Input : {source}")
    print(f"[CLEAN] Output: {destination}")
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
        description="Phase 6 canonical text cleaning"
    )
    parser.add_argument("--paper-id", required=True)
    args = parser.parse_args()

    clean_paper(args.paper_id)


if __name__ == "__main__":
    main()
