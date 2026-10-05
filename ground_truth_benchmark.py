from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluation.extraction_metrics.extraction_benchmark import ExtractionBenchmark


SUPPORTED_GT_KEYS = {
    "paper_id",
    "filename",
    "metadata",
    "abstract",
    "figures",
    "tables",
    "counts",
    "source_exceptions",
    # Backward-compatible full GT keys:
    "sections",
    "paragraphs",
    "references",
    "formulas",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def validate_gt(gt: dict[str, Any], path: Path) -> None:
    unknown = set(gt) - SUPPORTED_GT_KEYS
    if unknown:
        raise ValueError(f"{path}: unsupported top-level keys: {sorted(unknown)}")

    metadata = gt.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"{path}: metadata must be an object")

    for key in ("figures", "tables"):
        if key in gt and not isinstance(gt[key], list):
            raise ValueError(f"{path}: {key} must be a list")

    counts = gt.get("counts")
    if counts is not None:
        if not isinstance(counts, dict):
            raise ValueError(f"{path}: counts must be an object")
        for key in ("numbered_figures", "numbered_tables"):
            if key in counts:
                value = counts[key]
                if not isinstance(value, int) or value < 0:
                    raise ValueError(f"{path}: counts.{key} must be a non-negative integer")

        for key, records_key in (("numbered_figures", "figures"), ("numbered_tables", "tables")):
            if key in counts and records_key in gt and counts[key] != len(gt[records_key]):
                raise ValueError(
                    f"{path}: counts.{key}={counts[key]} but len({records_key})={len(gt[records_key])}"
                )

    if "source_exceptions" in gt and not isinstance(gt["source_exceptions"], list):
        raise ValueError(f"{path}: source_exceptions must be a list")


def find_canonical(root: Path, paper_id: str) -> Path:
    preferred = [root / "canonical" / f"{paper_id}.json", root / "canonical_json" / f"{paper_id}.json"]
    for path in preferred:
        if path.is_file():
            return path

    candidates = []
    for path in root.rglob(f"{paper_id}.json"):
        rel = path.relative_to(root).as_posix().lower()
        if "/ground_truth/" in f"/{rel}" or rel.startswith("ground_truth/"):
            continue
        candidates.append(path)

    if len(candidates) == 1:
        return candidates[0]

    ranked = sorted(
        candidates,
        key=lambda p: (
            0 if "canonical" in p.as_posix().lower() else 1,
            0 if "output" in p.as_posix().lower() else 1,
            len(p.parts),
        ),
    )
    if ranked:
        best = ranked[0]
        tied = [p for p in ranked if (
            ("canonical" in p.as_posix().lower()) == ("canonical" in best.as_posix().lower())
            and ("output" in p.as_posix().lower()) == ("output" in best.as_posix().lower())
            and len(p.parts) == len(best.parts)
        )]
        if len(tied) == 1:
            return best

    if not candidates:
        raise FileNotFoundError(f"No canonical JSON found for {paper_id} under {root}")
    raise RuntimeError(
        f"Multiple canonical JSON candidates found for {paper_id}:\n"
        + "\n".join(f"  {p}" for p in candidates)
        + "\nSet the canonical location explicitly or remove stale duplicates."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate targeted ground truth against canonical paper JSONs.")
    parser.add_argument("--storage", type=Path, default=Path("storage"))
    parser.add_argument("--gt-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    storage = args.storage
    gt_dir = args.gt_dir or (storage / "ground_truth")
    output = args.output or (storage / "evaluation" / "ground_truth_accuracy.json")

    if not gt_dir.is_dir():
        raise FileNotFoundError(f"Ground-truth directory not found: {gt_dir}")

    gt_files = sorted(gt_dir.glob("paper_*.json"))
    if not gt_files:
        raise FileNotFoundError(f"No paper_*.json ground-truth files found in {gt_dir}")

    evaluator = ExtractionBenchmark()
    reports = []
    failures = []

    for gt_path in gt_files:
        try:
            gt = load_json(gt_path)
            validate_gt(gt, gt_path)
            paper_id = str(gt.get("paper_id") or gt_path.stem)
            canonical_path = find_canonical(storage, paper_id)
            canonical = load_json(canonical_path)
            result = evaluator.evaluate(canonical=canonical, ground_truth=gt)
            reports.append({
                "paper_id": paper_id,
                "ground_truth": str(gt_path),
                "canonical": str(canonical_path),
                "result": result,
            })
        except Exception as exc:
            failures.append({"ground_truth": str(gt_path), "error": str(exc)})

    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "evaluation_type": "targeted_ground_truth_accuracy",
        "papers_evaluated": len(reports),
        "papers_failed": len(failures),
        "reports": reports,
        "failures": failures,
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Evaluated: {len(reports)} papers")
    for item in reports:
        result = item["result"]
        print(f"{item['paper_id']}: {result['accuracy_score']:.4f}")
        for name, metric in result["metrics"].items():
            if isinstance(metric, dict) and "score" in metric:
                print(f"  {name}: {metric['score']:.4f}")
            elif isinstance(metric, dict) and name == "metadata":
                print(f"  metadata: {metric['score']:.4f}")
            elif isinstance(metric, (int, float)):
                print(f"  {name}: {metric:.4f}")

        for name in ("figures", "tables"):
            artifact = result["metrics"].get(name)
            if isinstance(artifact, dict) and (artifact.get("missing_labels") or artifact.get("unexpected_labels")):
                print(f"  {name} missing: {artifact.get('missing_labels', [])}")
                print(f"  {name} unexpected: {artifact.get('unexpected_labels', [])}")

    for failure in failures:
        print(f"FAILED: {failure['ground_truth']}: {failure['error']}")

    print(f"Output: {output}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
