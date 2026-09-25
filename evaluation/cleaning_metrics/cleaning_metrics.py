from __future__ import annotations

import re
from collections import Counter
from typing import Any

NUMERIC_TOKEN_RE = re.compile(r"(?<![\w])(?:\d+(?:\.\d+)?%?)(?![\w])")


class CleaningEvaluator:
    """Evaluate preservation of information during Phase 6 cleaning."""

    def evaluate(self, original: dict[str, Any], cleaned: dict[str, Any]) -> dict[str, Any]:
        orig_paras = self._items(original, "paragraphs")
        clean_paras = self._items(cleaned, "paragraphs")
        orig_nums = self._numeric_tokens(original)
        clean_nums = self._numeric_tokens(cleaned)

        return {
            "evaluation_type": "cleaning_integrity",
            "metrics": {
                "paragraph_retention": self._ratio(len(clean_paras), len(orig_paras)),
                "numeric_token_retention": self._ratio(
                    sum((orig_nums & clean_nums).values()),
                    sum(orig_nums.values()),
                ),
                "logical_id_retention": self._logical_id_retention(original, cleaned),
                "artifact_provenance_coverage": self._provenance_coverage(cleaned),
            },
            "counts": {
                "original_paragraphs": len(orig_paras),
                "cleaned_paragraphs": len(clean_paras),
                "original_numeric_tokens": sum(orig_nums.values()),
                "cleaned_numeric_tokens": sum(clean_nums.values()),
            },
            "notes": [
                "Cleaning integrity is not ground-truth extraction accuracy.",
                "Numeric token preservation is used as a semantic safety guard.",
            ],
        }

    @staticmethod
    def _items(data: dict[str, Any], key: str) -> list[dict[str, Any]]:
        value = data.get(key)
        return [x for x in value if isinstance(x, dict)] if isinstance(value, list) else []

    @staticmethod
    def _ratio(a: int, b: int) -> float | None:
        return round(a / b, 4) if b else None

    @classmethod
    def _numeric_tokens(cls, data: Any) -> Counter[str]:
        out: Counter[str] = Counter()

        def walk(x: Any) -> None:
            if isinstance(x, dict):
                for k, v in x.items():
                    if k in {"source_refs", "coords", "bbox", "page", "pages"}:
                        continue
                    walk(v)
            elif isinstance(x, list):
                for v in x:
                    walk(v)
            elif isinstance(x, str):
                out.update(NUMERIC_TOKEN_RE.findall(x))

        walk(data)
        return out

    @staticmethod
    def _logical_id_retention(original: dict[str, Any], cleaned: dict[str, Any]) -> float | None:
        pairs = (("tables", "table_id"), ("figures", "figure_id"), ("formulas", "formula_id"))
        orig_ids, clean_ids = set(), set()

        for collection, field in pairs:
            for source, target in ((original, orig_ids), (cleaned, clean_ids)):
                items = source.get(collection)
                if not isinstance(items, list):
                    continue
                target.update(
                    str(item[field])
                    for item in items
                    if isinstance(item, dict) and item.get(field)
                )

        return round(len(orig_ids & clean_ids) / len(orig_ids), 4) if orig_ids else None

    @staticmethod
    def _provenance_coverage(cleaned: dict[str, Any]) -> float | None:
        artifacts = []
        for key in ("tables", "figures", "formulas"):
            value = cleaned.get(key)
            if isinstance(value, list):
                artifacts.extend(x for x in value if isinstance(x, dict))
        if not artifacts:
            return None
        valid = sum(1 for x in artifacts if x.get("source_refs") or x.get("source"))
        return round(valid / len(artifacts), 4)
