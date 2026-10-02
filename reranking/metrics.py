from __future__ import annotations

from typing import Any


def candidate_coverage(
    candidates: list[dict[str, Any]],
    qrels: dict[tuple[str, str], int],
    candidate_k: int,
) -> dict[str, float | int]:
    """Measure how much labelled evidence exists in the pre-rerank pool."""
    if not isinstance(candidate_k, int) or isinstance(candidate_k, bool) or candidate_k < 1:
        raise ValueError("candidate_k must be a positive integer")
    relevant = {key for key, relevance in qrels.items() if relevance > 0}
    found = {
        (item["paper_id"], item["chunk_id"])
        for item in candidates[:candidate_k]
    }
    hits = len(relevant.intersection(found))
    return {
        f"candidate_recall@{candidate_k}": hits / len(relevant) if relevant else 0.0,
        f"candidate_hit@{candidate_k}": int(hits > 0),
    }
