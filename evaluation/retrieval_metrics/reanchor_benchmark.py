from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK = ROOT / "storage" / "evaluation" / "retrieval_metrics" / "retrieval_benchmark_v1.json"
DEFAULT_CHUNKS_DIR = ROOT / "storage" / "chunks"
DEFAULT_LEGACY_CATALOG = ROOT / "storage" / "evaluation" / "retrieval_metrics" / "chunk_catalog.json"
DEFAULT_HISTORY_DIR = ROOT / "storage" / "evaluation" / "retrieval_metrics"
DEFAULT_REVIEW = ROOT / "storage" / "evaluation" / "retrieval_metrics" / "retrieval_benchmark_reanchor_review.json"
DEFAULT_OUTPUT = ROOT / "storage" / "evaluation" / "retrieval_metrics" / "retrieval_benchmark_v2.json"

_TOKEN_RE = re.compile(r"[\w]+(?:[-./:+][\w]+)*", re.UNICODE)


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {path}: {exc}") from exc


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(_TOKEN_RE.findall(text))


def tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(unicodedata.normalize("NFKC", text).casefold())


def infer_paper_id(scope_paper_id: str | None, chunk_id: str) -> str | None:
    """Infer the source paper from a chunk ID when the benchmark query is corpus-wide."""
    if isinstance(scope_paper_id, str) and scope_paper_id.strip():
        return scope_paper_id
    match = re.match(r"^(paper_\d+)_", chunk_id)
    return match.group(1) if match else None


def content_score(old_text: str, new_text: str) -> tuple[float, dict[str, float | int]]:
    old_norm = normalize_text(old_text)
    new_norm = normalize_text(new_text)
    if not old_norm or not new_norm:
        return 0.0, {"overlap_tokens": 0, "containment": 0.0, "jaccard": 0.0, "sequence": 0.0}
    if old_norm in new_norm or new_norm in old_norm:
        old_t = tokens(old_text)
        new_t = tokens(new_text)
        overlap = sum((Counter(old_t) & Counter(new_t)).values())
        containment = overlap / max(1, min(len(old_t), len(new_t)))
        return 1.0, {
            "overlap_tokens": overlap,
            "containment": round(containment, 6),
            "jaccard": round(containment / max(1.0, 2.0 - containment), 6),
            "sequence": 1.0,
        }

    old_t = tokens(old_text)
    new_t = tokens(new_text)
    old_counts = Counter(old_t)
    new_counts = Counter(new_t)
    overlap = sum((old_counts & new_counts).values())
    containment = overlap / max(1, min(len(old_t), len(new_t)))
    union = sum((old_counts | new_counts).values())
    jaccard = overlap / max(1, union)
    sequence = SequenceMatcher(None, old_norm, new_norm, autojunk=False).ratio()
    length_similarity = min(len(old_t), len(new_t)) / max(1, max(len(old_t), len(new_t)))

    score = (
        0.55 * containment
        + 0.20 * jaccard
        + 0.15 * sequence
        + 0.10 * length_similarity
    )
    return float(score), {
        "overlap_tokens": overlap,
        "containment": round(containment, 6),
        "jaccard": round(jaccard, 6),
        "sequence": round(sequence, 6),
        "length_similarity": round(length_similarity, 6),
    }


def walk_dicts(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_dicts(child)


def load_legacy_texts(
    benchmark: dict[str, Any],
    legacy_catalog: Path | None,
    history_dir: Path | None,
) -> dict[tuple[str, str], dict[str, Any]]:
    needed: dict[tuple[str, str], dict[str, Any]] = {}
    for query in benchmark.get("queries", []):
        scope = query.get("scope_paper_id")
        for qrel in query.get("relevant_chunks", []):
            paper_id = infer_paper_id(qrel.get("paper_id") or scope, qrel["chunk_id"])
            if paper_id is not None:
                needed[(paper_id, qrel["chunk_id"])] = {
                    "paper_id": paper_id,
                    "chunk_id": qrel["chunk_id"],
                }

    sources: list[Path] = []
    if legacy_catalog and legacy_catalog.is_file():
        sources.append(legacy_catalog)
    if history_dir and history_dir.is_dir():
        sources.extend(
            p for p in sorted(history_dir.glob("*.json"))
            if p.resolve() != (legacy_catalog.resolve() if legacy_catalog and legacy_catalog.exists() else None)
            and p.name != Path(DEFAULT_BENCHMARK).name
        )

    for source in sources:
        try:
            raw = load_json(source)
        except (OSError, ValueError):
            continue
        for item in walk_dicts(raw):
            paper_id = item.get("paper_id")
            chunk_id = item.get("chunk_id")
            text = item.get("text")
            if not isinstance(paper_id, str) or not isinstance(chunk_id, str) or not isinstance(text, str) or not text.strip():
                continue
            key = (paper_id, chunk_id)
            if key not in needed or needed[key].get("text"):
                continue
            needed[key] = {
                "paper_id": paper_id,
                "chunk_id": chunk_id,
                "text": text,
                "chunk_type": item.get("chunk_type"),
                "order_index": item.get("order_index"),
                "metadata": item.get("metadata", {}),
                "source": str(source),
            }
        if all("text" in value for value in needed.values()):
            break

    return needed


def unwrap_chunks(raw: Any, source: Path, fallback_paper_id: str) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        chunks = raw
    elif isinstance(raw, dict) and isinstance(raw.get("chunks"), list):
        chunks = raw["chunks"]
    else:
        raise ValueError(f"Unsupported chunk-file schema: {source}")

    output: list[dict[str, Any]] = []
    for item in chunks:
        if not isinstance(item, dict):
            continue
        chunk_id = item.get("chunk_id")
        text = item.get("text")
        if not isinstance(chunk_id, str) or not isinstance(text, str) or not text.strip():
            continue
        paper_id = item.get("paper_id") or fallback_paper_id
        if not isinstance(paper_id, str):
            continue
        output.append({
            "paper_id": paper_id,
            "chunk_id": chunk_id,
            "text": text,
            "chunk_type": item.get("chunk_type"),
            "section": item.get("section") or item.get("metadata", {}).get("section") if isinstance(item.get("metadata"), dict) else item.get("section"),
            "order_index": item.get("order_index"),
            "metadata": item.get("metadata", {}),
        })
    return output


def load_current_chunks(chunks_dir: Path) -> dict[str, list[dict[str, Any]]]:
    if not chunks_dir.is_dir():
        raise FileNotFoundError(f"Chunks directory not found: {chunks_dir}")
    result: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(chunks_dir.glob("paper_*_chunks.json")):
        fallback = path.name.removesuffix("_chunks.json")
        if not fallback.startswith("paper_"):
            continue
        raw = load_json(path)
        items = unwrap_chunks(raw, path, fallback)
        if items:
            result.setdefault(fallback, []).extend(items)
    if not result:
        raise RuntimeError(f"No readable chunk files found in {chunks_dir}")
    return result


def rank_candidates(old_text: str, candidates: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for candidate in candidates:
        score, details = content_score(old_text, candidate["text"])
        ranked.append({
            "chunk_id": candidate["chunk_id"],
            "score": round(score, 6),
            "details": details,
            "chunk_type": candidate.get("chunk_type"),
            "section": candidate.get("section"),
            "order_index": candidate.get("order_index"),
            "text_preview": " ".join(candidate["text"].split())[:600],
        })
    ranked.sort(key=lambda x: (-x["score"], -(x["details"].get("overlap_tokens", 0)), x["chunk_id"]))
    return ranked[:top_n]


def validate_exact_ids(benchmark: dict[str, Any], current: dict[str, list[dict[str, Any]]]) -> set[tuple[str, str]]:
    index = {(pid, c["chunk_id"]) for pid, chunks in current.items() for c in chunks}
    stale: set[tuple[str, str]] = set()
    for query in benchmark["queries"]:
        scope = query.get("scope_paper_id")
        for qrel in query["relevant_chunks"]:
            pid = qrel.get("paper_id") or scope
            if pid is None:
                matches = [(p, qrel["chunk_id"]) for p in current if any(c["chunk_id"] == qrel["chunk_id"] for c in current[p])]
                if not matches or len(matches) != 1:
                    stale.add(("<corpus-wide>", qrel["chunk_id"]))
            elif (pid, qrel["chunk_id"]) not in index:
                stale.add((pid, qrel["chunk_id"]))
    return stale


def build_review(
    benchmark: dict[str, Any],
    current: dict[str, list[dict[str, Any]]],
    legacy: dict[tuple[str, str], dict[str, Any]],
    *,
    top_n: int,
    min_confidence: float,
    min_margin: float,
) -> dict[str, Any]:
    current_index = {(pid, c["chunk_id"]): c for pid, chunks in current.items() for c in chunks}
    stale_entries: list[dict[str, Any]] = []
    total_qrels = 0
    valid_qrels = 0

    for query in benchmark["queries"]:
        scope = query.get("scope_paper_id")
        for qrel in query["relevant_chunks"]:
            total_qrels += 1
            chunk_id = qrel["chunk_id"]
            paper_id = qrel.get("paper_id") or scope

            exact_key = (paper_id, chunk_id) if paper_id is not None else None
            if exact_key is not None and exact_key in current_index:
                valid_qrels += 1
                continue

            source_paper_id = infer_paper_id(paper_id, chunk_id)
            legacy_item = legacy.get((source_paper_id, chunk_id)) if source_paper_id is not None else legacy.get((paper_id, chunk_id))
            old_text = legacy_item.get("text") if legacy_item else None
            candidates = current.get(source_paper_id, []) if source_paper_id is not None else [c for values in current.values() for c in values]

            if old_text:
                ranked = rank_candidates(old_text, candidates, top_n)
                source = "historical_chunk_text"
            else:
                query_tokens = set(tokens(query["query"]))
                scored = []
                for c in candidates:
                    ct = set(tokens(c["text"]))
                    overlap = len(query_tokens & ct)
                    scored.append((overlap / max(1, len(query_tokens)), c))
                scored.sort(key=lambda x: (-x[0], x[1]["chunk_id"]))
                ranked = [
                    {
                        "chunk_id": c["chunk_id"],
                        "score": round(score * 0.5, 6),
                        "details": {"overlap_tokens": int(score * max(1, len(query_tokens))), "query_overlap": round(score, 6)},
                        "chunk_type": c.get("chunk_type"),
                        "section": c.get("section"),
                        "order_index": c.get("order_index"),
                        "text_preview": " ".join(c["text"].split())[:600],
                    }
                    for score, c in scored[:top_n]
                ]
                source = "query_fallback"

            expected_type = None
            if legacy_item:
                expected_type = legacy_item.get("chunk_type")
            if expected_type is None:
                if "_figure_caption_" in chunk_id:
                    expected_type = "figure_caption"
                elif "_table_caption_" in chunk_id:
                    expected_type = "table_caption"
                elif "_body_" in chunk_id:
                    expected_type = "body"
                elif "_heading_" in chunk_id:
                    expected_type = "heading"
                elif "_reference_" in chunk_id:
                    expected_type = "reference"

            compatible = [item for item in ranked if expected_type is None or item.get("chunk_type") == expected_type]
            preferred = compatible or ranked
            top_score = preferred[0]["score"] if preferred else 0.0
            second_score = preferred[1]["score"] if len(preferred) > 1 else 0.0
            margin = top_score - second_score
            details = preferred[0]["details"] if preferred else {}
            # Exact historical-text matches are safe to auto-anchor when the candidate
            # is from the original paper and the chunk type agrees. Do not require a
            # margin: identical text often coexists with a short heading that scores 1.0.
            exact_compatible = bool(
                old_text and preferred and expected_type is not None
                and preferred[0].get("chunk_type") == expected_type
                and top_score >= 0.999999
                and details.get("containment", 0.0) >= 0.999999
                and details.get("sequence", 0.0) >= 0.999999
                and details.get("overlap_tokens", 0) >= 10
            )
            high_confidence = exact_compatible or bool(
                old_text and preferred and top_score >= min_confidence
                and margin >= min_margin
                and details.get("overlap_tokens", 0) >= 20
            )

            stale_entries.append({
                "query_id": query["id"],
                "query": query["query"],
                "scope_paper_id": paper_id,
                "source_paper_id": source_paper_id,
                "expected_chunk_type": expected_type,
                "old_chunk_id": chunk_id,
                "relevance": qrel["relevance"],
                "legacy_text_available": bool(old_text),
                "legacy_source": legacy_item.get("source") if legacy_item else None,
                "legacy_text_preview": " ".join(old_text.split())[:1000] if old_text else None,
                "candidate_source": source,
                "recommended_chunk_id": preferred[0]["chunk_id"] if high_confidence and preferred else None,
                "confidence": top_score,
                "margin_over_second": round(margin, 6),
                "candidates": ranked,
                "selected_chunk_ids": [preferred[0]["chunk_id"]] if high_confidence and preferred else [],
                "status": "auto_candidate" if high_confidence else "needs_review",
            })

    return {
        "schema_version": 2,
        "purpose": "Repair benchmark anchors after chunk-ID changes without changing relevance labels",
        "benchmark_source": str(DEFAULT_BENCHMARK),
        "summary": {
            "total_qrels": total_qrels,
            "valid_exact_qrels": valid_qrels,
            "stale_qrels": len(stale_entries),
            "auto_candidates": sum(x["status"] == "auto_candidate" for x in stale_entries),
            "needs_review": sum(x["status"] == "needs_review" for x in stale_entries),
        },
        "entries": stale_entries,
        "instructions": "Exact same-text candidates with matching chunk type and source paper are auto-selected. Review remaining selected_chunk_ids. Each must refer to a current chunk in storage/chunks. Preserve relevance values; only replace obsolete chunk IDs. Then run with --apply-review.",
    }


def apply_review(benchmark: dict[str, Any], review: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    mapping: dict[tuple[str, str, str], list[str]] = {}
    errors: list[str] = []
    for entry in review.get("entries", []):
        key = (entry.get("query_id"), entry.get("scope_paper_id"), entry.get("old_chunk_id"))
        selected = entry.get("selected_chunk_ids")
        if not isinstance(selected, list) or not selected or not all(isinstance(x, str) and x.strip() for x in selected):
            errors.append(f"{entry.get('query_id')}/{entry.get('old_chunk_id')}: selected_chunk_ids is empty")
            continue
        mapping[key] = selected

    output = json.loads(json.dumps(benchmark))
    unresolved = 0
    for query in output["queries"]:
        scope = query.get("scope_paper_id")
        new_qrels: list[dict[str, Any]] = []
        for qrel in query["relevant_chunks"]:
            old_id = qrel["chunk_id"]
            pid = qrel.get("paper_id") or scope
            key = (query["id"], pid, old_id)
            if key not in mapping:
                new_qrels.append(qrel)
                continue
            for new_id in mapping[key]:
                replacement = dict(qrel)
                replacement["chunk_id"] = new_id
                new_qrels.append(replacement)
        query["relevant_chunks"] = new_qrels

    # Rewritten benchmark keeps the original benchmark ID but records lineage.
    output["benchmark_id"] = str(output.get("benchmark_id", "benchmark")) + "_v2_reanchored"
    output["status"] = "draft_needs_human_review"
    output["reanchored_from"] = "retrieval_benchmark_v1"
    output["reanchor_review_file"] = "retrieval_benchmark_reanchor_review.json"
    return output, errors


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-anchor retrieval benchmark judgments to the current chunk corpus")
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--chunks-dir", type=Path, default=DEFAULT_CHUNKS_DIR)
    parser.add_argument("--legacy-catalog", type=Path, default=DEFAULT_LEGACY_CATALOG)
    parser.add_argument("--history-dir", type=Path, default=DEFAULT_HISTORY_DIR)
    parser.add_argument("--review-output", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--min-confidence", type=float, default=0.88)
    parser.add_argument("--min-margin", type=float, default=0.08)
    parser.add_argument("--apply-review", type=Path, help="Apply manually reviewed JSON mapping to produce benchmark v2")
    args = parser.parse_args()

    benchmark_path = args.benchmark.resolve()
    benchmark = load_json(benchmark_path)
    current = load_current_chunks(args.chunks_dir.resolve())

    if args.apply_review:
        review = load_json(args.apply_review.resolve())
        output, errors = apply_review(benchmark, review)
        # Validate every qrel resolves against the current corpus.
        index = {(pid, c["chunk_id"]) for pid, chunks in current.items() for c in chunks}
        unresolved: list[str] = []
        for query in output["queries"]:
            scope = query.get("scope_paper_id")
            for qrel in query["relevant_chunks"]:
                pid = qrel.get("paper_id") or scope
                if pid is not None and (pid, qrel["chunk_id"]) not in index:
                    unresolved.append(f"{query['id']}:{pid}:{qrel['chunk_id']}")
        if errors or unresolved:
            if errors:
                print("Review errors:")
                for error in errors[:20]:
                    print(f"  - {error}")
            if unresolved:
                print(f"Unresolved current chunk references: {len(unresolved)}")
                for item in unresolved[:20]:
                    print(f"  - {item}")
            return 2
        write_json(args.output.resolve(), output)
        print(f"[REANCHOR] Wrote {args.output.resolve()}")
        return 0

    legacy = load_legacy_texts(benchmark, args.legacy_catalog.resolve(), args.history_dir.resolve())
    review = build_review(
        benchmark,
        current,
        legacy,
        top_n=max(1, args.top_n),
        min_confidence=args.min_confidence,
        min_margin=args.min_margin,
    )
    review["benchmark_source"] = str(benchmark_path)
    review["chunks_dir"] = str(args.chunks_dir.resolve())
    write_json(args.review_output.resolve(), review)

    s = review["summary"]
    print("[REANCHOR] Benchmark validation")
    print(f"  qrels:            {s['total_qrels']}")
    print(f"  exact valid:     {s['valid_exact_qrels']}")
    print(f"  stale:            {s['stale_qrels']}")
    print(f"  auto candidates:  {s['auto_candidates']}")
    print(f"  needs review:     {s['needs_review']}")
    print(f"  review file:      {args.review_output.resolve()}")

    if s["stale_qrels"] == 0:
        print("[REANCHOR] No re-anchoring needed.")
        return 0
    if s["needs_review"] == 0:
        print("[REANCHOR] All stale anchors have high-confidence candidates.")
        print("          Inspect selected_chunk_ids, then run --apply-review to create v2.")
        return 0
    print("[REANCHOR] Manual review is required before producing benchmark v2.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
