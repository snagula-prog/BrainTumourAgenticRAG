from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from config.settings import settings
from embeddings.model_registry import accepted_model_aliases, available_model_aliases, get_model_profile
from retrieval.chroma_retriever import DenseChromaRetriever


ROOT = Path(__file__).resolve().parents[2]
EVALUATION_DIR = (ROOT / settings.evaluation_dir).resolve()
BENCHMARK_DIR = Path(getattr(settings, "retrieval_metrics_dir", EVALUATION_DIR / "retrieval_metrics"))
if not BENCHMARK_DIR.is_absolute():
    BENCHMARK_DIR = (ROOT / BENCHMARK_DIR).resolve()
RUNS_DIR = EVALUATION_DIR / "runs"
DEFAULT_BENCHMARK = BENCHMARK_DIR / "retrieval_benchmark_v1.json"
TOP_K = 10
METRIC_KS = (5, 10)
METRIC_NAMES = ("recall@5", "recall@10", "hit@5", "hit@10", "mrr@10", "ndcg@10")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            suffix=".tmp", delete=False,
        ) as file:
            temporary = Path(file.name)
            json.dump(data, file, indent=2, ensure_ascii=False)
            file.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def load_benchmark(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Benchmark not found: {path}")
    try:
        benchmark = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid benchmark JSON ({path}): {exc}") from exc
    if not isinstance(benchmark, dict) or benchmark.get("schema_version") != 1:
        raise ValueError("Expected a benchmark object with schema_version 1")
    queries = benchmark.get("queries")
    if not isinstance(queries, list) or not queries:
        raise ValueError("Benchmark must contain a non-empty queries list")

    query_ids: set[str] = set()
    for index, query in enumerate(queries):
        if not isinstance(query, dict):
            raise ValueError(f"queries[{index}] must be an object")
        query_id, query_text = query.get("id"), query.get("query")
        if not isinstance(query_id, str) or not query_id.strip() or query_id in query_ids:
            raise ValueError(f"queries[{index}] has an invalid or duplicate id")
        query_ids.add(query_id)
        if not isinstance(query_text, str) or not query_text.strip():
            raise ValueError(f"{query_id}: query must be non-empty")
        scope = query.get("scope_paper_id")
        if scope is not None and (not isinstance(scope, str) or not scope.strip()):
            raise ValueError(f"{query_id}: invalid scope_paper_id")
        qrels = query.get("relevant_chunks")
        if not isinstance(qrels, list) or not qrels:
            raise ValueError(f"{query_id}: relevant_chunks must be a non-empty list")
        seen: set[tuple[str | None, str]] = set()
        for qrel_index, qrel in enumerate(qrels):
            if not isinstance(qrel, dict):
                raise ValueError(f"{query_id}: relevant_chunks[{qrel_index}] must be an object")
            chunk_id, relevance, qrel_paper = (
                qrel.get("chunk_id"), qrel.get("relevance"), qrel.get("paper_id")
            )
            if not isinstance(chunk_id, str) or not chunk_id.strip():
                raise ValueError(f"{query_id}: invalid chunk_id in qrel {qrel_index}")
            if not isinstance(relevance, int) or isinstance(relevance, bool) or relevance not in (1, 2, 3):
                raise ValueError(f"{query_id}/{chunk_id}: relevance must be an integer from 1 to 3")
            if qrel_paper is not None and (not isinstance(qrel_paper, str) or not qrel_paper.strip()):
                raise ValueError(f"{query_id}/{chunk_id}: invalid paper_id")
            if scope is not None and qrel_paper is not None and qrel_paper != scope:
                raise ValueError(f"{query_id}/{chunk_id}: qrel paper_id conflicts with scope")
            identity = (qrel_paper, chunk_id)
            if identity in seen:
                raise ValueError(f"{query_id}: duplicate qrel {identity}")
            seen.add(identity)
    return benchmark


def get_chunk_inventory(retriever: DenseChromaRetriever) -> set[tuple[str, str]]:
    inventory: set[tuple[str, str]] = set()
    offset = 0
    page_size = 1000
    while True:
        batch = retriever.collection.get(
            limit=page_size, offset=offset, include=["metadatas"]
        )
        ids = batch.get("ids", [])
        for metadata in batch.get("metadatas", []) or []:
            if not isinstance(metadata, dict):
                continue
            paper_id = metadata.get("paper_id")
            chunk_id = metadata.get("chunk_id")
            if isinstance(paper_id, str) and isinstance(chunk_id, str):
                inventory.add((paper_id, chunk_id))
        offset += len(ids)
        if len(ids) < page_size:
            break
    return inventory


def resolve_qrels(
    query: dict[str, Any], inventory: set[tuple[str, str]]
) -> dict[tuple[str, str], int]:
    scope = query.get("scope_paper_id")
    resolved: dict[tuple[str, str], int] = {}
    for qrel in query["relevant_chunks"]:
        chunk_id = qrel["chunk_id"]
        paper_id = qrel.get("paper_id") or scope
        if paper_id is not None:
            key = (paper_id, chunk_id)
            if key not in inventory:
                raise ValueError(
                    f"{query['id']}: labelled chunk {chunk_id!r} is not indexed "
                    f"under paper {paper_id!r}"
                )
        else:
            matches = [key for key in inventory if key[1] == chunk_id]
            if len(matches) != 1:
                raise ValueError(
                    f"{query['id']}: corpus-wide chunk {chunk_id!r} matches "
                    f"{len(matches)} indexed chunks; add paper_id to the qrel"
                )
            key = matches[0]
        resolved[key] = qrel["relevance"]
    return resolved


def _dcg(relevances: list[int]) -> float:
    return sum(
        (2**relevance - 1) / np.log2(rank + 2)
        for rank, relevance in enumerate(relevances)
    )


def score_query(
    ranked: list[dict[str, Any]], qrels: dict[tuple[str, str], int]
) -> dict[str, float | int]:
    if not qrels:
        raise ValueError("A benchmark query has no relevance judgments")
    ranked_relevances = [
        qrels.get((item["paper_id"], item["chunk_id"]), 0)
        for item in ranked[:TOP_K]
    ]
    metrics: dict[str, float | int] = {}
    for k in METRIC_KS:
        found = sum(rel > 0 for rel in ranked_relevances[:k])
        metrics[f"recall@{k}"] = found / len(qrels)
        metrics[f"hit@{k}"] = int(found > 0)

    first_relevant_rank = next(
        (rank for rank, relevance in enumerate(ranked_relevances, 1) if relevance > 0),
        None,
    )
    metrics["mrr@10"] = 1.0 / first_relevant_rank if first_relevant_rank else 0.0
    ideal = sorted(qrels.values(), reverse=True)[:TOP_K]
    ideal_dcg = _dcg(ideal)
    metrics["ndcg@10"] = _dcg(ranked_relevances) / ideal_dcg if ideal_dcg else 0.0
    return metrics


def mean_metrics(results: list[dict[str, Any]]) -> dict[str, float]:
    if not results:
        return {}
    return {
        name: round(float(np.mean([result["metrics"][name] for result in results])), 6)
        for name in METRIC_NAMES
    }


def evaluate_model(
    model_alias: str,
    benchmark: dict[str, Any],
    benchmark_path: Path,
    *,
    retriever: DenseChromaRetriever | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    profile = get_model_profile(model_alias)
    retriever_was_injected = retriever is not None

    # Reuse an already initialized dense retriever when the caller supplies one
    # (e.g. the dense+hybrid comparison evaluator). Validate its profile when
    # that information is exposed so a retriever for the wrong embedding model
    # cannot silently contaminate the evaluation.
    if retriever is None:
        retriever = DenseChromaRetriever(model_alias=model_alias)
    else:
        injected_alias = getattr(
            getattr(retriever, "profile", None),
            "alias",
            None,
        )
        if injected_alias is not None and injected_alias != profile.alias:
            raise ValueError(
                "Injected retriever/model mismatch: "
                f"requested {profile.alias!r}, got {injected_alias!r}"
            )

    inventory = get_chunk_inventory(retriever)
    scoped_papers = {
        query["scope_paper_id"]
        for query in benchmark["queries"]
        if query.get("scope_paper_id") is not None
    }
    present_papers = {paper for paper, _ in inventory}
    missing_papers = sorted(scoped_papers - present_papers)
    if missing_papers:
        raise ValueError(f"Benchmark references unindexed papers: {missing_papers}")

    query_results: list[dict[str, Any]] = []
    for query in benchmark["queries"]:
        qrels = resolve_qrels(query, inventory)
        response = retriever.query_with_stats(
            query["query"],
            top_k=TOP_K,
            paper_id=query.get("scope_paper_id"),
        )
        ranked = response["results"]
        metrics = score_query(ranked, qrels)
        query_results.append({
            "query_id": query["id"],
            "query": query["query"],
            "category": query.get("category", "uncategorized"),
            "scope_paper_id": query.get("scope_paper_id"),
            "relevant_chunk_count": len(qrels),
            "metrics": metrics,
            "timing": response["timing"],
            "diagnostics": response["diagnostics"],
            "ranked_results": [
                {
                    "rank": rank,
                    **item,
                    "relevance": qrels.get((item["paper_id"], item["chunk_id"]), 0),
                }
                for rank, item in enumerate(ranked, start=1)
            ],
        })

    categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in query_results:
        categories[result["category"]].append(result)
    category_metrics = {
        category: {
            "query_count": len(items),
            "metrics": mean_metrics(items),
        }
        for category, items in sorted(categories.items())
    }
    encode_total = sum(item["timing"]["query_encoding_seconds"] for item in query_results)
    retrieve_total = sum(item["timing"]["retrieval_seconds"] for item in query_results)
    benchmark_hash = sha256_file(benchmark_path)
    model_and_corpus = {
        "model_alias": profile.alias,
        "model_name": profile.model_name,
        "query_instruction": profile.query_instruction,
        "resolved_revision": retriever.encoder.model_revision,
        "dimension": retriever.encoder.dimension,
        "collection_name": profile.collection_name,
        "collection_count": retriever.segment_count,
        "benchmark_sha256": benchmark_hash,
    }
    dataset_fingerprint = sha256_json(model_and_corpus)
    result = {
        "schema_version": 2,
        "evaluation_backend": "chromadb_dense_parent_chunk_retriever",
        "retriever": {
            "type": "dense_chromadb",
            "distance_metric": "cosine",
            "segment_aggregation": "max_similarity_per_parent_chunk",
            "deduplication": "paper_id_and_chunk_id",
            "model_alias": profile.alias,
            "collection_name": profile.collection_name,
        },
        "benchmark_id": benchmark.get("benchmark_id"),
        "benchmark_status": benchmark.get("status"),
        "model": {
            "alias": profile.alias,
            "model_name": profile.model_name,
            "query_instruction": profile.query_instruction,
            "resolved_revision": retriever.encoder.model_revision,
            "dimension": retriever.encoder.dimension,
        },
        "evaluation_runtime": {
            "retriever_reused": retriever_was_injected,
        },
        "corpus": {
            "paper_ids": retriever.paper_ids,
            "paper_count": len(retriever.paper_ids),
            "segment_count": retriever.segment_count,
            "chunk_count": len(inventory),
        },
        "summary": {
            "query_count": len(query_results),
            "metrics": mean_metrics(query_results),
            "by_category": category_metrics,
            "timing": {
                "query_encoding_total_seconds": round(encode_total, 6),
                "retrieval_total_seconds": round(retrieve_total, 6),
                "evaluation_total_seconds": round(encode_total + retrieve_total, 6),
            },
        },
        "dataset_fingerprint": dataset_fingerprint,
        "benchmark_source": {
            "file": str(benchmark_path),
            "sha256": benchmark_hash,
        },
        "queries": query_results,
    }
    run_id = (
        f"retrieval_evaluation_{profile.alias}_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_"
        + uuid.uuid4().hex[:8]
    )
    result["run_id"] = run_id
    result["created_at"] = utc_now()
    result_path = BENCHMARK_DIR / f"retrieval_results_{run_id}.json"
    write_json_atomic(result_path, result)
    run_record = {
        "schema_version": 1,
        "run_id": run_id,
        "phase": "retrieval_evaluation",
        "status": "completed",
        "started_at": result["created_at"],
        "finished_at": utc_now(),
        "benchmark_id": benchmark.get("benchmark_id"),
        "benchmark_status": benchmark.get("status"),
        "benchmark_sha256": benchmark_hash,
        "model": result["model"],
        "retriever": result["retriever"],
        "corpus": result["corpus"],
        "dataset_fingerprint": dataset_fingerprint,
        "metrics": result["summary"]["metrics"],
        "timing": result["summary"]["timing"],
        "result_file": str(result_path),
    }
    write_json_atomic(RUNS_DIR / f"{run_id}.json", run_record)
    return result, run_record


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the ChromaDB retriever against the judged benchmark"
    )
    parser.add_argument(
        "--model", choices=(*accepted_model_aliases(), "all"), default="bge",
        help="Model profile to evaluate, or all to compare every registered model",
    )
    parser.add_argument(
        "--benchmark", type=Path, default=DEFAULT_BENCHMARK,
        help="Benchmark JSON (default: storage/evaluation/retrieval_metrics/retrieval_benchmark_v1.json)",
    )
    args = parser.parse_args()
    benchmark_path = args.benchmark if args.benchmark.is_absolute() else ROOT / args.benchmark
    benchmark_path = benchmark_path.resolve()

    try:
        benchmark = load_benchmark(benchmark_path)
        aliases = available_model_aliases() if args.model == "all" else (args.model,)
        results: list[dict[str, Any]] = []
        for alias in aliases:
            print(f"\n[RETRIEVAL-EVAL:{alias}] Starting")
            result, _ = evaluate_model(alias, benchmark, benchmark_path)
            results.append(result)
            print(f"[RETRIEVAL-EVAL:{alias}] Completed ({result['summary']['query_count']} queries)")
            for name, value in result["summary"]["metrics"].items():
                print(f"  {name}: {value:.4f}")

        if len(results) > 1:
            comparison_id = (
                "retrieval_comparison_"
                + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                + "_"
                + uuid.uuid4().hex[:8]
            )
            comparison = {
                "schema_version": 1,
                "comparison_id": comparison_id,
                "created_at": utc_now(),
                "benchmark_id": benchmark.get("benchmark_id"),
                "benchmark_status": benchmark.get("status"),
                "models": [
                    {
                        "alias": item["model"]["alias"],
                        "model_name": item["model"]["model_name"],
                        "result_file": f"retrieval_results_{item['run_id']}.json",
                        "metrics": item["summary"]["metrics"],
                    }
                    for item in results
                ],
                "note": "Metrics are macro-averaged over the same benchmark. Relevance judgments are draft and require human review.",
            }
            comparison_path = BENCHMARK_DIR / f"{comparison_id}.json"
            write_json_atomic(comparison_path, comparison)
            print(f"\n[RETRIEVAL-EVAL] Comparison: {comparison_path}")
        return 0
    except Exception as exc:
        print(f"[RETRIEVAL-EVAL] FAILED: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
