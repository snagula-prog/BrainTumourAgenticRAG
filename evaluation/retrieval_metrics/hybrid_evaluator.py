from __future__ import annotations

import argparse
import json
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.settings import settings
from embeddings.model_registry import (
    accepted_model_aliases,
    available_model_aliases,
    get_model_profile,
)
from evaluation.retrieval_metrics.evaluator import (
    BENCHMARK_DIR,
    DEFAULT_BENCHMARK,
    METRIC_NAMES,
    RUNS_DIR,
    TOP_K,
    get_chunk_inventory,
    load_benchmark,
    mean_metrics,
    resolve_qrels,
    score_query,
    sha256_file,
    sha256_json,
    utc_now,
    write_json_atomic,
)
from retrieval.chroma_retriever import DenseChromaRetriever
from retrieval.hybrid_retriever import HybridRetriever


ROOT = Path(__file__).resolve().parents[2]


def evaluate_hybrid_model(
    model_alias: str,
    benchmark: dict[str, Any],
    benchmark_path: Path,
    *,
    candidate_k: int = 50,
    rrf_k: int = 60,
    dense_retriever: DenseChromaRetriever | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    profile = get_model_profile(model_alias)
    dense = dense_retriever or DenseChromaRetriever(model_alias=model_alias)
    retriever = HybridRetriever(
        model_alias=model_alias,
        dense_retriever=dense,
        candidate_k=candidate_k,
        rrf_k=rrf_k,
    )
    inventory = get_chunk_inventory(dense)
    scoped_papers = {
        query["scope_paper_id"]
        for query in benchmark["queries"]
        if query.get("scope_paper_id") is not None
    }
    missing_papers = sorted(scoped_papers - {paper for paper, _ in inventory})
    if missing_papers:
        raise ValueError(f"Benchmark references unindexed papers: {missing_papers}")

    query_results: list[dict[str, Any]] = []
    for query in benchmark["queries"]:
        qrels = resolve_qrels(query, inventory)
        response = retriever.query_with_stats(
            query["query"], top_k=TOP_K, paper_id=query.get("scope_paper_id")
        )
        ranked = response["results"]
        query_results.append({
            "query_id": query["id"],
            "query": query["query"],
            "category": query.get("category", "uncategorized"),
            "scope_paper_id": query.get("scope_paper_id"),
            "relevant_chunk_count": len(qrels),
            "metrics": score_query(ranked, qrels),
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
        category: {"query_count": len(items), "metrics": mean_metrics(items)}
        for category, items in sorted(categories.items())
    }
    encode_total = sum(item["timing"]["query_encoding_seconds"] for item in query_results)
    retrieve_total = sum(item["timing"]["retrieval_seconds"] for item in query_results)
    benchmark_hash = sha256_file(benchmark_path)
    retriever_config = {
        "type": "hybrid_dense_bm25_rrf",
        "dense_backend": "chromadb_cosine_parent_chunk",
        "sparse_backend": "in_memory_bm25_parent_chunk",
        "segment_aggregation": "max_similarity_per_parent_chunk",
        "fusion": "reciprocal_rank_fusion",
        "rrf_k": rrf_k,
        "candidate_k_per_channel": candidate_k,
        "deduplication": "paper_id_and_chunk_id",
    }
    model_config = {
        "alias": profile.alias,
        "model_name": profile.model_name,
        "query_instruction": profile.query_instruction,
        "resolved_revision": dense.encoder.model_revision,
        "dimension": dense.encoder.dimension,
        "collection_name": profile.collection_name,
    }
    dataset_fingerprint = sha256_json({
        "model": model_config,
        "retriever": retriever_config,
        "collection_count": dense.segment_count,
        "benchmark_sha256": benchmark_hash,
    })
    result = {
        "schema_version": 1,
        "evaluation_backend": "chromadb_dense_plus_bm25_rrf_parent_chunk_retriever",
        "retriever": retriever_config,
        "benchmark_id": benchmark.get("benchmark_id"),
        "benchmark_status": benchmark.get("status"),
        "model": model_config,
        "corpus": {
            "paper_ids": dense.paper_ids,
            "paper_count": len(dense.paper_ids),
            "segment_count": dense.segment_count,
            "chunk_count": len(inventory),
            "bm25_parent_chunk_count": len(retriever.documents),
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
        "benchmark_source": {"file": str(benchmark_path), "sha256": benchmark_hash},
        "queries": query_results,
    }
    run_id = (
        f"hybrid_retrieval_evaluation_{profile.alias}_"
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
        "phase": "hybrid_retrieval_evaluation",
        "status": "completed",
        "started_at": result["created_at"],
        "finished_at": utc_now(),
        "benchmark_id": result["benchmark_id"],
        "benchmark_status": result["benchmark_status"],
        "benchmark_sha256": benchmark_hash,
        "model": model_config,
        "retriever": retriever_config,
        "corpus": result["corpus"],
        "dataset_fingerprint": dataset_fingerprint,
        "metrics": result["summary"]["metrics"],
        "timing": result["summary"]["timing"],
        "result_file": str(result_path),
    }
    write_json_atomic(RUNS_DIR / f"{run_id}.json", run_record)
    return result, run_record


def _write_comparison(
    benchmark: dict[str, Any],
    benchmark_path: Path,
    model_results: list[dict[str, Any]],
    candidate_k: int,
    rrf_k: int,
) -> Path:
    comparison_id = (
        "dense_hybrid_comparison_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_"
        + uuid.uuid4().hex[:8]
    )
    output = {
        "schema_version": 1,
        "comparison_id": comparison_id,
        "created_at": utc_now(),
        "benchmark_id": benchmark.get("benchmark_id"),
        "benchmark_status": benchmark.get("status"),
        "benchmark_sha256": sha256_file(benchmark_path),
        "hybrid_config": {"candidate_k_per_channel": candidate_k, "rrf_k": rrf_k},
        "models": model_results,
        "note": (
            "Dense and hybrid runs use the same ChromaDB collection and benchmark. "
            "Hybrid results fuse dense and BM25 parent-chunk ranks using RRF. "
            "Benchmark relevance judgments are draft and require human review."
        ),
    }
    path = BENCHMARK_DIR / f"{comparison_id}.json"
    write_json_atomic(path, output)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare dense-only and dense+BM25 RRF retrieval on the judged benchmark"
    )
    parser.add_argument(
        "--model", choices=(*accepted_model_aliases(), "all"), default="all",
        help="Model profile to evaluate, or all active profiles",
    )
    parser.add_argument(
        "--strategy", choices=("dense", "hybrid", "both"), default="both",
        help="Retrieval strategies to evaluate (default: both)",
    )
    parser.add_argument("--candidate-k", type=int, default=50,
                        help="Candidates requested from each hybrid channel (default: 50)")
    parser.add_argument("--rrf-k", type=int, default=60,
                        help="RRF rank constant (default: 60)")
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK,
                        help="Path to retrieval benchmark JSON")
    args = parser.parse_args()
    if args.candidate_k < 1 or args.rrf_k < 1:
        parser.error("--candidate-k and --rrf-k must be positive integers")

    benchmark_path = args.benchmark if args.benchmark.is_absolute() else ROOT / args.benchmark
    benchmark_path = benchmark_path.resolve()
    try:
        benchmark = load_benchmark(benchmark_path)
        aliases = available_model_aliases() if args.model == "all" else (args.model,)
        comparisons: list[dict[str, Any]] = []
        for alias in aliases:
            print(f"\n[RETRIEVAL:{alias}] Initializing dense retriever")
            dense = DenseChromaRetriever(model_alias=alias)
            entry: dict[str, Any] = {"alias": alias, "model_name": get_model_profile(alias).model_name}
            if args.strategy in ("dense", "both"):
                # Reuse the same evaluator semantics as the existing dense baseline.
                from evaluation.retrieval_metrics.evaluator import evaluate_model
                dense_result, _ = evaluate_model(
                    alias, benchmark, benchmark_path, retriever=dense
                )
                entry["dense"] = {
                    "result_file": f"retrieval_results_{dense_result['run_id']}.json",
                    "metrics": dense_result["summary"]["metrics"],
                }
                print(f"[RETRIEVAL:{alias}:dense] {entry['dense']['metrics']}")
            if args.strategy in ("hybrid", "both"):
                hybrid_result, _ = evaluate_hybrid_model(
                    alias, benchmark, benchmark_path,
                    candidate_k=args.candidate_k,
                    rrf_k=args.rrf_k,
                    dense_retriever=dense,
                )
                entry["hybrid"] = {
                    "result_file": f"retrieval_results_{hybrid_result['run_id']}.json",
                    "metrics": hybrid_result["summary"]["metrics"],
                }
                print(f"[RETRIEVAL:{alias}:hybrid] {entry['hybrid']['metrics']}")
            if "dense" in entry and "hybrid" in entry:
                entry["delta_hybrid_minus_dense"] = {
                    name: round(entry["hybrid"]["metrics"][name] - entry["dense"]["metrics"][name], 6)
                    for name in METRIC_NAMES
                }
            comparisons.append(entry)

        if len(comparisons) >= 1 and args.strategy == "both":
            comparison_path = _write_comparison(
                benchmark, benchmark_path, comparisons, args.candidate_k, args.rrf_k
            )
            print(f"\n[RETRIEVAL] Dense vs hybrid comparison: {comparison_path}")
        return 0
    except Exception as exc:
        print(f"[RETRIEVAL] FAILED: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
