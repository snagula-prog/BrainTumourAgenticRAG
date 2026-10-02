from __future__ import annotations

import argparse
import gc
import json
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

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
from reranking.metrics import candidate_coverage
from reranking.cross_encoder import (
    CrossEncoderReranker,
    available_reranker_aliases,
    get_reranker_profile,
)


ROOT = Path(__file__).resolve().parents[2]
COVERAGE_METRICS = ("candidate_recall", "candidate_hit")


def delta_metrics(
    after: dict[str, Any], before: dict[str, Any]
) -> dict[str, float]:
    return {
        key: round(float(after[key]) - float(before[key]), 6)
        for key in METRIC_NAMES
    }


def _brief_candidate(item: dict[str, Any], relevance: int, rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "paper_id": item["paper_id"],
        "chunk_id": item["chunk_id"],
        "retrieval_score": item.get("score"),
        "relevance": relevance,
        "retrieval_channels": item.get("retrieval_channels"),
        "dense_rank": item.get("dense_rank"),
        "bm25_rank": item.get("bm25_rank"),
        "rrf_score": item.get("rrf_score"),
    }


def _evaluate_configuration(
    *,
    model_alias: str,
    strategy: str,
    reranker: CrossEncoderReranker,
    benchmark: dict[str, Any],
    benchmark_path: Path,
    candidate_k: int,
    rrf_k: int,
    dense_retriever: DenseChromaRetriever | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    profile = get_model_profile(model_alias)
    dense = dense_retriever or DenseChromaRetriever(model_alias=model_alias)
    hybrid = None
    if strategy == "hybrid":
        hybrid = HybridRetriever(
            model_alias=model_alias,
            dense_retriever=dense,
            candidate_k=candidate_k,
            rrf_k=rrf_k,
        )
    elif strategy != "dense":
        raise ValueError(f"Unknown retrieval strategy: {strategy}")

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
        started = time.perf_counter()
        if strategy == "dense":
            response = dense.query_with_stats(
                query["query"], top_k=candidate_k,
                paper_id=query.get("scope_paper_id"),
            )
            candidates = response["results"]
            retrieval_timing = response.get("timing", {})
            retrieval_diagnostics = response.get("diagnostics", {})
        else:
            assert hybrid is not None
            response = hybrid.query_with_stats(
                query["query"], top_k=candidate_k,
                paper_id=query.get("scope_paper_id"),
                candidate_k=candidate_k,
            )
            candidates = response["results"]
            retrieval_timing = response.get("timing", {})
            retrieval_diagnostics = response.get("diagnostics", {})

        # Record the unreranked order from the same candidate pool.
        initial = candidates[:TOP_K]
        initial_metrics = score_query(initial, qrels)
        coverage = candidate_coverage(candidates, qrels, candidate_k)

        rerank_started = time.perf_counter()
        ranked, rerank_diagnostics = reranker.rerank(
            query["query"], candidates, top_k=TOP_K
        )
        rerank_seconds = time.perf_counter() - rerank_started
        reranked_metrics = score_query(ranked, qrels)

        query_results.append({
            "query_id": query["id"],
            "query": query["query"],
            "category": query.get("category", "uncategorized"),
            "scope_paper_id": query.get("scope_paper_id"),
            "relevant_chunk_count": len(qrels),
            "candidate_count": len(candidates),
            "candidate_coverage": coverage,
            "initial_metrics": initial_metrics,
            "reranked_metrics": reranked_metrics,
            "delta_reranked_minus_initial": delta_metrics(reranked_metrics, initial_metrics),
            "timing": {
                "query_encoding_seconds": float(retrieval_timing.get("query_encoding_seconds", 0.0)),
                "retrieval_seconds": float(retrieval_timing.get("retrieval_seconds", 0.0)),
                "reranking_seconds": round(rerank_seconds, 6),
                "total_seconds": round(time.perf_counter() - started, 6),
            },
            "diagnostics": {
                "retrieval": retrieval_diagnostics,
                "reranking": rerank_diagnostics,
            },
            "candidate_results": [
                _brief_candidate(item, qrels.get((item["paper_id"], item["chunk_id"]), 0), rank)
                for rank, item in enumerate(candidates, 1)
            ],
            "ranked_results": [
                {
                    "rank": rank,
                    **item,
                    "relevance": qrels.get((item["paper_id"], item["chunk_id"]), 0),
                }
                for rank, item in enumerate(ranked, 1)
            ],
        })

    categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in query_results:
        categories[result["category"]].append(result)
    category_summary = {}
    for category, items in sorted(categories.items()):
        category_summary[category] = {
            "query_count": len(items),
            "initial_metrics": mean_metrics([
                {"metrics": item["initial_metrics"]} for item in items
            ]),
            "reranked_metrics": mean_metrics([
                {"metrics": item["reranked_metrics"]} for item in items
            ]),
        }

    initial_mean = mean_metrics([
        {"metrics": item["initial_metrics"]} for item in query_results
    ])
    reranked_mean = mean_metrics([
        {"metrics": item["reranked_metrics"]} for item in query_results
    ])
    coverage_mean = {
        name: round(float(np.mean([
            item["candidate_coverage"][name] for item in query_results
        ])), 6)
        for name in (
            f"candidate_recall@{candidate_k}",
            f"candidate_hit@{candidate_k}",
        )
    }
    query_encode_total = sum(item["timing"]["query_encoding_seconds"] for item in query_results)
    retrieval_total = sum(item["timing"]["retrieval_seconds"] for item in query_results)
    reranking_total = sum(item["timing"]["reranking_seconds"] for item in query_results)
    benchmark_hash = sha256_file(benchmark_path)
    reranker_config = {
        "alias": reranker.model_alias,
        "model_name": reranker.model_name,
        "resolved_revision": reranker.model_revision,
        "device": reranker.device_name,
        "batch_size": reranker.batch_size,
        "max_length": reranker.max_length,
        "max_query_tokens": reranker.max_query_tokens,
        "window_overlap": reranker.window_overlap,
        "long_passage_aggregation": "maximum_score_over_overlapping_windows",
    }
    retrieval_config = {
        "type": "chromadb_dense" if strategy == "dense" else "hybrid_dense_bm25_rrf",
        "candidate_k": candidate_k,
        "rrf_k": rrf_k if strategy == "hybrid" else None,
        "dense_backend": "chromadb_cosine_parent_chunk",
        "sparse_backend": "in_memory_bm25_parent_chunk" if strategy == "hybrid" else None,
        "segment_aggregation": "max_similarity_per_parent_chunk",
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
    fingerprint = sha256_json({
        "model": model_config,
        "retrieval": retrieval_config,
        "reranker": reranker_config,
        "benchmark_sha256": benchmark_hash,
        "collection_count": dense.segment_count,
    })
    result = {
        "schema_version": 1,
        "evaluation_backend": "retrieval_plus_cross_encoder_reranking",
        "retrieval_strategy": strategy,
        "retrieval": retrieval_config,
        "reranker": reranker_config,
        "benchmark_id": benchmark.get("benchmark_id"),
        "benchmark_status": benchmark.get("status"),
        "model": model_config,
        "corpus": {
            "paper_ids": dense.paper_ids,
            "paper_count": len(dense.paper_ids),
            "segment_count": dense.segment_count,
            "chunk_count": len(inventory),
        },
        "summary": {
            "query_count": len(query_results),
            "initial_metrics": initial_mean,
            "candidate_coverage": coverage_mean,
            "reranked_metrics": reranked_mean,
            "delta_reranked_minus_initial": delta_metrics(reranked_mean, initial_mean),
            "by_category": category_summary,
            "timing": {
                "query_encoding_total_seconds": round(query_encode_total, 6),
                "retrieval_total_seconds": round(retrieval_total, 6),
                "reranking_total_seconds": round(reranking_total, 6),
                "evaluation_total_seconds": round(query_encode_total + retrieval_total + reranking_total, 6),
            },
        },
        "dataset_fingerprint": fingerprint,
        "benchmark_source": {"file": str(benchmark_path), "sha256": benchmark_hash},
        "queries": query_results,
    }
    run_id = (
        f"reranking_evaluation_{profile.alias}_{strategy}_{reranker.model_alias}_"
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
        "phase": "retrieval_reranking_evaluation",
        "status": "completed",
        "created_at": result["created_at"],
        "finished_at": utc_now(),
        "benchmark_id": result["benchmark_id"],
        "benchmark_status": result["benchmark_status"],
        "benchmark_sha256": benchmark_hash,
        "model": model_config,
        "retrieval": retrieval_config,
        "reranker": reranker_config,
        "corpus": result["corpus"],
        "dataset_fingerprint": fingerprint,
        "initial_metrics": initial_mean,
        "candidate_coverage": coverage_mean,
        "reranked_metrics": reranked_mean,
        "timing": result["summary"]["timing"],
        "result_file": str(result_path),
    }
    write_json_atomic(RUNS_DIR / f"{run_id}.json", run_record)
    return result, run_record


def _write_comparison(
    benchmark: dict[str, Any],
    benchmark_path: Path,
    rows: list[dict[str, Any]],
    *,
    candidate_k: int,
    rrf_k: int,
) -> Path:
    comparison_id = (
        "reranking_comparison_"
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
        "benchmark_sha256": sha256_file(benchmark_path),
        "candidate_k": candidate_k,
        "rrf_k": rrf_k,
        "runs": rows,
        "note": (
            "Initial and reranked metrics are computed from the same candidate pool. "
            "Candidate coverage measures the maximum recall available to reranking. "
            "The relevance benchmark is draft and requires human review."
        ),
    }
    path = BENCHMARK_DIR / f"{comparison_id}.json"
    write_json_atomic(path, comparison)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate dense/hybrid retrieval with cross-encoder reranking"
    )
    parser.add_argument(
        "--model", choices=(*accepted_model_aliases(), "all"), default="all",
        help="Embedding model profile to use for candidate retrieval",
    )
    parser.add_argument(
        "--strategy", choices=("dense", "hybrid", "both"), default="both",
        help="Candidate retrieval strategy (default: both)",
    )
    parser.add_argument(
        "--reranker", choices=(*available_reranker_aliases(), "all"), default="minilm-msmarco",
        help="Cross-encoder profile, or all",
    )
    parser.add_argument("--candidate-k", type=int, default=50,
                        help="Candidate pool size before reranking (default: 50)")
    parser.add_argument("--rrf-k", type=int, default=60,
                        help="RRF constant for hybrid retrieval (default: 60)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Reranker inference batch size (default: 8)")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Maximum reranker pair length (default: 512)")
    parser.add_argument("--max-query-tokens", type=int, default=128,
                        help="Maximum query tokens reserved for reranking")
    parser.add_argument("--device", default="cpu", help="Reranker device (default: cpu)")
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK,
                        help="Path to retrieval benchmark JSON")
    args = parser.parse_args()
    if min(args.candidate_k, args.rrf_k, args.batch_size, args.max_length, args.max_query_tokens) < 1:
        parser.error("candidate, RRF, batch, and token limits must be positive")

    benchmark_path = args.benchmark if args.benchmark.is_absolute() else ROOT / args.benchmark
    benchmark_path = benchmark_path.resolve()
    try:
        benchmark = load_benchmark(benchmark_path)
        model_aliases = available_model_aliases() if args.model == "all" else (args.model,)
        reranker_aliases = available_reranker_aliases() if args.reranker == "all" else (args.reranker,)
        strategies = ("dense", "hybrid") if args.strategy == "both" else (args.strategy,)
        comparison_rows: list[dict[str, Any]] = []

        for reranker_alias in reranker_aliases:
            print(f"\n[RERANK:{reranker_alias}] Loading cross-encoder")
            reranker = CrossEncoderReranker(
                reranker_alias, device=args.device, batch_size=args.batch_size,
                max_length=args.max_length, max_query_tokens=args.max_query_tokens,
            )
            for model_alias in model_aliases:
                print(f"\n[RERANK:{reranker_alias}] Initializing {model_alias}")
                dense = DenseChromaRetriever(model_alias=model_alias)
                for strategy in strategies:
                    print(f"[RERANK:{reranker_alias}:{model_alias}:{strategy}] Starting")
                    result, _ = _evaluate_configuration(
                        model_alias=model_alias,
                        strategy=strategy,
                        reranker=reranker,
                        benchmark=benchmark,
                        benchmark_path=benchmark_path,
                        candidate_k=args.candidate_k,
                        rrf_k=args.rrf_k,
                        dense_retriever=dense,
                    )
                    comparison_rows.append({
                        "embedding_alias": model_alias,
                        "embedding_model": result["model"]["model_name"],
                        "strategy": strategy,
                        "reranker_alias": reranker_alias,
                        "reranker_model": result["reranker"]["model_name"],
                        "result_file": f"retrieval_results_{result['run_id']}.json",
                        "initial_metrics": result["summary"]["initial_metrics"],
                        "candidate_coverage": result["summary"]["candidate_coverage"],
                        "reranked_metrics": result["summary"]["reranked_metrics"],
                        "delta_reranked_minus_initial": result["summary"]["delta_reranked_minus_initial"],
                        "timing": result["summary"]["timing"],
                    })
                    print(f"[RERANK:{reranker_alias}:{model_alias}:{strategy}] "
                          f"initial={result['summary']['initial_metrics']} "
                          f"reranked={result['summary']['reranked_metrics']}")
                del dense
                gc.collect()
            del reranker
            gc.collect()

        comparison_path = _write_comparison(
            benchmark, benchmark_path, comparison_rows,
            candidate_k=args.candidate_k, rrf_k=args.rrf_k,
        )
        print(f"\n[RERANK] Comparison: {comparison_path}")
        return 0
    except Exception as exc:
        print(f"[RERANK] FAILED: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
