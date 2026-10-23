"""CLI entry point for the new deterministic Plan-Execute-Verify research agent.

Run from the repository root:
    python run_agent.py "Compare sensitivity and specificity in paper_010 and paper_011"

The agent uses the existing Chroma index, dense BGE-base retrieval, MiniLM reranking,
and Qwen 3 4B Instruct via Ollama. The reranker defaults to CPU; pass --device cuda
or set RERANKER_DEVICE=cuda only when CUDA/PyTorch support is confirmed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import Any

DEFAULT_QUERY = "Compare sensitivity and specificity in paper_010 and paper_011"
LLM_MODEL = "qwen3:4b-instruct"


def build_research_retriever(device: str = "cpu") -> Any:
    """Create the evaluated retrieval setup against the existing Chroma collection."""
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be 'cpu' or 'cuda'")

    from retrieval.research_retriever import ResearchRetriever

    return ResearchRetriever(
        model_alias="bge-base",
        strategy="dense",
        candidate_k=50,
        reranker_alias="minilm-msmarco",
        reranker_device=device,
        reranker_batch_size=8,
    )


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    return str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Plan-Execute-Verify research agent")
    parser.add_argument(
        "query",
        nargs="*",
        help="Research question (defaults to a sensitivity/specificity comparison for paper_010 and paper_011)",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default=os.getenv("RERANKER_DEVICE", "cpu").strip().lower(),
        help="Cross-encoder device; defaults to CPU until CUDA is confirmed",
    )
    parser.add_argument(
        "--debug-artifacts",
        action="store_true",
        help="Include extracted facts, evidence previews, comparison data, and answer draft in the JSON report",
    )
    args = parser.parse_args(argv)
    query = " ".join(args.query).strip() or DEFAULT_QUERY

    # This is a model default, not a forced override: an explicit OLLAMA_MODEL
    # environment variable remains respected by the shared LLM factory.
    os.environ.setdefault("OLLAMA_MODEL", LLM_MODEL)

    config = {
        "llm_model": os.getenv("OLLAMA_MODEL", LLM_MODEL),
        "retrieval_strategy": "dense",
        "embedding_alias": "bge-base",
        "reranker_alias": "minilm-msmarco",
        "candidate_k": 50,
        "final_top_k_per_paper": 10,
        "reranker_device": args.device,
        "reranker_batch_size": 8,
    }
    print("Initializing research agent with configuration:", json.dumps(config), flush=True)

    try:
        from agent.runtime import create_research_agent

        retriever = build_research_retriever(device=args.device)
        agent = create_research_agent(retriever)
        run = agent.run(query)
    except Exception as exc:
        print(f"Agent startup/run failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    execution = run.execution
    report = {
        "query": query,
        "config": config,
        "analysis": _jsonable(run.analysis),
        "plan": _jsonable(run.plan),
        "execution": {
            "status": execution.status,
            "succeeded": execution.succeeded,
            "elapsed_seconds": execution.elapsed_seconds,
            "error": execution.error,
            "step_records": _jsonable(execution.step_records),
            "artifact_keys": sorted(execution.artifacts),
        },
        "answer_bundle": _jsonable(run.answer_bundle),
    }
    if args.debug_artifacts:
        evidence = execution.artifacts.get("evidence", [])
        report["debug_artifacts"] = {
            "facts": _jsonable(execution.artifacts.get("facts", [])),
            "facts_diagnostics": _jsonable(execution.artifacts.get("facts_diagnostics", {})),
            "normalized_facts": _jsonable(execution.artifacts.get("normalized_facts", [])),
            "comparison": _jsonable(execution.artifacts.get("comparison", [])),
            "derived_values": _jsonable(execution.artifacts.get("derived_values", [])),
            "validated_derived_values": _jsonable(execution.artifacts.get("validated_derived_values", [])),
            "answer_draft": _jsonable(execution.artifacts.get("answer_draft", {})),
            "evidence": [
                {
                    "evidence_id": item.get("evidence_id"),
                    "paper_id": item.get("paper_id"),
                    "chunk_id": item.get("chunk_id"),
                    "page": item.get("page"),
                    "text_preview": str(item.get("text", ""))[:800],
                }
                for item in evidence[:20]
                if isinstance(item, Mapping)
            ] if isinstance(evidence, list) else [],
        }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if execution.succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
