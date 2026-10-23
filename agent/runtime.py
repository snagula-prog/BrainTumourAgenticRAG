"""Small orchestration entry point for the new Plan-Execute-Verify path."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.executor import ExecutionResult, Executor
from agent.runtime_adapters import build_runtime_handlers
from llm.base import ChatModel
from llm.factory import create_chat_model
from planner import DeterministicPlanner, Plan
from query_analysis import QueryAnalysis, QueryAnalyzer


@dataclass
class ResearchAgentRun:
    query: str
    analysis: QueryAnalysis
    plan: Plan
    execution: ExecutionResult

    @property
    def answer_bundle(self) -> dict[str, Any] | None:
        value = self.execution.artifacts.get("answer_bundle")
        return value if isinstance(value, dict) else None


class ResearchAgent:
    """Coordinate analyzer -> deterministic planner -> executor."""

    def __init__(self, analyzer: Any, planner: DeterministicPlanner, executor: Executor) -> None:
        self.analyzer = analyzer
        self.planner = planner
        self.executor = executor

    def run(self, query: str) -> ResearchAgentRun:
        analysis = self.analyzer.analyze(query)
        plan = self.planner.plan(analysis)
        execution = self.executor.execute(plan, query=query, analysis=analysis)
        return ResearchAgentRun(query=query, analysis=analysis, plan=plan, execution=execution)


def create_research_agent(
    retriever_or_tool: Any,
    *,
    model: ChatModel | None = None,
    valid_paper_ids: set[str] | frozenset[str] | None = None,
) -> ResearchAgent:
    """Build a Qwen-backed agent using the project's existing retrieval path.

    Pass an already-created ResearchRetriever instance or its existing RetrievalTool.
    This function does not construct or reconfigure Chroma/hybrid retrieval.
    """
    from agent.tools.retrieval import RetrievalTool

    shared_model = model or create_chat_model()
    retrieval_tool = retriever_or_tool
    if not callable(getattr(retrieval_tool, "execute", None)):
        retrieval_tool = RetrievalTool(retriever_or_tool)

    if valid_paper_ids is None:
        from query_analysis.bootstrap import load_valid_paper_ids
        valid_paper_ids = load_valid_paper_ids()

    analyzer = QueryAnalyzer(shared_model, valid_paper_ids=valid_paper_ids)
    planner = DeterministicPlanner()
    executor = Executor(build_runtime_handlers(retrieval_tool, shared_model))
    return ResearchAgent(analyzer, planner, executor)
