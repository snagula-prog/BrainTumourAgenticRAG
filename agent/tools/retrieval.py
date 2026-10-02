from __future__ import annotations

from typing import Any

from retrieval.research_retriever import ResearchRetriever


class RetrievalTool:
    """Expose ResearchRetriever as a validated agent tool."""

    name = "retrieve_papers"
    description = (
        "Retrieve relevant passages from research papers. "
        "Supports dense or hybrid retrieval and optional reranking. "
        "Can restrict retrieval to a specific paper."
    )

    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The question or search query.",
            },
            "top_k": {
                "type": "integer",
                "minimum": 1,
                "description": "Number of ranked passages to return.",
            },
            "paper_id": {
                "type": "string",
                "minLength": 1,
                "description": "Optional paper ID to restrict retrieval.",
            },
            "candidate_k": {
                "type": "integer",
                "minimum": 1,
                "description": "Optional candidate pool size.",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(self, retriever: ResearchRetriever) -> None:
        self._retriever = retriever

    @staticmethod
    def _positive_int(value: Any, name: str) -> int:
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
        return value

    def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be an object")

        allowed = {"query", "top_k", "paper_id", "candidate_k"}
        unknown = set(arguments) - allowed
        if unknown:
            raise ValueError(
                f"Unexpected arguments: {', '.join(sorted(unknown))}"
            )

        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        top_k = self._positive_int(
            arguments.get("top_k", 10), "top_k"
        )

        paper_id = arguments.get("paper_id")
        if paper_id is not None:
            if not isinstance(paper_id, str) or not paper_id.strip():
                raise ValueError(
                    "paper_id must be a non-empty string when supplied"
                )
            paper_id = paper_id.strip()

        candidate_k = arguments.get("candidate_k")
        if candidate_k is not None:
            candidate_k = self._positive_int(
                candidate_k, "candidate_k"
            )

        response = self._retriever.query_with_stats(
            query=query.strip(),
            top_k=top_k,
            paper_id=paper_id,
            candidate_k=candidate_k,
        )

        if not isinstance(response, dict) or "results" not in response:
            raise RuntimeError("Retriever returned an invalid response")

        return {
            "query": query.strip(),
            "results": response["results"],
            "timing": response.get("timing", {}),
            "diagnostics": response.get("diagnostics", {}),
        }