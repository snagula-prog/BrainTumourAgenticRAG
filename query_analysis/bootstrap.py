import json
from pathlib import Path

from llm.factory import create_chat_model
from query_analysis import QueryAnalyzer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPERS_REGISTRY_PATH = (
    PROJECT_ROOT / "storage" / "registry" / "papers.json"
)


def load_valid_paper_ids() -> frozenset[str]:
    if not PAPERS_REGISTRY_PATH.exists():
        raise FileNotFoundError(
            f"Paper registry not found: {PAPERS_REGISTRY_PATH}"
        )

    with PAPERS_REGISTRY_PATH.open("r", encoding="utf-8") as file:
        registry = json.load(file)

    if not isinstance(registry, dict):
        raise ValueError("papers.json must contain a JSON object.")

    valid_ids = {
        key
        for key, record in registry.items()
        if isinstance(record, dict)
        and record.get("paper_id") == key
        and record.get("status") == "extracted"
    }

    if not valid_ids:
        raise ValueError("No extracted papers found in the registry.")

    return frozenset(valid_ids)


def create_query_analyzer() -> QueryAnalyzer:
    return QueryAnalyzer(
        create_chat_model(),
        valid_paper_ids=load_valid_paper_ids(),
    )