import pytest

from agent.tools.retrieval import RetrievalTool


class FakeRetriever:
    def __init__(self):
        self.calls = []

    def query_with_stats(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "results": [
                {
                    "paper_id": "paper_001",
                    "chunk_id": "paper_001_body_0001",
                    "text": "Example evidence.",
                    "metadata": {"pages": [2]},
                    "rank": 1,
                }
            ],
            "timing": {"total_seconds": 0.1},
            "diagnostics": {"returned_count": 1},
        }


@pytest.fixture
def setup_tool():
    retriever = FakeRetriever()
    return RetrievalTool(retriever), retriever


def test_retrieval_uses_defaults(setup_tool):
    tool, retriever = setup_tool

    result = tool.execute({"query": "What is segmentation?"})

    assert retriever.calls == [{
        "query": "What is segmentation?",
        "top_k": 10,
        "paper_id": None,
        "candidate_k": None,
    }]
    assert result["results"][0]["chunk_id"] == (
        "paper_001_body_0001"
    )


def test_retrieval_passes_optional_arguments(setup_tool):
    tool, retriever = setup_tool

    tool.execute({
        "query": "What is segmentation?",
        "top_k": 5,
        "paper_id": "paper_001",
        "candidate_k": 20,
    })

    assert retriever.calls[0]["top_k"] == 5
    assert retriever.calls[0]["paper_id"] == "paper_001"
    assert retriever.calls[0]["candidate_k"] == 20


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"query": ""},
        {"query": "   "},
        {"query": "test", "top_k": 0},
        {"query": "test", "top_k": True},
        {"query": "test", "candidate_k": -1},
        {"query": "test", "paper_id": "  "},
        {"query": "test", "unknown": "value"},
    ],
)
def test_invalid_arguments_are_rejected(setup_tool, arguments):
    tool, _ = setup_tool

    with pytest.raises(ValueError):
        tool.execute(arguments)


def test_retrieval_preserves_provenance(setup_tool):
    tool, _ = setup_tool

    result = tool.execute({"query": "test"})

    evidence = result["results"][0]
    assert evidence["paper_id"] == "paper_001"
    assert evidence["chunk_id"] == "paper_001_body_0001"
    assert evidence["metadata"]["pages"] == [2]


def test_retrieval_preserves_diagnostics(setup_tool):
    tool, _ = setup_tool

    result = tool.execute({"query": "test"})

    assert result["timing"]["total_seconds"] == 0.1
    assert result["diagnostics"]["returned_count"] == 1