# tests/test_phase8_chunking.py
from chunking.text_chunker import SectionAwareChunker


def test_section_boundaries():
    """Verify that chunks never blend paragraphs from different sections."""
    mock_doc = {
        "paragraphs": [
            {"text": "Short intro to methods.", "section": "Methods"},
            {"text": "Dataset setup details.", "section": "Methods"},
            {"text": "Results showed 98% accuracy.", "section": "Results"},
        ]
    }
    
    chunker = SectionAwareChunker(max_words=300)
    chunks = chunker.chunk("test_001", mock_doc)
    
    body_chunks = [c for c in chunks if c["chunk_type"] == "body"]
    
    # We expect 2 chunks: one for Methods (combining the two short paras) and one for Results.
    assert len(body_chunks) == 2
    assert body_chunks[0]["metadata"]["section"] == "Methods"
    assert "intro to methods" in body_chunks[0]["text"]
    assert "Dataset setup" in body_chunks[0]["text"]
    
    assert body_chunks[1]["metadata"]["section"] == "Results"
    assert "98% accuracy" in body_chunks[1]["text"]
    
    # Verify Context Injection prepended the section name
    assert body_chunks[0]["text"].startswith("[Methods]")


def test_large_paragraph_splitting():
    """Verify that a single massive paragraph is split to respect token limits."""
    large_text = "Word. " * 400
    mock_doc = {
        "paragraphs": [
            {"text": large_text, "section": "Discussion"}
        ]
    }
    
    chunker = SectionAwareChunker(max_words=300)
    chunks = chunker.chunk("test_002", mock_doc)
    
    body_chunks = [c for c in chunks if c["chunk_type"] == "body"]
    
    # The 400-word paragraph should be split into at least 2 chunks.
    assert len(body_chunks) >= 2
    for chunk in body_chunks:
        assert chunk["word_count"] <= 300
        assert chunk["metadata"]["section"] == "Discussion"


def test_atomic_artifact_chunking():
    """Verify tables, figures, and formulas are chunked cleanly and atomically."""
    mock_doc = {
        "tables": [{"label": "Table 1", "caption": "Model stats", "content": "A | B\n1 | 2"}],
        "formulas": [{"formula_id": "eq_1", "source": "E = mc^2"}]
    }
    
    chunker = SectionAwareChunker()
    chunks = chunker.chunk("test_003", mock_doc)
    
    table_chunks = [c for c in chunks if c["chunk_type"] == "table"]
    formula_chunks = [c for c in chunks if c["chunk_type"] == "formula"]
    
    assert len(table_chunks) == 1
    assert "Table 1" in table_chunks[0]["text"]
    assert "A | B" in table_chunks[0]["text"]
    
    assert len(formula_chunks) == 1
    assert "E = mc^2" in formula_chunks[0]["text"]
    
    # Verify deterministic IDs
    assert formula_chunks[0]["chunk_id"].startswith("test_003_formula_")


def main():
    test_section_boundaries()
    test_large_paragraph_splitting()
    test_atomic_artifact_chunking()
    print("Phase 8 Chunking Validation: ALL PASSED")


if __name__ == "__main__":
    main()