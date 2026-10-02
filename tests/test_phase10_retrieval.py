from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

# Keep the pure retrieval/profile tests runnable in minimal environments where
# model packages are not installed. Real project runs use the actual library.
if importlib.util.find_spec("sentence_transformers") is None:
    stub = types.ModuleType("sentence_transformers")
    stub.SentenceTransformer = object  # type: ignore[attr-defined]
    sys.modules["sentence_transformers"] = stub

from config.settings import settings
from embeddings.migrate_legacy import migrate_legacy_bge
from embeddings.model_registry import (
    available_model_aliases,
    get_model_profile,
    model_embeddings_dir,
)
from evaluation.retrieval_metrics.evaluator import score_query
from retrieval.chroma_retriever import DenseChromaRetriever


class FakeEncoder:
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    dimension = 2
    model_revision = "fake-revision"

    def encode_queries(self, queries):
        return np.asarray([[1.0, 0.0] for _ in queries], dtype=np.float32)


class FakeCollection:
    metadata = {
        "hnsw:space": "cosine",
        "model_alias": "minilm",
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "resolved_revision": "fake-revision",
        "dimension": 2,
    }

    def __init__(self):
        self.rows = [
            ("a0", [0.99, 0.141067], {"paper_id": "paper_001", "chunk_id": "chunk_a", "segment_index": 0, "segment_count": 2, "source_metadata_json": "{}"}, "A segment 0"),
            ("a1", [0.98, 0.198997], {"paper_id": "paper_001", "chunk_id": "chunk_a", "segment_index": 1, "segment_count": 2, "source_metadata_json": "{}"}, "A segment 1"),
            ("b0", [0.95, 0.31225], {"paper_id": "paper_001", "chunk_id": "chunk_b", "segment_index": 0, "segment_count": 1, "source_metadata_json": "{}"}, "B segment"),
            ("c0", [0.90, 0.43589], {"paper_id": "paper_002", "chunk_id": "chunk_c", "segment_index": 0, "segment_count": 1, "source_metadata_json": "{}"}, "C segment"),
        ]

    def count(self):
        return len(self.rows)

    def get(self, limit=None, offset=0, include=None, where=None):
        rows = [r for r in self.rows if not where or r[2]["paper_id"] == where["paper_id"]]
        rows = rows[offset:offset + limit] if limit is not None else rows[offset:]
        return {"ids": [r[0] for r in rows], "metadatas": [r[2] for r in rows]}

    def query(self, query_embeddings, n_results, include, where=None):
        rows = [r for r in self.rows if not where or r[2]["paper_id"] == where["paper_id"]]
        q = np.asarray(query_embeddings[0], dtype=np.float32)
        rows.sort(key=lambda r: -float(np.dot(q, np.asarray(r[1])) / (np.linalg.norm(q) * np.linalg.norm(r[1]))))
        rows = rows[:n_results]
        distances = [1.0 - float(np.dot(q, np.asarray(r[1])) / (np.linalg.norm(q) * np.linalg.norm(r[1]))) for r in rows]
        return {
            "ids": [[r[0] for r in rows]],
            "metadatas": [[r[2] for r in rows]],
            "documents": [[r[3] for r in rows]],
            "distances": [distances],
        }


class RetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_chunks_dir = settings.chunks_dir
        cls.temp_dir = tempfile.TemporaryDirectory()
        settings.chunks_dir = cls.temp_dir.name
        Path(cls.temp_dir.name, "paper_001_chunks.json").write_text(json.dumps([
            {"paper_id": "paper_001", "chunk_id": "chunk_a", "chunk_type": "body", "text": "Full parent text A", "metadata": {"section": "A"}},
            {"paper_id": "paper_001", "chunk_id": "chunk_b", "chunk_type": "body", "text": "Full parent text B", "metadata": {"section": "B"}},
        ]), encoding="utf-8")
        Path(cls.temp_dir.name, "paper_002_chunks.json").write_text(json.dumps([
            {"paper_id": "paper_002", "chunk_id": "chunk_c", "chunk_type": "body", "text": "Full parent text C", "metadata": {"section": "C"}},
        ]), encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        settings.chunks_dir = cls.original_chunks_dir
        cls.temp_dir.cleanup()

    def test_overfetches_to_unique_parent_chunks(self):
        retriever = DenseChromaRetriever(
            "minilm", encoder=FakeEncoder(), collection=FakeCollection(), initial_overfetch=1
        )
        response = retriever.query_with_stats("test query", top_k=2)
        self.assertEqual([r["chunk_id"] for r in response["results"]], ["chunk_a", "chunk_b"])
        self.assertEqual(response["diagnostics"]["requested_segments"], 4)
        self.assertEqual(response["results"][0]["text"], "Full parent text A")

    def test_paper_scope_filters_results(self):
        retriever = DenseChromaRetriever(
            "minilm", encoder=FakeEncoder(), collection=FakeCollection(), initial_overfetch=1
        )
        results = retriever.query("test query", top_k=10, paper_id="paper_001")
        self.assertTrue(results)
        self.assertTrue(all(item["paper_id"] == "paper_001" for item in results))

    def test_all_registered_models_have_unique_directories_and_collections(self):
        profiles = [get_model_profile(alias) for alias in available_model_aliases()]
        root = Path("/tmp/model-output")
        paths = [model_embeddings_dir(root, profile) for profile in profiles]
        collections = [profile.collection_name for profile in profiles]
        self.assertEqual(len(paths), len(set(paths)))
        self.assertEqual(len(collections), len(set(collections)))
        self.assertEqual(model_embeddings_dir(root, get_model_profile("bge")), root / "bge-small-en-v1.5")
        self.assertEqual(get_model_profile("bge-small").alias, "bge")
        self.assertEqual(get_model_profile("bge-base").expected_dimension, 768)
        self.assertEqual(get_model_profile("bge-large").expected_dimension, 1024)
        self.assertEqual(set(available_model_aliases()), {"bge", "minilm", "bge-base", "bge-large"})
        self.assertTrue(get_model_profile("bge").query_instruction)
        self.assertEqual(get_model_profile("minilm").query_instruction, "")
        for retired in ("gte-modernbert", "gte-modernbert-base", "qwen3-0.6b", "qwen3-embedding-0.6b"):
            with self.subTest(retired=retired):
                with self.assertRaises(ValueError):
                    get_model_profile(retired)

    def test_legacy_bge_migration_is_verified_and_reversible_by_copy(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            vector = root / "paper_001_embeddings.npz"
            vector.write_bytes(b"test-vector-payload")
            import hashlib
            manifest = {
                "schema_version": 2,
                "paper_id": "paper_001",
                "embedding": {"model_name": "BAAI/bge-small-en-v1.5", "dimension": 384},
                "vectors": {"file": vector.name, "sha256": hashlib.sha256(vector.read_bytes()).hexdigest()},
            }
            manifest_path = root / "paper_001_embeddings.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with patch.object(settings, "embeddings_dir", str(root)):
                dry = migrate_legacy_bge(dry_run=True)
                self.assertEqual(dry["status"], "dry_run")
                copied = migrate_legacy_bge(move=False)
                target = root / "bge-small-en-v1.5"
                self.assertEqual(copied["status"], "copied")
                self.assertEqual((target / vector.name).read_bytes(), vector.read_bytes())
                self.assertTrue(vector.exists())
                moved = migrate_legacy_bge(move=True)
                self.assertEqual(moved["status"], "moved")
                self.assertFalse(vector.exists())
                self.assertFalse(manifest_path.exists())

    def test_metrics_at_expected_ranks(self):
        ranked = [
            {"paper_id": "p", "chunk_id": "a"},
            {"paper_id": "p", "chunk_id": "b"},
        ]
        qrels = {("p", "a"): 3, ("p", "b"): 1, ("p", "c"): 2}
        metrics = score_query(ranked, qrels)
        self.assertAlmostEqual(metrics["recall@5"], 2 / 3)
        self.assertEqual(metrics["hit@5"], 1)
        self.assertEqual(metrics["mrr@10"], 1.0)
        self.assertGreater(metrics["ndcg@10"], 0.0)


if __name__ == "__main__":
    unittest.main()
