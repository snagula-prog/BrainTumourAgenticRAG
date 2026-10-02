from __future__ import annotations

import unittest

from reranking.metrics import candidate_coverage
from reranking.cross_encoder import CrossEncoderReranker, get_reranker_profile


class FakeTokenizer:
    def __init__(self):
        self.pad_token_id = 0

    def encode(self, text, add_special_tokens=False, truncation=False):
        return [len(token) for token in text.split()]

    def num_special_tokens_to_add(self, pair=True):
        return 3 if pair else 2

    def prepare_for_model(self, ids, pair_ids=None, **kwargs):
        data = [101] + ids + [102] + (pair_ids or []) + [102]
        return {"input_ids": data, "attention_mask": [1] * len(data)}

    def pad(self, features, padding=True, return_tensors="pt"):
        import torch
        width = max(len(item["input_ids"]) for item in features)
        ids, masks = [], []
        for item in features:
            missing = width - len(item["input_ids"])
            ids.append(item["input_ids"] + [0] * missing)
            masks.append(item["attention_mask"] + [0] * missing)
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks)}


class FakeConfig:
    max_position_embeddings = 64
    _commit_hash = "fake-revision"


class FakeOutput:
    def __init__(self, logits):
        self.logits = logits


class FakeModel:
    config = FakeConfig()

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, **batch):
        import torch
        # Deterministic score: longer second sequence gets a higher score.
        lengths = batch["attention_mask"].sum(dim=1).float()
        return FakeOutput(lengths.unsqueeze(1))


class RerankingTests(unittest.TestCase):
    def test_profiles_resolve(self):
        self.assertEqual(get_reranker_profile("minilm-msmarco").model_name,
                         "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.assertEqual(get_reranker_profile("bge-reranker-base").model_name,
                         "BAAI/bge-reranker-base")

    def test_long_passage_windows_cover_tail(self):
        ids = list(range(100))
        windows = CrossEncoderReranker._windows(ids, size=32, overlap=8)
        self.assertEqual(windows[0], ids[:32])
        self.assertEqual(windows[-1][-1], 99)
        self.assertEqual(windows[1][0], 24)

    def test_candidate_coverage(self):
        candidates = [
            {"paper_id": "p1", "chunk_id": "a"},
            {"paper_id": "p1", "chunk_id": "b"},
        ]
        qrels = {("p1", "b"): 3, ("p1", "c"): 2}
        result = candidate_coverage(candidates, qrels, 2)
        self.assertEqual(result["candidate_recall@2"], 0.5)
        self.assertEqual(result["candidate_hit@2"], 1)

    def test_reranker_sorts_by_cross_encoder_score(self):
        reranker = CrossEncoderReranker(
            "minilm-msmarco", tokenizer=FakeTokenizer(), model=FakeModel(),
            max_length=64, max_query_tokens=8, window_overlap=4,
        )
        results, diag = reranker.rerank("query", [
            {"paper_id": "p1", "chunk_id": "a", "text": "short" , "score": 0.9},
            {"paper_id": "p1", "chunk_id": "b", "text": "long passage with more tokens", "score": 0.5},
        ], top_k=2)
        self.assertEqual(results[0]["chunk_id"], "b")
        self.assertEqual(results[0]["rank"], 1)
        self.assertEqual(results[0]["retrieval_rank"], 2)
        self.assertEqual(diag["documents_scored"], 2)

    def test_validation(self):
        with self.assertRaises(ValueError):
            CrossEncoderReranker("minilm-msmarco", batch_size=0)


if __name__ == "__main__":
    unittest.main()
