from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from reranking.cross_encoder import CrossEncoderReranker


class FakeBackendEncoding:
    def __init__(self, ids):
        self.ids = ids


class FakeBackendTokenizer:
    def encode(self, text, add_special_tokens=False):
        # Simulate a 593-token parent chunk and a short query.
        count = 5 if text == "query" else 593
        return FakeBackendEncoding([i % 100 for i in range(count)])


class FakeTokenizer:
    backend_tokenizer = FakeBackendTokenizer()
    pad_token_id = 0
    padding_side = "right"

    def num_special_tokens_to_add(self, pair=True):
        return 3

    def prepare_for_model(
        self, query_ids, *, pair_ids, add_special_tokens, truncation, padding,
        return_attention_mask,
    ):
        ids = [101, *query_ids, 102, *pair_ids, 102]
        return {
            "input_ids": ids,
            "token_type_ids": [0] * len(ids),
            "attention_mask": [1] * len(ids),
        }

    def pad(self, *args, **kwargs):
        raise AssertionError("Fast-tokenizer .pad() should not be used by the reranker")


class FakeModel:
    def __init__(self):
        self.config = SimpleNamespace(max_position_embeddings=64, _commit_hash="test")
        self.seen_lengths = []

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, **batch):
        self.seen_lengths.append(int(batch["input_ids"].shape[1]))
        logits = torch.zeros((batch["input_ids"].shape[0], 1), dtype=torch.float32)
        return SimpleNamespace(logits=logits)


class CrossEncoderLongPassageTests(unittest.TestCase):
    def test_long_parent_is_windowed_without_encode_then_pad_path(self):
        model = FakeModel()
        reranker = CrossEncoderReranker(
            model_alias="minilm-msmarco",
            device="cpu",
            batch_size=2,
            max_length=64,
            max_query_tokens=16,
            window_overlap=8,
            tokenizer=FakeTokenizer(),
            model=model,
        )
        scores, diagnostics = reranker.score_documents(
            "query", [{"text": "a long parent passage"}]
        )
        self.assertEqual(len(scores), 1)
        self.assertGreater(diagnostics["passage_windows_scored"], 1)
        self.assertTrue(model.seen_lengths)
        self.assertLessEqual(max(model.seen_lengths), 64)


if __name__ == "__main__":
    unittest.main(verbosity=2)
