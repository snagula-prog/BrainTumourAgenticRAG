from __future__ import annotations

import unittest
from unittest.mock import patch

from run_agent import build_research_retriever


class RunAgentConfigTests(unittest.TestCase):
    @patch("retrieval.research_retriever.ResearchRetriever")
    def test_uses_evaluated_dense_bge_minilm_configuration(self, retriever_cls):
        instance = object()
        retriever_cls.return_value = instance
        self.assertIs(build_research_retriever("cpu"), instance)
        retriever_cls.assert_called_once_with(
            model_alias="bge-base",
            strategy="dense",
            candidate_k=50,
            reranker_alias="minilm-msmarco",
            reranker_device="cpu",
            reranker_batch_size=8,
        )

    def test_rejects_unknown_device(self):
        with self.assertRaisesRegex(ValueError, "cpu.*cuda"):
            build_research_retriever("auto")


if __name__ == "__main__":
    unittest.main(verbosity=2)
