from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class RerankerProfile:
    alias: str
    model_name: str
    revision: str = "main"
    description: str = ""


_PROFILES: dict[str, RerankerProfile] = {
    "minilm-msmarco": RerankerProfile(
        alias="minilm-msmarco",
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Lightweight English MS MARCO passage cross-encoder",
    ),
    "bge-reranker-base": RerankerProfile(
        alias="bge-reranker-base",
        model_name="BAAI/bge-reranker-base",
        description="BGE base cross-encoder reranker",
    ),
}


def available_reranker_aliases() -> tuple[str, ...]:
    return tuple(_PROFILES)


def get_reranker_profile(alias: str) -> RerankerProfile:
    if not isinstance(alias, str) or not alias.strip():
        raise ValueError("Reranker alias must be a non-empty string")
    key = alias.strip().lower()
    if key not in _PROFILES:
        raise ValueError(
            f"Unknown reranker {alias!r}. Choose: "
            + ", ".join((*available_reranker_aliases(), "all"))
        )
    return _PROFILES[key]


class CrossEncoderReranker:
    """Pairwise cross-encoder that reorders retrieved parent chunks.

    Long parent chunks are split into overlapping passage windows to avoid
    silently discarding evidence at the end of a chunk. The maximum score
    across a chunk's windows is used as its reranker score.
    """

    def __init__(
        self,
        model_alias: str = "minilm-msmarco",
        *,
        device: str = "cpu",
        batch_size: int = 8,
        max_length: int = 512,
        max_query_tokens: int = 128,
        window_overlap: int = 64,
        tokenizer: Any | None = None,
        model: Any | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if max_length < 32:
            raise ValueError("max_length must be at least 32")
        if max_query_tokens < 1:
            raise ValueError("max_query_tokens must be at least 1")
        if window_overlap < 0:
            raise ValueError("window_overlap cannot be negative")

        self.profile = get_reranker_profile(model_alias)
        self.model_alias = self.profile.alias
        self.model_name = self.profile.model_name
        self.device_name = device
        self.batch_size = batch_size
        self.requested_max_length = max_length
        self.max_query_tokens = max_query_tokens
        self.window_overlap = window_overlap

        if tokenizer is None or model is None:
            try:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
            except ImportError as exc:
                raise RuntimeError(
                    "Reranking requires torch and transformers. Install the project's "
                    "sentence-transformers dependencies or `pip install transformers`."
                ) from exc
            if device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError(f"CUDA device requested but unavailable: {device}")
            self._torch = torch
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, revision=self.profile.revision
            ) if tokenizer is None else tokenizer
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, revision=self.profile.revision
            ) if model is None else model
            self.model.to(device)
            self.model.eval()
        else:
            import torch
            self._torch = torch
            self.tokenizer = tokenizer
            self.model = model
            self.model.to(device)
            self.model.eval()

        config_limit = getattr(self.model.config, "max_position_embeddings", max_length)
        # RoBERTa-family models sometimes report 514 due to their position
        # embedding offset; 512 remains a safe default for these profiles.
        usable_config_limit = min(int(config_limit), 512) if config_limit else 512
        self.max_length = min(max_length, usable_config_limit)
        if self.max_length < 32:
            raise ValueError(f"Model sequence limit is unexpectedly small: {self.max_length}")
        self.model_revision = getattr(self.model.config, "_commit_hash", None)
        self.special_pair_tokens = int(self.tokenizer.num_special_tokens_to_add(pair=True))

    def _token_ids(self, text: str) -> list[int]:
        value = self.tokenizer.encode(
            text, add_special_tokens=False, truncation=False
        )
        return list(value)

    @staticmethod
    def _windows(token_ids: list[int], size: int, overlap: int) -> list[list[int]]:
        if size < 1:
            raise ValueError("Passage token budget is too small")
        if not token_ids:
            return [[]]
        actual_overlap = min(overlap, size - 1)
        windows: list[list[int]] = []
        start = 0
        while start < len(token_ids):
            end = min(start + size, len(token_ids))
            windows.append(token_ids[start:end])
            if end >= len(token_ids):
                break
            next_start = end - actual_overlap
            if next_start <= start:
                raise RuntimeError("Reranker token window iteration did not advance")
            start = next_start
        return windows

    def _score_window_pairs(self, pairs: list[tuple[list[int], list[int]]]) -> np.ndarray:
        if not pairs:
            return np.empty((0,), dtype=np.float32)
        scores: list[np.ndarray] = []
        for offset in range(0, len(pairs), self.batch_size):
            batch_pairs = pairs[offset:offset + self.batch_size]
            features = []
            for query_ids, document_ids in batch_pairs:
                encoded = self.tokenizer.prepare_for_model(
                    query_ids,
                    pair_ids=document_ids,
                    add_special_tokens=True,
                    truncation=False,
                    padding=False,
                    return_attention_mask=True,
                )
                if len(encoded["input_ids"]) > self.max_length:
                    raise RuntimeError("Prepared reranker pair exceeds max_length")
                features.append(encoded)
            batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
            batch = {key: value.to(self.device_name) for key, value in batch.items()}
            with self._torch.inference_mode():
                logits = self.model(**batch).logits
            logits = logits.detach().float().cpu().numpy()
            if logits.ndim == 1:
                batch_scores = logits
            elif logits.ndim == 2 and logits.shape[1] == 1:
                batch_scores = logits[:, 0]
            elif logits.ndim == 2 and logits.shape[1] == 2:
                # For binary classifiers, rank by the positive-vs-negative logit.
                batch_scores = logits[:, 1] - logits[:, 0]
            else:
                raise RuntimeError(f"Unsupported reranker logit shape: {logits.shape}")
            scores.append(np.asarray(batch_scores, dtype=np.float32))
        output = np.concatenate(scores)
        if output.shape != (len(pairs),) or not np.isfinite(output).all():
            raise RuntimeError("Reranker produced invalid relevance scores")
        return output

    def score_documents(
        self, query: str, documents: Sequence[dict[str, Any]]
    ) -> tuple[list[float], dict[str, int]]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be non-empty")
        if not documents:
            return [], {"documents_scored": 0, "passage_windows_scored": 0}

        query_ids = self._token_ids(query.strip())[:self.max_query_tokens]
        if not query_ids:
            raise ValueError("Query produced no tokens")
        document_budget = self.max_length - len(query_ids) - self.special_pair_tokens
        if document_budget < 1:
            raise ValueError("Query leaves no token budget for the passage")

        pairs: list[tuple[list[int], list[int]]] = []
        owners: list[int] = []
        for index, document in enumerate(documents):
            text = document.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Candidate document {index} has empty text")
            doc_ids = self._token_ids(text)
            windows = self._windows(doc_ids, document_budget, self.window_overlap)
            for window in windows:
                pairs.append((query_ids, window))
                owners.append(index)

        window_scores = self._score_window_pairs(pairs)
        best = np.full((len(documents),), -np.inf, dtype=np.float32)
        for owner, score in zip(owners, window_scores, strict=True):
            if score > best[owner]:
                best[owner] = score
        if not np.isfinite(best).all():
            raise RuntimeError("A candidate document did not receive a finite reranker score")
        return best.astype(float).tolist(), {
            "documents_scored": len(documents),
            "passage_windows_scored": len(pairs),
        }

    def rerank(
        self, query: str, documents: Sequence[dict[str, Any]], *, top_k: int = 10
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        scores, diagnostics = self.score_documents(query, documents)
        ranked = []
        for original_rank, (document, score) in enumerate(zip(documents, scores, strict=True), 1):
            item = dict(document)
            item["retrieval_rank"] = original_rank
            item["retrieval_score"] = item.get("score")
            item["reranker_score"] = score
            item["score"] = score
            ranked.append(item)
        ranked.sort(key=lambda item: (
            -item["reranker_score"],
            item["retrieval_rank"],
            item.get("paper_id", ""),
            item.get("chunk_id", ""),
        ))
        for final_rank, item in enumerate(ranked[:top_k], 1):
            item["rank"] = final_rank
        return ranked[:top_k], diagnostics
