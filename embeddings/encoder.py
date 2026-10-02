from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class EmbeddingEncoder:
    """Local Sentence Transformers encoder with token-window support."""

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        batch_size: int = 16,
        normalize: bool = True,
        revision: str | None = None,
        query_instruction: str = "",
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if not isinstance(query_instruction, str):
            raise TypeError("query_instruction must be a string")

        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        self.requested_revision = revision
        self.query_instruction = query_instruction
        self.model = SentenceTransformer(
            model_name,
            revision=revision,
            device=device,
        )
        self.model.eval()
        self.tokenizer = self.model[0].tokenizer
        dimension = self.model.get_sentence_embedding_dimension()
        if dimension is None or dimension < 1:
            raise RuntimeError("Invalid embedding dimension")
        self.dimension = int(dimension)
        self.max_seq_length = int(self.model.max_seq_length)
        self.model_revision = self._get_model_revision()

    def _get_model_revision(self) -> str | None:
        try:
            config = self.model[0].auto_model.config
            return getattr(config, "_commit_hash", None)
        except (AttributeError, TypeError):
            return None

    def _raw_token_ids(self, text: str) -> list[int]:
        """Tokenize without adding special tokens or truncating."""
        backend = getattr(self.tokenizer, "backend_tokenizer", None)
        if backend is not None:
            return backend.encode(text, add_special_tokens=False).ids
        return self.tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=False,
        )

    def split_passage(
        self,
        text: str,
        overlap_tokens: int = 64,
    ) -> list[dict[str, Any]]:
        """Split a passage into overlapping token windows."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Passage text cannot be empty")
        special_count = self.tokenizer.num_special_tokens_to_add(pair=False)
        window_size = self.max_seq_length - special_count
        if window_size < 1:
            raise ValueError("Model has no usable content-token capacity")
        if not isinstance(overlap_tokens, int) or isinstance(overlap_tokens, bool):
            raise TypeError("overlap_tokens must be an integer")
        if overlap_tokens < 0 or overlap_tokens >= window_size:
            raise ValueError(
                f"overlap_tokens must be between 0 and {window_size - 1}"
            )

        token_ids = self._raw_token_ids(text)
        if not token_ids:
            raise ValueError("Passage produced no tokens")
        windows: list[dict[str, Any]] = []
        start = 0
        while start < len(token_ids):
            end = min(start + window_size, len(token_ids))
            ids = token_ids[start:end]
            prepared_ids = self.tokenizer.build_inputs_with_special_tokens(ids)
            if len(prepared_ids) > self.max_seq_length:
                raise RuntimeError("Window exceeds the model sequence-length limit")
            segment_text = self.tokenizer.decode(
                ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            windows.append({
                "input_ids": ids,
                "token_start": start,
                "token_end": end,
                "token_count": len(ids),
                "text": segment_text,
            })
            if end == len(token_ids):
                break
            next_start = end - overlap_tokens
            if next_start <= start:
                raise RuntimeError("Token-window iteration did not advance")
            start = next_start
        return windows

    def encode_token_windows(
        self,
        windows: Sequence[dict[str, Any]],
    ) -> np.ndarray:
        """Encode pre-tokenized passage windows without re-tokenizing."""
        if not windows:
            return np.empty((0, self.dimension), dtype=np.float32)
        results: list[np.ndarray] = []
        for offset in range(0, len(windows), self.batch_size):
            batch_windows = windows[offset:offset + self.batch_size]
            prepared = []
            for window in batch_windows:
                ids = window.get("input_ids")
                if not isinstance(ids, list) or not ids:
                    raise ValueError("Invalid or empty token window")
                features = self.tokenizer.prepare_for_model(
                    ids,
                    add_special_tokens=True,
                    truncation=False,
                    padding=False,
                    return_attention_mask=True,
                )
                if len(features["input_ids"]) > self.max_seq_length:
                    raise ValueError("Prepared window exceeds model token limit")
                prepared.append(features)
            batch = self.tokenizer.pad(
                prepared,
                padding=True,
                return_tensors="pt",
            )
            batch = {key: value.to(self.model.device) for key, value in batch.items()}
            with torch.inference_mode():
                output = self.model(batch)
                embeddings = output["sentence_embedding"]
                if self.normalize:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            results.append(
                embeddings.detach().cpu().numpy().astype(np.float32, copy=False)
            )
        vectors = np.concatenate(results, axis=0)
        if vectors.shape != (len(windows), self.dimension):
            raise RuntimeError(f"Unexpected embedding shape: {vectors.shape}")
        if not np.isfinite(vectors).all():
            raise RuntimeError("Embeddings contain non-finite values")
        return vectors

    def token_lengths(self, texts: Sequence[str]) -> list[int]:
        """Return token counts including special tokens."""
        return [
            len(self.tokenizer.build_inputs_with_special_tokens(self._raw_token_ids(text)))
            for text in texts
        ]

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        clean_texts: list[str] = []
        for index, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Empty text at index {index}")
            clean_texts.append(text.strip())
        if not clean_texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        lengths = self.token_lengths(clean_texts)
        overlong = [
            (index, length)
            for index, length in enumerate(lengths)
            if length > self.max_seq_length
        ]
        if overlong:
            raise ValueError(
                f"{len(overlong)} input(s) exceed the {self.max_seq_length}-token limit. "
                "Use split_passage() for long passages."
            )
        vectors = self.model.encode(
            clean_texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.shape != (len(clean_texts), self.dimension):
            raise RuntimeError(f"Unexpected embedding shape: {vectors.shape}")
        if not np.isfinite(vectors).all():
            raise RuntimeError("Embeddings contain non-finite values")
        return vectors

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_queries(self, queries: Sequence[str]) -> np.ndarray:
        prepared: list[str] = []
        for index, query in enumerate(queries):
            if not isinstance(query, str) or not query.strip():
                raise ValueError(f"Empty query at index {index}")
            prepared.append(self.query_instruction + query.strip())
        return self._encode(prepared)
