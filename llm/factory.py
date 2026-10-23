"""Factory for the application's configured chat model."""
from __future__ import annotations

from .ollama_client import OllamaChatModel, OllamaConfig


def create_chat_model() -> OllamaChatModel:
    """Return the configured model, defaulting to qwen3:4b-instruct."""
    return OllamaChatModel(OllamaConfig.from_env())
