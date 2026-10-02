from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config.settings import settings


BGE_QUERY_INSTRUCTION = (
    "Represent this sentence for searching relevant passages: "
)
@dataclass(frozen=True)
class ModelProfile:
    """Model-specific embedding, storage, and ChromaDB settings."""

    alias: str
    model_name: str
    revision: str | None
    query_instruction: str
    storage_subdir: str
    collection_name: str
    expected_dimension: int | None = None
    batch_size: int | None = None


# `bge` remains the stable public alias for backwards compatibility.
# Every model, including the original BGE-small baseline, stores artifacts
# beneath a dedicated directory under settings.embeddings_dir.
_PROFILES: dict[str, dict[str, Any]] = {
    "bge": {
        "model_name": "BAAI/bge-small-en-v1.5",
        "revision": "main",
        "query_instruction": BGE_QUERY_INSTRUCTION,
        "storage_subdir": "bge-small-en-v1.5",
        "collection_name": "research_papers_bge_small_en_v1_5",
        "expected_dimension": 384,
        "batch_size": None,
    },
    "minilm": {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "revision": "main",
        "query_instruction": "",
        "storage_subdir": "all-MiniLM-L6-v2",
        "collection_name": "research_papers_all_minilm_l6_v2",
        "expected_dimension": 384,
        "batch_size": None,
    },
    "bge-base": {
        "model_name": "BAAI/bge-base-en-v1.5",
        "revision": "main",
        "query_instruction": BGE_QUERY_INSTRUCTION,
        "storage_subdir": "bge-base-en-v1.5",
        "collection_name": "research_papers_bge_base_en_v1_5",
        "expected_dimension": 768,
        "batch_size": 8,
    },
    "bge-large": {
        "model_name": "BAAI/bge-large-en-v1.5",
        "revision": "main",
        "query_instruction": BGE_QUERY_INSTRUCTION,
        "storage_subdir": "bge-large-en-v1.5",
        "collection_name": "research_papers_bge_large_en_v1_5",
        "expected_dimension": 1024,
        "batch_size": 4,
    },
}
_ALIASES = {
    "bge-small": "bge",
    "bge-small-en-v1.5": "bge",
    "all-minilm-l6-v2": "minilm",
}


def normalize_model_alias(alias: str) -> str:
    if not isinstance(alias, str) or not alias.strip():
        raise ValueError("Model alias must be a non-empty string")
    key = alias.strip().lower()
    return _ALIASES.get(key, key)


def available_model_aliases() -> tuple[str, ...]:
    """Return canonical aliases only; aliases are accepted separately."""
    return tuple(_PROFILES)


def accepted_model_aliases() -> tuple[str, ...]:
    return (*available_model_aliases(), *_ALIASES.keys())


def get_model_profile(alias: str) -> ModelProfile:
    key = normalize_model_alias(alias)
    if key not in _PROFILES:
        choices = ", ".join(accepted_model_aliases())
        raise ValueError(f"Unknown embedding model alias {alias!r}. Choose: {choices}")
    item = _PROFILES[key]

    # Respect the existing central setting for the original BGE-small baseline,
    # but reject accidental changes that would mislabel its directory/collection.
    model_name = item["model_name"]
    revision = item["revision"]
    if key == "bge":
        configured_name = getattr(settings, "embedding_model_name", model_name)
        if configured_name != model_name:
            raise ValueError(
                "The 'bge' profile is the BAAI/bge-small-en-v1.5 baseline, but "
                f"config/settings.py selects {configured_name!r}. Restore the "
                "baseline setting or create a separate model profile."
            )
        revision = getattr(settings, "embedding_model_revision", revision)

    return ModelProfile(
        alias=key,
        model_name=model_name,
        revision=revision,
        query_instruction=item["query_instruction"],
        storage_subdir=item["storage_subdir"],
        collection_name=item["collection_name"],
        expected_dimension=item["expected_dimension"],
        batch_size=item["batch_size"],
    )


def model_embeddings_dir(root, profile: ModelProfile):
    """Return a stable, model-specific artifact directory."""
    return (root / profile.storage_subdir).resolve()
