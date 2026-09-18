# config/settings.py

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    # =========================================================
    # LLM
    # =========================================================

    ollama_base_url: str = (
        "http://localhost:11434"
    )

    ollama_model: str = (
        "qwen2.5:7b-instruct"
    )

    # =========================================================
    # EMBEDDINGS
    # =========================================================

    embedding_model: str = (
        "pritamdeka/S-PubMedBert-MS-MARCO"
    )

    # =========================================================
    # GROBID
    # =========================================================

    grobid_base_url: str = (
        "http://localhost:8070"
    )

    grobid_timeout: int = 120

    # =========================================================
    # DIRECTORIES
    # =========================================================

    # Persistent vector database.
    chroma_persist_dir: str = (
        "./storage/chroma"
    )

    # Uploaded/source PDFs.
    papers_dir: str = (
        "./papers"
    )

    # Paper-level registry.
    registry_dir: str = (
        "./storage/registry"
    )

    # Raw GROBID TEI output.
    grobid_dir: str = (
        "./storage/grobid"
    )

    # Clean canonical representation used by later stages.
    canonical_dir: str = (
        "./storage/canonical"
    )

    # Structured Docling artifacts.
    artifacts_dir: str = (
        "./storage/artifacts"
    )

    # Extraction/retrieval evaluation results.
    evaluation_dir: str = (
        "./storage/evaluation"
    )

    # Future chunked representation.
    chunks_dir: str = (
        "./storage/chunks"
    )

    # =========================================================
    # ENVIRONMENT
    # =========================================================

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()


# =============================================================
# CREATE DIRECTORIES
# =============================================================

for directory in (
    settings.chroma_persist_dir,
    settings.registry_dir,
    settings.papers_dir,
    settings.grobid_dir,
    settings.canonical_dir,
    settings.artifacts_dir,
    settings.evaluation_dir,
    settings.chunks_dir,
):

    Path(directory).mkdir(
        parents=True,
        exist_ok=True,
    )