"""
Central configuration for the Brain Tumor Research Agent.

Every other module imports `settings` from here instead of reading
os.environ directly. This keeps configuration in one place and makes
it trivially swappable (e.g. changing the embedding model or LLM
model) without hunting through the codebase.
"""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    grobid_base_url: str = "http://localhost:8070"
    grobid_timeout: int = 120

    raw_pds_dir: str = "./storage/raw_pds"
    registry_dir: str = "./storage/registry"
    grobid_dir: str = "./storage/grobid"
    artifacts_dir: str = "./storage/artifacts"
    canonical_dir: str = "./storage/canonical"
    evaluation_dir: str = "./storage/evaluation"
    extraction_metrics_dir: str = "./storage/evaluation/extraction_metrics"
    chunks_dir: str = "./storage/chunks"

    # Embeddings
    embeddings_dir: str = "./storage/embeddings"
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    embedding_model_revision: str = "main"
    embedding_batch_size: int = 16
    embedding_overlap_tokens: int = 64
    embedding_device: str = "cpu"
    embedding_normalize: bool = True
    embedding_include_chunk_types: list[str] = [
        "abstract",
        "body",
        "table_caption",
        "figure_caption",
        "index_terms",
    ]
    chroma_persist_dir: str = "./storage/chroma"


    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()

# Ensure critical directories exist at import time so the rest of the
# app never has to worry about "does this folder exist yet".
for directory in (
    settings.raw_pds_dir,
    settings.registry_dir,
    settings.grobid_dir,
    settings.artifacts_dir,
    settings.canonical_dir,
    settings.evaluation_dir,
    settings.extraction_metrics_dir,
    settings.chunks_dir,
    settings.chroma_persist_dir,
):
    Path(directory).mkdir(parents=True, exist_ok=True)