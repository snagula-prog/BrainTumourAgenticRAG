from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkingConfig:
    """
    Parameters used by the text chunker.

    Different experiments should change these values,
    not create different source files.
    """

    chunk_size: int = 450
    overlap: int = 70

    tokenizer_name: str = (
        "microsoft/"
        "BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    )

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError(
                "chunk_size must be greater than zero"
            )

        if self.overlap < 0:
            raise ValueError(
                "overlap cannot be negative"
            )

        if self.overlap >= self.chunk_size:
            raise ValueError(
                "overlap must be smaller than chunk_size"
            )

    @property
    def config_id(self) -> str:
        return (
            f"tok{self.chunk_size}"
            f"_ov{self.overlap}"
        )