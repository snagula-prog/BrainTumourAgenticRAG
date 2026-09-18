from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from chunking.config import ChunkingConfig
from chunking.text_chunker import TextChunker
from chunking.figure_chunker import FigureChunker
from chunking.table_chunker import TableChunker
from chunking.formula_chunker import FormulaChunker
from chunking.reference_chunker import ReferenceChunker


class ChunkingPipeline:

    def __init__(
        self,
        config: ChunkingConfig | None = None,
    ) -> None:

        self.config = (
            config
            or ChunkingConfig()
        )

        self.text_chunker = (
            TextChunker(
                self.config
            )
        )

        self.figure_chunker = (
            FigureChunker()
        )

        self.table_chunker = (
            TableChunker()
        )

        self.formula_chunker = (
            FormulaChunker()
        )

        self.reference_chunker = (
            ReferenceChunker()
        )

    # ============================================================
    # DOCUMENT
    # ============================================================

    def process(
        self,
        canonical: dict[str, Any],
    ) -> dict[str, Any]:

        paper_id = canonical.get(
            "paper_id"
        )

        if not paper_id:
            raise ValueError(
                "Canonical document missing paper_id"
            )

        text_units = (
            self.text_chunker.chunk(
                canonical.get(
                    "paragraphs",
                    [],
                )
            )
        )

        figure_units = (
            self.figure_chunker.chunk(
                canonical.get(
                    "figures",
                    [],
                )
            )
        )

        table_units = (
            self.table_chunker.chunk(
                canonical.get(
                    "tables",
                    [],
                )
            )
        )

        formula_units = (
            self.formula_chunker.chunk(
                canonical.get(
                    "formulas",
                    [],
                )
            )
        )

        reference_units = (
            self.reference_chunker.chunk(
                canonical.get(
                    "references",
                    [],
                )
            )
        )

        units = (
            text_units
            + figure_units
            + table_units
            + formula_units
            + reference_units
        )

        self._assign_ids(
            paper_id,
            units,
        )

        return {
            "paper_id": paper_id,
            "filename": canonical.get(
                "filename"
            ),
            "chunking": {
                "config_id": (
                    self.config.config_id
                ),
                "chunk_size": (
                    self.config.chunk_size
                ),
                "overlap": (
                    self.config.overlap
                ),
                "tokenizer": (
                    self.config.tokenizer_name
                ),
            },
            "counts": {
                "total_units": len(
                    units
                ),
                "text": len(
                    text_units
                ),
                "figure": len(
                    figure_units
                ),
                "table": len(
                    table_units
                ),
                "formula": len(
                    formula_units
                ),
                "reference": len(
                    reference_units
                ),
            },
            "units": units,
        }

    # ============================================================
    # IDs
    # ============================================================

    @staticmethod
    def _assign_ids(
        paper_id: str,
        units: list[dict[str, Any]],
    ) -> None:

        counters: dict[
            str,
            int,
        ] = {}

        for unit in units:

            unit_type = unit[
                "unit_type"
            ]

            counters[
                unit_type
            ] = (
                counters.get(
                    unit_type,
                    0,
                )
                + 1
            )

            unit[
                "unit_id"
            ] = (
                f"{paper_id}_"
                f"{unit_type}_"
                f"{counters[unit_type]:03d}"
            )

    # ============================================================
    # FILE
    # ============================================================

    def process_file(
        self,
        canonical_path: str | Path,
        output_path: str | Path,
    ) -> dict[str, Any]:

        canonical_path = Path(
            canonical_path
        )

        output_path = Path(
            output_path
        )

        with canonical_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            canonical = json.load(
                file
            )

        result = self.process(
            canonical
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                result,
                file,
                ensure_ascii=False,
                indent=2,
            )

        return result