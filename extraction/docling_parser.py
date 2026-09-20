# ingestion/docling_parser.py

from __future__ import annotations

from pathlib import Path
from typing import Any

from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)
from docling.datamodel.base_models import (
    InputFormat,
)
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableFormerMode,
)

import time


class DoclingError(RuntimeError):
    pass


class DoclingParser:
    """
    Thin wrapper around Docling.

    Docling is used for structured document information such as:
        - tables
        - figures
        - formulas
        - document provenance

    It is NOT the canonical scientific-text parser.
    GROBID remains responsible for the main scientific text.
    """

    def __init__(
        self,
        do_formula_enrichment: bool = False,
        do_table_structure: bool = True,
        do_ocr: bool = False,
        table_mode: str = "fast",
        document_timeout: float = 600.0,
    ) -> None:

        # Base converter: cheap pass. Docling's layout model
        # (RT-DETR) natively detects a "formula" region class as
        # part of ordinary layout detection -- this happens here
        # regardless of do_formula_enrichment. What enrichment adds
        # on top is the (GPU-heavy) LaTeX recognition model that
        # reads the *content* of regions already labeled "formula".
        # So this cheap pass is enough to tell us whether a document
        # has any formulas at all, before paying for recognition.
        
        table_mode_normalized = (
            str(table_mode)
            .strip()
            .lower()
        )

        if table_mode_normalized == "fast":
            self._table_mode = TableFormerMode.FAST
        elif table_mode_normalized == "accurate":
            self._table_mode = TableFormerMode.ACCURATE
        else:
            raise ValueError(
                "table_mode must be 'fast' or 'accurate'"
            )

        self._document_timeout = float(
            document_timeout
        )
        
        options = PdfPipelineOptions()

        options.do_ocr = do_ocr
        options.do_formula_enrichment = False
        options.do_table_structure = (
            do_table_structure
        )
        options.document_timeout = (
            self._document_timeout
        )

        options.table_structure_options.mode = (
            self._table_mode
        )

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=options
                )
            }
        )

        # Requested enrichment setting, applied conditionally by
        # parse() rather than unconditionally at construction time.
        self._formula_enrichment_requested = (
            do_formula_enrichment
        )
        self._do_table_structure = (
            do_table_structure
        )
        self._do_ocr = do_ocr

        # Heavy converter (formula recognition enabled) is only
        # built the first time a document actually needs it, then
        # reused for the rest of the run so the model is loaded
        # onto the GPU at most once per process.
        self._formula_converter: (
            DocumentConverter | None
        ) = None

    # =========================================================
    # LAZY HEAVY CONVERTER
    # =========================================================

    def _get_formula_converter(
        self,
    ) -> DocumentConverter:

        if self._formula_converter is None:

            options = PdfPipelineOptions()

            options.do_ocr = self._do_ocr
            options.do_formula_enrichment = True
            options.do_table_structure = (
                self._do_table_structure
            )
            
            options.document_timeout = (
                self._document_timeout
            )

            options.table_structure_options.mode = (
                self._table_mode
            )

            self._formula_converter = (
                DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(
                            pipeline_options=options
                        )
                    }
                )
            )

        return self._formula_converter

    # =========================================================
    # FORMULA REGION DETECTION (cheap, label-only)
    # =========================================================

    @staticmethod
    def _has_formula_regions(
        document: Any,
    ) -> bool:

        for item in getattr(
            document,
            "texts",
            [],
        ):

            label = str(
                getattr(
                    item,
                    "label",
                    "",
                )
            ).lower()

            if "formula" in label:
                return True

        return False

    # =========================================================
    # PARSE
    # =========================================================

    def parse(
        self,
        file_path: str | Path,
    ) -> tuple[Any, dict[str, Any]]:

        file_path = Path(
            file_path
        )

        if not file_path.exists():

            raise DoclingError(
                f"PDF not found: {file_path}"
            )

        try:

            start_time = time.perf_counter()

            print(
                f"[DOCLING] Starting: "
                f"{file_path.name}"
            )

            result = self.converter.convert(
                str(file_path)
            )

            elapsed = (
                time.perf_counter()
                - start_time
            )

            print(
                f"[DOCLING] Completed: "
                f"{file_path.name} "
                f"({elapsed:.1f}s)"
            )

            document = result.document

            metadata = {
                "input_file": str(
                    file_path
                ),
                "status": str(
                    getattr(
                        result,
                        "status",
                        "unknown",
                    )
                ),
                "formula_enrichment_ran": False,
                "elapsed_seconds": round(
                    elapsed,
                    2,
                ),
                "table_mode": (
                    self._table_mode.value
                    if hasattr(
                        self._table_mode,
                        "value",
                    )
                    else str(
                        self._table_mode
                    )
                ),
                "document_timeout": (
                    self._document_timeout
                ),
            }
            # Only pay for the GPU-heavy recognition pass if the
            # cheap layout pass actually found formula regions, and
            # only if the caller asked for enrichment at all.
            if (
                self._formula_enrichment_requested
                and self._has_formula_regions(
                    document
                )
            ):

                formula_converter = (
                    self._get_formula_converter()
                )

                result = formula_converter.convert(
                    str(file_path)
                )

                document = result.document

                metadata["formula_enrichment_ran"] = (
                    True
                )

            return (
                document,
                metadata,
            )

        except Exception as exc:

            raise DoclingError(
                f"Docling failed for "
                f"{file_path.name}: {exc}"
            ) from exc