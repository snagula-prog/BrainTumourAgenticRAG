# ingestion/pipeline.py

from __future__ import annotations

import json
import re
from pathlib import Path

from config.settings import settings

from ingestion.deduplication import (
    compute_file_hash,
    load_registry,
    find_by_hash,
    register_new_paper,
    save_registry,
)

from ingestion.grobid_parser import (
    GrobidParser,
    parse_grobid_tei,
)

from ingestion.pymupdf_audit import (
    PyMuPDFAudit,
)

from ingestion.docling_parser import (
    DoclingParser,
)

from ingestion.docling_artifacts import (
    DoclingArtifactExtractor,
)

from ingestion.canonical_builder import (
    CanonicalBuilder,
)

from evaluation.extraction_quality_metrics import (
    QualityMetrics,
)


class IngestionPipeline:
    """
    Complete PDF ingestion pipeline.

    Input:
        PDF file

    Flow:
        PDF
          ↓
        SHA-256 / duplicate check
          ↓
        stable paper_id
          ↓
        GROBID
          ↓
        PyMuPDF audit
          ↓
        Docling artifacts
          ↓
        extraction quality evaluation
          ↓
        canonical document
          ↓
        registry update

    Chunking, embeddings and retrieval are intentionally not
    performed here yet.
    """

    def __init__(self) -> None:

        self.grobid = GrobidParser(
            base_url=settings.grobid_base_url,
            timeout_seconds=settings.grobid_timeout,
            coordinate_elements=[
                "head",
                "p",
                "figure",
                "table",
            ],
            segment_sentences=True,
        )

        self.docling = DoclingParser(
            do_formula_enrichment=True,
            do_table_structure=True,
            do_ocr=False,
        )

        self.artifacts = DoclingArtifactExtractor()

        self.quality = QualityMetrics()

        self.builder = CanonicalBuilder()

    # =========================================================
    # PUBLIC API
    # =========================================================

    def process_single_pdf(
        self,
        pdf_path: Path,
    ) -> dict:
        """
        Process one PDF.

        Idempotent:
        uploading/running the same PDF again will detect its
        SHA-256 hash and skip duplicate processing.
        """

        pdf_path = Path(pdf_path).resolve()

        if not pdf_path.exists():
            raise FileNotFoundError(
                f"PDF not found: {pdf_path}"
            )

        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(
                f"Expected a PDF file, got: "
                f"{pdf_path.name}"
            )

        print()
        print("=" * 70)
        print(f"[INGEST] {pdf_path.name}")
        print("=" * 70)

        # -----------------------------------------------------
        # 1. HASH + DUPLICATE CHECK
        # -----------------------------------------------------

        print("[1/6] Checking duplicate...")

        file_hash = compute_file_hash(
            pdf_path
        )

        registry = load_registry()

        existing_paper_id = find_by_hash(
            registry,
            file_hash,
        )

        if existing_paper_id:

            existing_record = registry[
                existing_paper_id
            ]

            print(
                f"[SKIP] Already registered as "
                f"{existing_paper_id}"
            )

            return {
                "skipped": True,
                "paper_id": existing_paper_id,
                "record": existing_record,
            }

        # -----------------------------------------------------
        # 2. REGISTER PAPER
        # -----------------------------------------------------

        print("[2/6] Registering paper...")

        record = register_new_paper(
            pdf_path,
            file_hash,
        )

        paper_id = record[
            "paper_id"
        ]

        print(
            f"[REGISTERED] {paper_id}"
        )

        self._update_registry_status(
            paper_id,
            "processing",
        )

        try:

            # -------------------------------------------------
            # 3. GROBID
            # -------------------------------------------------

            print("[3/6] GROBID...")

            tei_xml = self.grobid.process_pdf(
                pdf_path
            )

            grobid = parse_grobid_tei(
                tei_xml
            )

            # -------------------------------------------------
            # 4. PYMUPDF + DOCLING
            # -------------------------------------------------

            print("[4/6] PyMuPDF audit...")

            pymupdf = PyMuPDFAudit(
                pdf_path
            ).extract()

            print("[5/6] Docling artifacts...")

            document, docling_metadata = (
                self.docling.parse(
                    pdf_path
                )
            )

            raw_docling = (
                document.export_to_dict()
            )

            artifacts = (
                self.artifacts.extract(
                    raw_docling
                )
            )
            text_blocks = artifacts.get("text_blocks", [])

            print(
                "[DEBUG TEXT BLOCK]",
                json.dumps(
                    text_blocks[0] if text_blocks else {"error": "No text_blocks found"},
                    indent=2,
                    default=str,
                )
            )

            # -------------------------------------------------
            # YEAR CORRECTION
            # -------------------------------------------------

            pdf_year = (
                self._extract_pdf_year(
                    pymupdf
                )
            )

            if pdf_year:
                grobid["year"] = pdf_year

            # -------------------------------------------------
            # QUALITY
            # -------------------------------------------------

            print(
                "[6/6] Evaluating extraction quality..."
            )

            quality = self.quality.evaluate(
                grobid=grobid,
                pymupdf=pymupdf,
                artifacts=artifacts,
            )

            # -------------------------------------------------
            # CANONICAL DOCUMENT
            # -------------------------------------------------

            canonical = self.builder.build(
                paper_id=paper_id,
                filename=pdf_path.name,
                grobid=grobid,
                pymupdf=pymupdf,
                artifacts=artifacts,
                quality=quality,
            )

            # -------------------------------------------------
            # SAVE EVERYTHING
            # -------------------------------------------------

            self._save_outputs(
                paper_id=paper_id,
                pdf_path=pdf_path,
                tei_xml=tei_xml,
                raw_docling=raw_docling,
                docling_metadata=docling_metadata,
                canonical=canonical,
                quality=quality,
            )

            # -------------------------------------------------
            # UPDATE REGISTRY
            # -------------------------------------------------

            self._update_registry_after_success(
                paper_id=paper_id,
                canonical=canonical,
                pymupdf=pymupdf,
                quality=quality,
            )

            self._print_result(
                canonical
            )

            return {
                "skipped": False,
                "paper_id": paper_id,
                "record": self._get_registry_record(
                    paper_id
                ),
                "canonical": canonical,
                "quality": quality,
            }

        except Exception as exc:

            self._update_registry_status(
                paper_id,
                "failed",
                error=str(exc),
            )

            print(
                f"[FAILED] {paper_id}: "
                f"{type(exc).__name__}: {exc}"
            )

            raise

    # =========================================================
    # PROCESS ALL PDFs
    # =========================================================

    def run_ingestion(
        self,
    ) -> list[dict]:
        """
        Process every PDF currently in papers/.

        This is useful for:
          - first-time ingestion
          - rebuilding from the current papers directory

        Duplicate PDFs are automatically skipped.
        """

        papers_dir = Path(
            settings.papers_dir
        )

        pdf_files = sorted(
            papers_dir.glob("*.pdf")
        )

        if not pdf_files:

            print(
                f"No PDFs found in "
                f"{papers_dir.resolve()}"
            )

            return []

        print()
        print("=" * 70)
        print(
            f"FOUND {len(pdf_files)} PDF(S)"
        )
        print("=" * 70)

        results = []

        success = 0
        skipped = 0
        failed = 0

        for pdf_path in pdf_files:

            try:

                result = (
                    self.process_single_pdf(
                        pdf_path
                    )
                )

                results.append(
                    result
                )

                if result.get(
                    "skipped"
                ):
                    skipped += 1
                else:
                    success += 1

            except Exception as exc:

                failed += 1

                results.append(
                    {
                        "skipped": False,
                        "filename": pdf_path.name,
                        "error": str(exc),
                    }
                )

        print()
        print("=" * 70)
        print("INGESTION COMPLETE")
        print("=" * 70)
        print(
            f"Processed : {success}"
        )
        print(
            f"Skipped   : {skipped}"
        )
        print(
            f"Failed    : {failed}"
        )
        print()
        print(
            f"Canonical : "
            f"{settings.canonical_dir}"
        )
        print(
            f"Evaluation: "
            f"{settings.evaluation_dir}"
        )
        print(
            f"Artifacts : "
            f"{settings.artifacts_dir}"
        )
        print(
            f"GROBID   : "
            f"{settings.grobid_dir}"
        )
        print("=" * 70)

        return results

    # =========================================================
    # REGISTRY
    # =========================================================

    @staticmethod
    def _get_registry_record(
        paper_id: str,
    ) -> dict:

        registry = load_registry()

        return registry.get(
            paper_id,
            {},
        )

    @staticmethod
    def _update_registry_status(
        paper_id: str,
        status: str,
        error: str | None = None,
    ) -> None:

        registry = load_registry()

        if paper_id not in registry:
            return

        registry[paper_id]["status"] = status

        if error:
            registry[paper_id][
                "error"
            ] = error

        else:
            registry[paper_id].pop(
                "error",
                None,
            )

        save_registry(
            registry
        )

    def _update_registry_after_success(
        self,
        *,
        paper_id: str,
        canonical: dict,
        pymupdf: dict,
        quality: dict,
    ) -> None:

        registry = load_registry()

        if paper_id not in registry:
            return

        metadata = canonical.get(
            "metadata",
            {},
        )

        registry[paper_id].update(
            {
                "status": (
                    "extracted"
                ),
                "title": (
                    metadata.get(
                        "title"
                    )
                ),
                "authors": (
                    metadata.get(
                        "authors"
                    )
                ),
                "year": (
                    metadata.get(
                        "year"
                    )
                ),
                "num_pages": (
                    pymupdf.get(
                        "num_pages"
                    )
                ),
                "quality_score": (
                    quality.get(
                        "integrity_score"
                    )
                ),
                "quality_status": (
                    quality.get(
                        "status"
                    )
                ),
                "num_tables": (
                    quality.get(
                        "artifact_counts",
                        {},
                    ).get(
                        "tables",
                        0,
                    )
                ),
                "num_figures": (
                    quality.get(
                        "artifact_counts",
                        {},
                    ).get(
                        "figures",
                        0,
                    )
                ),
            }
        )

        save_registry(
            registry
        )

    # =========================================================
    # YEAR
    # =========================================================

    @staticmethod
    def _extract_pdf_year(
        pymupdf: dict,
    ) -> int | None:

        metadata = pymupdf.get(
            "metadata",
            {},
        )

        value = metadata.get(
            "creation_date"
        )

        if not value:
            return None

        match = re.match(
            r"^D:(\d{4})",
            str(value),
        )

        if match:
            return int(
                match.group(1)
            )

        return None

    # =========================================================
    # OUTPUTS
    # =========================================================

    def _save_outputs(
        self,
        *,
        paper_id: str,
        pdf_path: Path,
        tei_xml: str,
        raw_docling: dict,
        docling_metadata: dict,
        canonical: dict,
        quality: dict,
    ) -> None:

        # -----------------------------------------------------
        # GROBID TEI
        # -----------------------------------------------------

        grobid_path = (
            Path(
                settings.grobid_dir
            )
            / f"{paper_id}.tei.xml"
        )

        grobid_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        grobid_path.write_text(
            tei_xml,
            encoding="utf-8",
        )

        # -----------------------------------------------------
        # DOCLING RAW DOCUMENT
        # -----------------------------------------------------

        artifacts_path = (
            Path(
                settings.artifacts_dir
            )
            / f"{paper_id}_docling.json"
        )

        self._save_json(
            artifacts_path,
            {
                "paper_id": paper_id,
                "filename": pdf_path.name,
                "metadata": docling_metadata,
                "docling_document": raw_docling,
            },
        )

        # -----------------------------------------------------
        # CANONICAL DOCUMENT
        # -----------------------------------------------------

        canonical_path = (
            Path(
                settings.canonical_dir
            )
            / f"{paper_id}.json"
        )

        self._save_json(
            canonical_path,
            canonical,
        )

        # -----------------------------------------------------
        # EVALUATION
        # -----------------------------------------------------

        evaluation_path = (
            Path(
                settings.evaluation_dir
            )
            / f"{paper_id}_quality.json"
        )

        self._save_json(
            evaluation_path,
            quality,
        )

    # =========================================================
    # JSON
    # =========================================================

    @staticmethod
    def _save_json(
        path: Path,
        data: dict,
    ) -> None:

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=2,
            )

    # =========================================================
    # DISPLAY
    # =========================================================

    @staticmethod
    def _print_result(
        canonical: dict,
    ) -> None:

        metadata = canonical.get(
            "metadata",
            {},
        )

        quality = canonical.get(
            "quality",
            {},
        )

        artifacts = quality.get(
            "artifact_counts",
            {},
        )

        authors = metadata.get(
            "authors",
            [],
        )

        print()
        print("-" * 70)
        print(
            f"Paper ID   : "
            f"{canonical.get('paper_id')}"
        )
        print(
            f"Title      : "
            f"{metadata.get('title')}"
        )
        print(
            f"Authors    : "
            f"{len(authors)}"
        )
        print(
            f"Year       : "
            f"{metadata.get('year')}"
        )
        print(
            f"Sections   : "
            f"{len(canonical.get('sections', []))}"
        )
        print(
            f"References : "
            f"{len(canonical.get('references', []))}"
        )
        print(
            f"Tables     : "
            f"{artifacts.get('tables', 0)}"
        )
        print(
            f"Figures    : "
            f"{artifacts.get('figures', 0)}"
        )
        print(
            f"Formulas   : "
            f"{artifacts.get('formulas', 0)}"
        )
        print(
            f"Quality    : "
            f"{quality.get('integrity_score', 0):.3f}"
        )
        print(
            f"Status     : "
            f"{quality.get('status', 'unknown')}"
        )

        issues = quality.get(
            "issues",
            [],
        )

        if issues:

            print(
                "Issues     : "
                + ", ".join(
                    issues
                )
            )

        print("-" * 70)


def process_single_pdf(
    file_path: Path,
) -> dict:
    """
    Convenience function so other modules, including the
    Streamlit upload interface later, can simply call:

        process_single_pdf(pdf_path)
    """

    pipeline = IngestionPipeline()

    return pipeline.process_single_pdf(
        file_path
    )


def run_ingestion() -> list[dict]:
    """
    Convenience function for processing all PDFs in papers/.
    """

    pipeline = IngestionPipeline()

    return pipeline.run_ingestion()


if __name__ == "__main__":

    run_ingestion()