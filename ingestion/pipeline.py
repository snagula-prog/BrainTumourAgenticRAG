from __future__ import annotations

import argparse
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

from evaluation.artifact_quality import (
    ArtifactQualityEvaluator,
)


class IngestionPipeline:
    """
    Complete PDF ingestion pipeline.

    Normal ingestion:
        duplicate PDF -> skip

    Reprocessing:
        duplicate PDF -> reuse existing paper_id -> rebuild in place

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
        preliminary extraction quality
          ↓
        canonical document
          ↓
        canonical artifact quality evaluation
          ↓
        final canonical document
          ↓
        registry update

    Chunking, embeddings and retrieval remain outside this pipeline.
    """

    def __init__(self) -> None:

        self.grobid = GrobidParser(
            base_url=(
                settings.grobid_base_url
            ),
            timeout_seconds=(
                settings.grobid_timeout
            ),
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

        self.artifacts = (
            DoclingArtifactExtractor()
        )

        self.quality = (
            QualityMetrics()
        )

        self.builder = (
            CanonicalBuilder()
        )

        self.artifact_quality = (
            ArtifactQualityEvaluator()
        )

    # =========================================================
    # PUBLIC API
    # =========================================================

    def process_single_pdf(
        self,
        pdf_path: Path,
        *,
        force_reprocess: bool = False,
    ) -> dict:

        pdf_path = (
            Path(pdf_path)
            .resolve()
        )

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
        print(
            f"[INGEST] {pdf_path.name}"
        )
        print("=" * 70)

        # -----------------------------------------------------
        # 1. HASH + DUPLICATE CHECK
        # -----------------------------------------------------

        print(
            "[1/6] Checking duplicate..."
        )

        file_hash = compute_file_hash(
            pdf_path
        )

        registry = load_registry()

        existing_paper_id = (
            find_by_hash(
                registry,
                file_hash,
            )
        )

        if (
            existing_paper_id
            and not force_reprocess
        ):

            existing_record = registry[
                existing_paper_id
            ]

            print(
                f"[SKIP] Already registered as "
                f"{existing_paper_id}"
            )

            return {
                "skipped": True,
                "paper_id": (
                    existing_paper_id
                ),
                "record": existing_record,
            }

        if (
            existing_paper_id
            and force_reprocess
        ):

            paper_id = (
                existing_paper_id
            )

            print(
                f"[REPROCESS] Reusing existing "
                f"paper_id: {paper_id}"
            )

            self._update_registry_status(
                paper_id,
                "processing",
            )

        else:

            print(
                "[2/6] Registering paper..."
            )

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

            print(
                "[3/6] GROBID..."
            )

            tei_xml = (
                self.grobid.process_pdf(
                    pdf_path
                )
            )

            grobid = parse_grobid_tei(
                tei_xml
            )

            # -------------------------------------------------
            # 4. PYMuPDF
            # -------------------------------------------------

            print(
                "[4/6] PyMuPDF audit..."
            )

            pymupdf = (
                PyMuPDFAudit(
                    pdf_path
                ).extract()
            )

            # -------------------------------------------------
            # 5. DOCLING
            # -------------------------------------------------

            print(
                "[5/6] Docling artifacts..."
            )

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
                    raw_docling,
                    fallback_pdf_path=pdf_path,
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
                grobid["year"] = (
                    pdf_year
                )

            # -------------------------------------------------
            # 6. PRELIMINARY QUALITY
            # -------------------------------------------------

            print(
                "[6/6] Evaluating extraction quality..."
            )

            quality = (
                self.quality.evaluate(
                    grobid=grobid,
                    pymupdf=pymupdf,
                    artifacts=artifacts,
                )
            )

            # -------------------------------------------------
            # PROVISIONAL CANONICAL DOCUMENT
            # -------------------------------------------------

            canonical = (
                self.builder.build(
                    paper_id=paper_id,
                    filename=pdf_path.name,
                    grobid=grobid,
                    pymupdf=pymupdf,
                    artifacts=artifacts,
                    quality=quality,
                )
            )

            # -------------------------------------------------
            # CANONICAL COUNTS + ARTIFACT QUALITY
            # -------------------------------------------------

            canonical_counts = self._canonical_counts(
                canonical
            )

            artifact_integrity = (
                self.artifact_quality.evaluate(
                    canonical
                )
            )

            # Keep parser/extraction counts from QualityMetrics
            # separate from the counts that survived into the
            # final canonical representation.
            quality["canonical_counts"] = (
                canonical_counts
            )

            quality["artifact_integrity"] = (
                artifact_integrity
            )

            quality["issues"] = self._merge_issues(
                quality.get("issues", []),
                artifact_integrity.get("issues", []),
            )

            # Canonical must contain the final quality object.
            canonical["quality"] = (
                quality
            )

            # -------------------------------------------------
            # SAVE EVERYTHING
            # -------------------------------------------------

            self._save_outputs(
                paper_id=paper_id,
                pdf_path=pdf_path,
                tei_xml=tei_xml,
                raw_docling=raw_docling,
                docling_metadata=(
                    docling_metadata
                ),
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
                "record": (
                    self._get_registry_record(
                        paper_id
                    )
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
        *,
        force_reprocess: bool = False,
    ) -> list[dict]:

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

        results: list[dict] = []

        success = 0
        skipped = 0
        failed = 0

        for pdf_path in pdf_files:

            try:

                result = (
                    self.process_single_pdf(
                        pdf_path,
                        force_reprocess=(
                            force_reprocess
                        ),
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
                        "filename": (
                            pdf_path.name
                        ),
                        "error": str(exc),
                    }
                )

        print()
        print("=" * 70)
        print(
            "INGESTION COMPLETE"
        )
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

        registry[
            paper_id
        ][
            "status"
        ] = status

        if error:

            registry[
                paper_id
            ][
                "error"
            ] = error

        else:

            registry[
                paper_id
            ].pop(
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

        canonical_counts = quality.get(
            "canonical_counts",
            {},
        )

        registry[
            paper_id
        ].update(
            {
                "status": "extracted",

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
                    canonical_counts.get(
                        "tables",
                        0,
                    )
                ),

                "num_figures": (
                    canonical_counts.get(
                        "figures",
                        0,
                    )
                ),

                "num_formulas": (
                    canonical_counts.get(
                        "formulas",
                        0,
                    )
                ),
            }
        )

        save_registry(
            registry
        )

    # =========================================================
    # CANONICAL COUNTS / ISSUES
    # =========================================================

    @staticmethod
    def _count_records(value: object) -> int:
        return len(value) if isinstance(value, list) else 0

    @classmethod
    def _canonical_counts(
        cls,
        canonical: dict,
    ) -> dict[str, int]:

        return {
            "sections": cls._count_records(
                canonical.get("sections")
            ),
            "paragraphs": cls._count_records(
                canonical.get("paragraphs")
            ),
            "references": cls._count_records(
                canonical.get("references")
            ),
            "figures": cls._count_records(
                canonical.get("figures")
            ),
            "tables": cls._count_records(
                canonical.get("tables")
            ),
            "formulas": cls._count_records(
                canonical.get("formulas")
            ),
        }

    @staticmethod
    def _merge_issues(
        existing: object,
        added: object,
    ) -> list[str]:

        merged: list[str] = []

        for source in (existing, added):
            if not isinstance(source, list):
                continue

            for issue in source:
                if not isinstance(issue, str):
                    continue
                if issue not in merged:
                    merged.append(issue)

        return merged

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
                "filename": (
                    pdf_path.name
                ),
                "metadata": (
                    docling_metadata
                ),
                "docling_document": (
                    raw_docling
                ),
            },
        )

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

        canonical_counts = quality.get(
            "canonical_counts",
            {},
        )

        artifact_integrity = quality.get(
            "artifact_integrity",
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
            f"{canonical_counts.get('sections', 0)}"
        )

        print(
            f"Paragraphs : "
            f"{canonical_counts.get('paragraphs', 0)}"
        )

        print(
            f"References : "
            f"{canonical_counts.get('references', 0)}"
        )

        print(
            f"Tables     : "
            f"{canonical_counts.get('tables', 0)}"
        )

        print(
            f"Figures    : "
            f"{canonical_counts.get('figures', 0)}"
        )

        print(
            f"Formulas   : "
            f"{canonical_counts.get('formulas', 0)}"
        )

        print(
            f"Quality    : "
            f"{quality.get('integrity_score', 0):.3f}"
        )

        print(
            f"Status     : "
            f"{quality.get('status', 'unknown')}"
        )

        print(
            f"Artifact   : "
            f"{artifact_integrity.get('status', 'unknown')}"
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
    *,
    force_reprocess: bool = False,
) -> dict:

    pipeline = (
        IngestionPipeline()
    )

    return (
        pipeline.process_single_pdf(
            file_path,
            force_reprocess=(
                force_reprocess
            ),
        )
    )


def run_ingestion(
    *,
    force_reprocess: bool = False,
) -> list[dict]:

    pipeline = (
        IngestionPipeline()
    )

    return (
        pipeline.run_ingestion(
            force_reprocess=(
                force_reprocess
            ),
        )
    )


def _parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Brain Tumor paper ingestion pipeline"
        )
    )

    parser.add_argument(
        "--reprocess",
        action="store_true",
        help=(
            "Reprocess already-registered PDFs "
            "in place using their existing paper_id."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = _parse_args()

    run_ingestion(
        force_reprocess=(
            args.reprocess
        )
    )
