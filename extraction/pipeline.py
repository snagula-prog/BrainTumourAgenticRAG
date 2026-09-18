# extraction/pipeline.py

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from config.settings import settings

from extraction.deduplication import (
    compute_file_hash,
    load_registry,
    find_by_hash,
    register_new_paper,
    save_registry,
)

from extraction.grobid_parser import (
    GrobidParser,
    parse_grobid_tei,
)

from extraction.pymupdf_audit import (
    PyMuPDFAudit,
)

from extraction.docling_parser import (
    DoclingParser,
)

from extraction.docling_artifacts import (
    DoclingArtifactExtractor,
)

from extraction.canonical_builder import (
    CanonicalBuilder,
)

from evaluation.extraction_metrics.extraction_quality_metrics import (
    QualityMetrics,
)


class ExtractionPipeline:
    """
    Complete PDF extraction pipeline.

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

        # Normal extraction deliberately keeps formula enrichment off.
        # Formula enrichment is an explicit, cached one-time stage.
        self.docling = DoclingParser(
            do_formula_enrichment=False,
            do_table_structure=True,
            do_ocr=False,
        )

        self.artifacts = DoclingArtifactExtractor()

        self.quality = QualityMetrics()

        self.builder = CanonicalBuilder()

    # =========================================================
    # PIPELINE STAGES
    # =========================================================

    STAGES = (
        "grobid",
        "pymupdf",
        "docling",
        "formula",
        "canonical",
        "evaluation",
    )

    STAGE_ALIASES = {
        "reconstruction": "canonical",
        "artifacts": "docling",
        "formulas": "formula",
        "quality": "evaluation",
    }

    # =========================================================
    # PUBLIC API
    # =========================================================

    def process_single_pdf(
        self,
        pdf_path: Path,
        *,
        force: bool = False,
        reprocess_existing: bool = False,
        from_stage: str | None = None,
        enrich_formulas: bool = False,
    ) -> dict:
        """
        Process one PDF with optional stage-aware reuse.

        Normal mode:
            new PDF -> ingest; existing PDF -> skip.

        Reprocess mode:
            reuse completed parser stages and rerun downstream stages.

        Force mode:
            ignore caches and rebuild every stage.

        ``from_stage`` selects the first stage to rebuild while
        reusing valid prerequisite caches.
        """

        pdf_path = Path(
            pdf_path
        ).resolve()

        if not pdf_path.exists():
            raise FileNotFoundError(
                f"PDF not found: {pdf_path}"
            )

        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(
                f"Expected a PDF file, got: "
                f"{pdf_path.name}"
            )

        normalized_stage = (
            self._normalize_stage(
                from_stage
            )
            if from_stage
            else None
        )

        if force:
            requested_stage = "grobid"
        elif normalized_stage:
            requested_stage = normalized_stage
        elif enrich_formulas:
            requested_stage = "formula"
        elif reprocess_existing:
            requested_stage = "canonical"
        else:
            requested_stage = "grobid"

        print()
        print("=" * 70)
        print(f"[EXTRACT] {pdf_path.name}")
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

        if existing_paper_id and not (
            force
            or reprocess_existing
            or normalized_stage
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
                "paper_id": existing_paper_id,
                "record": existing_record,
            }

        # -----------------------------------------------------
        # 2. REGISTER OR REUSE PAPER ID
        # -----------------------------------------------------

        if existing_paper_id:

            paper_id = existing_paper_id

            print(
                f"[REPROCESS] Existing paper "
                f"{paper_id}"
            )

        else:

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

        if force:

            start_stage = "grobid"

        else:

            start_stage = (
                self._resolve_start_stage(
                    paper_id,
                    requested_stage,
                )
            )

        if start_stage != requested_stage:

            print(
                f"[RESUME] Requested stage "
                f"'{requested_stage}', but earliest "
                f"missing prerequisite is "
                f"'{start_stage}'."
            )

        print(
            f"[PLAN] Starting from stage: "
            f"{start_stage}"
        )

        start_rank = self.STAGES.index(
            start_stage
        )

        def should_run(
            stage: str,
        ) -> bool:

            return (
                self.STAGES.index(stage)
                >= start_rank
            )

        current_stage = start_stage

        try:

            # -------------------------------------------------
            # 3. GROBID
            # -------------------------------------------------

            current_stage = "grobid"

            print("[3/6] GROBID...")

            if should_run("grobid"):

                tei_xml = (
                    self.grobid.process_pdf(
                        pdf_path
                    )
                )

                self._save_grobid_cache(
                    paper_id=paper_id,
                    tei_xml=tei_xml,
                )

                print(
                    "[CACHE] Saved GROBID TEI"
                )

            else:

                tei_xml = (
                    self._load_grobid_cache(
                        paper_id
                    )
                )

                print(
                    "[CACHE] Loaded GROBID TEI"
                )

            grobid = parse_grobid_tei(
                tei_xml
            )

            # -------------------------------------------------
            # 4. PYMUPDF
            # -------------------------------------------------

            current_stage = "pymupdf"

            print(
                "[4/6] PyMuPDF audit..."
            )

            if should_run("pymupdf"):

                pymupdf = (
                    PyMuPDFAudit(
                        pdf_path
                    ).extract()
                )

                self._save_pymupdf_cache(
                    paper_id=paper_id,
                    pdf_path=pdf_path,
                    file_hash=file_hash,
                    pymupdf=pymupdf,
                )

                print(
                    "[CACHE] Saved PyMuPDF audit"
                )

            else:

                pymupdf = (
                    self._load_pymupdf_cache(
                        paper_id
                    )
                )

                print(
                    "[CACHE] Loaded PyMuPDF audit"
                )

            # -------------------------------------------------
            # 5. DOCLING + ARTIFACT EXTRACTION
            # -------------------------------------------------

            current_stage = "docling"

            print(
                "[5/6] Docling artifacts..."
            )

            if should_run("docling"):

                (
                    document,
                    docling_metadata,
                ) = self.docling.parse(
                    pdf_path
                )

                raw_docling = (
                    document.export_to_dict()
                )

                self._save_docling_cache(
                    paper_id=paper_id,
                    pdf_path=pdf_path,
                    file_hash=file_hash,
                    raw_docling=raw_docling,
                    docling_metadata=docling_metadata,
                )

                # Formula output depends on the Docling base artifact.
                # Rebuilds invalidate the old enrichment cache.
                self._delete_formula_cache(paper_id)

                print(
                    "[CACHE] Saved Docling artifact"
                )

            else:

                (
                    raw_docling,
                    docling_metadata,
                ) = self._load_docling_cache(
                    paper_id
                )

                print(
                    "[CACHE] Loaded Docling artifact"
                )

            # -------------------------------------------------
            # 5.1 FORMULA ENRICHMENT (OPTIONAL / CACHED)
            # -------------------------------------------------

            current_stage = "formula"

            formula_cache_exists = self._stage_cache_exists(
                paper_id,
                "formula",
            )

            formula_requested = (
                normalized_stage == "formula"
                or enrich_formulas
            )

            if should_run("formula") and formula_requested:

                if force or normalized_stage == "formula" or not formula_cache_exists:
                    print(
                        "[5.1/6] Formula enrichment (one-time)..."
                    )

                    formula_parser = DoclingParser(
                        do_formula_enrichment=True,
                        do_table_structure=True,
                        do_ocr=False,
                    )

                    (
                        formula_document,
                        formula_metadata,
                    ) = formula_parser.parse(
                        pdf_path
                    )

                    formula_docling = formula_document.export_to_dict()

                    self._save_formula_cache(
                        paper_id=paper_id,
                        pdf_path=pdf_path,
                        file_hash=file_hash,
                        raw_docling=formula_docling,
                        docling_metadata=formula_metadata,
                    )

                    raw_docling = formula_docling
                    docling_metadata = formula_metadata

                    print(
                        "[CACHE] Saved formula-enriched Docling artifact"
                    )

                else:
                    (
                        raw_docling,
                        docling_metadata,
                    ) = self._load_formula_cache(
                        paper_id
                    )

                    print(
                        "[CACHE] Loaded formula-enriched Docling artifact"
                    )

            elif formula_cache_exists:

                (
                    raw_docling,
                    docling_metadata,
                ) = self._load_formula_cache(
                    paper_id
                )

                print(
                    "[CACHE] Using existing formula-enriched Docling artifact"
                )

            # Artifact extraction is cheap and deterministic, so
            # it is rebuilt from the selected Docling cache.
            artifacts = (
                self.artifacts.extract(
                    raw_docling
                )
            )

            text_blocks = artifacts.get(
                "text_blocks",
                [],
            )

            print(
                "[DEBUG TEXT BLOCK]",
                json.dumps(
                    (
                        text_blocks[0]
                        if text_blocks
                        else {
                            "error":
                                "No text_blocks found"
                        }
                    ),
                    indent=2,
                    default=str,
                ),
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
            # 6. QUALITY + CANONICAL
            # -------------------------------------------------

            if should_run("canonical"):

                current_stage = "evaluation"

                print(
                    "[6/6] Evaluating extraction quality..."
                )

                quality = self.quality.evaluate(
                    grobid=grobid,
                    pymupdf=pymupdf,
                    artifacts=artifacts,
                )

                # Persist the evaluation before canonical building,
                # so an interrupted canonical stage still leaves the
                # expensive parser work and quality result available.
                self._save_quality_cache(
                    paper_id=paper_id,
                    quality=quality,
                )

                current_stage = "canonical"

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

                self._save_canonical_cache(
                    canonical=canonical,
                )

            elif should_run("evaluation"):

                current_stage = "evaluation"

                print(
                    "[6/6] Evaluating extraction quality..."
                )

                quality = self.quality.evaluate(
                    grobid=grobid,
                    pymupdf=pymupdf,
                    artifacts=artifacts,
                )

                canonical = (
                    self._load_canonical_cache(
                        paper_id
                    )
                )

                canonical["quality"] = quality

                self._save_quality_cache(
                    paper_id=paper_id,
                    quality=quality,
                )

                self._save_canonical_cache(
                    canonical=canonical,
                )

            else:

                canonical = (
                    self._load_canonical_cache(
                        paper_id
                    )
                )

                quality = canonical.get(
                    "quality"
                )

                if quality is None:

                    quality = (
                        self._load_quality_cache(
                            paper_id
                        )
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
                "start_stage": start_stage,
            }

        except Exception as exc:

            self._update_registry_status(
                paper_id,
                "failed",
                error=(
                    f"[{current_stage}] "
                    f"{exc}"
                ),
            )

            print(
                f"[FAILED] {paper_id} "
                f"at {current_stage}: "
                f"{type(exc).__name__}: {exc}"
            )

            raise


    # =========================================================
    # PROCESS ALL PDFs
    # =========================================================

    def run_extraction(
        self,
        *,
        file_path: Path | None = None,
        paper_id: str | None = None,
        force: bool = False,
        reprocess_existing: bool = False,
        from_stage: str | None = None,
        enrich_formulas: bool = False,
    ) -> list[dict]:
        """
        Extract all PDFs, or one specifically selected PDF/paper.

        ``file_path`` and ``paper_id`` are mutually exclusive.
        """

        if (
            file_path is not None
            and paper_id is not None
        ):
            raise ValueError(
                "Use either file_path or paper_id, "
                "not both."
            )

        if paper_id is not None:

            registry = load_registry()

            record = registry.get(
                paper_id
            )

            if not record:
                raise ValueError(
                    f"Unknown paper_id: {paper_id}"
                )

            filename = record.get(
                "filename"
            )

            if not filename:
                raise ValueError(
                    f"Registry record for {paper_id} "
                    "has no filename."
                )

            file_path = (
                Path(
                    settings.raw_pds_dir
                )
                / filename
            )

            if not file_path.exists():
                raise FileNotFoundError(
                    f"Registered PDF not found: "
                    f"{file_path}"
                )

            registered_hash = record.get(
                "file_hash"
            )

            current_hash = compute_file_hash(
                file_path
            )

            if (
                registered_hash
                and current_hash != registered_hash
            ):
                raise ValueError(
                    f"File hash mismatch for "
                    f"{paper_id}. The PDF appears "
                    "to have changed."
                )

        if file_path is not None:

            pdf_files = [
                Path(
                    file_path
                ).resolve()
            ]

        else:

            raw_pds_dir = Path(
                settings.raw_pds_dir
            )

            pdf_files = sorted(
                raw_pds_dir.glob("*.pdf")
            )

        if not pdf_files:

            print(
                f"No PDFs found in "
                f"{Path(settings.raw_pds_dir).resolve()}"
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
                        pdf_path,
                        force=force,
                        reprocess_existing=reprocess_existing,
                        from_stage=from_stage,
                        enrich_formulas=enrich_formulas,
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
        print("EXTRACTION COMPLETE")
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
                    len(
                        canonical.get(
                            "tables",
                            [],
                        )
                    )
                ),
                "num_figures": (
                    len(
                        canonical.get(
                            "figures",
                            [],
                        )
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

    def _save_grobid_cache(
        self,
        *,
        paper_id: str,
        tei_xml: str,
    ) -> None:

        path = (
            Path(
                settings.grobid_dir
            )
            / f"{paper_id}.tei.xml"
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        path.write_text(
            tei_xml,
            encoding="utf-8",
        )

    def _load_grobid_cache(
        self,
        paper_id: str,
    ) -> str:

        path = (
            Path(
                settings.grobid_dir
            )
            / f"{paper_id}.tei.xml"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Missing GROBID cache: {path}"
            )

        return path.read_text(
            encoding="utf-8"
        )

    def _save_pymupdf_cache(
        self,
        *,
        paper_id: str,
        pdf_path: Path,
        file_hash: str,
        pymupdf: dict,
    ) -> None:

        path = (
            Path(
                settings.artifacts_dir
            )
            / f"{paper_id}_pymupdf.json"
        )

        self._save_json(
            path,
            {
                "paper_id": paper_id,
                "filename": pdf_path.name,
                "file_hash": file_hash,
                "pymupdf": pymupdf,
            },
        )

    def _load_pymupdf_cache(
        self,
        paper_id: str,
    ) -> dict:

        path = (
            Path(
                settings.artifacts_dir
            )
            / f"{paper_id}_pymupdf.json"
        )

        data = self._load_json(
            path
        )

        if "pymupdf" in data:
            return data["pymupdf"]

        return data

    def _save_docling_cache(
        self,
        *,
        paper_id: str,
        pdf_path: Path,
        file_hash: str,
        raw_docling: dict,
        docling_metadata: dict,
    ) -> None:

        path = (
            Path(
                settings.artifacts_dir
            )
            / f"{paper_id}_docling.json"
        )

        self._save_json(
            path,
            {
                "paper_id": paper_id,
                "filename": pdf_path.name,
                "file_hash": file_hash,
                "metadata": docling_metadata,
                "docling_document": raw_docling,
            },
        )

    def _load_docling_cache(
        self,
        paper_id: str,
    ) -> tuple[dict, dict]:

        path = (
            Path(
                settings.artifacts_dir
            )
            / f"{paper_id}_docling.json"
        )

        data = self._load_json(
            path
        )

        return (
            data.get(
                "docling_document",
                {},
            ),
            data.get(
                "metadata",
                {},
            ),
        )

    def _save_formula_cache(
        self,
        *,
        paper_id: str,
        pdf_path: Path,
        file_hash: str,
        raw_docling: dict,
        docling_metadata: dict,
    ) -> None:

        path = (
            Path(
                settings.artifacts_dir
            )
            / f"{paper_id}_docling_formula.json"
        )

        self._save_json(
            path,
            {
                "paper_id": paper_id,
                "filename": pdf_path.name,
                "file_hash": file_hash,
                "formula_enriched": True,
                "metadata": docling_metadata,
                "docling_document": raw_docling,
            },
        )

    def _load_formula_cache(
        self,
        paper_id: str,
    ) -> tuple[dict, dict]:

        path = (
            Path(
                settings.artifacts_dir
            )
            / f"{paper_id}_docling_formula.json"
        )

        data = self._load_json(
            path
        )

        return (
            data.get(
                "docling_document",
                {},
            ),
            data.get(
                "metadata",
                {},
            ),
        )

    @staticmethod
    def _delete_formula_cache(
        paper_id: str,
    ) -> None:

        path = (
            Path(
                settings.artifacts_dir
            )
            / f"{paper_id}_docling_formula.json"
        )

        if path.exists():
            path.unlink()

    def _save_canonical_cache(
        self,
        *,
        canonical: dict,
    ) -> None:

        paper_id = canonical.get(
            "paper_id"
        )

        if not paper_id:
            raise ValueError(
                "Canonical document has no paper_id."
            )

        path = (
            Path(
                settings.canonical_dir
            )
            / f"{paper_id}.json"
        )

        self._save_json(
            path,
            canonical,
        )

    def _load_canonical_cache(
        self,
        paper_id: str,
    ) -> dict:

        path = (
            Path(
                settings.canonical_dir
            )
            / f"{paper_id}.json"
        )

        return self._load_json(
            path
        )

    def _save_quality_cache(
        self,
        *,
        paper_id: str,
        quality: dict,
    ) -> None:

        path = (
            Path(
                settings.extraction_metrics_dir
            )
            / f"{paper_id}_quality.json"
        )

        self._save_json(
            path,
            quality,
        )

    def _load_quality_cache(
        self,
        paper_id: str,
    ) -> dict:

        path = (
            Path(
                settings.extraction_metrics_dir
            )
            / f"{paper_id}_quality.json"
        )

        return self._load_json(
            path
        )

    # =========================================================
    # CACHE / STAGE HELPERS
    # =========================================================

    @classmethod
    def _normalize_stage(
        cls,
        stage: str,
    ) -> str:

        normalized = (
            str(stage)
            .strip()
            .lower()
        )

        normalized = cls.STAGE_ALIASES.get(
            normalized,
            normalized,
        )

        if normalized not in cls.STAGES:

            supported = ", ".join(
                (
                    *cls.STAGES,
                    *cls.STAGE_ALIASES.keys(),
                )
            )

            raise ValueError(
                f"Unknown stage '{stage}'. "
                f"Supported stages: {supported}"
            )

        return normalized

    @classmethod
    def _resolve_start_stage(
        cls,
        paper_id: str,
        requested_stage: str,
    ) -> str:

        prerequisites = {
            "grobid": [],
            "pymupdf": [
                "grobid",
            ],
            "docling": [
                "grobid",
                "pymupdf",
            ],
            "formula": [
                "grobid",
                "pymupdf",
                "docling",
            ],
            "canonical": [
                "grobid",
                "pymupdf",
                "docling",
            ],
            "evaluation": [
                "grobid",
                "pymupdf",
                "docling",
                "canonical",
            ],
        }

        for stage in prerequisites[
            requested_stage
        ]:

            if not cls._stage_cache_exists(
                paper_id,
                stage,
            ):
                return stage

        return requested_stage

    @staticmethod
    def _stage_cache_exists(
        paper_id: str,
        stage: str,
    ) -> bool:

        if stage == "grobid":

            return (
                Path(
                    settings.grobid_dir
                )
                / f"{paper_id}.tei.xml"
            ).exists()

        if stage == "pymupdf":

            return (
                Path(
                    settings.artifacts_dir
                )
                / f"{paper_id}_pymupdf.json"
            ).exists()

        if stage == "docling":

            return (
                Path(
                    settings.artifacts_dir
                )
                / f"{paper_id}_docling.json"
            ).exists()

        if stage == "formula":

            return (
                Path(
                    settings.artifacts_dir
                )
                / f"{paper_id}_docling_formula.json"
            ).exists()

        if stage == "canonical":

            return (
                Path(
                    settings.canonical_dir
                )
                / f"{paper_id}.json"
            ).exists()

        if stage == "evaluation":

            return (
                Path(
                    settings.extraction_metrics_dir
                )
                / f"{paper_id}_quality.json"
            ).exists()

        return False

    # =========================================================
    # JSON
    # =========================================================

    @staticmethod
    def _load_json(
        path: Path,
    ) -> dict:

        if not path.exists():
            raise FileNotFoundError(
                f"Missing cache artifact: {path}"
            )

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(
                file
            )

        if not isinstance(
            data,
            dict,
        ):
            raise ValueError(
                f"Expected a JSON object in {path}"
            )

        return data

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

        logical_tables = canonical.get(
            "tables",
            [],
        )

        logical_figures = canonical.get(
            "figures",
            [],
        )

        logical_formulas = canonical.get(
            "formulas",
            [],
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
            f"{len(logical_tables)}"
        )
        print(
            f"Figures    : "
            f"{len(logical_figures)}"
        )
        print(
            f"Formulas   : "
            f"{len(logical_formulas)}"
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
    *,
    force: bool = False,
    reprocess_existing: bool = False,
    from_stage: str | None = None,
    enrich_formulas: bool = False,
) -> dict:
    """
    Convenience function for extracting one PDF.
    """

    pipeline = ExtractionPipeline()

    return pipeline.process_single_pdf(
        file_path,
        force=force,
        reprocess_existing=reprocess_existing,
        from_stage=from_stage,
        enrich_formulas=enrich_formulas,
    )


def run_extraction(
    *,
    file_path: Path | None = None,
    paper_id: str | None = None,
    force: bool = False,
    reprocess_existing: bool = False,
    from_stage: str | None = None,
    enrich_formulas: bool = False,
) -> list[dict]:
    """
    Convenience function for batch or targeted extraction.
    """

    pipeline = ExtractionPipeline()

    return pipeline.run_extraction(
        file_path=file_path,
        paper_id=paper_id,
        force=force,
        reprocess_existing=reprocess_existing,
        from_stage=from_stage,
        enrich_formulas=enrich_formulas,
    )


def _parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Research Paper Agentic RAG "
            "extraction pipeline"
        )
    )

    selection = parser.add_mutually_exclusive_group()

    selection.add_argument(
        "--file",
        type=Path,
        help="Process one specific PDF.",
    )

    selection.add_argument(
        "--paper-id",
        help="Process one registered paper by paper_id.",
    )

    parser.add_argument(
        "--reprocess-existing",
        action="store_true",
        help=(
            "Reuse completed extraction stages and "
            "rerun downstream processing."
        ),
    )
    

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Ignore all existing caches and "
            "rebuild every base extraction stage."
        ),
    )

    parser.add_argument(
        "--enrich-formulas",
        action="store_true",
        help=(
            "Run formula enrichment once and cache the "
            "enriched Docling artifact. Reusing the paper "
            "does not repeat this expensive pass."
        ),
    )

    parser.add_argument(
        "--from-stage",
        choices=(
            "grobid",
            "pymupdf",
            "docling",
            "formula",
            "formulas",
            "artifacts",
            "reconstruction",
            "canonical",
            "quality",
            "evaluation",
        ),
        help=(
            "Rebuild starting at this stage while "
            "reusing valid prerequisites."
        ),
    )

    args = parser.parse_args()

    if args.force and args.reprocess_existing:
        print(
            "[INFO] --force overrides "
            "--reprocess-existing"
        )

    return args


if __name__ == "__main__":

    args = _parse_args()

    run_extraction(
        file_path=args.file,
        paper_id=args.paper_id,
        force=args.force,
        reprocess_existing=args.reprocess_existing,
        from_stage=args.from_stage,
        enrich_formulas=args.enrich_formulas,
    )
