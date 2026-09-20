from __future__ import annotations

import json
from difflib import SequenceMatcher
import re
from pathlib import Path
from typing import Any

from extraction.artifact_reconstruction import (
    reconstruct_artifacts,
)
from extraction.table_fallback import TableFallbackReconstructor


class CanonicalBuilder:
    """
    Builds the canonical representation from parser outputs.

    GROBID:
        title, authors, year, DOI, abstract, sections,
        paragraphs, references, figures, tables

    PyMuPDF:
        fallback source for abstract recovery

    Docling:
        page/bounding-box provenance for text and headings
    """

    # =========================================================
    # MAIN
    # =========================================================

    def build(
        self,
        *,
        paper_id: str,
        filename: str,
        grobid: dict[str, Any],
        pymupdf: dict[str, Any],
        artifacts: dict[str, Any],
        quality: dict[str, Any],
        source_pdf: Path | None = None,
        table_fallback_cache_path: Path | None = None,
        source_file_hash: str | None = None,
    ) -> dict[str, Any]:

        title = self._clean_text(
            grobid.get("title")
        )

        authors = self._clean_authors(
            grobid.get(
                "authors",
                [],
            )
        )

        year = self._clean_year(
            grobid.get("year")
        )

        doi = self._clean_text(
            grobid.get("doi")
        ) or None

        abstract = self._extract_abstract(
            grobid=grobid,
            pymupdf=pymupdf,
        )

        text_blocks = artifacts.get(
            "text_blocks",
            [],
        )

        sections = self._build_sections(
            raw_sections=grobid.get(
                "sections",
                [],
            ),
            text_blocks=text_blocks,
        )

        section_lookup = {
            self._normalize(
                section["heading"]
            ): section["heading"]
            for section in sections
        }

        paragraphs = self._build_paragraphs(
            paragraphs=grobid.get(
                "paragraphs",
                [],
            ),
            section_lookup=section_lookup,
            abstract=abstract,
            text_blocks=text_blocks,
        )
        
        reconstructed = reconstruct_artifacts(
            grobid=grobid,
            artifacts=artifacts,
        )

        # Formulas are parser-level artifacts extracted from Docling.
        # Rebuild them into canonical logical formula records here.
        raw_formulas = reconstructed.get(
            "formulas"
        )
        if not isinstance(raw_formulas, list) or not raw_formulas:
            raw_formulas = artifacts.get(
                "formulas",
                [],
            )

        formulas = self._build_formulas(
            raw_formulas
            if isinstance(raw_formulas, list)
            else []
        )

        figures = self._clean_artifacts(
            reconstructed.get(
                "figures",
                [],
            ),
            artifact_type="figure",
        )

        tables = self._clean_artifacts(
            reconstructed.get(
                "tables",
                [],
            ),
            artifact_type="table",
        )

        # The reconstruction layer may preserve only logical table identity.
        # Merge parser-level structured_data/table-fallback evidence back into
        # the canonical table so downstream querying can use real cells.
        tables = self._merge_table_structure_data(
            tables=tables,
            raw_tables=artifacts.get(
                "tables",
                [],
            ),
            source_pdf=source_pdf,
            fallback_cache_path=table_fallback_cache_path,
            source_file_hash=source_file_hash,
        )

        residual_visuals = reconstructed.get(
            "visual_artifacts"
        )
        if not isinstance(residual_visuals, list):
            residual_visuals = reconstructed.get(
                "unclassified_visuals",
                [],
            )

        visual_artifacts = self._classify_visual_artifacts(
            residual_visuals
            if isinstance(residual_visuals, list)
            else []
        )

        unclassified_visuals = [
            item
            for item in visual_artifacts
            if item.get("classification") == "unknown"
        ]

        artifact_reconstruction = reconstructed.get(
            "stats",
            {},
        )

        references = self._clean_references(
            grobid.get(
                "references",
                [],
            )
        )

        canonical_text = self._build_canonical_text(
            title=title,
            abstract=abstract,
            sections=sections,
            paragraphs=paragraphs,
        )

        return {
            "paper_id": paper_id,
            "filename": filename,
            "metadata": {
                "title": title,
                "authors": authors,
                "year": year,
                "doi": doi,
            },
            "abstract": abstract,
            "sections": sections,
            "paragraphs": paragraphs,
            "formulas": formulas,
            "figures": figures,
            "tables": tables,
            "visual_artifacts": visual_artifacts,
            "unclassified_visuals": unclassified_visuals,
            "artifact_reconstruction": artifact_reconstruction,
            "references": references,
            "canonical_text": canonical_text,
            "quality": quality,
        }

    # =========================================================
    # ABSTRACT
    # =========================================================

    @classmethod
    def _extract_abstract(
        cls,
        *,
        grobid: dict[str, Any],
        pymupdf: dict[str, Any],
    ) -> str:

        # 1. GROBID structured abstract
        abstract = cls._clean_text(
            grobid.get("abstract")
        )

        if abstract:
            return abstract

        # 2. GROBID paragraph fallback
        paragraphs = grobid.get(
            "paragraphs",
            [],
        )

        abstract_parts: list[str] = []
        abstract_started = False

        for paragraph in paragraphs:

            text = cls._clean_text(
                paragraph.get("text")
            )

            if not text:
                continue

            section = cls._normalize(
                paragraph.get(
                    "section",
                    "",
                )
            )

            if section == "abstract":
                abstract_started = True
                abstract_parts.append(
                    text
                )
                continue

            if abstract_started:

                if cls._is_major_section(
                    section
                ):
                    break

                abstract_parts.append(
                    text
                )

        abstract = cls._clean_text(
            " ".join(
                abstract_parts
            )
        )

        if abstract:
            return abstract

        # 3. PyMuPDF fallback
        return cls._extract_abstract_from_pdf(
            pymupdf.get(
                "text",
                "",
            )
        )

    @classmethod
    def _extract_abstract_from_pdf(
        cls,
        text: str,
    ) -> str:

        text = cls._clean_text(
            text
        )

        if not text:
            return ""

        pattern = (
            r"\babstract\b"
            r"\s*:?\s*"
            r"(?P<body>.*?)"
            r"(?="
            r"\bintroduction\b"
            r"|\bkeywords?\b"
            r"|\bbackground\b"
            r"|\brelated\s+work\b"
            r"|\bmaterials?\s+and\s+methods?\b"
            r"|\bmethods?\b"
            r")"
        )

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if not match:
            return ""

        abstract = cls._clean_text(
            match.group("body")
        )

        if len(abstract) < 80:
            return ""

        return abstract

    # =========================================================
    # SECTIONS
    # =========================================================

    @classmethod
    def _build_sections(
        cls,
        *,
        raw_sections: list[dict[str, Any]],
        text_blocks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:

        result: list[
            dict[str, Any]
        ] = []

        seen: set[str] = set()

        for section in raw_sections:

            heading = cls._clean_text(
                section.get(
                    "heading"
                )
            )

            if not heading:
                continue

            if cls._is_garbage_heading(
                heading
            ):
                continue

            key = cls._normalize(
                heading
            )

            if key in seen:
                continue

            seen.add(
                key
            )

            coords = section.get(
                "coords",
                [],
            )

            if not coords:
                coords = cls._match_docling_coords(
                    heading,
                    text_blocks,
                    heading_only=True,
                )

            result.append(
                {
                    "heading": heading,
                    "level": section.get(
                        "level",
                        1,
                    ),
                    "path": section.get(
                        "path",
                        [heading],
                    ),
                    "coords": coords,
                }
            )

        return result

    # =========================================================
    # PARAGRAPHS
    # =========================================================

    @classmethod
    def _build_paragraphs(
        cls,
        *,
        paragraphs: list[dict[str, Any]],
        section_lookup: dict[str, str],
        abstract: str,
        text_blocks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:

        result: list[
            dict[str, Any]
        ] = []

        seen: set[
            tuple[str, int | None]
        ] = set()

        last_section: str | None = None
        last_section_path: list[str] = []

        abstract_normalized = cls._normalize(
            abstract
        )

        for paragraph in paragraphs:

            text = cls._clean_text(
                paragraph.get("text")
            )

            if not text:
                continue

            normalized_text = cls._normalize(
                text
            )

            # Do not duplicate abstract into body.
            if (
                abstract_normalized
                and normalized_text
                == abstract_normalized
            ):
                continue

            section_value = cls._clean_text(
                paragraph.get(
                    "section"
                )
            )

            section_key = cls._normalize(
                section_value
            )

            if section_key:

                section = section_lookup.get(
                    section_key,
                    section_value,
                )

                last_section = section

                raw_path = paragraph.get(
                    "section_path"
                )

                if isinstance(
                    raw_path,
                    list,
                ):
                    last_section_path = [
                        cls._clean_text(
                            value
                        )
                        for value in raw_path
                        if cls._clean_text(
                            value
                        )
                    ]

            else:

                section = last_section

                raw_path = paragraph.get(
                    "section_path"
                )

                if (
                    not raw_path
                    and last_section_path
                ):
                    raw_path = (
                        last_section_path.copy()
                    )

            coords = paragraph.get(
                "coords",
                [],
            )

            if not coords:
                coords = cls._match_docling_coords(
                    text,
                    text_blocks,
                )

            page = paragraph.get(
                "page"
            )

            if page is None and coords:
                page = coords[0].get(
                    "page"
                )

            dedup_key = (
                normalized_text,
                page,
            )

            if dedup_key in seen:
                continue

            seen.add(
                dedup_key
            )

            sentences = paragraph.get(
                "sentences"
            )

            if not isinstance(
                sentences,
                list,
            ):
                sentences = []

            sentences = [
                cls._clean_text(
                    sentence
                )
                for sentence in sentences
                if cls._clean_text(
                    sentence
                )
            ]

            result.append(
                {
                    "text": text,
                    "sentences": sentences,
                    "section": section,
                    "section_path": (
                        raw_path
                        if isinstance(
                            raw_path,
                            list,
                        )
                        else (
                            [section]
                            if section
                            else []
                        )
                    ),
                    "coords": coords,
                    "page": page,
                }
            )

        return result
    
    def _build_formulas(
        self,
        raw_formulas: list[dict],
    ) -> list[dict]:

        formulas = []

        seen = set()

        for index, formula in enumerate(
            raw_formulas,
            start=1,
        ):
            if not isinstance(formula, dict):
                continue

            text = (
                formula.get("text")
                or formula.get("orig")
                or ""
            )

            text = self._clean_formula(text)

            if not text:
                continue

            normalized = self._normalize_formula(
                text
            )

            if not normalized:
                continue

            # Preserve provenance already normalized by the artifact extractor.
            coords = formula.get(
                "coords",
                [],
            )

            if not isinstance(
                coords,
                list,
            ):
                coords = []

            page = formula.get(
                "page"
            )

            if page is None and coords:
                page = coords[0].get(
                    "page"
                )

            # Fall back to raw Docling provenance when necessary.
            if not coords:
                provenance = formula.get(
                    "prov",
                    [],
                )

                for prov in provenance:
                    if not isinstance(
                        prov,
                        dict,
                    ):
                        continue

                    bbox = prov.get(
                        "bbox"
                    )

                    if not isinstance(
                        bbox,
                        dict,
                    ):
                        continue

                    prov_page = prov.get(
                        "page_no"
                    )

                    if page is None:
                        page = prov_page

                    coords.append({
                        "page": prov_page,
                        "x": bbox.get("l"),
                        "y": bbox.get("b"),
                        "width": (
                            bbox.get("r", 0)
                            - bbox.get("l", 0)
                        ),
                        "height": (
                            bbox.get("t", 0)
                            - bbox.get("b", 0)
                        ),
                        "coord_origin": bbox.get(
                            "coord_origin"
                        ),
                    })

            dedup_key = (
                normalized,
                page,
            )

            if dedup_key in seen:
                continue

            seen.add(dedup_key)

            source_refs = formula.get(
                "source_refs",
                [],
            )

            if not isinstance(
                source_refs,
                list,
            ):
                source_refs = []

            label = self._clean_text(
                formula.get("label")
            )

            formulas.append({
                "formula_id": (
                    f"formula_{len(formulas) + 1:03d}"
                ),
                "label": label or None,
                "text": text,
                "page": page,
                "coords": coords,
                "source_refs": source_refs,
                "source": "docling",
                "artifact_type": "equation",
            })

        return formulas

    # =========================================================
    # ARTIFACTS
    # =========================================================

    @classmethod
    def _match_raw_table(
        cls,
        table: dict[str, Any],
        raw_tables: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """
        Match a logical reconstructed table to the corresponding
        parser-level Docling table.

        Matching priority:
          1. source reference / self_ref
          2. caption similarity
          3. normalized label
          4. page overlap

        Returns None when the match is ambiguous.
        """

        if not raw_tables:
            return None

        # 1. Exact source-reference match.
        table_refs = table.get("source_refs", [])
        if isinstance(table_refs, list):
            for source_ref in table_refs:
                if not isinstance(source_ref, dict):
                    continue

                ref = source_ref.get("ref")
                if not ref:
                    continue

                for raw in raw_tables:
                    if raw.get("self_ref") == ref:
                        return raw

        target_caption = cls._normalize(
            table.get("caption", "")
        )

        target_label = cls._normalize(
            table.get("label", "")
        )

        # Remove digits so "table 1" and "table" can still match.
        target_label_base = re.sub(
            r"\d+",
            "",
            target_label,
        ).strip()

        target_pages = set()

        for key in ("pages", "page"):
            value = table.get(key)

            if isinstance(value, list):
                target_pages.update(
                    p for p in value
                    if isinstance(p, int)
                )

            elif isinstance(value, int):
                target_pages.add(value)

        for region in (
            table.get("regions")
            or table.get("coords")
            or []
        ):
            if not isinstance(region, dict):
                continue

            page = region.get("page")

            if isinstance(page, int):
                target_pages.add(page)

        candidates = []

        for raw in raw_tables:
            if not isinstance(raw, dict):
                continue

            score = 0.0

            raw_caption = cls._normalize(
                raw.get("caption", "")
            )

            raw_label = cls._normalize(
                raw.get("label", "")
            )

            raw_label_base = re.sub(
                r"\d+",
                "",
                raw_label,
            ).strip()

            # Caption is usually the strongest semantic match.
            if target_caption and raw_caption:
                if target_caption == raw_caption:
                    score = max(score, 0.95)
                elif (
                    target_caption in raw_caption
                    or raw_caption in target_caption
                ):
                    score = max(score, 0.85)
                else:
                    similarity = SequenceMatcher(
                        None,
                        target_caption,
                        raw_caption,
                    ).ratio()

                    if similarity >= 0.80:
                        score = max(
                            score,
                            0.80 * similarity,
                        )

            # Label fallback.
            if (
                target_label_base
                and raw_label_base
                and target_label_base
                == raw_label_base
            ):
                score = max(score, 0.65)

            # Page overlap fallback.
            raw_pages = set()

            for key in ("pages", "page"):
                value = raw.get(key)

                if isinstance(value, list):
                    raw_pages.update(
                        p for p in value
                        if isinstance(p, int)
                    )

                elif isinstance(value, int):
                    raw_pages.add(value)

            for region in (
                raw.get("regions")
                or raw.get("coords")
                or []
            ):
                if not isinstance(region, dict):
                    continue

                page = region.get("page")

                if isinstance(page, int):
                    raw_pages.add(page)

            if target_pages and raw_pages:
                if target_pages & raw_pages:
                    score = max(score, 0.55)

            if score > 0:
                candidates.append(
                    (
                        score,
                        raw,
                    )
                )

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        best_score, best_raw = candidates[0]

        # Require a reasonably defensible match.
        if best_score < 0.55:
            return None

        # Reject ambiguous matches.
        if len(candidates) > 1:
            second_score = candidates[1][0]

            if (
                abs(best_score - second_score)
                < 0.05
            ):
                return None

        return best_raw

    @staticmethod
    def _table_structure_is_usable(
        structure: Any,
    ) -> bool:
        """
        Return True only when actual row/column content exists.

        Empty grids, metadata-only structures and bounding boxes
        are not considered usable table structure.
        """

        if not isinstance(
            structure,
            dict,
        ):
            return False

        rows = structure.get(
            "rows"
        )

        if not isinstance(
            rows,
            list,
        ):
            rows = structure.get(
                "grid"
            )

        if not isinstance(
            rows,
            list,
        ) or not rows:
            return False

        valid_rows = [
            row
            for row in rows
            if isinstance(row, list)
        ]

        if not valid_rows:
            return False

        max_columns = max(
            (
                len(row)
                for row in valid_rows
            ),
            default=0,
        )

        if max_columns < 1:
            return False

        nonempty_cells = 0

        for row in valid_rows:
            for cell in row:
                if isinstance(
                    cell,
                    dict,
                ):
                    value = (
                        cell.get("text")
                        or cell.get("value")
                        or ""
                    )
                else:
                    value = cell

                if str(value).strip():
                    nonempty_cells += 1

        # At least two pieces of actual cell content.
        return nonempty_cells >= 2

    @classmethod
    def _normalize_raw_table_structure(
        cls,
        structure: Any,
    ) -> dict[str, Any] | None:
        """
        Convert Docling-native grid structure into the canonical
        rows/columns representation.
        """

        if not isinstance(
            structure,
            dict,
        ):
            return None

        grid = structure.get(
            "grid"
        )

        if not isinstance(
            grid,
            list,
        ):
            return None

        rows = []

        for row in grid:
            if not isinstance(
                row,
                list,
            ):
                continue

            normalized_row = []

            for cell in row:
                if isinstance(
                    cell,
                    dict,
                ):
                    value = (
                        cell.get("text")
                        or cell.get("value")
                        or ""
                    )
                else:
                    value = cell

                normalized_row.append(
                    cls._clean_text(value)
                )

            if any(
                cell.strip()
                for cell in normalized_row
            ):
                rows.append(
                    normalized_row
                )

        if not rows:
            return None

        num_cols = max(
            (
                len(row)
                for row in rows
            ),
            default=0,
        )

        if num_cols == 0:
            return None

        # Normalize all rows to the same width.
        for row in rows:
            if len(row) < num_cols:
                row.extend(
                    [""] *
                    (num_cols - len(row))
                )

        return {
            "columns": [
                f"column_{index}"
                for index in range(
                    1,
                    num_cols + 1,
                )
            ],
            "rows": rows,
            "num_rows": len(rows),
            "num_cols": num_cols,
            "method": "docling_native",
        }
    @classmethod
    def _merge_table_structure_data(
        cls,
        *,
        tables: list[dict[str, Any]],
        raw_tables: list[dict[str, Any]],
        source_pdf: Path | None = None,
        fallback_cache_path: Path | None = None,
        source_file_hash: str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_raw = [
            item
            for item in raw_tables
            if isinstance(item, dict)
        ] if isinstance(raw_tables, list) else []

        # First preserve parser-provided table structure.
        for table in tables:
            raw = cls._match_raw_table(
                table,
                normalized_raw,
            )
            if raw is None:
                continue

            structured = raw.get(
                "structured_data"
            )
            if not cls._table_structure_is_usable(
                structured
            ):
                structured = cls._normalize_raw_table_structure(
                    raw.get("structure")
                )
            
            
            if cls._table_structure_is_usable(
                structured
            ):
                table["structure_status"] = raw.get(
                    "structure_status",
                    "structured",
                )

                if table.get("structure_source") is None:
                    table["structure_source"] = (
                        structured.get(
                            "method",
                            "docling_native",
                        )
                        if isinstance(structured, dict)
                        else "docling_native"
    )

                for key in (
                    "structure_source",
                    "table_fallback",
                    "content",
                ):
                    if raw.get(key) is not None:
                        table[key] = raw.get(key)

        # Then repair only tables whose structure is still missing.
        if source_pdf is None:
            return tables

        cache = cls._load_table_fallback_cache(
            fallback_cache_path,
            source_file_hash,
        )
        cache_dirty = False
        reconstructor = TableFallbackReconstructor()

        for index, table in enumerate(tables, start=1):
            if cls._table_structure_is_usable(
                table.get("structured_data")
            ):
                continue

            key = cls._table_cache_key(
                table,
                index,
            )

            cached = cache.get(key)
            if isinstance(cached, dict):
                cls._apply_table_fallback_result(
                    table,
                    cached,
                )
                continue

            result = reconstructor.reconstruct(
                pdf_path=source_pdf,
                table=table,
            )
            if result is None:
                continue

            cls._apply_table_fallback_result(
                table,
                result,
            )
            cache[key] = result
            cache_dirty = True
            
        # Normalize table status after all reconstruction attempts.
        for table in tables:

            structured_data = table.get(
                "structured_data"
            )

            if cls._table_structure_is_usable(
                structured_data
            ):
                table["structure_status"] = (
                    "structured"
                )
                continue

            content = cls._clean_text(
                table.get("content")
            )

            if content:
                table["structure_status"] = (
                    "text_only"
                )
            else:
                table["structure_status"] = (
                    "detected_no_content"
                )

            # Do not claim native structure when none exists.
            if not table.get(
                "structured_data"
            ):
                table["structured_data"] = None

            if table.get(
                "structure_source"
            ) == "docling_native" and not cls._table_structure_is_usable(
                table.get("structured_data")
            ):
                table["structure_source"] = None

        if cache_dirty and fallback_cache_path is not None:
            fallback_cache_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            fallback_cache_path.write_text(
                json.dumps(
                    {
                        "file_hash": source_file_hash,
                        "tables": cache,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

        return tables

    @classmethod
    def _apply_table_fallback_result(
        cls,
        table: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        structured = dict(
            table.get("structured_data")
            or {}
        )
        structured.update(
            {
                "columns": result.get("columns", []),
                "rows": result.get("rows", []),
                "num_rows": result.get("num_rows", 0),
                "num_cols": result.get("num_cols", 0),
                "method": result.get("method"),
                "confidence": result.get("confidence"),
                "pages": result.get("pages", []),
            }
        )

        if cls._table_structure_is_usable(
            structured
        ):
            table["structured_data"] = structured
            table["structure_status"] = "structured"
        else:
            return
        table["structure_source"] = result.get(
            "method",
            "table_fallback",
        )
        table["table_fallback"] = {
            "used": True,
            "method": result.get("method"),
            "confidence": result.get("confidence"),
            "occupancy": result.get("occupancy"),
            "region_results": result.get(
                "region_results",
                [],
            ),
        }
        table["content"] = "\n".join(
            " | ".join(row)
            for row in result.get("rows", [])
        )

    @staticmethod
    def _table_cache_key(
        table: dict[str, Any],
        index: int,
    ) -> str:
        table_id = table.get("table_id")
        if table_id:
            return str(table_id)

        label = CanonicalBuilder._normalize(
            table.get("label", "")
        )
        caption = CanonicalBuilder._normalize(
            table.get("caption", "")
        )
        pages = (
            tuple(table.get("pages", []))
            if isinstance(
                table.get("pages"),
                list,
            )
            else ()
        )
        return f"{label}|{caption}|{pages}|{index}"

    @staticmethod
    def _load_table_fallback_cache(
        cache_path: Path | None,
        source_file_hash: str | None,
    ) -> dict[str, Any]:
        if cache_path is None or not cache_path.exists():
            return {}

        try:
            data = json.loads(
                cache_path.read_text(
                    encoding="utf-8"
                )
            )
        except (OSError, ValueError, TypeError):
            return {}

        if source_file_hash is not None and data.get(
            "file_hash"
        ) != source_file_hash:
            return {}

        tables = data.get("tables")
        return tables if isinstance(
            tables,
            dict,
        ) else {}

    @classmethod
    def _clean_artifacts(
        cls,
        items: list[dict[str, Any]],
        *,
        artifact_type: str | None = None,
    ) -> list[dict[str, Any]]:

        result: list[
            dict[str, Any]
        ] = []

        for item in items:

            cleaned = dict(
                item
            )

            for key in (
                "label",
                "caption",
            ):

                if key in cleaned:
                    cleaned[key] = (
                        cls._clean_text(
                            cleaned[key]
                        )
                    )

            if artifact_type:
                cleaned["artifact_type"] = (
                    artifact_type
                )

            result.append(
                cleaned
            )

        return result

    @classmethod
    def _classify_visual_artifacts(
        cls,
        items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:

        allowed = {
            "diagram",
            "equation",
            "page_furniture",
            "image_fragment",
            "unknown",
        }

        result: list[
            dict[str, Any]
        ] = []

        for index, item in enumerate(
            items,
            start=1,
        ):
            if not isinstance(
                item,
                dict,
            ):
                continue

            cleaned = dict(
                item
            )

            existing = cls._normalize(
                cleaned.get(
                    "classification"
                )
            )

            if existing in allowed:
                classification = existing
                evidence = list(
                    cleaned.get(
                        "classification_evidence",
                        [],
                    )
                    if isinstance(
                        cleaned.get(
                            "classification_evidence"
                        ),
                        list,
                    )
                    else []
                )
            else:
                classification, evidence = (
                    cls._infer_visual_classification(
                        cleaned
                    )
                )

            cleaned["visual_id"] = (
                cleaned.get(
                    "visual_id"
                )
                or f"visual_{index:03d}"
            )
            cleaned["classification"] = (
                classification
            )
            cleaned["artifact_type"] = (
                classification
            )
            cleaned["classification_evidence"] = (
                evidence
            )

            result.append(
                cleaned
            )

        return result

    @classmethod
    def _infer_visual_classification(
        cls,
        item: dict[str, Any],
    ) -> tuple[str, list[str]]:

        text = cls._normalize(
            " ".join(
                str(value or "")
                for value in (
                    item.get("label"),
                    item.get("caption"),
                    item.get("reason"),
                )
            )
        )

        signals = list(
            item.get(
                "signals",
                [],
            )
            if isinstance(
                item.get(
                    "signals"
                ),
                list,
            )
            else []
        )

        evidence: list[str] = []

        if "repeated_small_region" in signals:
            evidence.append(
                "repeated_small_region"
            )
            return (
                "page_furniture",
                evidence,
            )

        if re.search(
            r"\b(?:equation|formula)\b",
            text,
        ):
            evidence.append(
                "equation_text_signal"
            )
            return (
                "equation",
                evidence,
            )

        if re.search(
            r"\b(?:diagram|flowchart|"
            r"architecture|schematic|framework|"
            r"pipeline|workflow|block diagram)\b",
            text,
        ):
            evidence.append(
                "diagram_text_signal"
            )
            return (
                "diagram",
                evidence,
            )

        coords = (
            item.get("regions")
            or item.get("coords")
            or []
        )

        if isinstance(
            coords,
            list,
        ) and coords:
            areas = []

            for coord in coords:
                if not isinstance(
                    coord,
                    dict,
                ):
                    continue

                try:
                    width = float(
                        coord.get(
                            "w",
                            coord.get(
                                "width",
                                0,
                            ),
                        )
                    )
                    height = float(
                        coord.get(
                            "h",
                            coord.get(
                                "height",
                                0,
                            ),
                        )
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    continue

                areas.append(
                    width * height
                )

            if areas:
                median_area = sorted(
                    areas
                )[len(areas) // 2]

                if median_area < 8000:
                    evidence.append(
                        "small_visual_region"
                    )
                    return (
                        "image_fragment",
                        evidence,
                    )

        return (
            "unknown",
            evidence,
        )

    # =========================================================
    # REFERENCES
    # =========================================================

    @classmethod
    def _clean_references(
        cls,
        references: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:

        result: list[
            dict[str, Any]
        ] = []

        seen: set[str] = set()

        for reference in references:

            title = cls._clean_text(
                reference.get(
                    "title"
                )
            )

            raw = cls._clean_text(
                reference.get(
                    "raw"
                )
            )

            authors = reference.get(
                "authors",
                [],
            )

            if not isinstance(
                authors,
                list,
            ):
                authors = []

            authors = [
                cls._clean_text(
                    author
                )
                for author in authors
                if cls._clean_text(
                    author
                )
            ]

            if not title and not raw:
                continue

            if raw and len(raw) < 15:
                continue

            key = cls._normalize(
                raw or title
            )

            if key in seen:
                continue

            seen.add(
                key
            )

            result.append(
                {
                    "title": title,
                    "authors": authors,
                    "raw": raw,
                }
            )

        return result

    # =========================================================
    # CANONICAL TEXT
    # =========================================================

    @classmethod
    def _build_canonical_text(
        cls,
        *,
        title: str,
        abstract: str,
        sections: list[dict[str, Any]],
        paragraphs: list[dict[str, Any]],
    ) -> str:

        parts: list[str] = []

        if title:
            parts.append(
                title
            )

        if abstract:
            parts.append(
                "## Abstract"
            )
            parts.append(
                abstract
            )

        paragraphs_by_section: dict[
            str,
            list[str],
        ] = {}

        for paragraph in paragraphs:

            text = cls._clean_text(
                paragraph.get(
                    "text"
                )
            )

            if not text:
                continue

            section = paragraph.get(
                "section"
            )

            key = (
                section
                if section
                else "Unsectioned"
            )

            paragraphs_by_section.setdefault(
                key,
                [],
            ).append(
                text
            )

        emitted_sections: set[str] = set()

        for section in sections:

            heading = section[
                "heading"
            ]

            parts.append(
                f"## {heading}"
            )

            emitted_sections.add(
                heading
            )

            for text in paragraphs_by_section.get(
                heading,
                [],
            ):
                parts.append(
                    text
                )

        orphan_text = paragraphs_by_section.get(
            "Unsectioned",
            [],
        )

        if orphan_text:

            parts.append(
                "## Unsectioned"
            )

            parts.extend(
                orphan_text
            )

        for (
            section_name,
            texts,
        ) in paragraphs_by_section.items():

            if section_name in emitted_sections:
                continue

            if section_name == "Unsectioned":
                continue

            parts.append(
                f"## {section_name}"
            )

            parts.extend(
                texts
            )

        return "\n\n".join(
            part
            for part in parts
            if part
        )
    

    # =========================================================
    # AUTHORS / YEAR
    # =========================================================

    @classmethod
    def _clean_authors(
        cls,
        authors: list[Any],
    ) -> list[str]:

        result: list[str] = []

        for author in authors:

            value = cls._clean_text(
                author
            )

            if not value:
                continue

            value = value.strip(
                " ,;."
            )

            if value:
                result.append(
                    value
                )

        output: list[str] = []
        seen: set[str] = set()

        for author in result:

            key = cls._normalize(
                author
            )

            if key in seen:
                continue

            seen.add(
                key
            )

            output.append(
                author
            )

        return output

    @staticmethod
    def _clean_year(
        year: Any,
    ) -> int | None:

        if year is None:
            return None

        try:

            value = int(
                year
            )

            if 1900 <= value <= 2100:
                return value

        except (
            TypeError,
            ValueError,
        ):
            pass

        match = re.search(
            r"(19|20)\d{2}",
            str(year),
        )

        if match:
            return int(
                match.group(0)
            )

        return None

    # =========================================================
    # HEADING FILTERS
    # =========================================================

    @classmethod
    def _is_garbage_heading(
        cls,
        heading: str,
    ) -> bool:

        value = cls._normalize(
            heading
        )

        if not value:
            return True

        if len(value) > 180:
            return True

        if value in {
            "•",
            "-",
            "_",
        }:
            return True

        if re.fullmatch(
            r"[\d\s%=+\-*/().]+",
            value,
        ):
            return True

        if re.search(
            r"scientific reports\s*\|",
            value,
        ):
            return True

        if re.search(
            r"proceedings of the .*conference",
            value,
        ):
            return True

        if re.search(
            r"\b(copyright|correspondence to)\b",
            value,
        ):
            return True

        if (
            len(
                heading.split()
            ) <= 10
            and re.search(
                r"[=^_{}]",
                heading,
            )
        ):
            return True

        return False

    @staticmethod
    def _is_major_section(
        section: str,
    ) -> bool:

        return section in {
            "abstract",
            "introduction",
            "related work",
            "background",
            "materials and methods",
            "materials method",
            "methods",
            "methodology",
            "experiments",
            "results",
            "results and discussion",
            "discussion",
            "conclusion",
            "conclusions",
            "limitations",
            "references",
        }

    # =========================================================
    # DOCLING COORDINATE MATCHING
    # =========================================================

    @classmethod
    def _match_docling_coords(
        cls,
        target_text: str,
        text_blocks: list[dict[str, Any]],
        heading_only: bool = False,
    ) -> list[dict[str, Any]]:

        target = cls._normalize(
            target_text
        )

        if not target:
            return []

        candidates: list[
            tuple[
                float,
                list[dict[str, Any]],
            ]
        ] = []

        for block in text_blocks:

            label = cls._normalize(
                block.get(
                    "label",
                    "",
                )
            )

            if heading_only and (
                label != "section_header"
            ):
                continue

            block_text = cls._normalize(
                block.get(
                    "text",
                    "",
                )
            )

            if not block_text:
                continue

            score = cls._text_match_score(
                target,
                block_text,
            )

            if score >= 0.45:

                candidates.append(
                    (
                        score,
                        block.get(
                            "coords",
                            [],
                        ),
                    )
                )

        if not candidates:
            return []

        candidates.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        if heading_only:
            return candidates[0][1]

        selected: list[
            dict[str, Any]
        ] = []

        best_score = candidates[0][0]

        for score, coords in candidates:

            if score < (
                best_score * 0.80
            ):
                continue

            selected.extend(
                coords
            )

            if len(selected) >= 20:
                break

        return selected

    @staticmethod
    def _text_match_score(
        target: str,
        candidate: str,
    ) -> float:

        if not target or not candidate:
            return 0.0

        if target == candidate:
            return 1.0

        if (
            target in candidate
            or candidate in target
        ):
            return 0.90

        target_tokens = set(
            target.split()
        )

        candidate_tokens = set(
            candidate.split()
        )

        if not target_tokens:
            return 0.0

        overlap = (
            len(
                target_tokens
                & candidate_tokens
            )
            / len(target_tokens)
        )

        sequence = SequenceMatcher(
            None,
            target,
            candidate,
        ).ratio()

        return (
            0.65 * overlap
            + 0.35 * sequence
        )

    # =========================================================
    # TEXT
    # =========================================================

    @staticmethod
    def _clean_text(
        value: Any,
    ) -> str:

        if value is None:
            return ""

        return re.sub(
            r"\s+",
            " ",
            str(value)
            .replace(
                "\xa0",
                " ",
            ),
        ).strip()

    @staticmethod
    def _normalize(
        text: Any,
    ) -> str:

        return re.sub(
            r"\s+",
            " ",
            str(text or "")
            .lower()
            .strip(),
        )
        
    @staticmethod
    def _clean_formula(
        text: str,
    ) -> str:

        text = str(text)

        # Remove Docling/PDF control artifacts.
        text = text.replace(
            "\x08",
            " ",
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()
    
    @staticmethod
    def _normalize_formula(
        text: str,
    ) -> str:

        text = text.lower()

        text = re.sub(
            r"\s+",
            "",
            text,
        )

        return text