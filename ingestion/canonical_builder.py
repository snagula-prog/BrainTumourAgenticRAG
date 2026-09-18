from __future__ import annotations

from difflib import SequenceMatcher
import hashlib
import json
import re
from typing import Any


class CanonicalBuilder:
    """
    Builds the canonical representation from parser outputs.

    Source responsibilities:

    GROBID
        title
        authors
        year
        DOI
        abstract
        sections
        paragraphs
        references

    PyMuPDF
        abstract fallback

    Docling artifacts
        text coordinates
        heading coordinates
        tables
        figures
        formulas

    Design rules:
        - prose remains paragraph-oriented
        - tables remain atomic structured artifacts
        - figures remain atomic artifacts
        - formulas remain atomic artifacts
        - references remain separate
        - provenance is preserved
        - artifacts are never collapsed merely because page/caption is missing
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
    ) -> dict[str, Any]:

        title = self._clean_text(
            grobid.get("title")
        )

        authors = self._clean_authors(
            grobid.get("authors", [])
        )

        year = self._clean_year(
            grobid.get("year")
        )

        doi = (
            self._clean_text(
                grobid.get("doi")
            )
            or None
        )

        abstract = self._extract_abstract(
            grobid=grobid,
            pymupdf=pymupdf,
        )

        text_blocks = artifacts.get(
            "text_blocks",
            [],
        )

        if not isinstance(text_blocks, list):
            text_blocks = []

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

        formulas = self._build_formulas(
            artifacts.get(
                "formulas",
                [],
            )
        )

        figures = self._build_figures(
            artifacts.get(
                "figures",
                [],
            )
        )

        tables = self._build_tables(
            artifacts.get(
                "tables",
                [],
            )
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

        abstract = cls._clean_text(
            grobid.get("abstract")
        )

        if abstract:
            return abstract

        paragraphs = grobid.get(
            "paragraphs",
            [],
        )

        if not isinstance(paragraphs, list):
            paragraphs = []

        abstract_parts: list[str] = []
        abstract_started = False

        for paragraph in paragraphs:

            if not isinstance(paragraph, dict):
                continue

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
                abstract_parts.append(text)
                continue

            if abstract_started:

                if cls._is_major_section(section):
                    break

                abstract_parts.append(text)

        abstract = cls._clean_text(
            " ".join(abstract_parts)
        )

        if abstract:
            return abstract

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

        text = cls._clean_text(text)

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

        result: list[dict[str, Any]] = []
        seen: set[str] = set()

        for section in raw_sections:

            if not isinstance(section, dict):
                continue

            heading = cls._clean_text(
                section.get("heading")
            )

            if not heading:
                continue

            if cls._is_garbage_heading(heading):
                continue

            key = cls._normalize(heading)

            if key in seen:
                continue

            seen.add(key)

            coords = cls._extract_coords(section)

            if not coords:
                coords = cls._match_docling_coords(
                    heading,
                    text_blocks,
                    heading_only=True,
                )

            path = section.get(
                "path",
                [heading],
            )

            if not isinstance(path, list):
                path = [heading]

            path = [
                cls._clean_text(value)
                for value in path
                if cls._clean_text(value)
            ]

            if not path:
                path = [heading]

            result.append(
                {
                    "heading": heading,
                    "level": section.get("level", 1),
                    "path": path,
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

        result: list[dict[str, Any]] = []

        seen: set[
            tuple[str, int | None]
        ] = set()

        last_section: str | None = None
        last_section_path: list[str] = []

        abstract_normalized = cls._normalize(
            abstract
        )

        for paragraph in paragraphs:

            if not isinstance(paragraph, dict):
                continue

            text = cls._clean_text(
                paragraph.get("text")
            )

            if not text:
                continue

            normalized_text = cls._normalize(
                text
            )

            if (
                abstract_normalized
                and normalized_text == abstract_normalized
            ):
                continue

            section_value = cls._clean_text(
                paragraph.get("section")
            )

            section_key = cls._normalize(
                section_value
            )

            raw_path = paragraph.get(
                "section_path"
            )

            if section_key:

                section = section_lookup.get(
                    section_key,
                    section_value,
                )

                last_section = section

                if isinstance(raw_path, list):

                    last_section_path = [
                        cls._clean_text(value)
                        for value in raw_path
                        if cls._clean_text(value)
                    ]

            else:

                section = last_section

                if (
                    not isinstance(raw_path, list)
                    and last_section_path
                ):
                    raw_path = last_section_path.copy()

            if not isinstance(raw_path, list):

                raw_path = (
                    last_section_path.copy()
                    if last_section_path
                    else (
                        [section]
                        if section
                        else []
                    )
                )

            raw_path = [
                cls._clean_text(value)
                for value in raw_path
                if cls._clean_text(value)
            ]

            coords = cls._extract_coords(
                paragraph
            )

            if not coords:
                coords = cls._match_docling_coords(
                    text,
                    text_blocks,
                )

            page = paragraph.get("page")

            if page is None and coords:
                page = coords[0].get("page")

            dedup_key = (
                normalized_text,
                page,
            )

            if dedup_key in seen:
                continue

            seen.add(dedup_key)

            sentences = paragraph.get(
                "sentences"
            )

            if not isinstance(sentences, list):
                sentences = []

            sentences = [
                cls._clean_text(sentence)
                for sentence in sentences
                if cls._clean_text(sentence)
            ]

            result.append(
                {
                    "text": text,
                    "sentences": sentences,
                    "section": section,
                    "section_path": raw_path,
                    "coords": coords,
                    "page": page,
                }
            )

        return result

    # =========================================================
    # FORMULAS
    # =========================================================

    @classmethod
    def _build_formulas(
        cls,
        raw_formulas: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:

        formulas: list[dict[str, Any]] = []
        seen: set[str] = set()

        if not isinstance(raw_formulas, list):
            return formulas

        for index, formula in enumerate(
            raw_formulas,
            start=1,
        ):

            if not isinstance(formula, dict):
                continue

            text = (
                formula.get("text")
                or formula.get("orig")
                or formula.get("value")
                or formula.get("content")
                or ""
            )

            text = cls._clean_formula(text)

            if not text:
                continue

            normalized = cls._normalize_formula(
                text
            )

            if not normalized:
                continue

            page = cls._extract_page(formula)
            coords = cls._extract_coords(formula)

            # Critical:
            # only remove a formula when the actual content
            # and provenance match. A missing page must NOT
            # cause unrelated formulas to collapse.
            fingerprint = cls._artifact_fingerprint(
                item=formula,
                content=normalized,
                page=page,
                coords=coords,
            )

            if fingerprint in seen:
                continue

            seen.add(fingerprint)

            section_path = formula.get(
                "section_path",
                [],
            )

            if not isinstance(section_path, list):
                section_path = []

            formulas.append(
                {
                    "formula_id": (
                        formula.get("formula_id")
                        or f"formula_{index:03d}"
                    ),
                    "text": text,
                    "equation_number": (
                        formula.get("equation_number")
                        or formula.get("label")
                        or formula.get("number")
                    ),
                    "section": (
                        cls._clean_text(
                            formula.get("section")
                        )
                        or None
                    ),
                    "section_path": [
                        cls._clean_text(value)
                        for value in section_path
                        if cls._clean_text(value)
                    ],
                    "page": page,
                    "coords": coords,
                    "source": "docling",
                }
            )

        return formulas

    # =========================================================
    # TABLES
    # =========================================================

    @classmethod
    def _build_tables(
        cls,
        raw_tables: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:

        result: list[dict[str, Any]] = []
        seen: set[str] = set()

        if not isinstance(raw_tables, list):
            return result

        for index, table in enumerate(
            raw_tables,
            start=1,
        ):

            if not isinstance(table, dict):
                continue

            label = cls._clean_text(
                table.get("label")
            )

            caption = cls._clean_text(
                table.get("caption")
            )

            section = cls._clean_text(
                table.get("section")
            )

            section_path = table.get(
                "section_path",
                [],
            )

            if not isinstance(section_path, list):
                section_path = []

            page = cls._extract_page(table)
            coords = cls._extract_coords(table)

            rows = cls._extract_table_rows(table)
            cells = cls._extract_table_cells(table)

            content = cls._extract_table_content(
                table,
                rows=rows,
                cells=cells,
            )

            # Critical:
            # do NOT use only (label, caption, page) as the
            # deduplication key. For many Docling records these
            # fields are empty/null, which previously collapsed
            # all tables into one.
            fingerprint = cls._artifact_fingerprint(
                item=table,
                content=content,
                page=page,
                coords=coords,
                fallback_index=index,
            )

            if fingerprint in seen:
                continue

            seen.add(fingerprint)

            structure_status = (
                "structured"
                if rows or cells or content.strip()
                else "bbox_only"
            )

            table_id = (
                table.get("table_id")
                or f"table_{index:03d}"
            )

            result.append(
                {
                    "table_id": table_id,
                    "label": label or None,
                    "caption": caption or None,
                    "section": section or None,
                    "section_path": [
                        cls._clean_text(value)
                        for value in section_path
                        if cls._clean_text(value)
                    ],
                    "rows": rows,
                    "cells": cells,
                    "content": content,
                    "text": content,
                    "coords": coords,
                    "page": page,
                    "source": "docling",
                    "structure_status": structure_status,
                }
            )

        return result

    # =========================================================
    # TABLE EXTRACTION
    # =========================================================

    @classmethod
    def _find_nested_value(
        cls,
        obj: Any,
        keys: tuple[str, ...],
        *,
        max_depth: int = 3,
        _depth: int = 0,
    ) -> Any:

        if _depth > max_depth:
            return None

        if isinstance(obj, dict):

            for key in keys:

                value = obj.get(key)

                if value is not None:
                    return value

            for value in obj.values():

                found = cls._find_nested_value(
                    value,
                    keys,
                    max_depth=max_depth,
                    _depth=_depth + 1,
                )

                if found is not None:
                    return found

        elif isinstance(obj, list):

            for value in obj:

                found = cls._find_nested_value(
                    value,
                    keys,
                    max_depth=max_depth,
                    _depth=_depth + 1,
                )

                if found is not None:
                    return found

        return None

    @classmethod
    def _extract_table_rows(
        cls,
        table: dict[str, Any],
    ) -> list[Any]:

        direct = table.get("rows")

        if isinstance(direct, list):
            return direct

        candidates = (
            "rows",
            "data",
            "table_data",
            "grid",
            "structure",
        )

        found = cls._find_nested_value(
            table,
            candidates,
        )

        if isinstance(found, list):
            return found

        return []

    @classmethod
    def _extract_table_cells(
        cls,
        table: dict[str, Any],
    ) -> list[Any]:

        direct = table.get("cells")

        if isinstance(direct, list):
            return direct

        found = cls._find_nested_value(
            table,
            (
                "cells",
                "table_cells",
            ),
        )

        if isinstance(found, list):
            return found

        rows = cls._extract_table_rows(table)

        if rows:

            flattened: list[Any] = []

            for row in rows:

                if isinstance(row, list):
                    flattened.extend(row)

                elif isinstance(row, dict):

                    # Some row representations contain
                    # cell arrays under one of these keys.
                    nested = cls._find_nested_value(
                        row,
                        (
                            "cells",
                            "items",
                            "values",
                        ),
                        max_depth=1,
                    )

                    if isinstance(nested, list):
                        flattened.extend(nested)

            if flattened:
                return flattened

        return []

    @classmethod
    def _extract_table_content(
        cls,
        table: dict[str, Any],
        *,
        rows: list[Any],
        cells: list[Any],
    ) -> str:

        # Existing textual representation.
        for key in (
            "content",
            "text",
            "markdown",
            "md",
        ):

            value = table.get(key)

            if isinstance(value, str):

                cleaned = cls._clean_multiline_text(
                    value
                )

                if cleaned:
                    return cleaned

        # Some artifact extractors use a nested
        # textual table representation.
        nested_content = cls._find_nested_value(
            table,
            (
                "markdown",
                "content",
                "text",
            ),
            max_depth=2,
        )

        if isinstance(
            nested_content,
            str,
        ):

            cleaned = cls._clean_multiline_text(
                nested_content
            )

            if cleaned:
                return cleaned

        # Structured rows.
        if rows:

            row_texts: list[str] = []

            for row in rows:

                if isinstance(row, list):

                    values = [
                        cls._cell_to_text(cell)
                        for cell in row
                    ]

                    values = [
                        value
                        for value in values
                        if value
                    ]

                    if values:
                        row_texts.append(
                            " | ".join(values)
                        )

                elif isinstance(row, dict):

                    nested_cells = cls._find_nested_value(
                        row,
                        (
                            "cells",
                            "items",
                            "values",
                        ),
                        max_depth=1,
                    )

                    if isinstance(
                        nested_cells,
                        list,
                    ):

                        values = [
                            cls._cell_to_text(cell)
                            for cell in nested_cells
                        ]

                        values = [
                            value
                            for value in values
                            if value
                        ]

                        if values:
                            row_texts.append(
                                " | ".join(values)
                            )
                        continue

                    values: list[str] = []

                    for key, value in row.items():

                        # Avoid serializing structural metadata.
                        if key in {
                            "prov",
                            "bbox",
                            "coords",
                            "page",
                            "row_span",
                            "col_span",
                            "start_row_offset_idx",
                            "end_row_offset_idx",
                            "start_col_offset_idx",
                            "end_col_offset_idx",
                        }:
                            continue

                        key_text = cls._clean_text(key)
                        value_text = cls._cell_to_text(value)

                        if not value_text:
                            continue

                        if key_text:
                            values.append(
                                f"{key_text}: {value_text}"
                            )
                        else:
                            values.append(value_text)

                    if values:
                        row_texts.append(
                            " | ".join(values)
                        )

                else:

                    value = cls._cell_to_text(row)

                    if value:
                        row_texts.append(value)

            if row_texts:
                return "\n".join(row_texts)

        # Cell fallback.
        if cells:

            structured_cells: list[
                tuple[int, int, str]
            ] = []

            plain_cells: list[str] = []

            for cell_index, cell in enumerate(cells):

                value = cls._cell_to_text(cell)

                if not value:
                    continue

                row_index = (
                    cls._first_int(
                        cell,
                        (
                            "start_row_offset_idx",
                            "row_idx",
                            "row_index",
                            "row",
                        ),
                    )
                    if isinstance(cell, dict)
                    else None
                )

                col_index = (
                    cls._first_int(
                        cell,
                        (
                            "start_col_offset_idx",
                            "col_idx",
                            "col_index",
                            "column",
                            "col",
                        ),
                    )
                    if isinstance(cell, dict)
                    else None
                )

                if (
                    row_index is not None
                    and col_index is not None
                ):

                    structured_cells.append(
                        (
                            row_index,
                            col_index,
                            value,
                        )
                    )

                else:
                    plain_cells.append(value)

            if structured_cells:

                grouped: dict[
                    int,
                    list[tuple[int, str]],
                ] = {}

                for row_index, col_index, value in (
                    structured_cells
                ):

                    grouped.setdefault(
                        row_index,
                        [],
                    ).append(
                        (
                            col_index,
                            value,
                        )
                    )

                lines: list[str] = []

                for row_index in sorted(
                    grouped
                ):

                    ordered = sorted(
                        grouped[row_index],
                        key=lambda item: item[0],
                    )

                    lines.append(
                        " | ".join(
                            value
                            for _, value in ordered
                        )
                    )

                if plain_cells:
                    lines.append(
                        " | ".join(
                            plain_cells
                        )
                    )

                return "\n".join(lines)

            if plain_cells:
                return " | ".join(
                    plain_cells
                )

        return ""

    @classmethod
    def _cell_to_text(
        cls,
        cell: Any,
    ) -> str:

        if cell is None:
            return ""

        if isinstance(
            cell,
            str,
        ):
            return cls._clean_text(cell)

        if isinstance(
            cell,
            (int, float, bool),
        ):
            return str(cell)

        if isinstance(
            cell,
            dict,
        ):

            for key in (
                "text",
                "value",
                "content",
                "raw_text",
                "label",
            ):

                value = cell.get(key)

                if isinstance(
                    value,
                    (str, int, float, bool),
                ):

                    cleaned = cls._clean_text(
                        value
                    )

                    if cleaned:
                        return cleaned

            return ""

        return cls._clean_text(cell)

    @classmethod
    def _first_int(
        cls,
        item: dict[str, Any],
        keys: tuple[str, ...],
    ) -> int | None:

        for key in keys:

            value = item.get(key)

            if isinstance(
                value,
                int,
            ):
                return value

            if isinstance(
                value,
                float,
            ) and value.is_integer():
                return int(value)

            if value is not None:

                match = re.search(
                    r"-?\d+",
                    str(value),
                )

                if match:
                    try:
                        return int(
                            match.group(0)
                        )
                    except ValueError:
                        pass

        return None

    # =========================================================
    # FIGURES
    # =========================================================

    @classmethod
    def _build_figures(
        cls,
        raw_figures: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:

        result: list[dict[str, Any]] = []
        seen: set[str] = set()

        if not isinstance(raw_figures, list):
            return result

        for index, figure in enumerate(
            raw_figures,
            start=1,
        ):

            if not isinstance(figure, dict):
                continue

            caption = cls._clean_text(
                figure.get("caption")
            )

            label = cls._clean_text(
                figure.get("label")
            )

            section = cls._clean_text(
                figure.get("section")
            )

            section_path = figure.get(
                "section_path",
                [],
            )

            if not isinstance(section_path, list):
                section_path = []

            page = cls._extract_page(figure)
            coords = cls._extract_coords(figure)

            fingerprint = cls._artifact_fingerprint(
                item=figure,
                content=(
                    caption
                    or label
                ),
                page=page,
                coords=coords,
                fallback_index=index,
            )

            if fingerprint in seen:
                continue

            seen.add(fingerprint)

            result.append(
                {
                    "figure_id": (
                        figure.get("figure_id")
                        or f"figure_{index:03d}"
                    ),
                    "label": label or None,
                    "caption": caption or None,
                    "caption_status": (
                        "available"
                        if caption
                        else "missing"
                    ),
                    "section": section or None,
                    "section_path": [
                        cls._clean_text(value)
                        for value in section_path
                        if cls._clean_text(value)
                    ],
                    "page": page,
                    "coords": coords,
                    "source": "docling",
                }
            )

        return result

    # =========================================================
    # REFERENCES
    # =========================================================

    @classmethod
    def _clean_references(
        cls,
        references: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:

        result: list[dict[str, Any]] = []
        seen: set[str] = set()

        for reference in references:

            if not isinstance(reference, dict):
                continue

            title = cls._clean_text(
                reference.get("title")
            )

            raw = cls._clean_text(
                reference.get("raw")
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
                cls._clean_text(author)
                for author in authors
                if cls._clean_text(author)
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

            seen.add(key)

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
            parts.append(title)

        if abstract:
            parts.append("## Abstract")
            parts.append(abstract)

        paragraphs_by_section: dict[
            str,
            list[str],
        ] = {}

        for paragraph in paragraphs:

            text = cls._clean_text(
                paragraph.get("text")
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
            ).append(text)

        emitted_sections: set[str] = set()

        for section in sections:

            heading = section["heading"]

            parts.append(
                f"## {heading}"
            )

            emitted_sections.add(
                heading
            )

            for text in (
                paragraphs_by_section.get(
                    heading,
                    [],
                )
            ):
                parts.append(text)

        orphan_text = (
            paragraphs_by_section.get(
                "Unsectioned",
                [],
            )
        )

        if orphan_text:

            parts.append(
                "## Unsectioned"
            )

            parts.extend(orphan_text)

        for section_name, texts in (
            paragraphs_by_section.items()
        ):

            if section_name in emitted_sections:
                continue

            if section_name == "Unsectioned":
                continue

            parts.append(
                f"## {section_name}"
            )

            parts.extend(texts)

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

            if isinstance(author, dict):

                value = (
                    author.get("name")
                    or author.get("full_name")
                    or author.get("text")
                    or ""
                )

            else:
                value = author

            value = cls._clean_text(value)

            if not value:
                continue

            value = value.strip(" ,;.")

            if value:
                result.append(value)

        output: list[str] = []
        seen: set[str] = set()

        for author in result:

            key = cls._normalize(author)

            if key in seen:
                continue

            seen.add(key)
            output.append(author)

        return output

    @staticmethod
    def _clean_year(
        year: Any,
    ) -> int | None:

        if year is None:
            return None

        try:

            value = int(year)

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

        value = cls._normalize(heading)

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
            len(heading.split()) <= 10
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
    # PROVENANCE
    # =========================================================

    @classmethod
    def _extract_page(
        cls,
        item: dict[str, Any],
    ) -> int | None:

        page = item.get("page")

        if isinstance(page, int):
            return page

        if isinstance(page, float) and page.is_integer():
            return int(page)

        coords = item.get("coords")

        if isinstance(coords, list):

            for coord in coords:

                if not isinstance(coord, dict):
                    continue

                value = coord.get("page")

                if isinstance(value, int):
                    return value

                if (
                    isinstance(value, float)
                    and value.is_integer()
                ):
                    return int(value)

        provenance = item.get("prov")

        if isinstance(provenance, list):

            for prov in provenance:

                if not isinstance(prov, dict):
                    continue

                page_no = prov.get("page_no")

                if isinstance(page_no, int):
                    return page_no

                if (
                    isinstance(page_no, float)
                    and page_no.is_integer()
                ):
                    return int(page_no)

        return None

    @classmethod
    def _extract_coords(
        cls,
        item: dict[str, Any],
    ) -> list[dict[str, Any]]:

        coords = item.get("coords")

        if isinstance(coords, list):

            normalized: list[
                dict[str, Any]
            ] = []

            for coord in coords:

                if not isinstance(coord, dict):
                    continue

                normalized_coord = (
                    cls._normalize_coord(
                        coord
                    )
                )

                if normalized_coord:
                    normalized.append(
                        normalized_coord
                    )

            if normalized:
                return normalized

        provenance = item.get("prov")

        if not isinstance(provenance, list):
            return []

        output: list[
            dict[str, Any]
        ] = []

        for prov in provenance:

            if not isinstance(prov, dict):
                continue

            page = prov.get("page_no")
            bbox = prov.get("bbox")

            if not isinstance(bbox, dict):
                continue

            left = cls._to_float(
                bbox.get("l")
            )

            right = cls._to_float(
                bbox.get("r")
            )

            top = cls._to_float(
                bbox.get("t")
            )

            bottom = cls._to_float(
                bbox.get("b")
            )

            if (
                left is None
                or right is None
                or top is None
                or bottom is None
            ):
                continue

            output.append(
                {
                    "page": (
                        int(page)
                        if isinstance(
                            page,
                            (int, float),
                        )
                        and float(page).is_integer()
                        else page
                    ),
                    "x": left,
                    "y": bottom,
                    "w": abs(right - left),
                    "h": abs(top - bottom),
                    "coord_origin": (
                        bbox.get(
                            "coord_origin"
                        )
                    ),
                }
            )

        return output

    @classmethod
    def _normalize_coord(
        cls,
        coord: dict[str, Any],
    ) -> dict[str, Any] | None:

        page = coord.get("page")

        if page is None:
            page = coord.get("page_no")

        x = coord.get("x")

        if x is None:
            x = coord.get("left")

        y = coord.get("y")

        if y is None:
            y = coord.get("bottom")

        w = coord.get("w")

        if w is None:
            w = coord.get("width")

        h = coord.get("h")

        if h is None:
            h = coord.get("height")

        x = cls._to_float(x)
        y = cls._to_float(y)
        w = cls._to_float(w)
        h = cls._to_float(h)

        if (
            x is None
            or y is None
            or w is None
            or h is None
        ):
            return None

        return {
            "page": page,
            "x": x,
            "y": y,
            "w": abs(w),
            "h": abs(h),
            "coord_origin": coord.get(
                "coord_origin"
            ),
        }

    # =========================================================
    # ARTIFACT FINGERPRINTING
    # =========================================================

    @classmethod
    def _artifact_fingerprint(
        cls,
        *,
        item: dict[str, Any],
        content: str,
        page: int | None,
        coords: list[dict[str, Any]],
        fallback_index: int = 0,
    ) -> str:

        normalized_content = cls._normalize(
            content
        )

        normalized_coords = []

        for coord in coords:

            normalized_coords.append(
                {
                    "page": coord.get("page"),
                    "x": cls._round_float(
                        coord.get("x")
                    ),
                    "y": cls._round_float(
                        coord.get("y")
                    ),
                    "w": cls._round_float(
                        coord.get("w")
                    ),
                    "h": cls._round_float(
                        coord.get("h")
                    ),
                }
            )

        source_id = (
            item.get("id")
            or item.get("self_ref")
            or item.get("ref")
            or item.get("table_id")
            or item.get("figure_id")
            or item.get("formula_id")
        )

        payload = {
            "source_id": source_id,
            "label": cls._normalize(
                item.get("label")
            ),
            "caption": cls._normalize(
                item.get("caption")
            ),
            "content": normalized_content,
            "page": page,
            "coords": normalized_coords,
        }

        # When no distinguishing information exists at all,
        # preserve the source record instead of collapsing it.
        has_distinguishing_information = any(
            (
                source_id,
                payload["label"],
                payload["caption"],
                normalized_content,
                page is not None,
                bool(normalized_coords),
            )
        )

        if not has_distinguishing_information:
            payload["fallback_index"] = fallback_index

        raw = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
        )

        return hashlib.sha256(
            raw.encode("utf-8")
        ).hexdigest()

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

            if not isinstance(block, dict):
                continue

            label = cls._normalize(
                block.get(
                    "label",
                    "",
                )
            )

            if (
                heading_only
                and label != "section_header"
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
                        cls._extract_coords(
                            block
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

            selected.extend(coords)

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
    # TEXT HELPERS
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
    def _clean_multiline_text(
        value: Any,
    ) -> str:

        if value is None:
            return ""

        text = str(value).replace(
            "\xa0",
            " ",
        )

        lines = [
            re.sub(
                r"[ \t]+",
                " ",
                line,
            ).strip()
            for line in text.splitlines()
        ]

        lines = [
            line
            for line in lines
            if line
        ]

        return "\n".join(lines).strip()

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

    @staticmethod
    def _to_float(
        value: Any,
    ) -> float | None:

        try:

            if value is None:
                return None

            return float(value)

        except (
            TypeError,
            ValueError,
        ):
            return None

    @staticmethod
    def _round_float(
        value: Any,
    ) -> float | None:

        if value is None:
            return None

        try:
            return round(
                float(value),
                4,
            )
        except (
            TypeError,
            ValueError,
        ):
            return None
