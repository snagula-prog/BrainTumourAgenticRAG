from __future__ import annotations

from difflib import SequenceMatcher
import re
from typing import Any


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
        
        formulas = self._build_formulas(
        artifacts.get("formulas", [])
        )

        figures = self._clean_artifacts(
            artifacts.get(
                "figures",
                [],
            )
        )

        tables = self._clean_artifacts(
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

            # Docling commonly keeps the rendered/processed
            # value in `text` and the extracted source in `orig`.
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

            # Avoid duplicate formula records.
            page = None
            coords = []

            provenance = formula.get(
                "prov",
                [],
            )

            if provenance:
                first_prov = provenance[0]

                page = first_prov.get(
                    "page_no"
                )

                bbox = first_prov.get(
                    "bbox"
                )

                if bbox:
                    coords.append({
                        "page": page,
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

            formulas.append({
                "formula_id": (
                    f"formula_{index:03d}"
                ),
                "text": text,
                "page": page,
                "coords": coords,
                "source": "docling",
            })

        return formulas

    # =========================================================
    # ARTIFACTS
    # =========================================================

    @classmethod
    def _clean_artifacts(
        cls,
        items: list[dict[str, Any]],
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

            result.append(
                cleaned
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