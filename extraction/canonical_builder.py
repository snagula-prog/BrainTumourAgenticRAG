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
        fallback source for abstract recovery and conservative page-local
        reading-order repair when GROBID emits layout-order fragments.

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

        text_blocks = artifacts.get(
            "text_blocks",
            [],
        )

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

        # GROBID can fail completely on publisher/title-page layouts. Recover
        # only missing metadata from the already extracted Docling/PDF evidence
        # rather than overriding successful scholarly metadata.
        recovered_meta = self._recover_missing_metadata(
            title=title,
            authors=authors,
            year=year,
            pymupdf=pymupdf,
            text_blocks=text_blocks,
        )
        title = recovered_meta["title"]
        authors = recovered_meta["authors"]
        year = recovered_meta["year"]

        doi = self._clean_text(
            grobid.get("doi")
        ) or None

        keywords = self._extract_keywords(
            grobid=grobid,
            pymupdf=pymupdf,
        )

        abstract = self._extract_abstract(
            grobid=grobid,
            pymupdf=pymupdf,
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

        ordered_paragraphs, reading_order_report = (
            self._repair_paragraph_reading_order(
                paragraphs=grobid.get(
                    "paragraphs",
                    [],
                ),
                pymupdf=pymupdf,
            )
        )

        paragraphs = self._build_paragraphs(
            paragraphs=ordered_paragraphs,
            section_lookup=section_lookup,
            abstract=abstract,
            text_blocks=text_blocks,
        )

        biographies = self._extract_author_biographies(
            grobid=grobid,
            pymupdf=pymupdf,
            authors=authors,
        )
        if biographies:
            bio_heading = "Author Biographies"
            if not any(s.get("heading") == bio_heading for s in sections):
                sections.append({
                    "heading": bio_heading,
                    "level": 1,
                    "path": [bio_heading],
                    "coords": [],
                })
            paragraphs.extend(biographies)

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

        paragraphs, table_derived_paragraphs = self._strip_table_derived_paragraphs(
        paragraphs, tables
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

        artifact_reconstruction = dict(reconstructed.get("stats", {}))
        artifact_reconstruction["table_derived_paragraphs_removed"] = len(
            table_derived_paragraphs
        )

        references, reference_recovery = self._recover_references(
            grobid_references=grobid.get(
                "references",
                [],
            ),
            paragraphs=paragraphs,
            pymupdf=pymupdf,
        )

        # Reference-list fragments sometimes leak into GROBID body paragraphs.
        # Remove only paragraphs that are overwhelmingly reference records;
        # ordinary prose containing bracketed citations is preserved.
        paragraphs, reference_contamination_removed = (
            self._remove_reference_contamination(paragraphs)
        )
        reference_recovery["contamination_paragraphs_removed"] = (
            len(reference_contamination_removed)
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
                "keywords": keywords,
            },
            "reading_order_repair": reading_order_report,
            "abstract": abstract,
            "sections": sections,
            "paragraphs": paragraphs,
            "formulas": formulas,
            "figures": figures,
            "tables": tables,
            "visual_artifacts": visual_artifacts,
            "unclassified_visuals": unclassified_visuals,
            "artifact_reconstruction": artifact_reconstruction,
            "table_derived_paragraphs": table_derived_paragraphs,
            "references": references,
            "reference_recovery": reference_recovery,
            "canonical_text": canonical_text,
            "quality": quality,
        }

    # =========================================================
    # METADATA RECOVERY
    # =========================================================

    @classmethod
    def _recover_missing_metadata(
        cls,
        *,
        title: str,
        authors: list[str],
        year: int | None,
        pymupdf: dict[str, Any],
        text_blocks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Recover only metadata fields that GROBID left empty.

        This is intentionally conservative and document-generic.  It uses
        first-page Docling text blocks for title/author recovery and explicit
        publication-date phrases for the year.
        """
        recovered_title = title
        recovered_authors = list(authors)
        recovered_year = year

        page_one_blocks: list[dict[str, Any]] = []
        for block in text_blocks if isinstance(text_blocks, list) else []:
            if not isinstance(block, dict):
                continue
            coords = block.get("coords") or []
            if any(isinstance(c, dict) and c.get("page") == 1 for c in coords):
                page_one_blocks.append(block)

        def block_y(block: dict[str, Any]) -> float:
            coords = block.get("coords") or []
            for coord in coords:
                if isinstance(coord, dict) and coord.get("page") == 1:
                    try:
                        return float(coord.get("y", 0.0)) + float(coord.get("h", 0.0))
                    except (TypeError, ValueError):
                        return 0.0
            return 0.0

        if not recovered_title:
            title_candidates: list[tuple[float, str]] = []
            for block in page_one_blocks:
                text = cls._clean_text(block.get("text"))
                label = cls._clean_text(block.get("label")).lower()
                if len(text.split()) < 4 or len(text) > 220:
                    continue
                if label not in {"section_header", "title", "heading"}:
                    continue
                lower = text.casefold()
                if re.search(r"\b(?:abstract|index terms|keywords|doi|volume|vol\.|published on)\b", lower):
                    continue
                if re.match(r"^(?:[ivxlcdm]+\.|\d+(?:\.\d+)*\.)\s*\S", text, re.IGNORECASE):
                    continue
                score = min(len(text.split()), 30) + min(block_y(block) / 100.0, 8.0)
                title_candidates.append((score, text))
            if title_candidates:
                recovered_title = max(title_candidates, key=lambda item: item[0])[1]

        if not recovered_authors and recovered_title:
            title_y = 0.0
            for block in page_one_blocks:
                if cls._clean_text(block.get("text")) == recovered_title:
                    title_y = block_y(block)
                    break

            author_candidates: list[tuple[float, str]] = []
            for block in page_one_blocks:
                text = cls._clean_text(block.get("text"))
                y = block_y(block)
                if y >= title_y or not text or len(text.split()) < 2 or len(text.split()) > 18:
                    continue
                label = cls._clean_text(block.get("label")).lower()
                if label not in {"text", "author", "authors"}:
                    continue
                lower = text.casefold()
                if any(term in lower for term in ("department", "university", "journal", "email", "e-mail", "doi", "published", "volume")):
                    continue
                if re.search(r"@|\bhttps?://|\b(?:school|faculty|institute)\b", lower):
                    continue
                if " and " in lower or re.search(r"\b[A-Z]\.\s*[A-Z]\w+", text):
                    author_candidates.append((abs(title_y - y), text))
            if author_candidates:
                best = min(author_candidates, key=lambda item: item[0])[1]
                recovered_authors = cls._clean_authors(
                    [part.strip() for part in re.split(r"\s+and\s+|\s*,\s*", best) if part.strip()]
                )

        if recovered_year is None:
            pdf_text = str(pymupdf.get("text") or "") if isinstance(pymupdf, dict) else ""
            published_match = re.search(
                r"\bpublished\s+(?:on|:)?\s*[^\n]{0,80}?(20\d{2})\b",
                pdf_text,
                re.IGNORECASE,
            )
            if published_match:
                recovered_year = int(published_match.group(1))

        return {
            "title": recovered_title,
            "authors": recovered_authors,
            "year": recovered_year,
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
        abstract_start_section = ""

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

            # Some GROBID outputs fail to classify the abstract and instead
            # leave it as the first body paragraph, e.g.
            # ``Abstract-Brain Tumor ...``.  Recognize that document-local
            # marker before relying on the PDF fallback.
            abstract_marker = re.match(
                r"^abstract\s*[-–—:]?\s*(.*)$",
                text,
                flags=re.IGNORECASE,
            )
            if abstract_marker:
                body = cls._clean_text(abstract_marker.group(1))
                if body:
                    abstract_started = True
                    abstract_start_section = section
                    abstract_parts.append(body)
                    continue

            if section == "abstract":
                abstract_started = True
                abstract_start_section = section
                abstract_parts.append(
                    text
                )
                continue

            if abstract_started:

                # Do not let the index-terms line become part of the abstract.
                if section.startswith("index terms") or section.startswith("keywords"):
                    break

                if cls._is_major_section(
                    section
                ):
                    break

                # GROBID may place an unclassified abstract paragraph in a
                # synthetic/garbage section, followed immediately by the real
                # first section such as ``I. INTRODUCTION``. Stop at that
                # transition rather than swallowing the entire body.
                if (
                    abstract_start_section
                    and section
                    and section != abstract_start_section
                    and re.match(r"^(?:[ivxlcdm]+\.|\d+(?:\.\d+)*\.)\s*\S", section, re.IGNORECASE)
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

    @classmethod
    def _extract_keywords(
        cls,
        *,
        grobid: dict[str, Any],
        pymupdf: dict[str, Any],
    ) -> list[str]:
        raw_kw = grobid.get("keywords")
        if raw_kw:
            if isinstance(raw_kw, list):
                return [cls._clean_text(k) for k in raw_kw if cls._clean_text(k)]
            if isinstance(raw_kw, str):
                return [cls._clean_text(k) for k in re.split(r"[,;]", raw_kw) if cls._clean_text(k)]

        pdf_text = ""
        if isinstance(pymupdf, dict):
            pdf_text = pymupdf.get("text", "")
            if not pdf_text and "pages" in pymupdf and pymupdf["pages"]:
                pdf_text = pymupdf["pages"][0].get("text", "")

        m = re.search(
            r"\b(?:INDEX\s+TERMS|KEYWORDS?)\b\s*[-–—:]?\s*"
            r"(?P<body>.*?)"
            r"(?=\n\s*(?:[IVXLCDM]+(?:\.[IVXLCDM]+)*\.|\d+(?:\.\d+)*\.)\s+[A-Z]|$)",
            pdf_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if m:
            extracted = m.group("body").strip()
            parts = [cls._clean_text(p) for p in re.split(r"[,;]", extracted) if cls._clean_text(p)]
            if parts:
                return parts
        return []

    @classmethod
    def _extract_author_biographies(
        cls,
        *,
        grobid: dict[str, Any],
        pymupdf: dict[str, Any],
        authors: list[str],
    ) -> list[dict[str, Any]]:
        """Extract biographies only when there is strong document evidence.

        Priority:
        1. Structured biographies emitted by the GROBID parser, after
           re-validating author association.
        2. A conservative PyMuPDF fallback restricted to late-page back matter.

        No document-specific author names, page numbers, or publisher strings
        are assumed here.
        """

        def normalize_name(value: Any) -> str:
            text = cls._clean_text(value)
            text = re.sub(r"\d+", " ", text)
            text = re.sub(r"[^\w\s'’.-]", " ", text, flags=re.UNICODE)
            return re.sub(r"\s+", " ", text).strip().casefold()

        def name_tokens(value: Any) -> set[str]:
            return set(re.findall(r"[\w’'-]+", normalize_name(value)))

        author_values = [
            cls._clean_text(value)
            for value in authors
            if isinstance(value, str) and cls._clean_text(value)
        ]

        def matches_author(candidate: Any) -> bool:
            candidate_norm = normalize_name(candidate)
            if not candidate_norm:
                return False
            candidate_tokens = name_tokens(candidate)
            for author in author_values:
                author_norm = normalize_name(author)
                author_tokens = name_tokens(author)
                if not author_tokens or not candidate_tokens:
                    continue
                shared = author_tokens & candidate_tokens
                coverage = len(shared) / len(author_tokens)
                similarity = SequenceMatcher(
                    None,
                    author_norm,
                    candidate_norm,
                ).ratio()
                if len(shared) >= 2 and coverage >= 0.66:
                    return True
                if similarity >= 0.82:
                    return True
            return False

        bio_signal_re = re.compile(
            r"\b(?:received|was born|is currently|currently pursuing|"
            r"currently working|research interests?|has published|"
            r"has authored|earned|obtained|degrees? in)\b",
            re.IGNORECASE,
        )

        def clean_biography(text: Any) -> str:
            value = cls._clean_text(text)
            if not value:
                return ""
            value = re.sub(
                r"\b\d{4,5}\s+VOLUME\s+\d+\s*,?\s*\d{1,2}\b",
                " ",
                value,
                flags=re.IGNORECASE,
            )
            value = re.sub(
                r"\bVOLUME\s+\d+\s*,?\s*\d{1,2}\b",
                " ",
                value,
                flags=re.IGNORECASE,
            )
            value = re.sub(r"[\r\n\t]+", " ", value)
            value = re.sub(r" {2,}", " ", value).strip()
            return value

        # ---- 1. Structured GROBID biographies ----------------------------
        structured = grobid.get("author_biographies")
        if isinstance(structured, list):
            result: list[dict[str, Any]] = []
            seen: set[str] = set()
            for item in structured:
                if not isinstance(item, dict):
                    continue
                text = clean_biography(item.get("text"))
                author = cls._clean_text(item.get("author"))
                if len(text.split()) < 15 or len(text.split()) > 300:
                    continue
                if not author or not matches_author(author):
                    continue
                if len(bio_signal_re.findall(text)) < 2:
                    continue
                key = cls._normalize(text)
                if key in seen:
                    continue
                seen.add(key)
                result.append(
                    {
                        "text": text,
                        "sentences": [
                            part.strip()
                            for part in re.split(
                                r"(?<=[.!?])\s+",
                                text,
                            )
                            if part.strip()
                        ],
                        "author": author,
                        "section": "Author Biographies",
                        "section_path": ["Author Biographies"],
                        "coords": (
                            item.get("coords")
                            if isinstance(item.get("coords"), list)
                            else []
                        ),
                        "page": item.get("page"),
                    }
                )
            if result:
                return result

        # ---- 2. Conservative PDF back-matter fallback ---------------------
        pages = pymupdf.get("pages", []) if isinstance(pymupdf, dict) else []
        if not pages or not author_values:
            return []

        # Author biographies are normally back matter. Restrict fallback to
        # the final quarter of pages so title/abstract authors cannot be
        # mistaken for biographies.
        start_index = max(0, len(pages) - max(2, (len(pages) + 3) // 4))
        back_matter_pages = pages[start_index:]

        author_pattern = re.compile(
            "|".join(
                sorted(
                    (re.escape(value) for value in author_values),
                    key=len,
                    reverse=True,
                )
            ),
            re.IGNORECASE,
        )

        result: list[dict[str, Any]] = []
        seen: set[str] = set()

        for page in back_matter_pages:
            if not isinstance(page, dict):
                continue
            page_text = str(page.get("text") or "")
            if not page_text:
                continue

            matches = list(author_pattern.finditer(page_text))
            for index, match in enumerate(matches):
                # Only consider an author occurrence when it appears reasonably
                # close to the start of the candidate block. This avoids taking
                # an author mentioned in references and turning the remainder of
                # the page into a biography.
                start_pos = match.start()
                end_pos = (
                    matches[index + 1].start()
                    if index + 1 < len(matches)
                    else len(page_text)
                )
                candidate = clean_biography(
                    page_text[start_pos:end_pos]
                )
                if not candidate:
                    continue

                words = candidate.split()
                if len(words) < 15 or len(words) > 300:
                    continue
                if len(bio_signal_re.findall(candidate)) < 2:
                    continue
                if not matches_author(match.group(0)):
                    continue

                # Reject obvious article-body/reference blocks.
                candidate_lower = candidate.casefold()
                if "abstract" in candidate_lower[:120]:
                    continue
                if "references" in candidate_lower[:120]:
                    continue
                if re.search(
                    r"\b(?:https?://|doi:\s*10\.)",
                    candidate,
                    re.IGNORECASE,
                ):
                    continue

                key = cls._normalize(candidate)
                if key in seen:
                    continue
                seen.add(key)

                result.append(
                    {
                        "text": candidate,
                        "sentences": [
                            part.strip()
                            for part in re.split(
                                r"(?<=[.!?])\s+",
                                candidate,
                            )
                            if part.strip()
                        ],
                        "author": cls._clean_text(match.group(0)),
                        "section": "Author Biographies",
                        "section_path": ["Author Biographies"],
                        "coords": [],
                        "page": page.get("page_number"),
                    }
                )

        return result

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

            normalized_heading = cls._normalize(heading)

            # Index-term/keyword lines belong to metadata, not the section
            # hierarchy. Garbage-only headings are also excluded before they
            # can affect section lookup.
            if normalized_heading.startswith(("index terms", "index term", "keywords")):
                continue

            if cls._is_garbage_heading(
                heading
            ):
                continue

            raw_path = section.get("path")
            path_key = tuple(
                cls._normalize(item)
                for item in raw_path
                if cls._clean_text(item)
            ) if isinstance(raw_path, list) else (normalized_heading,)
            key = (path_key, int(section.get("level", 1) or 1))

            if key in seen:
                continue

            seen.add(key)

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

            text = re.sub(
                r"\s*The associate editor coordinating the review of this manuscript and approving it for publication was [^.]+\.\s*",
                " ",
                text
            )
            text = re.sub(r" {2,}", " ", text).strip()

            if not text or cls._is_noise_paragraph(text):
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

            # A parser may assign formula fragments / running banners as a
            # section. Treat those exactly like a missing section so the
            # previous real section remains active.
            if section_value and cls._is_garbage_heading(section_value):
                section_value = ""

            section_key = cls._normalize(
                section_value
            )

            raw_path = paragraph.get(
                "section_path"
            )

            clean_path: list[str] = []
            if isinstance(raw_path, list):
                for value in raw_path:
                    cleaned = cls._clean_text(value)
                    if cleaned and not cls._is_garbage_heading(cleaned):
                        clean_path.append(cleaned)

            if section_key:

                section = section_lookup.get(
                    section_key,
                    section_value,
                )

                last_section = section

                if clean_path:
                    last_section_path = clean_path.copy()
                else:
                    last_section_path = [section]

            else:

                section = last_section

                if not clean_path and last_section_path:
                    clean_path = last_section_path.copy()

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

            seen.add(dedup_key)

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
                    "paragraph_id": f"para_{len(result) + 1:04d}",
                    "text": text,
                    "sentences": sentences,
                    "section": section,
                    "section_path": (
                        clean_path
                        if clean_path
                        else ([section] if section else [])
                    ),
                    "coords": coords,
                    "page": page,
                }
            )

        return result

    # =========================================================
    # READING ORDER
    # =========================================================

    @classmethod
    def _repair_paragraph_reading_order(
        cls,
        *,
        paragraphs: list[dict[str, Any]],
        pymupdf: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Repair page-local paragraph order using PDF layout evidence.

        GROBID occasionally emits a page's column fragments in structural
        order rather than physical reading order. This fallback does not
        globally sort the document and does not rely on paper-specific text.
        It only reorders paragraph records on pages where enough paragraph text
        can be confidently matched back to PyMuPDF text blocks.
        """

        if not isinstance(paragraphs, list) or not paragraphs:
            return paragraphs, {
                "applied": False,
                "reason": "no_paragraphs",
                "pages_reordered": [],
                "paragraphs_reordered": 0,
            }

        pages = pymupdf.get("pages") if isinstance(pymupdf, dict) else None
        if not isinstance(pages, list) or not pages:
            return paragraphs, {
                "applied": False,
                "reason": "no_pymupdf_pages",
                "pages_reordered": [],
                "paragraphs_reordered": 0,
            }

        def normalize(value: Any) -> str:
            text = cls._clean_text(value)
            return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()

        def tokens(value: Any) -> list[str]:
            value = normalize(value)
            return value.split() if value else []

        def numeric_page(value: Any) -> int | None:
            try:
                page = int(value)
            except (TypeError, ValueError):
                return None
            return page if page > 0 else None

        # Build a compact searchable representation of PDF text blocks. The
        # same PyMuPDF block schema is already used by reference recovery.
        blocks_by_page: dict[int, list[dict[str, Any]]] = {}
        page_widths: dict[int, float] = {}

        for page_index, page in enumerate(pages, start=1):
            if not isinstance(page, dict):
                continue

            page_number = numeric_page(page.get("page_number")) or page_index
            try:
                page_widths[page_number] = float(page.get("width"))
            except (TypeError, ValueError):
                page_widths[page_number] = 0.0

            raw_blocks = page.get("blocks")
            if not isinstance(raw_blocks, list):
                continue

            page_blocks: list[dict[str, Any]] = []
            for block_index, block in enumerate(raw_blocks):
                if not isinstance(block, dict):
                    continue

                text = cls._clean_text(block.get("text"))
                if not text:
                    continue

                block_tokens = tokens(text)
                if len(block_tokens) < 3:
                    continue

                try:
                    x0 = float(block.get("x0", 0.0))
                    x1 = float(block.get("x1", x0))
                    y0 = float(block.get("y0", 0.0))
                except (TypeError, ValueError):
                    continue

                page_blocks.append({
                    "index": block_index,
                    "text": text,
                    "tokens": block_tokens,
                    "x0": x0,
                    "x1": x1,
                    "y0": y0,
                })

            if page_blocks:
                blocks_by_page[page_number] = page_blocks

        if not blocks_by_page:
            return paragraphs, {
                "applied": False,
                "reason": "no_text_blocks",
                "pages_reordered": [],
                "paragraphs_reordered": 0,
            }

        def match_score(
            paragraph_text: str,
            paragraph_tokens: list[str],
            block: dict[str, Any],
        ) -> tuple[float, int]:
            block_text = block["text"]
            block_tokens = block["tokens"]

            shared = set(paragraph_tokens) & set(block_tokens)
            shared_count = len(shared)
            if shared_count < 5:
                return 0.0, shared_count

            para_norm = normalize(paragraph_text)
            block_norm = normalize(block_text)

            if para_norm == block_norm:
                return 1.0, shared_count

            prefix_len = min(24, len(paragraph_tokens))
            suffix_len = min(24, len(paragraph_tokens))
            prefix = " ".join(paragraph_tokens[:prefix_len])
            suffix = " ".join(paragraph_tokens[-suffix_len:])

            prefix_tokens = set(prefix.split())
            suffix_tokens = set(suffix.split())
            prefix_overlap = (
                len(prefix_tokens & set(block_tokens)) / max(len(prefix_tokens), 1)
            )
            suffix_overlap = (
                len(suffix_tokens & set(block_tokens)) / max(len(suffix_tokens), 1)
            )

            block_coverage = shared_count / max(len(block_tokens), 1)
            paragraph_coverage = shared_count / max(len(paragraph_tokens), 1)

            score = max(
                0.60 * block_coverage + 0.40 * prefix_overlap,
                0.60 * block_coverage + 0.40 * suffix_overlap,
                0.50 * block_coverage + 0.25 * prefix_overlap + 0.25 * suffix_overlap,
                0.35 * paragraph_coverage,
            )

            # Character similarity provides a useful tie-break for near-identical
            # fragments without requiring any document-specific terms.
            if para_norm and block_norm:
                similarity = SequenceMatcher(
                    None,
                    para_norm[:400],
                    block_norm[:400],
                ).ratio()
                score = max(score, 0.55 * similarity + 0.45 * block_coverage)

            return score, shared_count

        mappings: dict[int, dict[str, Any]] = {}
        candidate_page_count = 0

        for paragraph_index, paragraph in enumerate(paragraphs):
            if not isinstance(paragraph, dict):
                continue

            text = cls._clean_text(paragraph.get("text"))
            paragraph_tokens = tokens(text)
            if len(paragraph_tokens) < 6:
                continue

            known_page = numeric_page(paragraph.get("page"))
            candidate_pages = (
                [known_page]
                if known_page in blocks_by_page
                else list(blocks_by_page)
            )

            best: tuple[float, int | None, dict[str, Any] | None, int] = (
                0.0,
                None,
                None,
                0,
            )

            for page_number in candidate_pages:
                if page_number is None:
                    continue
                for block in blocks_by_page.get(page_number, []):
                    score, shared_count = match_score(
                        text,
                        paragraph_tokens,
                        block,
                    )
                    if score > best[0]:
                        best = (
                            score,
                            page_number,
                            block,
                            shared_count,
                        )

            score, page_number, block, shared_count = best
            if block is None:
                continue

            # Confidence threshold intentionally requires both meaningful
            # lexical overlap and a reasonably strong block match.
            if score < 0.50 or shared_count < 5:
                continue

            mappings[paragraph_index] = {
                "page": page_number,
                "x0": block["x0"],
                "x1": block["x1"],
                "y0": block["y0"],
                "score": score,
            }
            candidate_page_count += 1

        if candidate_page_count < 2:
            return paragraphs, {
                "applied": False,
                "reason": "insufficient_confident_matches",
                "matched_paragraphs": candidate_page_count,
                "pages_reordered": [],
                "paragraphs_reordered": 0,
            }

        result = list(paragraphs)
        pages_reordered: list[int] = []
        moved_count = 0
        reordered_page_details: list[dict[str, Any]] = []

        mapped_by_page: dict[int, list[int]] = {}
        all_by_page: dict[int, list[int]] = {}

        for index, mapping in mappings.items():
            mapped_by_page.setdefault(mapping["page"], []).append(index)

        # Include known paragraph pages even when an individual paragraph did
        # not match; this lets us enforce a conservative page-level coverage
        # threshold before changing order.
        for index, paragraph in enumerate(paragraphs):
            if not isinstance(paragraph, dict):
                continue
            page_number = numeric_page(paragraph.get("page"))
            if page_number is not None:
                all_by_page.setdefault(page_number, []).append(index)
            elif index in mappings:
                all_by_page.setdefault(mappings[index]["page"], []).append(index)

        def page_reading_key(
            page_number: int,
            indices: list[int],
        ) -> dict[int, tuple[int, float, float, int]]:
            mapped = [mappings[i] for i in indices if i in mappings]
            page_width = page_widths.get(page_number, 0.0)
            if page_width <= 0:
                page_width = max(
                    (float(m["x1"]) for m in mapped),
                    default=600.0,
                )

            x_values = sorted(float(m["x0"]) for m in mapped)
            clusters: list[list[float]] = []
            gap_threshold = max(40.0, page_width * 0.12)
            for x in x_values:
                if not clusters or x - clusters[-1][-1] > gap_threshold:
                    clusters.append([x])
                else:
                    clusters[-1].append(x)

            # A large horizontal gap between text starts is enough evidence
            # for column-major order when the matched paragraph set spans both
            # sides. This also handles pages where one column contains only a
            # single matched paragraph.
            use_columns = len(clusters) >= 2

            column_bounds: list[tuple[float, float, int]] = []
            if use_columns:
                for column_index, cluster in enumerate(clusters):
                    column_bounds.append((min(cluster), max(cluster), column_index))

            keys: dict[int, tuple[int, float, float, int]] = {}
            for index in indices:
                mapping = mappings.get(index)
                if mapping is None:
                    continue

                x0 = float(mapping["x0"])
                y0 = float(mapping["y0"])
                column_index = 0
                if use_columns:
                    # Assign by nearest x-cluster center.
                    column_index = min(
                        column_bounds,
                        key=lambda item: abs(
                            x0 - ((item[0] + item[1]) / 2.0)
                        ),
                    )[2]

                keys[index] = (
                    column_index if use_columns else 0,
                    y0,
                    x0,
                    index,
                )

            return keys

        for page_number in sorted(mapped_by_page):
            mapped_indices = mapped_by_page[page_number]
            all_indices = all_by_page.get(page_number, mapped_indices)
            if len(mapped_indices) < 2:
                continue

            coverage = len(mapped_indices) / max(len(all_indices), 1)
            if coverage < 0.80:
                continue

            keys = page_reading_key(page_number, all_indices)
            ordered_mapped = sorted(
                mapped_indices,
                key=lambda index: keys.get(index, (9999, 999999.0, 999999.0, index)),
            )

            original_mapped = list(mapped_indices)
            if ordered_mapped == original_mapped:
                continue

            target_slots = sorted(mapped_indices)
            for slot, source_index in zip(target_slots, ordered_mapped):
                if result[slot] is not paragraphs[source_index]:
                    moved_count += 1
                result[slot] = paragraphs[source_index]

            pages_reordered.append(page_number)
            reordered_page_details.append({
                "page": page_number,
                "matched_paragraphs": len(mapped_indices),
                "page_paragraphs": len(all_indices),
                "coverage": coverage,
            })
            
        moved_count = sum(
            1 for index, item in enumerate(result[:len(paragraphs)])
            if index < len(paragraphs)
            and item is not paragraphs[index]
        )
        
        # Merge unambiguous continuations across adjacent PDF pages.
        # This preserves one logical paragraph across page breaks.

        page_text_by_number = {}

        for page_index, page in enumerate(pages, start=1):
            if not isinstance(page, dict):
                continue

            page_number = (
                numeric_page(page.get("page_number")) or page_index
            )
            page_tokens = tokens(page.get("text", ""))

            if page_tokens:
                page_text_by_number[page_number] = page_tokens

        def contains_sequence(
            haystack: list[str],
            needle: list[str],
        ) -> bool:
            if not needle or len(needle) > len(haystack):
                return False

            size = len(needle)
            return any(
                haystack[i:i + size] == needle
                for i in range(len(haystack) - size + 1)
            )

        def edge_match_size(
            page_tokens: list[str],
            paragraph_tokens: list[str],
            *,
            end: bool,
        ) -> int:
            # Use progressively shorter exact token matches to tolerate
            # minor PDF hyphenation and text-extraction differences.
            for size in range(
                min(8, len(paragraph_tokens)), 4, -1
            ):
                needle = (
                    paragraph_tokens[-size:]
                    if end
                    else paragraph_tokens[:size]
                )
                if contains_sequence(page_tokens, needle):
                    return size

            return 0

        # Find fragments whose text and actual PDF page boundary
        # support joining them.
        boundary_pages_by_pair: dict[
            tuple[int, int], set[int]
        ] = {}

        for left_index, left in enumerate(result):
            if not isinstance(left, dict):
                continue

            left_text = cls._clean_text(left.get("text"))
            left_tokens = tokens(left_text)

            if (
                not re.search(r"[,;:]\s*$", left_text)
                or len(left_tokens) < 5
            ):
                continue

            for right_index, right in enumerate(result):
                if left_index == right_index or not isinstance(right, dict):
                    continue

                right_text = cls._clean_text(right.get("text"))
                right_tokens = tokens(right_text)

                if (
                    not right_text
                    or not right_text[0].islower()
                    or len(right_tokens) < 5
                ):
                    continue

                for page_number, current_tokens in page_text_by_number.items():
                    next_tokens = page_text_by_number.get(
                        page_number + 1
                    )
                    if not next_tokens:
                        continue

                    tail_match = edge_match_size(
                        current_tokens[-120:],
                        left_tokens,
                        end=True,
                    )
                    head_match = edge_match_size(
                        next_tokens[:120],
                        right_tokens,
                        end=False,
                    )

                    if tail_match >= 5 and head_match >= 5:
                        boundary_pages_by_pair.setdefault(
                            (left_index, right_index), set()
                        ).add(page_number)

        # Merge only mutually unambiguous pairs.
        unique_pairs = {
            pair: next(iter(boundary_pages))
            for pair, boundary_pages in boundary_pages_by_pair.items()
            if len(boundary_pages) == 1
        }

        left_counts = {}
        right_counts = {}

        for left_index, right_index in unique_pairs:
            left_counts[left_index] = (
                left_counts.get(left_index, 0) + 1
            )
            right_counts[right_index] = (
                right_counts.get(right_index, 0) + 1
            )

        removed_indices = set()
        used_indices = set()
        continuation_merge_details = []

        for (left_index, right_index), boundary_page in sorted(
            unique_pairs.items()
        ):
            if (
                left_counts[left_index] != 1
                or right_counts[right_index] != 1
                or left_index in used_indices
                or right_index in used_indices
            ):
                continue

            left = result[left_index]
            right = result[right_index]

            left["text"] = (
                cls._clean_text(left.get("text"))
                + " "
                + cls._clean_text(right.get("text"))
            )

            # Preserve and join sentence data where available.
            left_sentences = left.get("sentences")
            right_sentences = right.get("sentences")

            if (
                isinstance(left_sentences, list)
                and isinstance(right_sentences, list)
            ):
                if left_sentences and right_sentences:
                    left_sentences[-1] = (
                        cls._clean_text(left_sentences[-1])
                        + " "
                        + cls._clean_text(right_sentences[0])
                    )
                    left_sentences.extend(right_sentences[1:])
                elif right_sentences:
                    left["sentences"] = right_sentences.copy()

            # Preserve provenance from both page fragments.
            left_coords = left.get("coords", [])
            right_coords = right.get("coords", [])

            if not isinstance(left_coords, list):
                left_coords = []
            if not isinstance(right_coords, list):
                right_coords = []

            combined_coords = []
            seen_coords = set()

            for coord in left_coords + right_coords:
                marker = json.dumps(
                    coord,
                    sort_keys=True,
                    ensure_ascii=False,
                )
                if marker not in seen_coords:
                    seen_coords.add(marker)
                    combined_coords.append(coord)

            left["coords"] = combined_coords

            removed_indices.add(right_index)
            used_indices.update((left_index, right_index))

            continuation_merge_details.append({
                "left_index": left_index,
                "right_index": right_index,
                "boundary_page": boundary_page,
            })

        if removed_indices:
            result = [
                paragraph
                for index, paragraph in enumerate(result)
                if index not in removed_indices
            ]

        continuation_merges = len(continuation_merge_details)

        report = {
            "applied": bool(
                pages_reordered or continuation_merges
            ),
            "reason": (
                "reordered_and_merged"
                if pages_reordered and continuation_merges
                else "merged_cross_page_continuation"
                if continuation_merges
                else "reordered_page_local_layout"
                if pages_reordered
                else "already_in_layout_order"
            ),
            "matched_paragraphs": candidate_page_count,
            "pages_reordered": pages_reordered,
            "paragraphs_reordered": moved_count,
            "cross_page_continuations_merged": continuation_merges,
            "continuation_merge_details": continuation_merge_details,
            "page_details": reordered_page_details,
        }

        return result, report

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

    # --- new helper on CanonicalBuilder ---

    TABLE_LABEL_SCAN_RE = re.compile(r"\bTABLE\s+\d+\b", re.IGNORECASE)

    @classmethod
    def _looks_like_table_dump(cls, text: str) -> bool:
        """
        Heuristic catch for GROBID emitting raw table content as a <p>
        instead of isolating it into a <table>. Two or more table labels
        inside one paragraph is a strong signal it's not prose.
        """
        return len(cls.TABLE_LABEL_SCAN_RE.findall(text)) >= 2


    @staticmethod
    def _inline_table_segment(
        text: str,
        table: dict[str, Any],
    ) -> tuple[str, str | None]:
        """Remove a small table dump embedded inside a prose paragraph.

        Only acts when a strong table-content prefix/suffix sequence is
        present, which avoids deleting ordinary prose that merely mentions a
        table. The original table artifact remains available as a structured
        table record.
        """
        content = str(table.get("content") or "").strip()
        if not content:
            return text, None

        tokens = re.findall(
            r"[A-Za-z0-9]+(?:[.'’/-][A-Za-z0-9]+)*|%",
            content,
        )

        # Do not construct large regexes for legitimately large tables.
        if len(tokens) < 10 or len(tokens) > 120:
            return text, None

        prefix = tokens[:6]
        suffix = tokens[-6:]
        sep = r"[\s|,;:()\[\]{}–—-]+"
        prefix_pat = sep.join(re.escape(t) for t in prefix)
        suffix_pat = sep.join(re.escape(t) for t in suffix)

        pattern = rf"{prefix_pat}.*?{suffix_pat}"
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if not match:
            return text, None

        # Require a nearby label/caption as a second signal before removing.
        label = str(table.get("label") or "").strip()
        caption = str(table.get("caption") or "").strip()
        nearby_end = min(len(text), match.end() + 180)
        nearby_start = max(0, match.start() - 180)
        nearby = text[nearby_start:nearby_end]

        label_hit = bool(label and re.search(re.escape(label), nearby, re.IGNORECASE))
        caption_hit = bool(caption and re.search(re.escape(caption), nearby, re.IGNORECASE))
        if not (label_hit or caption_hit):
            return text, None

        start = match.start()
        end = match.end()

        # Absorb an adjacent caption/label, whether it appears immediately
        # before or after the cell dump.
        if caption:
            cap_match_after = re.search(
                re.escape(caption),
                text[end:end + 220],
                re.IGNORECASE,
            )
            if cap_match_after:
                end += cap_match_after.end()
            else:
                cap_match_before = re.search(
                    re.escape(caption),
                    text[max(0, start - 220):start],
                    re.IGNORECASE,
                )
                if cap_match_before:
                    start = max(0, start - 220) + cap_match_before.start()

        cleaned = (text[:start].rstrip() + " " + text[end:].lstrip()).strip()
        return cleaned, text[start:end].strip()

    @classmethod
    def _strip_table_derived_paragraphs(
        cls,
        paragraphs: list[dict[str, Any]],
        tables: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Remove paragraphs that are actually table content folded into the
        body stream by the parser. Matches on:
        1. multi-label table dump heuristic
        2. paragraph text contained inside a reconstructed table's content
        3. paragraph text matching a table's caption (fuzzy)
        Returns (clean_paragraphs, removed_paragraphs) — removals are kept,
        not discarded, so this stays auditable.
        """
        table_contents = [
            cls._normalize(t.get("content", "")) for t in tables if t.get("content")
        ]
        table_captions = [
            cls._normalize(t.get("caption", "")) for t in tables if t.get("caption")
        ]

        kept, removed = [], []

        for paragraph in paragraphs:
            text = paragraph.get("text", "")
            inline_segments = []

            # Remove small structured-table dumps that were copied into body
            # paragraphs, but keep the surrounding prose intact.
            for table in tables:
                text, removed_segment = cls._inline_table_segment(text, table)
                if removed_segment:
                    inline_segments.append(removed_segment)

            if inline_segments:
                paragraph["text"] = text
                paragraph["table_derived_segments_removed"] = inline_segments
                paragraph["sentences"] = [
                    s.strip()
                    for s in re.split(r"(?<=[.!?])\s+", text)
                    if s.strip()
                ]

            # If paragraph starts with a table continuation header but contains body prose, strip the prefix
            table_prefix_match = re.match(
                r"^\s*TABLE\s+\d+\.?\s*(?:\([^\)]+\)\s*)?(?:[A-Za-z0-9\s,–-]+?\.)\s+(?=[A-Z])",
                text,
                re.IGNORECASE
            )
            if table_prefix_match:
                prefix = table_prefix_match.group(0)
                remaining = text[len(prefix):].strip()
                if len(remaining.split()) >= 15:
                    text = remaining
                    paragraph["text"] = text
                    if "sentences" in paragraph and paragraph["sentences"]:
                        paragraph["sentences"] = [
                            s for s in paragraph["sentences"]
                            if not re.match(r"^\s*TABLE\s+\d+", s, re.IGNORECASE)
                        ]

            normalized = cls._normalize(text)

            is_dump = cls._looks_like_table_dump(text)

            is_matched_content = any(
                normalized and normalized in table_text
                for table_text in table_contents
            )

            # Strip running volume headers/table-label artifacts when the
            # surrounding structure identifies them as non-prose content.
            norm_without_vol = re.sub(r"\bvolume\s+\d+,\s*\d+\b", "", normalized).strip()
            is_table_header_artifact = bool(re.match(r"^table\s+\d+\b", norm_without_vol) and len(norm_without_vol.split()) <= 20)

            is_matched_caption = any(
                normalized
                and (
                    normalized == caption
                    or norm_without_vol == caption
                    or SequenceMatcher(None, normalized, caption).ratio() >= 0.85
                    or SequenceMatcher(None, norm_without_vol, caption).ratio() >= 0.85
                )
                for caption in table_captions
            )

            if is_dump or is_matched_content or is_matched_caption or is_table_header_artifact:
                removed.append(paragraph)
            else:
                kept.append(paragraph)

        # Stitch split paragraphs across stripped table artifacts
        stitched = []
        i = 0
        while i < len(kept):
            p = kept[i]
            p_text = p.get("text", "").strip()
            if i + 1 < len(kept) and p_text and not p_text[-1] in ".!?\":":
                next_p = kept[i + 1]
                next_text = next_p.get("text", "").strip()
                if p.get("section") == next_p.get("section") and next_text and (next_text[0].islower() or len(p_text.split()) < 30):
                    merged_text = f"{p_text} {next_text}"
                    merged_p = dict(p)
                    merged_p["text"] = merged_text
                    merged_p["sentences"] = (p.get("sentences") or []) + (next_p.get("sentences") or [])
                    stitched.append(merged_p)
                    i += 2
                    continue
            stitched.append(p)
            i += 1
        kept = stitched

        return kept, removed

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
    def _dedupe_repeated_caption(cls, caption: str) -> str:
        """Remove an exact/near-exact duplicate caption prefix.

        GROBID can duplicate a caption when a figure description and body
        fragment are both serialized into the same record.  Only collapse the
        string when the two halves are highly similar, so legitimate mentions
        of another figure/table are preserved.
        """
        text = cls._clean_text(caption)
        if not text:
            return text

        matches = list(re.finditer(
            r"(?i)\b(?:figure|fig\.?|table|tab\.?)\s*[0-9ivx]+\b",
            text,
        ))
        if len(matches) < 2:
            return text

        first = text[:matches[1].start()].strip()
        second = text[matches[1].start():].strip()
        if not first or not second:
            return text

        similarity = SequenceMatcher(
            None,
            cls._normalize(first),
            cls._normalize(second),
        ).ratio()
        if similarity >= 0.96:
            return first

        return text

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
                    if key == "caption":
                        cleaned[key] = cls._dedupe_repeated_caption(
                            cleaned[key]
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

        result: list[dict[str, Any]] = []

        # Reference number is source metadata when available. Never allow
        # semantic deduplication to collapse two distinct numbered records.
        seen_numbered: set[int] = set()
        seen_unnumbered: set[str] = set()

        for reference in references:

            if not isinstance(reference, dict):
                continue

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

            number = reference.get("number")
            if isinstance(number, str) and number.strip().isdigit():
                number = int(number.strip())
            elif not isinstance(number, int):
                number = None

            if not title and not raw:
                continue

            if raw and len(raw) < 15:
                continue

            if number is not None:
                if number in seen_numbered:
                    continue
                seen_numbered.add(number)
            else:
                key = cls._normalize(
                    raw or title
                )

                if key in seen_unnumbered:
                    continue
                seen_unnumbered.add(key)

            cleaned = {
                "number": number,
                "title": title,
                "authors": authors,
                "raw": raw,
            }

            # Preserve optional provenance fields when a parser supplies them.
            for field in (
                "page",
                "source",
                "coords",
                "source_refs",
            ):
                if field in reference:
                    cleaned[field] = reference[field]

            result.append(cleaned)

        return result

    # =========================================================
    # REFERENCE RECOVERY
    # =========================================================

    REFERENCE_MARKER_RE = re.compile(
        r"(?<![\w\[])\b(?P<number>[1-9]\d{0,2})\.(?=\s*(?:[A-Za-zÀ-ÖØ-öø-ÿ]|https?://|www\.|doi\b))",
        re.IGNORECASE,
    )

    @classmethod
    def _split_numbered_reference_text(
        cls,
        text: str,
        *,
        min_number: int = 1,
        max_number: int = 999,
    ) -> list[tuple[int, str]]:
        """Split a flattened bibliography stream into numbered records."""
        if not isinstance(text, str) or not text.strip():
            return []

        matches = [
            m
            for m in cls.REFERENCE_MARKER_RE.finditer(text)
            if min_number <= int(m.group("number")) <= max_number
        ]

        entries: list[tuple[int, str]] = []
        for index, match in enumerate(matches):
            number = int(match.group("number"))
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            raw = cls._clean_text(text[start:end]).strip()
            if len(raw.split()) < 4:
                continue
            entries.append((number, raw))
        return entries

    BACKMATTER_HEADING_RE = re.compile(
        r"(?im)^\s*(?:acknowledg(?:e)?ments?|author(?:s)?\s+contributions?|funding|declarations?|"
        r"competing\s+interests?|conflicts?\s+of\s+interest|data\s+availability|"
        r"ethical?(?:\s+approval)?|ethics|additional\s+information|"
        r"reprints(?:\s+and\s+permissions)?|supplementary\s+(?:information|material))\b\s*:?.*$"
    )

    @classmethod
    def _reference_block_fingerprint(
        cls,
        text: str,
    ) -> str:
        """Normalize an edge block so repeated running headers/footers compare equal."""
        value = cls._clean_text(text).lower()
        value = re.sub(r"\b(?:19|20)\d{2}\b", "<year>", value)
        value = re.sub(r"\b\d{1,4}\b", "<n>", value)
        value = re.sub(r"https?://\S+", "<url>", value)
        value = re.sub(r"\s+", " ", value).strip()
        return value

    @classmethod
    def _collect_repeated_edge_block_fingerprints(
        cls,
        pages: list[dict[str, Any]],
    ) -> set[str]:
        """Find repeated top/bottom blocks without depending on journal wording."""
        if len(pages) < 3:
            return set()

        page_count_by_fp: dict[str, set[int]] = {}

        for page_index, page in enumerate(pages):
            if not isinstance(page, dict):
                continue

            blocks = page.get("blocks")
            if not isinstance(blocks, list):
                continue

            page_height = page.get("height")
            try:
                page_height = float(page_height) if page_height is not None else None
            except (TypeError, ValueError):
                page_height = None

            if page_height is None:
                y_values = []
                for block in blocks:
                    if not isinstance(block, dict):
                        continue
                    for key in ("y1", "y0"):
                        try:
                            y_values.append(float(block.get(key)))
                        except (TypeError, ValueError):
                            pass
                page_height = max(y_values, default=0.0)

            if page_height <= 0:
                continue

            for block in blocks:
                if not isinstance(block, dict):
                    continue

                value = cls._clean_text(block.get("text"))
                if not value:
                    continue

                try:
                    y0 = float(block.get("y0", 0.0))
                    y1 = float(block.get("y1", y0))
                except (TypeError, ValueError):
                    continue

                is_edge = (
                    y0 <= page_height * 0.12
                    or y1 >= page_height * 0.88
                )
                if not is_edge:
                    continue

                fingerprint = cls._reference_block_fingerprint(value)
                if len(fingerprint) < 8:
                    continue

                page_count_by_fp.setdefault(fingerprint, set()).add(page_index)

        threshold = max(2, int(len(pages) * 0.50 + 0.999))
        return {
            fp
            for fp, page_indexes in page_count_by_fp.items()
            if len(page_indexes) >= threshold
        }

    @classmethod
    def _strip_reference_page_noise(
        cls,
        page_text: str,
    ) -> str:
        """Remove generic running back-matter headings before bibliography splitting."""
        if not page_text:
            return ""
        match = cls.BACKMATTER_HEADING_RE.search(page_text)
        if match:
            page_text = page_text[:match.start()]
        return page_text.strip()

    @classmethod
    def _collect_pymupdf_text_for_references(
        cls,
        pymupdf: dict[str, Any],
    ) -> str:
        """Collect PDF text from the bibliography onward using page/block order."""
        if not isinstance(pymupdf, dict):
            return ""

        pages = pymupdf.get("pages")
        if not isinstance(pages, list) or not pages:
            text = cls._clean_text(pymupdf.get("text", ""))
            return cls._strip_reference_page_noise(text)

        repeated_edge_blocks = cls._collect_repeated_edge_block_fingerprints(pages)

        page_parts: list[str] = []
        reference_started = False

        for page in pages:
            if not isinstance(page, dict):
                continue

            blocks = page.get("blocks")
            texts: list[tuple[float, float, str]] = []

            if isinstance(blocks, list):
                page_height = page.get("height")
                try:
                    page_height = float(page_height) if page_height is not None else None
                except (TypeError, ValueError):
                    page_height = None

                if page_height is None:
                    y_values = []
                    for block in blocks:
                        if not isinstance(block, dict):
                            continue
                        for key in ("y1", "y0"):
                            try:
                                y_values.append(float(block.get(key)))
                            except (TypeError, ValueError):
                                pass
                    page_height = max(y_values, default=0.0)

                for block in blocks:
                    if not isinstance(block, dict):
                        continue

                    value = cls._clean_text(block.get("text"))
                    if not value:
                        continue

                    try:
                        x0 = float(block.get("x0", 0.0))
                        y0 = float(block.get("y0", 0.0))
                        y1 = float(block.get("y1", y0))
                    except (TypeError, ValueError):
                        x0, y0, y1 = 0.0, 0.0, 0.0

                    if page_height and (
                        y0 <= page_height * 0.12
                        or y1 >= page_height * 0.88
                    ):
                        fingerprint = cls._reference_block_fingerprint(value)
                        if fingerprint in repeated_edge_blocks:
                            continue

                    texts.append((x0, y0, value))

            if texts:
                # Preserve simple two-column reading order: top-to-bottom in
                # the left column, then top-to-bottom in the right column.
                xs = sorted(x for x, _, _ in texts)
                if len(xs) >= 4 and (xs[-1] - xs[0]) > 120:
                    midpoint = (xs[0] + xs[-1]) / 2.0
                    left = [item for item in texts if item[0] <= midpoint]
                    right = [item for item in texts if item[0] > midpoint]
                    ordered = sorted(left, key=lambda item: (item[1], item[0]))
                    ordered.extend(sorted(right, key=lambda item: (item[1], item[0])))
                else:
                    ordered = sorted(texts, key=lambda item: (item[1], item[0]))
                page_text = "\n".join(item[2] for item in ordered)
            else:
                page_text = cls._clean_text(page.get("text", ""))

            if not page_text:
                continue

            if not reference_started:
                if re.search(r"(?im)^\s*(?:##\s*)?references\s*$", page_text):
                    reference_started = True
                    page_text = re.split(
                        r"(?im)^\s*(?:##\s*)?references\s*$",
                        page_text,
                        maxsplit=1,
                    )[1]
                elif re.search(r"(?i)\bReferences\b", page_text) and len(pages) > 1:
                    marker = re.search(r"(?i)\bReferences\b", page_text)
                    if marker:
                        reference_started = True
                        page_text = page_text[marker.end():]

            if reference_started:
                backmatter = cls.BACKMATTER_HEADING_RE.search(page_text)
                if backmatter:
                    page_text = page_text[:backmatter.start()].strip()
                    if page_text:
                        page_parts.append(page_text)
                    # Once the bibliography ends, later pages are article
                    # back matter/license material, not references.
                    break

                page_text = cls._strip_reference_page_noise(page_text)
                if page_text:
                    page_parts.append(page_text)

        return "\n".join(page_parts).strip()

    @classmethod
    def _recover_references(
        cls,
        *,
        grobid_references: list[dict[str, Any]],
        paragraphs: list[dict[str, Any]],
        pymupdf: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Recover flattened/missing bibliography records without inventing content."""
        structured = cls._clean_references(grobid_references or [])

        by_number: dict[int, dict[str, Any]] = {}
        paragraph_reference_numbers: set[int] = set()
        embedded_from_grobid = 0

        # Recover explicit numbered records that GROBID flattened into a raw
        # bibliography string. We never assign numbers based on list position.
        for reference in structured:
            raw = str(reference.get("raw") or "")
            fragments = cls._split_numbered_reference_text(raw, min_number=91)
            for number, fragment in fragments:
                if number not in by_number:
                    by_number[number] = {
                        "number": number,
                        "title": "",
                        "authors": [],
                        "raw": f"{fragment}",
                    }
                    embedded_from_grobid += 1

        # Recover reference records that leaked into body paragraphs. A normal
        # prose paragraph usually has few numbered references; require a strong
        # concentration of bibliography markers before treating it as contamination.
        for paragraph in paragraphs:
            text = str(paragraph.get("text") or "")
            fragments = cls._split_numbered_reference_text(text, min_number=91)
            if len(fragments) < 4:
                continue
            high_numbered = [item for item in fragments if item[0] >= 91]
            if len(high_numbered) < 4:
                continue
            for number, fragment in high_numbered:
                paragraph_reference_numbers.add(number)
                by_number.setdefault(
                    number,
                    {
                        "number": number,
                        "title": "",
                        "authors": [],
                        "raw": fragment,
                    },
                )

        pdf_text = cls._collect_pymupdf_text_for_references(pymupdf)
        pdf_entries = cls._split_numbered_reference_text(pdf_text, min_number=1)

        # PDF extraction is the strongest recovery source when it contains a
        # sufficiently complete numbered bibliography. Use total coverage rather
        # than stopping at the first isolated gap.
        pdf_by_number = {number: raw for number, raw in pdf_entries}
        max_pdf_number = max(pdf_by_number, default=0)
        contiguous = 0
        if max_pdf_number:
            for number in range(1, max_pdf_number + 1):
                if number in pdf_by_number:
                    contiguous += 1
                else:
                    break

        coverage_ratio = (
            len(pdf_by_number) / max_pdf_number
            if max_pdf_number
            else 0.0
        )
        use_pdf_bibliography = (
            max_pdf_number >= 20
            and coverage_ratio >= 0.80
        )

        backfilled_reference_numbers: list[int] = []

        if use_pdf_bibliography:
            output = []
            for number in range(1, max_pdf_number + 1):
                raw = pdf_by_number.get(number)
                existing = by_number.get(number)

                if raw is None:
                    if existing and existing.get("raw"):
                        raw = str(existing.get("raw"))
                        backfilled_reference_numbers.append(number)
                    else:
                        continue

                output.append({
                    "number": number,
                    "title": existing.get("title", "") if existing else "",
                    "authors": existing.get("authors", []) if existing else [],
                    "raw": f"{raw}",
                    "source": "pymupdf_reference_section",
                })
            recovery_source = "pymupdf_reference_section"
        else:
            output = []
            for reference in structured:
                cleaned = dict(reference)
                if cleaned.get("number") is None:
                    cleaned["number"] = None
                output.append(cleaned)

            for number in sorted(by_number):
                candidate = by_number[number]
                if not candidate.get("raw"):
                    continue
                output.append(dict(candidate))
            recovery_source = "grobid_embedded_or_body"

        output = cls._clean_references(output)

        final_numbers = sorted(
            {
                int(reference["number"])
                for reference in output
                if isinstance(reference.get("number"), int)
            }
        )
        missing_reference_numbers = (
            [n for n in range(1, max_pdf_number + 1) if n not in set(final_numbers)]
            if max_pdf_number
            else []
        )

        report = {
            "source": recovery_source,
            "structured_reference_count": len(structured),
            "recovered_numbered_records": len(by_number),
            "embedded_grobid_records": embedded_from_grobid,
            "paragraph_reference_numbers": sorted(paragraph_reference_numbers),
            "pdf_reference_records": len(pdf_by_number),
            "pdf_max_reference_number": max_pdf_number,
            "pdf_contiguous_reference_count": contiguous,
            "pdf_coverage_ratio": coverage_ratio,
            "backfilled_reference_numbers": backfilled_reference_numbers,
            "missing_reference_numbers": missing_reference_numbers,
            "final_reference_count": len(output),
            "final_reference_numbers": final_numbers,
        }
        return output, report

    @classmethod
    def _remove_reference_contamination(
        cls,
        paragraphs: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        kept: list[dict[str, Any]] = []
        removed: list[dict[str, Any]] = []

        for paragraph in paragraphs:
            text = str(paragraph.get("text") or "").strip()
            fragments = cls._split_numbered_reference_text(text, min_number=91)
            if len(fragments) >= 4:
                high_numbered = [number for number, _ in fragments if number >= 91]
                starts_like_reference_list = bool(
                    re.match(r"^\s*(?:e\d+\s+)?(?:9[1-9]|1\d\d)\.\s*[A-ZÀ-ÖØ-Þ]", text)
                )
                if len(high_numbered) >= 4 and starts_like_reference_list:
                    removed.append(paragraph)
                    continue

            kept.append(paragraph)

        return kept, removed

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

    @staticmethod
    def _is_noise_paragraph(text: str) -> bool:
        value = re.sub(r"\s+", " ", str(text or "")).strip()

        if not value:
            return True

        # Standalone page-number/footer artifacts such as "1 3".
        if re.fullmatch(r"\d{1,4}(?:\s+\d{1,4}){1,2}", value):
            return True

        # Generic journal running banners accidentally emitted as body paragraphs.
        if re.search(r"^[^|\n]{2,80}\s*\|\s*\(\d{4}\)", value, re.IGNORECASE):
            return True

        return False

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

        if re.fullmatch(r"\d+\)", value.strip()):
            return True

        if re.fullmatch(
            r"[\d\s%=+\-*/().]+",
            value,
        ):
            return True

        # Generic publisher running-banner shape, independent of journal.
        if re.fullmatch(
            r"[a-z][^|\n]{1,100}\s*\|\s*(?:\(\d{4}\)|\d{4,5}(?:\s+volume\s+\d+(?:\s*,\s*\d+)?)?(?:\s*\|\s*\d+)?)",
            value,
            re.IGNORECASE,
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