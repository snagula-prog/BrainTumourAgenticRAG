# cleaning/scientific_cleaner.py

from __future__ import annotations

import copy
import difflib
import re
from typing import Any


class ScientificCleaner:
    """
    Scientific-paper cleaner designed for multiple publisher layouts.

    Strategy:
        title
        authors
        front matter
        abstract
        keywords
        body
        references

    The Docling hierarchy is NEVER physically modified.
    Text nodes are only blanked when excluded from retrieval.
    """

    YEAR_RE = re.compile(
        r"\b(?:19|20)\d{2}\b"
    )

    PDF_DATE_RE = re.compile(
        r"^D:(\d{4})"
    )

    SECTION_RE = re.compile(
        r"^\s*\d+(?:\.\d+)*\.?\s+.+$"
    )

    REFERENCE_RE = re.compile(
        r"^\s*(?:\[\d+\]|\d+\.)\s+"
    )

    KEYWORDS_RE = re.compile(
        r"^\s*keywords?\b",
        re.IGNORECASE,
    )

    REFERENCES_RE = re.compile(
        r"^\s*references\s*$",
        re.IGNORECASE,
    )

    PUBLISHER_PATTERNS = (
        "citation:",
        "academic editors:",
        "copyright:",
        "licensee",
        "published:",
        "received:",
        "accepted:",
        "https://www.mdpi.com",
        "https://creativecommons.org",
        "www.nature.com",
        "vol.:(0123456789)",
        "vol:.(1234567890)",
    )

    METADATA_LINES = {
        "proceeding paper",
        "open",
        "open access",
    }

    def __init__(
        self,
        parsed_document: dict[str, Any],
        paper_id: str,
    ) -> None:

        self.parsed_document = parsed_document
        self.paper_id = paper_id

    # =========================================================
    # MAIN
    # =========================================================

    def clean(self) -> dict[str, Any]:

        raw_document = self.parsed_document[
            "docling_document"
        ]

        document = copy.deepcopy(
            raw_document
        )

        text_items = document.get(
            "texts",
            [],
        )

        metadata = {
            "title": None,
            "authors": [],
            "year": None,
            "num_pages": self.parsed_document.get(
                "num_pages",
                0,
            ),
        }

        # -----------------------------------------------------
        # FIRST-PAGE ANALYSIS
        # -----------------------------------------------------

        first_page_items = [
            item
            for item in text_items
            if self._page_no(item) in (
                None,
                1,
            )
        ]

        title_index = self._find_title_index(
            first_page_items
        )

        if title_index is not None:
            metadata["title"] = self._clean_title(
                self._item_text(
                    first_page_items[
                        title_index
                    ]
                )
            )

        # -----------------------------------------------------
        # AUTHORS
        # -----------------------------------------------------

        metadata["authors"] = (
            self._extract_authors_from_first_page(
                first_page_items,
                title_index,
            )
        )

        if not metadata["authors"]:

            metadata["authors"] = (
                self._authors_from_pdf_metadata()
            )

        # -----------------------------------------------------
        # YEAR
        # -----------------------------------------------------

        metadata["year"] = (
            self._extract_year(
                document
            )
        )

        if metadata["year"] is None:

            metadata["year"] = (
                self._year_from_pdf_metadata()
            )

        # -----------------------------------------------------
        # ABSTRACT
        # -----------------------------------------------------

        abstract = (
            self._extract_abstract(
                first_page_items,
                title_index,
            )
        )

        # -----------------------------------------------------
        # PROCESS BODY
        # -----------------------------------------------------

        references: list[str] = []

        in_references = False

        for item in text_items:

            text = self._item_text(
                item
            )

            if not text:
                continue

            text = self._normalize_text(
                text
            )

            # -------------------------------------------------
            # REFERENCE SECTION
            # -------------------------------------------------

            if self.REFERENCES_RE.match(
                text
            ):

                in_references = True
                item["text"] = ""
                continue

            if in_references:

                if self._is_reference(
                    text
                ):
                    references.append(
                        text
                    )

                item["text"] = ""
                continue

            # -------------------------------------------------
            # FIRST-PAGE FRONT MATTER
            # -------------------------------------------------

            if self._is_front_matter(
                text,
                item,
            ):

                item["text"] = ""
                continue

            # -------------------------------------------------
            # CAPTIONS
            # -------------------------------------------------

            if self._is_caption(
                text
            ):

                item["text"] = ""
                continue

            # -------------------------------------------------
            # PUBLISHER BOILERPLATE
            # -------------------------------------------------

            text = (
                self._remove_publisher_text(
                    text
                )
            )

            if not text:

                item["text"] = ""
                continue

            # -------------------------------------------------
            # KEYWORDS
            # -------------------------------------------------

            if self.KEYWORDS_RE.match(
                text
            ):

                item["text"] = ""

                continue

            # -------------------------------------------------
            # HEADING
            # -------------------------------------------------

            text = (
                self._normalize_heading_text(
                    text
                )
            )

            # -------------------------------------------------
            # DUPLICATION
            # -------------------------------------------------

            text = (
                self._remove_near_duplicates(
                    text
                )
            )

            item["text"] = text

        # =====================================================
        # REFERENCES
        # =====================================================

        references = (
            self._deduplicate_references(
                references
            )
        )

        # =====================================================
        # ABSTRACT
        # =====================================================

        abstract = (
            self._remove_near_duplicates(
                self._normalize_text(
                    abstract
                )
            )
        )

        metadata["abstract"] = abstract

        # =====================================================
        # OUTPUT
        # =====================================================

        return {
            "paper_id": self.paper_id,
            "filename": self.parsed_document.get(
                "filename"
            ),
            "source_parser": "docling_cleaned",
            "title": metadata["title"],
            "authors": metadata["authors"],
            "year": metadata["year"],
            "num_pages": metadata["num_pages"],
            "abstract": abstract,
            "references": references,
            "metadata": metadata,
            "docling_document": document,
            "stats": {
                "total_text_nodes": len(
                    text_items
                ),
                "abstract_characters": len(
                    abstract
                ),
                "reference_count": len(
                    references
                ),
            },
        }

    # =========================================================
    # TITLE
    # =========================================================

    def _find_title_index(
        self,
        items: list[dict[str, Any]],
    ) -> int | None:

        for index, item in enumerate(
            items
        ):

            text = self._normalize_text(
                self._item_text(item)
            )

            if not text:
                continue

            if self._is_sane_title(
                text
            ):
                return index

        return None

    @staticmethod
    def _clean_title(
        text: str,
    ) -> str:

        text = text.strip()

        text = re.sub(
            r"\s*[†‡]+\s*$",
            "",
            text,
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    # =========================================================
    # AUTHORS
    # =========================================================

    def _extract_authors_from_first_page(
        self,
        items: list[dict[str, Any]],
        title_index: int | None,
    ) -> list[str]:

        if title_index is None:
            start = 0
        else:
            start = title_index + 1

        candidates: list[str] = []

        # Only inspect the first few nodes after title.
        for item in items[
            start : start + 8
        ]:

            text = self._normalize_text(
                self._item_text(item)
            )

            if not text:
                continue

            # Stop when obvious article body begins.
            if self._looks_like_body_sentence(
                text
            ):
                break

            # Skip obvious affiliation / metadata.
            lower = text.lower()

            if (
                "department" in lower
                or "university" in lower
                or "school of" in lower
                or "faculty" in lower
                or "email" in lower
                or "@" in lower
            ):
                continue

            if self.KEYWORDS_RE.match(
                text
            ):
                break

            if len(text) > 500:
                continue

            candidates.append(text)

        authors: list[str] = []

        for candidate in candidates:

            # Scientific Reports commonly uses:
            #
            # Author A1 & Author B2*
            #
            if "&" in candidate:

                parts = re.split(
                    r"\s*&\s*",
                    candidate,
                )

            else:

                parts = re.split(
                    r"\s*;\s*",
                    candidate,
                )

            for part in parts:

                part = re.sub(
                    r"\d+[\*\u2020\u2021]*",
                    "",
                    part,
                )

                part = re.sub(
                    r"[\*\u2020\u2021]+",
                    "",
                    part,
                )

                part = part.strip(
                    " ,."
                )

                if not part:
                    continue

                if self._looks_like_person_name(
                    part
                ):
                    authors.append(
                        part
                    )

        return self._unique_strings(
            authors
        )

    @staticmethod
    def _looks_like_person_name(
        text: str,
    ) -> bool:

        words = text.split()

        if not (
            2 <= len(words) <= 6
        ):
            return False

        if any(
            char.isdigit()
            for char in text
        ):
            return False

        return True

    def _authors_from_pdf_metadata(
        self,
    ) -> list[str]:

        metadata = (
            self.parsed_document.get(
                "docling_metadata",
                {},
            )
        )

        value = metadata.get(
            "pdf_author"
        )

        if not value:
            return []

        value = str(value)

        parts = re.split(
            r",\s*|\s+and\s+|\s*&\s*",
            value,
        )

        authors = []

        for part in parts:

            part = part.strip()

            if part:
                authors.append(
                    part
                )

        return self._unique_strings(
            authors
        )

    # =========================================================
    # ABSTRACT
    # =========================================================

    def _extract_abstract(
        self,
        items: list[dict[str, Any]],
        title_index: int | None,
    ) -> str:

        if title_index is None:
            start = 0
        else:
            start = title_index + 1

        abstract_parts: list[str] = []

        started = False

        for item in items[
            start:
        ]:

            text = self._normalize_text(
                self._item_text(item)
            )

            if not text:
                continue

            # -------------------------------------------------
            # Skip author/front-matter material.
            # -------------------------------------------------

            if not started:

                if self._looks_like_person_name_line(
                    text
                ):
                    continue

                if self._is_affiliation(
                    text
                ):
                    continue

                if self._is_correspondence(
                    text
                ):
                    continue

            # -------------------------------------------------
            # Keywords terminate abstract.
            # -------------------------------------------------

            keyword_match = (
                self.KEYWORDS_RE.search(
                    text
                )
            )

            if keyword_match:

                before = text[
                    :keyword_match.start()
                ].strip()

                if before:
                    abstract_parts.append(
                        before
                    )

                break

            # -------------------------------------------------
            # Some publishers place "Abstract:"
            # inline with the paragraph.
            # -------------------------------------------------

            abstract_marker = re.search(
                r"\babstract\s*:\s*",
                text,
                re.IGNORECASE,
            )

            if abstract_marker:

                text = text[
                    abstract_marker.end():
                ].strip()

                started = True

            # -------------------------------------------------
            # Detect beginning of actual prose.
            # -------------------------------------------------

            if not started:

                if self._looks_like_body_sentence(
                    text
                ):

                    started = True

                else:
                    continue

            if started:
                abstract_parts.append(
                    text
                )

        return " ".join(
            abstract_parts
        ).strip()

    @staticmethod
    def _looks_like_person_name_line(
        text: str,
    ) -> bool:

        if len(text) > 250:
            return False

        if (
            "@" in text
            or "department" in text.lower()
            or "university" in text.lower()
        ):
            return False

        # Name separator patterns.
        if (
            " & " in text
            or re.search(
                r"\b[A-Z][a-z]+\d",
                text,
            )
        ):
            return True

        return False

    @staticmethod
    def _looks_like_body_sentence(
        text: str,
    ) -> bool:

        if len(text) < 60:
            return False

        words = text.split()

        if len(words) < 10:
            return False

        return bool(
            re.search(
                r"[.!?]",
                text,
            )
        )

    @staticmethod
    def _is_affiliation(
        text: str,
    ) -> bool:

        lower = text.lower()

        patterns = (
            "department",
            "university",
            "school of",
            "faculty",
            "college",
            "institute",
            "email:",
            "@",
        )

        return any(
            pattern in lower
            for pattern in patterns
        )

    @staticmethod
    def _is_correspondence(
        text: str,
    ) -> bool:

        lower = text.lower()

        return (
            "email" in lower
            or "correspondence" in lower
            or "*" in text
        )

    # =========================================================
    # FRONT MATTER FILTER
    # =========================================================

    def _is_front_matter(
        self,
        text: str,
        item: dict[str, Any],
    ) -> bool:

        page_no = self._page_no(
            item
        )

        if page_no != 1:
            return False

        lower = text.lower()

        if lower in self.METADATA_LINES:
            return True

        if self._is_affiliation(
            text
        ):
            return True

        if self._is_correspondence(
            text
        ):
            return True

        # DOI / journal information.
        for pattern in self.PUBLISHER_PATTERNS:

            if pattern in lower:
                return True

        return False

    # =========================================================
    # PUBLISHER TEXT
    # =========================================================

    def _remove_publisher_text(
        self,
        text: str,
    ) -> str:

        result = text

        for pattern in self.PUBLISHER_PATTERNS:

            index = result.lower().find(
                pattern
            )

            if index != -1:

                result = result[
                    :index
                ].strip()

        return result.strip()

    # =========================================================
    # CAPTION
    # =========================================================

    @staticmethod
    def _is_caption(
        text: str,
    ) -> bool:

        return bool(
            re.match(
                r"^(?:Figure|Fig\.|Table)\s*\d+\b",
                text,
                re.IGNORECASE,
            )
        )

    # =========================================================
    # REFERENCE
    # =========================================================

    @staticmethod
    def _is_reference(
        text: str,
    ) -> bool:

        return bool(
            ScientificCleaner.REFERENCE_RE.match(
                text
            )
        )

    # =========================================================
    # HEADING
    # =========================================================

    @staticmethod
    def _normalize_heading_text(
        text: str,
    ) -> str:

        text = re.sub(
            r"\s+",
            " ",
            text,
        ).strip()

        # Example:
        # 3. Materials and Methods 3.
        # -> 3. Materials and Methods

        match = re.match(
            r"^(?P<num>\d+(?:\.\d+)*\.)"
            r"\s+"
            r"(?P<title>.*?)"
            r"\s+"
            r"(?P<trailing>\d+(?:\.\d+)*\.)$",
            text,
        )

        if match:

            num = match.group("num")
            trailing = match.group("trailing")

            if num == trailing:

                title = match.group(
                    "title"
                ).strip()

                return f"{num} {title}"

        return text

    # =========================================================
    # DEDUP
    # =========================================================

    def _remove_near_duplicates(
        self,
        text: str,
    ) -> str:

        sentences = re.split(
            r"(?<=[.!?])\s+(?=[A-Z0-9])",
            text,
        )

        result: list[str] = []

        for sentence in sentences:

            sentence = sentence.strip()

            if not sentence:
                continue

            normalized = (
                self._dedup_normalize(
                    sentence
                )
            )

            duplicate = False

            for previous in result[-15:]:

                previous_normalized = (
                    self._dedup_normalize(
                        previous
                    )
                )

                ratio = (
                    difflib.SequenceMatcher(
                        None,
                        previous_normalized,
                        normalized,
                    ).ratio()
                )

                if ratio >= 0.92:
                    duplicate = True
                    break

                if (
                    len(normalized) > 40
                    and (
                        normalized in previous_normalized
                        or previous_normalized
                        in normalized
                    )
                ):
                    duplicate = True
                    break

            if not duplicate:
                result.append(
                    sentence
                )

        return " ".join(
            result
        ).strip()

    @staticmethod
    def _dedup_normalize(
        text: str,
    ) -> str:

        text = text.lower()

        text = re.sub(
            r"[^\w\s]",
            "",
            text,
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    # =========================================================
    # REFERENCES
    # =========================================================

    def _deduplicate_references(
        self,
        references: list[str],
    ) -> list[str]:

        result: list[str] = []
        seen: set[str] = set()

        for reference in references:

            normalized = (
                self._dedup_normalize(
                    reference
                )
            )

            if not normalized:
                continue

            if normalized in seen:
                continue

            seen.add(
                normalized
            )

            result.append(
                reference
            )

        return result

    # =========================================================
    # YEAR
    # =========================================================

    def _extract_year(
        self,
        document: dict[str, Any],
    ) -> int | None:

        metadata = (
            self.parsed_document.get(
                "docling_metadata",
                {},
            )
        )

        for key in (
            "creation_date",
            "mod_date",
            "publication_date",
        ):

            value = metadata.get(
                key
            )

            if not value:
                continue

            value = str(value)

            if value.startswith(
                "D:"
            ):

                match = (
                    self.PDF_DATE_RE.match(
                        value
                    )
                )

                if match:
                    return int(
                        match.group(1)
                    )

            match = self.YEAR_RE.search(
                value
            )

            if match:
                return int(
                    match.group(0)
                )

        # Search first-page visible text.
        for item in document.get(
            "texts",
            [],
        ):

            if self._page_no(item) != 1:
                continue

            text = self._normalize_text(
                self._item_text(item)
            )

            if "published" in text.lower():

                match = (
                    self.YEAR_RE.search(
                        text
                    )
                )

                if match:
                    return int(
                        match.group(0)
                    )

        return None

    def _year_from_pdf_metadata(
        self,
    ) -> int | None:

        metadata = (
            self.parsed_document.get(
                "docling_metadata",
                {},
            )
        )

        return self._extract_year_from_value(
            metadata.get(
                "creation_date"
            )
        )

    @staticmethod
    def _extract_year_from_value(
        value: Any,
    ) -> int | None:

        if not value:
            return None

        value = str(
            value
        )

        if value.startswith(
            "D:"
        ):

            digits = value[2:]

            if (
                len(digits) >= 4
                and digits[:4].isdigit()
            ):
                return int(
                    digits[:4]
                )

        match = re.search(
            r"\b(?:19|20)\d{2}\b",
            value,
        )

        if match:
            return int(
                match.group(0)
            )

        return None

    # =========================================================
    # UTILS
    # =========================================================

    @staticmethod
    def _page_no(
        item: dict[str, Any],
    ) -> int | None:

        prov = item.get(
            "prov"
        )

        if not isinstance(
            prov,
            list,
        ) or not prov:
            return None

        first = prov[0]

        if not isinstance(
            first,
            dict,
        ):
            return None

        value = first.get(
            "page_no"
        )

        try:
            if value is not None:
                return int(value)
        except (
            TypeError,
            ValueError,
        ):
            pass

        return None

    @staticmethod
    def _item_text(
        item: dict[str, Any],
    ) -> str:

        text = str(
            item.get("text") or ""
        ).strip()

        if text:
            return text

        if (
            str(
                item.get("label") or ""
            ).lower()
            == "formula"
        ):

            orig = str(
                item.get("orig") or ""
            ).strip()

            if orig:
                return orig

        return ""

    @staticmethod
    def _normalize_text(
        text: str,
    ) -> str:

        text = text.replace(
            "\u00a0",
            " ",
        )

        text = text.replace(
            "\r",
            " ",
        )

        text = text.replace(
            "\n",
            " ",
        )

        return re.sub(
            r"\s+",
            " ",
            text,
        ).strip()

    @staticmethod
    def _is_sane_title(
        text: str,
    ) -> bool:

        if not (
            20 <= len(text) <= 300
        ):
            return False

        lower = text.lower()

        if (
            "pdftitle" in lower
            or text.startswith("\\")
        ):
            return False

        return True

    @staticmethod
    def _unique_strings(
        values: list[str],
    ) -> list[str]:

        seen: set[str] = set()
        result: list[str] = []

        for value in values:

            key = value.lower().strip()

            if not key:
                continue

            if key in seen:
                continue

            seen.add(key)
            result.append(
                value.strip()
            )

        return result