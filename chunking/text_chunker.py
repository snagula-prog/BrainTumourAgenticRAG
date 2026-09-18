# chunking/text_chunker.py

from __future__ import annotations

import re
from typing import Any

from transformers import AutoTokenizer


class ScientificTextChunker:

    SECTION_RE = re.compile(
        r"^\s*"
        r"(?P<number>\d+(?:\.\d+)*)"
        r"\.?\s+"
        r"(?P<title>.+?)"
        r"\s*$"
    )

    def __init__(
        self,
        *,
        embedding_model: str,
        max_tokens: int = 450,
        overlap_tokens: int = 60,
    ) -> None:

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                embedding_model
            )
        )

        self.max_tokens = max_tokens
        self.overlap_tokens = (
            overlap_tokens
        )

    # =========================================================
    # PUBLIC
    # =========================================================

    def chunk(
        self,
        cleaned: dict[str, Any],
    ) -> list[dict]:

        paper_id = cleaned[
            "paper_id"
        ]

        chunks = []

        # -----------------------------------------------------
        # ABSTRACT
        # -----------------------------------------------------

        abstract = (
            cleaned.get(
                "abstract"
            )
            or ""
        ).strip()

        if abstract:

            abstract_chunks = (
                self._chunk_text(
                    abstract
                )
            )

            for index, text in enumerate(
                abstract_chunks
            ):

                chunks.append(
                    self._record(
                        cleaned=cleaned,
                        paper_id=paper_id,
                        chunk_id=(
                            f"{paper_id}"
                            f"_abstract_"
                            f"{index:03d}"
                        ),
                        section="Abstract",
                        text=text,
                        page_no=1,
                        source="pymupdf",
                    )
                )

        # -----------------------------------------------------
        # BODY
        # -----------------------------------------------------

        sections = self._extract_sections(
            cleaned[
                "canonical_text"
            ]
        )

        index = len(chunks)

        for section in sections:

            pieces = self._chunk_text(
                section["text"]
            )

            for piece in pieces:

                chunks.append(
                    self._record(
                        cleaned=cleaned,
                        paper_id=paper_id,
                        chunk_id=(
                            f"{paper_id}"
                            f"_chunk_"
                            f"{index:03d}"
                        ),
                        section=section[
                            "heading"
                        ],
                        text=piece,
                        page_no=section.get(
                            "page_no"
                        ),
                        source="pymupdf",
                    )
                )

                index += 1

        return chunks

    # =========================================================
    # SECTIONS
    # =========================================================

    def _extract_sections(
        self,
        text: str,
    ) -> list[dict]:

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        sections = []

        current_heading = (
            "Introduction"
        )

        current_text: list[str] = []

        for line in lines:

            # Skip obvious metadata.
            if self._metadata_line(
                line
            ):
                continue

            match = self.SECTION_RE.match(
                line
            )

            if match:

                if current_text:

                    sections.append(
                        {
                            "heading": (
                                current_heading
                            ),
                            "text": " ".join(
                                current_text
                            ),
                            "page_no": None,
                        }
                    )

                current_heading = (
                    self._clean_heading(
                        line
                    )
                )

                current_text = []

                continue

            current_text.append(
                line
            )

        if current_text:

            sections.append(
                {
                    "heading": current_heading,
                    "text": " ".join(
                        current_text
                    ),
                    "page_no": None,
                }
            )

        return sections

    @staticmethod
    def _clean_heading(
        heading: str,
    ) -> str:

        heading = re.sub(
            r"\s+",
            " ",
            heading,
        ).strip()

        # Remove duplicate trailing number.
        match = re.match(
            r"^(?P<num>\d+(?:\.\d+)*\.)"
            r"\s+"
            r"(?P<title>.*?)"
            r"\s+"
            r"(?P<trailing>\d+(?:\.\d+)*\.)$",
            heading,
        )

        if match and (
            match.group("num")
            == match.group("trailing")
        ):

            return (
                f"{match.group('num')} "
                f"{match.group('title')}"
            )

        return heading

    # =========================================================
    # TOKEN CHUNKING
    # =========================================================

    def _chunk_text(
        self,
        text: str,
    ) -> list[str]:

        sentences = re.split(
            r"(?<=[.!?])\s+(?=[A-Z0-9])",
            text.strip(),
        )

        sentences = [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

        chunks = []
        current: list[str] = []
        current_tokens = 0

        for sentence in sentences:

            token_count = len(
                self.tokenizer.encode(
                    sentence,
                    add_special_tokens=False,
                )
            )

            if (
                current
                and
                current_tokens
                + token_count
                > self.max_tokens
            ):

                chunks.append(
                    " ".join(
                        current
                    )
                )

                overlap = []

                overlap_tokens = 0

                for previous in reversed(
                    current
                ):

                    count = len(
                        self.tokenizer.encode(
                            previous,
                            add_special_tokens=False,
                        )
                    )

                    if (
                        overlap_tokens
                        + count
                        > self.overlap_tokens
                    ):
                        break

                    overlap.insert(
                        0,
                        previous,
                    )

                    overlap_tokens += count

                current = overlap
                current_tokens = (
                    overlap_tokens
                )

            current.append(
                sentence
            )

            current_tokens += (
                token_count
            )

        if current:

            chunks.append(
                " ".join(
                    current
                )
            )

        return chunks

    # =========================================================
    # RECORD
    # =========================================================

    @staticmethod
    def _record(
        *,
        cleaned: dict,
        paper_id: str,
        chunk_id: str,
        section: str,
        text: str,
        page_no: int | None,
        source: str,
    ) -> dict:

        return {
            "chunk_id": chunk_id,
            "paper_id": paper_id,
            "section": section,
            "text": text,
            "page_no": page_no,
            "source": source,
            "title": cleaned.get(
                "title"
            ),
            "authors": cleaned.get(
                "authors",
                [],
            ),
            "year": cleaned.get(
                "year"
            ),
        }

    @staticmethod
    def _metadata_line(
        text: str,
    ) -> bool:

        lower = text.lower()

        return (
            lower.startswith(
                "scientific reports"
            )
            or lower.startswith(
                "eng. proc."
            )
            or lower.startswith(
                "www.nature.com"
            )
            or lower.startswith(
                "doi.org"
            )
        )