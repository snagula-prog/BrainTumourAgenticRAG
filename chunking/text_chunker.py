from __future__ import annotations

import re
from typing import Any

from transformers import AutoTokenizer

from chunking.config import ChunkingConfig


class TextChunker:
    """
    Section-aware, paragraph-aware, sentence-aware text chunker.

    It consumes canonical JSON paragraph records.

    It does NOT know about:
        GROBID
        Docling
        PyMuPDF
        ChromaDB
        embeddings
    """

    def __init__(
        self,
        config: ChunkingConfig,
    ) -> None:

        self.config = config

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                config.tokenizer_name
            )
        )

    # ============================================================
    # PUBLIC
    # ============================================================

    def chunk(
        self,
        paragraphs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:

        groups = self._group_by_section(
            paragraphs
        )

        chunks: list[dict[str, Any]] = []

        for group in groups:

            chunks.extend(
                self._chunk_group(
                    group
                )
            )

        return chunks

    # ============================================================
    # GROUPING
    # ============================================================

    @staticmethod
    def _group_by_section(
        paragraphs: list[dict[str, Any]],
    ) -> list[list[dict[str, Any]]]:

        groups: list[
            list[dict[str, Any]]
        ] = []

        current: list[
            dict[str, Any]
        ] = []

        current_key: tuple[
            str,
            tuple[str, ...],
        ] | None = None

        for paragraph in paragraphs:

            text = TextChunker._clean_text(
                paragraph.get("text")
            )

            if not text:
                continue

            section = str(
                paragraph.get(
                    "section"
                )
                or "Unclassified"
            ).strip()

            section_path = tuple(
                str(item).strip()
                for item in (
                    paragraph.get(
                        "section_path"
                    )
                    or [section]
                )
                if str(item).strip()
            )

            key = (
                section,
                section_path,
            )

            if (
                current_key is not None
                and key != current_key
            ):
                groups.append(
                    current
                )
                current = []

            current_key = key
            current.append(
                paragraph
            )

        if current:
            groups.append(
                current
            )

        return groups

    # ============================================================
    # GROUP CHUNKING
    # ============================================================

    def _chunk_group(
        self,
        paragraphs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:

        if not paragraphs:
            return []

        section = str(
            paragraphs[0].get(
                "section"
            )
            or "Unclassified"
        )

        section_path = (
            paragraphs[0].get(
                "section_path"
            )
            or [section]
        )

        chunks: list[
            dict[str, Any]
        ] = []

        current_sentences: list[str] = []
        current_paragraphs: list[
            dict[str, Any]
        ] = []

        current_tokens = 0

        for paragraph in paragraphs:

            text = self._clean_text(
                paragraph.get("text")
            )

            if not text:
                continue

            sentences = (
                self._split_sentences(
                    text
                )
            )

            if not sentences:
                sentences = [text]

            for sentence in sentences:

                sentence_tokens = (
                    self._token_count(
                        sentence
                    )
                )

                # ------------------------------------------------
                # Extremely long sentence
                # ------------------------------------------------

                if (
                    sentence_tokens
                    > self.config.chunk_size
                ):

                    if current_sentences:

                        chunks.append(
                            self._make_chunk(
                                current_sentences,
                                current_paragraphs,
                                section,
                                section_path,
                            )
                        )

                    current_sentences = []
                    current_paragraphs = []
                    current_tokens = 0

                    pieces = (
                        self._hard_token_split(
                            sentence
                        )
                    )

                    for piece in pieces:

                        chunks.append(
                            self._make_chunk(
                                [piece],
                                [paragraph],
                                section,
                                section_path,
                                boundary_type="token",
                            )
                        )

                    continue

                # ------------------------------------------------
                # Fits current chunk
                # ------------------------------------------------

                if (
                    current_tokens
                    + sentence_tokens
                    <= self.config.chunk_size
                ):

                    current_sentences.append(
                        sentence
                    )

                    if (
                        paragraph
                        not in current_paragraphs
                    ):
                        current_paragraphs.append(
                            paragraph
                        )

                    current_tokens += (
                        sentence_tokens
                    )

                    continue

                # ------------------------------------------------
                # Flush current chunk
                # ------------------------------------------------

                if current_sentences:

                    chunks.append(
                        self._make_chunk(
                            current_sentences,
                            current_paragraphs,
                            section,
                            section_path,
                        )
                    )

                # ------------------------------------------------
                # Add overlap
                # ------------------------------------------------

                overlap_sentences = (
                    self._get_overlap_sentences(
                        current_sentences
                    )
                )

                overlap_tokens = sum(
                    self._token_count(
                        sentence
                    )
                    for sentence
                    in overlap_sentences
                )

                current_sentences = (
                    overlap_sentences
                    + [sentence]
                )

                current_paragraphs = (
                    current_paragraphs[-1:]
                    if overlap_sentences
                    and current_paragraphs
                    else []
                )

                if (
                    paragraph
                    not in current_paragraphs
                ):
                    current_paragraphs.append(
                        paragraph
                    )

                current_tokens = (
                    overlap_tokens
                    + sentence_tokens
                )

        # --------------------------------------------------------
        # Final chunk
        # --------------------------------------------------------

        if current_sentences:

            chunks.append(
                self._make_chunk(
                    current_sentences,
                    current_paragraphs,
                    section,
                    section_path,
                )
            )

        return chunks

    # ============================================================
    # CHUNK CREATION
    # ============================================================

    def _make_chunk(
        self,
        sentences: list[str],
        paragraphs: list[dict[str, Any]],
        section: str,
        section_path: list[str],
        boundary_type: str = "sentence",
    ) -> dict[str, Any]:

        text = " ".join(
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ).strip()

        pages: list[int] = []
        coords: list[dict[str, Any]] = []

        for paragraph in paragraphs:

            page = paragraph.get(
                "page"
            )

            if isinstance(
                page,
                int,
            ):
                pages.append(
                    page
                )

            for coord in (
                paragraph.get(
                    "coords"
                )
                or []
            ):

                if isinstance(
                    coord,
                    dict,
                ):
                    coords.append(
                        coord
                    )

                    coord_page = (
                        coord.get(
                            "page"
                        )
                    )

                    if isinstance(
                        coord_page,
                        int,
                    ):
                        pages.append(
                            coord_page
                        )

        pages = sorted(
            set(pages)
        )

        return {
            "unit_type": "text",
            "section": section,
            "section_path": section_path,
            "text": text,
            "embedding_text": (
                f"Section: "
                f"{' > '.join(section_path)}\n"
                f"{text}"
            ),
            "token_count": (
                self._token_count(text)
            ),
            "boundary_type": boundary_type,
            "pages": pages,
            "page_start": (
                pages[0]
                if pages
                else None
            ),
            "page_end": (
                pages[-1]
                if pages
                else None
            ),
            "coords": coords,
            "source_paragraph_count": len(
                paragraphs
            ),
        }

    # ============================================================
    # SENTENCES
    # ============================================================

    @staticmethod
    def _split_sentences(
        text: str,
    ) -> list[str]:

        text = re.sub(
            r"\s+",
            " ",
            text.strip(),
        )

        if not text:
            return []

        parts = re.split(
            r"(?<=[.!?])\s+(?=[A-Z0-9(\[])",
            text,
        )

        return [
            part.strip()
            for part in parts
            if part.strip()
        ]

    # ============================================================
    # TOKEN SPLIT
    # ============================================================

    def _hard_token_split(
        self,
        text: str,
    ) -> list[str]:

        token_ids = (
            self.tokenizer.encode(
                text,
                add_special_tokens=False,
            )
        )

        step = (
            self.config.chunk_size
            - self.config.overlap
        )

        pieces: list[str] = []

        for start in range(
            0,
            len(token_ids),
            step,
        ):

            window = token_ids[
                start:
                start + self.config.chunk_size
            ]

            if not window:
                break

            piece = (
                self.tokenizer.decode(
                    window,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                ).strip()
            )

            if piece:
                pieces.append(
                    piece
                )

            if (
                start
                + self.config.chunk_size
                >= len(token_ids)
            ):
                break

        return pieces

    # ============================================================
    # OVERLAP
    # ============================================================

    def _get_overlap_sentences(
        self,
        sentences: list[str],
    ) -> list[str]:

        if not sentences:
            return []

        selected: list[str] = []
        token_count = 0

        for sentence in reversed(
            sentences
        ):

            count = (
                self._token_count(
                    sentence
                )
            )

            if (
                token_count + count
                > self.config.overlap
            ):
                break

            selected.insert(
                0,
                sentence
            )

            token_count += count

        return selected

    # ============================================================
    # TOKEN COUNT
    # ============================================================

    def _token_count(
        self,
        text: str,
    ) -> int:

        return len(
            self.tokenizer.encode(
                text,
                add_special_tokens=False,
            )
        )

    # ============================================================
    # CLEAN
    # ============================================================

    @staticmethod
    def _clean_text(
        text: Any,
    ) -> str:

        if text is None:
            return ""

        text = str(text)

        text = text.replace(
            "\u00ad",
            "",
        )

        return re.sub(
            r"\s+",
            " ",
            text,
        ).strip()