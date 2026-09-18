# cleaning/pdf_text_extractor.py

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import fitz


@dataclass
class PDFTextBlock:
    page_no: int
    block_no: int
    x0: float
    y0: float
    x1: float
    y1: float
    text: str


class PDFTextExtractor:
    """
    Canonical text extractor.

    PyMuPDF is used for the article's textual content because
    some PDFs have broken Unicode mappings in structural parsers.

    Docling remains responsible for structured artifacts.
    """

    HEADER_FOOTER_PATTERNS = (
        r"^eng\.\s*proc\.",
        r"^scientific reports\s*\|",
        r"^www\.nature\.com",
        r"^www\.nature\.com/",
        r"^\d+\s+of\s+\d+$",
        r"^vol\.:",
        r"^vol:\.",
    )

    def __init__(
        self,
        pdf_path: str | Path,
    ) -> None:

        self.pdf_path = Path(pdf_path)

        if not self.pdf_path.exists():
            raise FileNotFoundError(
                f"PDF not found: {self.pdf_path}"
            )

    # =========================================================
    # PUBLIC
    # =========================================================

    def extract(
        self,
    ) -> dict:

        blocks = self._extract_blocks()

        blocks = self._remove_headers_footers(
            blocks
        )

        blocks = self._deduplicate_blocks(
            blocks
        )

        pages = self._group_into_pages(
            blocks
        )

        full_text = "\n\n".join(
            page["text"]
            for page in pages
            if page["text"]
        )

        return {
            "pdf_path": str(
                self.pdf_path
            ),
            "num_pages": len(pages),
            "pages": pages,
            "text": full_text,
        }

    # =========================================================
    # BLOCK EXTRACTION
    # =========================================================

    def _extract_blocks(
        self,
    ) -> list[PDFTextBlock]:

        result: list[
            PDFTextBlock
        ] = []

        with fitz.open(
            self.pdf_path
        ) as pdf:

            for page_index, page in enumerate(
                pdf
            ):

                blocks = page.get_text(
                    "blocks"
                )

                for block_no, block in enumerate(
                    blocks
                ):

                    if len(block) < 5:
                        continue

                    x0, y0, x1, y1, text = (
                        block[:5]
                    )

                    text = self._normalize(
                        text
                    )

                    if not text:
                        continue

                    result.append(
                        PDFTextBlock(
                            page_no=page_index + 1,
                            block_no=block_no,
                            x0=float(x0),
                            y0=float(y0),
                            x1=float(x1),
                            y1=float(y1),
                            text=text,
                        )
                    )

        return result

    # =========================================================
    # READING ORDER
    # =========================================================

    @staticmethod
    def _group_into_pages(
        blocks: list[PDFTextBlock],
    ) -> list[dict]:

        pages: dict[
            int,
            list[PDFTextBlock]
        ] = {}

        for block in blocks:

            pages.setdefault(
                block.page_no,
                [],
            ).append(block)

        output: list[dict] = []

        for page_no in sorted(
            pages
        ):

            page_blocks = sorted(
                pages[page_no],
                key=lambda item: (
                    item.y0,
                    item.x0,
                ),
            )

            output.append(
                {
                    "page_no": page_no,
                    "blocks": [
                        {
                            "block_no": block.block_no,
                            "x0": block.x0,
                            "y0": block.y0,
                            "x1": block.x1,
                            "y1": block.y1,
                            "text": block.text,
                        }
                        for block in page_blocks
                    ],
                    "text": "\n\n".join(
                        block.text
                        for block in page_blocks
                    ),
                }
            )

        return output

    # =========================================================
    # HEADER / FOOTER
    # =========================================================

    def _remove_headers_footers(
        self,
        blocks: list[PDFTextBlock],
    ) -> list[PDFTextBlock]:

        result = []

        for block in blocks:

            text = block.text.strip()

            if self._is_header_footer(
                block,
                text,
            ):
                continue

            result.append(block)

        return result

    def _is_header_footer(
        self,
        block: PDFTextBlock,
        text: str,
    ) -> bool:

        lower = text.lower().strip()

        for pattern in (
            self.HEADER_FOOTER_PATTERNS
        ):

            if re.match(
                pattern,
                lower,
            ):
                return True

        # Page-number-only block.
        if re.fullmatch(
            r"\d+",
            text,
        ):
            if (
                block.y0 < 60
                or block.y1 > 760
            ):
                return True

        return False

    # =========================================================
    # DUPLICATE BLOCKS
    # =========================================================

    def _deduplicate_blocks(
        self,
        blocks: list[PDFTextBlock],
    ) -> list[PDFTextBlock]:

        result: list[
            PDFTextBlock
        ] = []

        seen_by_page: dict[
            int,
            set[str]
        ] = {}

        for block in blocks:

            normalized = self._dedup_key(
                block.text
            )

            if not normalized:
                continue

            page_seen = seen_by_page.setdefault(
                block.page_no,
                set(),
            )

            if normalized in page_seen:
                continue

            page_seen.add(
                normalized
            )

            result.append(block)

        return result

    @staticmethod
    def _dedup_key(
        text: str,
    ) -> str:

        text = text.lower()

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        text = re.sub(
            r"[^\w\s]",
            "",
            text,
        )

        return text.strip()

    # =========================================================
    # NORMALIZATION
    # =========================================================

    @staticmethod
    def _normalize(
        text: str,
    ) -> str:

        text = text.replace(
            "\r",
            "\n",
        )

        lines = []

        for line in text.splitlines():

            line = re.sub(
                r"[ \t]+",
                " ",
                line,
            ).strip()

            if line:
                lines.append(
                    line
                )

        return "\n".join(
            lines
        ).strip()