# ingestion/pymupdf_audit.py

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import fitz


class PyMuPDFAudit:
    """
    Independent PDF audit.

    PyMuPDF is NOT treated as the primary scientific parser.
    It is used to verify:
      - raw text availability
      - page/block counts
      - numeric content
      - metadata
      - possible extraction truncation
    """

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

    def extract(self) -> dict[str, Any]:

        metadata = self._extract_metadata()
        pages = self._extract_pages()

        full_text = "\n\n".join(
            page["text"]
            for page in pages
            if page["text"]
        )

        return {
            "metadata": metadata,
            "num_pages": len(pages),
            "pages": pages,
            "text": full_text,
            "stats": {
                "characters": len(full_text),
                "words": len(
                    full_text.split()
                ),
                "numbers": len(
                    re.findall(
                        r"\b\d+(?:\.\d+)?%?\b",
                        full_text,
                    )
                ),
                "blocks": sum(
                    len(page["blocks"])
                    for page in pages
                ),
            },
        }

    # =========================================================
    # METADATA
    # =========================================================

    def _extract_metadata(self) -> dict[str, Any]:

        with fitz.open(
            self.pdf_path
        ) as pdf:

            info = pdf.metadata or {}

        return {
            "title": info.get(
                "title"
            ),
            "author": info.get(
                "author"
            ),
            "subject": info.get(
                "subject"
            ),
            "keywords": info.get(
                "keywords"
            ),
            "creator": info.get(
                "creator"
            ),
            "producer": info.get(
                "producer"
            ),
            "creation_date": info.get(
                "creationDate"
            ),
            "modification_date": info.get(
                "modDate"
            ),
        }

    # =========================================================
    # PAGES
    # =========================================================

    def _extract_pages(self) -> list[dict]:

        pages = []

        with fitz.open(
            self.pdf_path
        ) as pdf:

            for page_number, page in enumerate(
                pdf,
                start=1,
            ):

                raw_blocks = page.get_text(
                    "blocks"
                )

                blocks = []

                for block_number, block in enumerate(
                    raw_blocks
                ):

                    if len(block) < 5:
                        continue

                    x0, y0, x1, y1, text = (
                        block[:5]
                    )

                    text = self._normalize_text(
                        text
                    )

                    if not text:
                        continue

                    blocks.append(
                        {
                            "block_no": (
                                block_number
                            ),
                            "x0": float(x0),
                            "y0": float(y0),
                            "x1": float(x1),
                            "y1": float(y1),
                            "text": text,
                        }
                    )

                # PDF coordinates are top-left based.
                blocks.sort(
                    key=lambda item: (
                        item["y0"],
                        item["x0"],
                    )
                )

                pages.append(
                    {
                        "page_no": page_number,
                        "blocks": blocks,
                        "text": "\n\n".join(
                            block["text"]
                            for block in blocks
                        ),
                    }
                )

        return pages

    # =========================================================
    # NORMALIZATION
    # =========================================================

    @staticmethod
    def _normalize_text(
        text: str,
    ) -> str:

        text = text.replace(
            "\r\n",
            "\n",
        )

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