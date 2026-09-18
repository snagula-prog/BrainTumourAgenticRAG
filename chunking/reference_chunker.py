from __future__ import annotations

from typing import Any


class ReferenceChunker:

    def chunk(
        self,
        references: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:

        units: list[
            dict[str, Any]
        ] = []

        for reference in references:

            if not isinstance(
                reference,
                dict,
            ):
                continue

            text = str(
                reference.get(
                    "text"
                )
                or reference.get(
                    "raw"
                )
                or ""
            ).strip()

            if not text:
                continue

            pages = self._pages(
                reference
            )

            units.append(
                {
                    "unit_type": "reference",
                    "section": "References",
                    "section_path": [
                        "References"
                    ],
                    "text": text,
                    "embedding_text": text,
                    "pages": pages,
                    "page_start": (
                        min(pages)
                        if pages
                        else None
                    ),
                    "page_end": (
                        max(pages)
                        if pages
                        else None
                    ),
                    "coords": (
                        reference.get(
                            "coords"
                        )
                        or []
                    ),
                    "reference_id": (
                        reference.get(
                            "reference_id"
                        )
                    ),
                    "retrieval_role": (
                        "citation"
                    ),
                }
            )

        return units

    @staticmethod
    def _pages(
        reference: dict[str, Any],
    ) -> list[int]:

        pages: list[int] = []

        for coord in (
            reference.get(
                "coords"
            )
            or []
        ):

            if isinstance(
                coord,
                dict,
            ):

                page = coord.get(
                    "page"
                )

                if isinstance(
                    page,
                    int,
                ):
                    pages.append(
                        page
                    )

        return sorted(
            set(pages)
        )