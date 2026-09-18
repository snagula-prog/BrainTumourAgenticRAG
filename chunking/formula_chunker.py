from __future__ import annotations

from typing import Any


class FormulaChunker:

    def chunk(
        self,
        formulas: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:

        units: list[
            dict[str, Any]
        ] = []

        for formula in formulas:

            if not isinstance(
                formula,
                dict,
            ):
                continue

            text = str(
                formula.get(
                    "text"
                )
                or formula.get(
                    "orig"
                )
                or ""
            ).strip()

            if not text:
                continue

            section_path = (
                formula.get(
                    "section_path"
                )
                or []
            )

            section = (
                formula.get(
                    "section"
                )
                or (
                    section_path[-1]
                    if section_path
                    else "Unclassified"
                )
            )

            pages = self._pages(
                formula
            )

            units.append(
                {
                    "unit_type": "formula",
                    "section": section,
                    "section_path": section_path,
                    "text": text,
                    "embedding_text": (
                        f"Formula\n"
                        f"Section: {section}\n"
                        f"{text}"
                    ),
                    "pages": pages,
                    "page_start": (
                        min(pages)
                        if pages
                        else formula.get(
                            "page"
                        )
                    ),
                    "page_end": (
                        max(pages)
                        if pages
                        else formula.get(
                            "page"
                        )
                    ),
                    "coords": (
                        formula.get(
                            "coords"
                        )
                        or []
                    ),
                    "formula_source": (
                        formula.get(
                            "source"
                        )
                    ),
                    "retrieval_role": (
                        "scientific"
                    ),
                }
            )

        return units

    @staticmethod
    def _pages(
        formula: dict[str, Any],
    ) -> list[int]:

        pages: list[int] = []

        page = formula.get(
            "page"
        )

        if isinstance(page, int):
            pages.append(
                page
            )

        for coord in (
            formula.get(
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