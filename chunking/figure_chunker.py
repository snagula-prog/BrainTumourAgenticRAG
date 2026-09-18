from __future__ import annotations

from typing import Any


class FigureChunker:

    def chunk(
        self,
        figures: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:

        units: list[
            dict[str, Any]
        ] = []

        for figure in figures:

            if not isinstance(
                figure,
                dict,
            ):
                continue

            caption = str(
                figure.get(
                    "caption"
                )
                or ""
            ).strip()

            label = str(
                figure.get(
                    "label"
                )
                or ""
            ).strip()

            if not caption and not label:
                continue

            section_path = (
                figure.get(
                    "section_path"
                )
                or []
            )

            section = (
                figure.get(
                    "section"
                )
                or (
                    section_path[-1]
                    if section_path
                    else "Unclassified"
                )
            )

            text = self._build_text(
                label=label,
                caption=caption,
            )

            pages = self._pages(
                figure
            )

            units.append(
                {
                    "unit_type": "figure",
                    "section": section,
                    "section_path": section_path,
                    "text": text,
                    "embedding_text": (
                        f"Figure\n"
                        f"Section: {section}\n"
                        f"{text}"
                    ),
                    "label": label,
                    "caption": caption,
                    "pages": pages,
                    "page_start": (
                        min(pages)
                        if pages
                        else figure.get("page")
                    ),
                    "page_end": (
                        max(pages)
                        if pages
                        else figure.get("page")
                    ),
                    "coords": (
                        figure.get(
                            "coords"
                        )
                        or []
                    ),
                    "retrieval_role": "visual",
                    "source_ref": {
                        "page": figure.get(
                            "page"
                        ),
                        "coords": (
                            figure.get(
                                "coords"
                            )
                            or []
                        ),
                    },
                }
            )

        return units

    @staticmethod
    def _build_text(
        *,
        label: str,
        caption: str,
    ) -> str:

        if label and caption:
            return (
                f"Figure {label}. "
                f"{caption}"
            )

        if caption:
            return caption

        return f"Figure {label}"

    @staticmethod
    def _pages(
        artifact: dict[str, Any],
    ) -> list[int]:

        pages: list[int] = []

        page = artifact.get(
            "page"
        )

        if isinstance(page, int):
            pages.append(page)

        for coord in (
            artifact.get(
                "coords"
            )
            or []
        ):

            if isinstance(
                coord,
                dict,
            ):
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

        return sorted(
            set(pages)
        )