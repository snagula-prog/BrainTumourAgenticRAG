from __future__ import annotations

from typing import Any


class TableChunker:

    def chunk(
        self,
        tables: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:

        units: list[
            dict[str, Any]
        ] = []

        for table in tables:

            if not isinstance(
                table,
                dict,
            ):
                continue

            label = str(
                table.get(
                    "label"
                )
                or ""
            ).strip()

            caption = str(
                table.get(
                    "caption"
                )
                or ""
            ).strip()

            content = self._extract_content(
                table
            )

            if not (
                label
                or caption
                or content
            ):
                continue

            section_path = (
                table.get(
                    "section_path"
                )
                or []
            )

            section = (
                table.get(
                    "section"
                )
                or (
                    section_path[-1]
                    if section_path
                    else "Unclassified"
                )
            )

            text_parts = []

            if label:
                text_parts.append(
                    f"Table {label}"
                )

            if caption:
                text_parts.append(
                    caption
                )

            if content:
                text_parts.append(
                    content
                )

            text = "\n".join(
                text_parts
            )

            pages = self._pages(
                table
            )

            units.append(
                {
                    "unit_type": "table",
                    "section": section,
                    "section_path": section_path,
                    "text": text,
                    "embedding_text": (
                        f"Table\n"
                        f"Section: {section}\n"
                        f"{text}"
                    ),
                    "label": label,
                    "caption": caption,
                    "table_content": content,
                    "pages": pages,
                    "page_start": (
                        min(pages)
                        if pages
                        else table.get(
                            "page"
                        )
                    ),
                    "page_end": (
                        max(pages)
                        if pages
                        else table.get(
                            "page"
                        )
                    ),
                    "coords": (
                        table.get(
                            "coords"
                        )
                        or []
                    ),
                    "retrieval_role": (
                        "structured"
                    ),
                    "source_ref": {
                        "page": table.get(
                            "page"
                        ),
                        "coords": (
                            table.get(
                                "coords"
                            )
                            or []
                        ),
                    },
                }
            )

        return units

    @staticmethod
    def _extract_content(
        table: dict[str, Any],
    ) -> str:

        content = table.get(
            "content"
        )

        if isinstance(
            content,
            str,
        ):
            return content.strip()

        # Allow future structured table
        # representation without changing
        # this module's public API.
        if isinstance(
            content,
            list,
        ):

            rows: list[str] = []

            for row in content:

                if isinstance(
                    row,
                    list,
                ):
                    rows.append(
                        " | ".join(
                            str(cell)
                            for cell
                            in row
                        )
                    )

                elif isinstance(
                    row,
                    dict,
                ):
                    rows.append(
                        " | ".join(
                            f"{key}: {value}"
                            for key, value
                            in row.items()
                        )
                    )

            return "\n".join(
                rows
            ).strip()

        return ""

    @staticmethod
    def _pages(
        table: dict[str, Any],
    ) -> list[int]:

        pages: list[int] = []

        page = table.get(
            "page"
        )

        if isinstance(page, int):
            pages.append(page)

        for coord in (
            table.get(
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