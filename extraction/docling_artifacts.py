from __future__ import annotations

from typing import Any


class DoclingArtifactExtractor:
    """
    Extract parser-level artifacts from the Docling export.

    This layer preserves what Docling actually detected.
    It does NOT decide whether every picture is a logical research figure
    or whether multiple table regions form one logical table.

    That reconstruction happens in artifact_reconstruction.py.
    """

    def extract(
        self,
        document_dict: dict[str, Any],
    ) -> dict[str, Any]:
        text_blocks = self._extract_text_blocks(
            document_dict
        )

        figures = self._extract_pictures(
            document_dict
        )

        tables = self._extract_tables(
            document_dict
        )

        formulas = self._extract_formulas(
            document_dict
        )

        return {
            "figures": figures,
            "tables": tables,
            "formulas": formulas,
            "text_blocks": text_blocks,
            "counts": {
                "figures": len(figures),
                "tables": len(tables),
                "formulas": len(formulas),
                "text_blocks": len(text_blocks),
            },
        }

    # =========================================================
    # TEXT
    # =========================================================

    def _extract_text_blocks(
        self,
        document_dict: dict[str, Any],
    ) -> list[dict[str, Any]]:
        result = []

        for item in document_dict.get(
            "texts",
            [],
        ):
            if not isinstance(item, dict):
                continue

            text = self._clean(
                item.get("text")
            )

            if not text:
                continue

            result.append(
                {
                    "self_ref": item.get(
                        "self_ref"
                    ),
                    "text": text,
                    "label": self._clean(
                        item.get("label")
                    ),
                    "coords": self._coords(
                        item.get("prov")
                    ),
                }
            )

        return result

    # =========================================================
    # PICTURES
    # =========================================================

    def _extract_pictures(
        self,
        document_dict: dict[str, Any],
    ) -> list[dict[str, Any]]:
        result = []

        for item in document_dict.get(
            "pictures",
            [],
        ):
            if not isinstance(item, dict):
                continue

            caption = self._resolve_caption(
                item,
                document_dict,
            )

            result.append(
                {
                    "self_ref": item.get(
                        "self_ref"
                    ),
                    "label": self._clean(
                        item.get(
                            "label",
                            "picture",
                        )
                    ),
                    "caption": caption,
                    "annotations": item.get(
                        "annotations",
                        [],
                    ),
                    "coords": self._coords(
                        item.get("prov")
                    ),
                    "source_refs": [
                        {
                            "source": "docling",
                            "ref": item.get(
                                "self_ref"
                            ),
                            "artifact_type": "picture",
                        }
                    ],
                }
            )

        return result

    # =========================================================
    # TABLES
    # =========================================================

    def _extract_tables(
        self,
        document_dict: dict[str, Any],
    ) -> list[dict[str, Any]]:
        result = []

        for item in document_dict.get(
            "tables",
            [],
        ):
            if not isinstance(item, dict):
                continue

            structure = self._extract_table_structure(
                item
            )

            content = self._table_content(
                structure
            )

            result.append(
                {
                    "self_ref": item.get(
                        "self_ref"
                    ),
                    "label": self._clean(
                        item.get(
                            "label",
                            "table",
                        )
                    ),
                    "caption": self._resolve_caption(
                        item,
                        document_dict,
                    ),
                    "coords": self._coords(
                        item.get("prov")
                    ),
                    "content": content,
                    "structure": structure,

                    # IMPORTANT:
                    # artifact_reconstruction.py looks for
                    # "structured_data", not just "structure".
                    "structured_data": structure,

                    # Expose the reconstructed rows directly as well.
                    "rows": (
                        structure.get(
                            "rows",
                            []
                        )
                        if structure
                        else []
                    ),
                    "cells": (
                        structure.get(
                            "table_cells",
                            []
                        )
                        if structure
                        else []
                    ),

                    "structure_status": (
                        "structured"
                        if structure
                        and structure.get("rows")
                        else "partial"
                        if structure
                        and structure.get("table_cells")
                        else "unavailable"
                    ),

                    "source_refs": [
                        {
                            "source": "docling",
                            "ref": item.get(
                                "self_ref"
                            ),
                            "artifact_type": "table",
                        }
                    ],
                }
            )

        return result

    def _extract_table_structure(
        self,
        item: dict[str, Any],
    ) -> dict[str, Any] | None:
        data = item.get(
            "data"
        )

        if not isinstance(
            data,
            dict,
        ):
            return None

        grid = data.get(
            "grid"
        )

        table_cells = data.get(
            "table_cells"
        )

        num_rows = data.get(
            "num_rows"
        )

        num_cols = data.get(
            "num_cols"
        )

        if not isinstance(
            grid,
            list,
        ):
            grid = None

        if not isinstance(
            table_cells,
            list,
        ):
            table_cells = None

        if grid is None and table_cells is None:
            return None

        # ---------------------------------------------------------
        # Prefer Docling's actual grid when available.
        # ---------------------------------------------------------

        rows = []

        if grid:
            for raw_row in grid:
                if not isinstance(
                    raw_row,
                    list,
                ):
                    continue

                row = []

                for raw_cell in raw_row:
                    if not isinstance(
                        raw_cell,
                        dict,
                    ):
                        continue

                    text = self._clean(
                        raw_cell.get(
                            "text",
                            raw_cell.get(
                                "value",
                                raw_cell.get(
                                    "content",
                                    "",
                                ),
                            ),
                        )
                    )

                    row.append(
                        {
                            "text": text,
                            "row_span": raw_cell.get(
                                "row_span",
                                1,
                            ),
                            "col_span": raw_cell.get(
                                "col_span",
                                1,
                            ),
                            "start_row_offset_idx": raw_cell.get(
                                "start_row_offset_idx"
                            ),
                            "end_row_offset_idx": raw_cell.get(
                                "end_row_offset_idx"
                            ),
                            "start_col_offset_idx": raw_cell.get(
                                "start_col_offset_idx"
                            ),
                            "end_col_offset_idx": raw_cell.get(
                                "end_col_offset_idx"
                            ),
                        }
                    )

                if row:
                    rows.append(
                        row
                    )

        # ---------------------------------------------------------
        # Some Docling versions populate table_cells but leave grid
        # empty. Reconstruct rows from the row/column offsets.
        # ---------------------------------------------------------

        if not rows and table_cells:

            grouped_rows = {}

            for raw_cell in table_cells:
                if not isinstance(
                    raw_cell,
                    dict,
                ):
                    continue

                row_idx = raw_cell.get(
                    "start_row_offset_idx"
                )

                col_idx = raw_cell.get(
                    "start_col_offset_idx"
                )

                try:
                    row_idx = int(
                        row_idx
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    continue

                try:
                    col_idx = int(
                        col_idx
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    col_idx = 0

                text = self._clean(
                    raw_cell.get(
                        "text",
                        raw_cell.get(
                            "value",
                            raw_cell.get(
                                "content",
                                "",
                            ),
                        ),
                    )
                )

                normalized_cell = {
                    "text": text,
                    "row_span": raw_cell.get(
                        "row_span",
                        1,
                    ),
                    "col_span": raw_cell.get(
                        "col_span",
                        1,
                    ),
                    "start_row_offset_idx": row_idx,
                    "end_row_offset_idx": raw_cell.get(
                        "end_row_offset_idx"
                    ),
                    "start_col_offset_idx": col_idx,
                    "end_col_offset_idx": raw_cell.get(
                        "end_col_offset_idx"
                    ),
                }

                grouped_rows.setdefault(
                    row_idx,
                    [],
                ).append(
                    (
                        col_idx,
                        normalized_cell,
                    )
                )

            for row_idx in sorted(
                grouped_rows
            ):
                ordered_cells = sorted(
                    grouped_rows[row_idx],
                    key=lambda value: value[0],
                )

                rows.append(
                    [
                        cell
                        for _, cell in ordered_cells
                    ]
                )

        return {
            "num_rows": num_rows,
            "num_cols": num_cols,
            "grid": grid,
            "table_cells": table_cells,
            "rows": rows,
        }

    @staticmethod
    def _table_content(
        structure: dict[str, Any] | None,
    ) -> str:
        if not structure:
            return ""

        rows = structure.get(
            "rows"
        )

        if not isinstance(
            rows,
            list,
        ):
            rows = []

        # Fallback for older grid representation.
        if not rows:
            grid = structure.get(
                "grid"
            )

            if isinstance(
                grid,
                list,
            ):
                rows = grid

        output_rows = []

        for row in rows:
            if not isinstance(
                row,
                list,
            ):
                continue

            values = []

            for cell in row:
                if not isinstance(
                    cell,
                    dict,
                ):
                    continue

                text = str(
                    cell.get(
                        "text",
                        "",
                    )
                ).strip()

                values.append(
                    text
                )

            if values:
                output_rows.append(
                    " | ".join(values)
                )

        return "\n".join(
            output_rows
        )

    # =========================================================
    # FORMULAS
    # =========================================================

    def _extract_formulas(
        self,
        document_dict: dict[str, Any],
    ) -> list[dict[str, Any]]:
        result = []

        for item in document_dict.get(
            "texts",
            [],
        ):
            if not isinstance(item, dict):
                continue

            label = self._clean(
                item.get(
                    "label",
                    ""
                )
            ).lower()

            if (
                "formula" not in label
                and "equation" not in label
            ):
                continue

            text = self._clean(
                item.get("text")
            )

            if not text:
                continue

            result.append(
                {
                    "formula_id": (
                        f"formula_{len(result) + 1:03d}"
                    ),
                    "text": text,
                    "orig": self._clean(
                        item.get("orig")
                    ),
                    "page": (
                        self._first_page(
                            item.get("prov")
                        )
                    ),
                    "coords": self._coords(
                        item.get("prov")
                    ),
                    "source_refs": [
                        {
                            "source": "docling",
                            "ref": item.get(
                                "self_ref"
                            ),
                            "artifact_type": "formula",
                        }
                    ],
                    "source": "docling",
                }
            )

        return result

    # =========================================================
    # CAPTIONS
    # =========================================================

    def _resolve_caption(
        self,
        item: dict[str, Any],
        document_dict: dict[str, Any],
    ) -> str:
        texts = document_dict.get(
            "texts",
            [],
        )

        by_ref = {}

        for text_item in texts:
            if not isinstance(
                text_item,
                dict,
            ):
                continue

            ref = text_item.get(
                "self_ref"
            )

            if ref:
                by_ref[ref] = text_item

        # Modern Docling representation:
        # item["captions"] -> [{"$ref": "#/texts/14"}]
        for ref_item in item.get(
            "captions",
            [],
        ):
            if not isinstance(
                ref_item,
                dict,
            ):
                continue

            ref = (
                ref_item.get("$ref")
                or ref_item.get("ref")
            )

            caption_item = by_ref.get(
                ref
            )

            if caption_item:
                text = self._clean(
                    caption_item.get("text")
                )

                if text:
                    return text

        # Older/local representation:
        # item["children"] -> text-like dicts
        for child in item.get(
            "children",
            [],
        ):
            if not isinstance(
                child,
                dict,
            ):
                continue

            text = self._clean(
                child.get("text")
            )

            if text:
                return text

        return ""

    # =========================================================
    # PROVENANCE
    # =========================================================

    @staticmethod
    def _coords(
        provenance: Any,
    ) -> list[dict[str, Any]]:
        if not isinstance(
            provenance,
            list,
        ):
            return []

        result = []

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

            page = prov.get(
                "page_no"
            )

            if page is None:
                continue

            result.append(
                {
                    "page": page,
                    "x": bbox.get(
                        "l",
                        0.0,
                    ),
                    "y": bbox.get(
                        "b",
                        0.0,
                    ),
                    "w": (
                        bbox.get(
                            "r",
                            0.0,
                        )
                        - bbox.get(
                            "l",
                            0.0,
                        )
                    ),
                    "h": (
                        bbox.get(
                            "t",
                            0.0,
                        )
                        - bbox.get(
                            "b",
                            0.0,
                        )
                    ),
                    "coord_origin": bbox.get(
                        "coord_origin",
                        "BOTTOMLEFT",
                    ),
                }
            )

        return result

    @staticmethod
    def _first_page(
        provenance: Any,
    ) -> int | None:
        coords = DoclingArtifactExtractor._coords(
            provenance
        )

        if not coords:
            return None

        return coords[0].get(
            "page"
        )

    # =========================================================
    # HELPERS
    # =========================================================

    @staticmethod
    def _clean(
        value: Any,
    ) -> str:
        if value is None:
            return ""

        return " ".join(
            str(value)
            .replace("\xa0", " ")
            .split()
        )