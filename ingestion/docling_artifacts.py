from __future__ import annotations

import re
from pathlib import Path
from typing import Any


class DoclingArtifactExtractor:
    """
    Extracts retrieval-ready artifact records from Docling's exported
    dictionary.

    Output schema:

        {
            "text_blocks": [...],
            "figures": [...],
            "tables": [...],
            "formulas": [...],
            "counts": {...}
        }

    Important design rule:
    Do not collapse distinct Docling artifacts merely because a caption,
    label, or page is missing.
    """

    def extract(
        self,
        raw_docling: dict[str, Any],
        *,
        fallback_pdf_path: str | Path | None = None,
    ) -> dict[str, Any]:

        document = self._get_document(
            raw_docling
        )

        texts = self._as_list(
            document.get("texts")
        )

        pictures = self._as_list(
            document.get("pictures")
        )

        tables = self._as_list(
            document.get("tables")
        )

        formulas = self._as_list(
            document.get("formulas")
        )

        text_lookup = {
            item.get("self_ref"): item
            for item in texts
            if isinstance(item, dict)
            and item.get("self_ref")
        }

        text_blocks = self._extract_text_blocks(
            texts
        )

        figures = self._extract_figures(
            pictures,
            text_lookup,
        )

        table_records = self._extract_tables(
            tables,
            text_lookup,
            fallback_pdf_path=fallback_pdf_path,
        )

        formula_records = self._extract_formulas(
            formulas,
            texts,
        )

        return {
            "text_blocks": text_blocks,
            "figures": figures,
            "tables": table_records,
            "formulas": formula_records,
            "counts": {
                "text_blocks": len(text_blocks),
                "figures": len(figures),
                "tables": len(table_records),
                "formulas": len(formula_records),
            },
        }

    # =========================================================
    # DOCUMENT
    # =========================================================

    @staticmethod
    def _get_document(
        raw_docling: dict[str, Any],
    ) -> dict[str, Any]:

        if not isinstance(
            raw_docling,
            dict,
        ):
            return {}

        nested = raw_docling.get(
            "docling_document"
        )

        if isinstance(
            nested,
            dict,
        ):
            return nested

        return raw_docling

    @staticmethod
    def _as_list(
        value: Any,
    ) -> list[Any]:

        return (
            value
            if isinstance(value, list)
            else []
        )

    # =========================================================
    # TEXT
    # =========================================================

    def _extract_text_blocks(
        self,
        texts: list[Any],
    ) -> list[dict[str, Any]]:

        result: list[dict[str, Any]] = []

        for item in texts:

            if not isinstance(
                item,
                dict,
            ):
                continue

            text = self._clean_text(
                item.get("text")
                or item.get("orig")
            )

            if not text:
                continue

            coords = self._extract_coords(
                item
            )

            result.append(
                {
                    "self_ref": item.get(
                        "self_ref"
                    ),
                    "label": self._clean_text(
                        item.get("label")
                    ),
                    "text": text,
                    "coords": coords,
                    "page": self._extract_page(
                        item,
                        coords,
                    ),
                    "source": "docling",
                }
            )

        return result

    # =========================================================
    # FIGURES
    # =========================================================

    def _extract_figures(
        self,
        pictures: list[Any],
        text_lookup: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:

        result: list[dict[str, Any]] = []

        for index, item in enumerate(
            pictures,
            start=1,
        ):

            if not isinstance(
                item,
                dict,
            ):
                continue

            coords = self._extract_coords(
                item
            )

            page = self._extract_page(
                item,
                coords,
            )

            caption = self._resolve_refs(
                item.get(
                    "captions",
                    [],
                ),
                text_lookup,
            )

            references = self._resolve_refs(
                item.get(
                    "references",
                    [],
                ),
                text_lookup,
            )

            label = self._clean_text(
                item.get("label")
            )

            source_ref = (
                item.get("self_ref")
                or f"#/pictures/{index - 1}"
            )

            result.append(
                {
                    "figure_id": (
                        f"figure_{index:03d}"
                    ),
                    "source_ref": source_ref,
                    "label": label or None,
                    "caption": (
                        caption or None
                    ),
                    "caption_status": (
                        "available"
                        if caption
                        else "missing"
                    ),
                    "page": page,
                    "coords": coords,
                    "references": references,
                    "source": "docling",
                    "renderable": bool(
                        coords
                    ),
                }
            )

        return result

    # =========================================================
    # TABLES
    # =========================================================

    def _extract_tables(
        self,
        tables: list[Any],
        text_lookup: dict[str, dict[str, Any]],
        *,
        fallback_pdf_path: str | Path | None = None,
    ) -> list[dict[str, Any]]:

        result: list[dict[str, Any]] = []

        for index, item in enumerate(
            tables,
            start=1,
        ):

            if not isinstance(
                item,
                dict,
            ):
                continue

            data = item.get(
                "data"
            )

            if not isinstance(
                data,
                dict,
            ):
                data = {}

            raw_cells = data.get(
                "table_cells"
            )

            if not isinstance(
                raw_cells,
                list,
            ):
                raw_cells = []

            cells = [
                self._normalize_table_cell(
                    cell
                )
                for cell in raw_cells
                if isinstance(
                    cell,
                    dict,
                )
            ]

            grid = data.get(
                "grid"
            )

            rows = self._normalize_grid(
                grid
            )

            if not rows and cells:
                rows = self._reconstruct_rows(
                    cells
                )

            caption = self._resolve_refs(
                item.get(
                    "captions",
                    [],
                ),
                text_lookup,
            )

            coords = self._extract_coords(
                item
            )

            page = self._extract_page(
                item,
                coords,
            )

            structure = {
                "num_rows": data.get(
                    "num_rows"
                ),
                "num_cols": data.get(
                    "num_cols"
                ),
                "orientation": data.get(
                    "orientation"
                ),
            }

            content = self._table_to_text(
                rows
            )

            if not content and cells:
                content = self._cells_to_text(
                    cells
                )

            source_ref = (
                item.get("self_ref")
                or f"#/tables/{index - 1}"
            )

            result.append(
                {
                    "table_id": (
                        f"table_{index:03d}"
                    ),
                    "source_ref": source_ref,
                    "label": (
                        self._clean_text(
                            item.get("label")
                        )
                        or None
                    ),
                    "caption": (
                        caption or None
                    ),
                    "page": page,
                    "coords": coords,
                    "rows": rows,
                    "cells": cells,
                    "content": content,
                    "text": content,
                    "structure": structure,
                    "structure_status": (
                        "structured"
                        if cells or rows or content.strip()
                        else "bbox_only"
                    ),
                    "source": "docling",
                }
            )

        # Docling can identify the table region while failing to populate
        # its structured table payload. In that case, use PyMuPDF's table
        # extraction as an independent fallback. This preserves the
        # Docling table identity/provenance while recovering the cell grid
        # from the PDF itself.
        if fallback_pdf_path and result:
            self._fill_unstructured_tables_from_pdf(
                result,
                fallback_pdf_path,
            )

        return result

    @classmethod
    def _normalize_grid(
        cls,
        grid: Any,
    ) -> list[list[dict[str, Any]]]:

        if not isinstance(
            grid,
            list,
        ):
            return []

        rows: list[
            list[dict[str, Any]]
        ] = []

        for raw_row in grid:

            if not isinstance(
                raw_row,
                list,
            ):
                continue

            row: list[
                dict[str, Any]
            ] = []

            for cell in raw_row:

                if not isinstance(
                    cell,
                    dict,
                ):
                    continue

                row.append(
                    cls._normalize_table_cell(
                        cell
                    )
                )

            if row:
                rows.append(row)

        return rows

    @classmethod
    def _normalize_table_cell(
        cls,
        cell: dict[str, Any],
    ) -> dict[str, Any]:

        text = cls._clean_text(
            cell.get("text")
            or cell.get("value")
            or cell.get("content")
        )

        bbox = cell.get(
            "bbox"
        )

        coords = (
            cls._bbox_to_coord(
                bbox
            )
            if isinstance(
                bbox,
                dict,
            )
            else None
        )

        return {
            "text": text,
            "row_span": cell.get(
                "row_span",
                1,
            ),
            "col_span": cell.get(
                "col_span",
                1,
            ),
            "start_row_offset_idx": cell.get(
                "start_row_offset_idx"
            ),
            "end_row_offset_idx": cell.get(
                "end_row_offset_idx"
            ),
            "start_col_offset_idx": cell.get(
                "start_col_offset_idx"
            ),
            "end_col_offset_idx": cell.get(
                "end_col_offset_idx"
            ),
            "column_header": bool(
                cell.get(
                    "column_header",
                    False,
                )
            ),
            "row_header": bool(
                cell.get(
                    "row_header",
                    False,
                )
            ),
            "row_section": bool(
                cell.get(
                    "row_section",
                    False,
                )
            ),
            "fillable": bool(
                cell.get(
                    "fillable",
                    False,
                )
            ),
            "coords": (
                coords
                if coords
                else []
            ),
        }

    @classmethod
    def _reconstruct_rows(
        cls,
        cells: list[dict[str, Any]],
    ) -> list[list[dict[str, Any]]]:

        grouped: dict[
            int,
            list[
                tuple[
                    int,
                    dict[str, Any],
                ]
            ],
        ] = {}

        for cell in cells:

            row_idx = cls._to_int(
                cell.get(
                    "start_row_offset_idx"
                )
            )

            col_idx = cls._to_int(
                cell.get(
                    "start_col_offset_idx"
                )
            )

            if row_idx is None:
                continue

            if col_idx is None:
                col_idx = 0

            grouped.setdefault(
                row_idx,
                [],
            ).append(
                (
                    col_idx,
                    cell,
                )
            )

        rows: list[
            list[dict[str, Any]]
        ] = []

        for row_idx in sorted(
            grouped
        ):

            ordered = sorted(
                grouped[row_idx],
                key=lambda item: item[0],
            )

            rows.append(
                [
                    cell
                    for _, cell in ordered
                ]
            )

        return rows

    @classmethod
    def _table_to_text(
        cls,
        rows: list[list[dict[str, Any]]],
    ) -> str:

        lines: list[str] = []

        for row in rows:

            values = [
                cls._clean_text(
                    cell.get("text")
                )
                for cell in row
                if isinstance(
                    cell,
                    dict,
                )
            ]

            values = [
                value
                for value in values
                if value
            ]

            if values:
                lines.append(
                    " | ".join(values)
                )

        return "\n".join(lines)

    @classmethod
    def _cells_to_text(
        cls,
        cells: list[dict[str, Any]],
    ) -> str:

        values = [
            cls._clean_text(
                cell.get("text")
            )
            for cell in cells
            if cls._clean_text(
                cell.get("text")
            )
        ]

        return " | ".join(values)

    # =========================================================
    # FORMULAS
    # =========================================================

    def _extract_formulas(
        self,
        formulas: list[Any],
        texts: list[Any],
    ) -> list[dict[str, Any]]:

        result: list[dict[str, Any]] = []
        seen_refs: set[str] = set()

        # Preferred representation: explicit Docling formula objects.
        for index, item in enumerate(
            formulas,
            start=1,
        ):

            if not isinstance(item, dict):
                continue

            text = (
                self._clean_formula(item.get("text"))
                or self._clean_formula(item.get("orig"))
            )

            if not text:
                # Some Docling versions keep the formula text in child
                # text objects rather than exporting a populated formula
                # list. The text fallback below handles that representation.
                continue

            coords = self._extract_coords(item)
            page = self._extract_page(item, coords)
            source_ref = (
                item.get("self_ref")
                or f"#/formulas/{index - 1}"
            )

            equation_number = (
                item.get("equation_number")
                or item.get("number")
            )

            if equation_number is None:
                match = re.search(
                    r"\(\s*(\d+)\s*\)\s*$",
                    text,
                )
                if match:
                    equation_number = match.group(1)

            result.append(
                {
                    "formula_id": f"formula_{len(result) + 1:03d}",
                    "source_ref": source_ref,
                    "text": text,
                    "orig": item.get("orig") or None,
                    "equation_number": equation_number,
                    "page": page,
                    "coords": coords,
                    "source": "docling",
                }
            )
            seen_refs.add(source_ref)

        # Compatibility fallback: in the observed Docling export, formulas
        # are sometimes plain text objects with label == "formula" and the
        # actual equation stored in text/orig. Treat these as formulas rather
        # than losing them entirely.
        for item in texts:

            if not isinstance(item, dict):
                continue

            label = self._clean_text(item.get("label")).lower()
            if label not in {"formula", "equation"}:
                continue

            source_ref = item.get("self_ref")
            if source_ref and source_ref in seen_refs:
                continue

            text = (
                self._clean_formula(item.get("text"))
                or self._clean_formula(item.get("orig"))
            )

            if not text:
                continue

            coords = self._extract_coords(item)
            page = self._extract_page(item, coords)

            equation_number = (
                item.get("equation_number")
                or item.get("number")
            )

            if equation_number is None:
                match = re.search(
                    r"\(\s*(\d+)\s*\)\s*$",
                    text,
                )
                if match:
                    equation_number = match.group(1)

            result.append(
                {
                    "formula_id": f"formula_{len(result) + 1:03d}",
                    "source_ref": (
                        source_ref
                        or f"#/texts/formula/{len(result) + 1:03d}"
                    ),
                    "text": text,
                    "orig": item.get("orig") or None,
                    "equation_number": equation_number,
                    "page": page,
                    "coords": coords,
                    "source": "docling",
                }
            )

            if source_ref:
                seen_refs.add(source_ref)

        return result

    # =========================================================
    # PYMuPDF TABLE FALLBACK
    # =========================================================

    def _fill_unstructured_tables_from_pdf(
        self,
        tables: list[dict[str, Any]],
        pdf_path: str | Path,
    ) -> None:
        """Fill bbox-only Docling tables using PyMuPDF table extraction."""

        try:
            import fitz
        except ImportError:
            return

        try:
            document = fitz.open(str(pdf_path))
        except Exception:
            return

        try:
            for table in tables:

                if table.get("structure_status") == "structured":
                    continue

                page_number = table.get("page")
                coords = table.get("coords") or []

                if page_number is None or not coords:
                    continue

                try:
                    page_index = int(page_number) - 1
                except (TypeError, ValueError):
                    continue

                if page_index < 0 or page_index >= len(document):
                    continue

                page = document[page_index]

                docling_rect = self._coord_to_top_left_rect(
                    coords[0],
                    page_height=float(page.rect.height),
                )

                if docling_rect is None:
                    continue

                finder = None
                for strategy in (None, "text"):
                    try:
                        kwargs = {"clip": docling_rect}
                        if strategy is not None:
                            kwargs["strategy"] = strategy
                        finder = page.find_tables(**kwargs)
                    except Exception:
                        finder = None
                    if finder is not None and finder.tables:
                        break

                if finder is None or not finder.tables:
                    continue

                # Pick the PDF table with the greatest overlap with the
                # Docling table region.
                best = max(
                    finder.tables,
                    key=lambda candidate: self._rect_overlap(
                        docling_rect,
                        self._fitz_table_bbox(candidate),
                    ),
                )

                overlap = self._rect_overlap(
                    docling_rect,
                    self._fitz_table_bbox(best),
                )

                if overlap <= 0.10:
                    continue

                extracted_rows = best.extract()
                if not extracted_rows:
                    continue

                rows = []
                cells = []

                for row_idx, raw_row in enumerate(extracted_rows):
                    normalized_row = []
                    row_cells = getattr(best, "rows", [])[row_idx].cells if row_idx < len(getattr(best, "rows", [])) else []

                    for col_idx, value in enumerate(raw_row):
                        text = self._clean_text(value)
                        cell_bbox = row_cells[col_idx] if col_idx < len(row_cells) else None
                        cell_coords = self._fitz_bbox_to_coord(
                            cell_bbox,
                            page_height=float(page.rect.height),
                        ) if cell_bbox else []

                        cell = {
                            "text": text,
                            "row_span": 1,
                            "col_span": 1,
                            "start_row_offset_idx": row_idx,
                            "end_row_offset_idx": row_idx + 1,
                            "start_col_offset_idx": col_idx,
                            "end_col_offset_idx": col_idx + 1,
                            "column_header": row_idx == 0,
                            "row_header": col_idx == 0,
                            "row_section": False,
                            "fillable": False,
                            "coords": cell_coords,
                        }
                        normalized_row.append(cell)
                        cells.append(cell)

                    if normalized_row:
                        rows.append(normalized_row)

                content = self._table_to_text(rows)
                if not content:
                    continue

                table["rows"] = rows
                table["cells"] = cells
                table["content"] = content
                table["text"] = content
                table["structure"] = {
                    "num_rows": len(rows),
                    "num_cols": max((len(row) for row in rows), default=0),
                    "orientation": "rot_0",
                    "source": "pymupdf_fallback",
                }
                table["structure_status"] = "structured_fallback"
                table["source"] = "docling+pymupdf_fallback"
                table["table_structure_source"] = "pymupdf"

        finally:
            document.close()

    @classmethod
    def _coord_to_top_left_rect(
        cls,
        coord: dict[str, Any],
        *,
        page_height: float,
    ):
        """Convert canonical/Docling coordinates into a fitz.Rect."""

        try:
            x = float(coord["x"])
            y = float(coord["y"])
            w = float(coord["w"])
            h = float(coord["h"])
        except (KeyError, TypeError, ValueError):
            return None

        origin = str(coord.get("coord_origin") or "BOTTOMLEFT").upper()

        if origin == "BOTTOMLEFT":
            top = page_height - (y + h)
            bottom = page_height - y
        else:
            top = y
            bottom = y + h

        return (x, top, x + w, bottom)

    @classmethod
    def _fitz_table_bbox(cls, table: Any):
        try:
            return tuple(table.bbox)
        except Exception:
            try:
                return tuple(table.bbox)
            except Exception:
                return None

    @classmethod
    def _fitz_bbox_to_coord(
        cls,
        bbox: Any,
        *,
        page_height: float,
    ) -> list[dict[str, Any]]:
        if not bbox or len(bbox) != 4:
            return []

        try:
            x0, y0, x1, y1 = [float(v) for v in bbox]
        except (TypeError, ValueError):
            return []

        return [{
            "page": None,
            "x": x0,
            "y": page_height - y1,
            "w": abs(x1 - x0),
            "h": abs(y1 - y0),
            "coord_origin": "BOTTOMLEFT",
        }]

    @staticmethod
    def _rect_overlap(a: Any, b: Any) -> float:
        if not a or not b:
            return 0.0

        try:
            ax0, ay0, ax1, ay1 = [float(v) for v in a]
            bx0, by0, bx1, by1 = [float(v) for v in b]
        except (TypeError, ValueError):
            return 0.0

        ix0 = max(ax0, bx0)
        iy0 = max(ay0, by0)
        ix1 = min(ax1, bx1)
        iy1 = min(ay1, by1)

        if ix1 <= ix0 or iy1 <= iy0:
            return 0.0

        intersection = (ix1 - ix0) * (iy1 - iy0)
        area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
        area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
        union = area_a + area_b - intersection

        return intersection / union if union > 0 else 0.0

    # =========================================================
    # REFERENCES / CAPTIONS
    # =========================================================

    @classmethod
    def _resolve_refs(
        cls,
        refs: Any,
        lookup: dict[str, dict[str, Any]],
    ) -> str:

        if not isinstance(
            refs,
            list,
        ):
            return ""

        values: list[str] = []

        for ref in refs:

            if isinstance(
                ref,
                dict,
            ):

                ref_id = ref.get(
                    "$ref"
                )

            else:
                ref_id = ref

            source = lookup.get(
                ref_id
            )

            if not source:
                continue

            text = cls._clean_text(
                source.get("text")
                or source.get("orig")
            )

            if text:
                values.append(text)

        return cls._clean_text(
            " ".join(values)
        )

    # =========================================================
    # PROVENANCE
    # =========================================================

    @classmethod
    def _extract_page(
        cls,
        item: dict[str, Any],
        coords: list[dict[str, Any]],
    ) -> int | None:

        value = item.get(
            "page"
        )

        if isinstance(
            value,
            int,
        ):
            return value

        if (
            isinstance(value, float)
            and value.is_integer()
        ):
            return int(value)

        if coords:

            value = coords[0].get(
                "page"
            )

            if isinstance(
                value,
                int,
            ):
                return value

        provenance = item.get(
            "prov",
        )

        if isinstance(
            provenance,
            list,
        ):

            for prov in provenance:

                if not isinstance(
                    prov,
                    dict,
                ):
                    continue

                value = prov.get(
                    "page_no"
                )

                if isinstance(
                    value,
                    int,
                ):
                    return value

                if (
                    isinstance(value, float)
                    and value.is_integer()
                ):
                    return int(value)

        return None

    @classmethod
    def _extract_coords(
        cls,
        item: dict[str, Any],
    ) -> list[dict[str, Any]]:

        direct = item.get(
            "coords"
        )

        if isinstance(
            direct,
            list,
        ):

            result = []

            for coord in direct:

                if not isinstance(
                    coord,
                    dict,
                ):
                    continue

                normalized = cls._normalize_coord(
                    coord
                )

                if normalized:
                    result.append(
                        normalized
                    )

            if result:
                return result

        provenance = item.get(
            "prov"
        )

        if not isinstance(
            provenance,
            list,
        ):
            return []

        result: list[
            dict[str, Any]
        ] = []

        for prov in provenance:

            if not isinstance(
                prov,
                dict,
            ):
                continue

            page = prov.get(
                "page_no"
            )

            bbox = prov.get(
                "bbox"
            )

            if not isinstance(
                bbox,
                dict,
            ):
                continue

            coord = cls._bbox_to_coord(
                bbox,
                page=page,
            )

            if coord:
                result.append(
                    coord
                )

        return result

    @classmethod
    def _normalize_coord(
        cls,
        coord: dict[str, Any],
    ) -> dict[str, Any] | None:

        page = coord.get(
            "page"
        )

        if page is None:
            page = coord.get(
                "page_no"
            )

        x = coord.get("x")
        y = coord.get("y")
        w = coord.get("w")
        h = coord.get("h")

        if w is None:
            w = coord.get(
                "width"
            )

        if h is None:
            h = coord.get(
                "height"
            )

        if (
            x is None
            or y is None
            or w is None
            or h is None
        ):
            return None

        try:

            return {
                "page": page,
                "x": float(x),
                "y": float(y),
                "w": abs(float(w)),
                "h": abs(float(h)),
                "coord_origin": (
                    coord.get(
                        "coord_origin"
                    )
                ),
            }

        except (
            TypeError,
            ValueError,
        ):
            return None

    @classmethod
    def _bbox_to_coord(
        cls,
        bbox: dict[str, Any],
        page: Any = None,
    ) -> dict[str, Any] | None:

        left = cls._to_float(
            bbox.get("l")
        )

        right = cls._to_float(
            bbox.get("r")
        )

        top = cls._to_float(
            bbox.get("t")
        )

        bottom = cls._to_float(
            bbox.get("b")
        )

        if (
            left is None
            or right is None
            or top is None
            or bottom is None
        ):
            return None

        return {
            "page": page,
            "x": left,
            "y": bottom,
            "w": abs(
                right - left
            ),
            "h": abs(
                top - bottom
            ),
            "coord_origin": (
                bbox.get(
                    "coord_origin"
                )
            ),
        }

    # =========================================================
    # HELPERS
    # =========================================================

    @staticmethod
    def _clean_text(
        value: Any,
    ) -> str:

        if value is None:
            return ""

        return re.sub(
            r"\s+",
            " ",
            str(value)
            .replace(
                "\xa0",
                " ",
            )
            .strip(),
        ).strip()

    @classmethod
    def _clean_formula(
        cls,
        value: Any,
    ) -> str:

        text = cls._clean_text(
            value
        )

        return text.replace(
            "\x08",
            " ",
        ).strip()

    @staticmethod
    def _to_int(
        value: Any,
    ) -> int | None:

        try:

            if value is None:
                return None

            return int(value)

        except (
            TypeError,
            ValueError,
        ):
            return None

    @staticmethod
    def _to_float(
        value: Any,
    ) -> float | None:

        try:

            if value is None:
                return None

            return float(value)

        except (
            TypeError,
            ValueError,
        ):
            return None
