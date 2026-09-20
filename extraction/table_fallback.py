from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TableFallbackConfig:
    """Controls for the targeted table fallback."""

    render_scale: float = 2.5
    min_text_words: int = 8
    min_ocr_confidence: float = 35.0
    column_gap_px: int = 80
    row_y_tolerance_px: int = 12
    row_start_coverage: float = 0.72
    min_rows: int = 2
    min_columns: int = 2
    ocr_enabled: bool = True


class TableFallbackReconstructor:
    """
    Lightweight, targeted fallback for Docling tables with missing structure.

    Strategy:
      1. Use only the table bounding box.
      2. Prefer native PDF text-layer words when available.
      3. Fall back to Tesseract OCR for that crop only.
      4. Infer columns from the header x-positions.
      5. Group OCR/PDF visual lines into logical rows using column coverage.

    This is heuristic and evidence-preserving. It never invents cell values.
    A weak result is returned as None rather than being forced into structure.
    """

    def __init__(
        self,
        config: TableFallbackConfig | None = None,
    ) -> None:
        self.config = config or TableFallbackConfig()

    def reconstruct(
        self,
        *,
        pdf_path: Path,
        table: dict[str, Any],
        column_anchors: list[float] | None = None,
    ) -> dict[str, Any] | None:
        regions = table.get("coords") or table.get("regions") or []
        if not isinstance(regions, list) or not regions:
            return None

        try:
            import fitz  # PyMuPDF
        except ImportError:
            return None

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            return None

        page_results: list[dict[str, Any]] = []

        try:
            with fitz.open(pdf_path) as document:
                anchors = list(column_anchors) if column_anchors else None
                for region_index, region in enumerate(regions):
                    result = self._reconstruct_region(
                        document=document,
                        region=region,
                        column_anchors=anchors,
                    )
                    if result and anchors is None:
                        anchors = result.get(
                            "column_anchors"
                        )
                    if result:
                        page_results.append(result)
        except Exception:
            return None

        if not page_results:
            return None

        return self._merge_region_results(page_results)

    def _reconstruct_region(
        self,
        *,
        document: Any,
        region: dict[str, Any],
        column_anchors: list[float] | None = None,
    ) -> dict[str, Any] | None:
        page_number = self._page_number(region)
        if page_number is None:
            return None

        page_index = page_number - 1
        if page_index < 0 or page_index >= len(document):
            return None

        page = document[page_index]
        rect = self._to_pdf_rect(page, region)
        if rect is None or rect.is_empty:
            return None

        words = self._pdf_text_words(page, rect)
        method = "pdf_text_layout"
        ocr_confidence = None

        # OCR is only invoked for this table crop when native text is absent.
        if len(words) < self.config.min_text_words and self.config.ocr_enabled:
            ocr_result = self._ocr_words(page, rect)
            if ocr_result:
                words = ocr_result["words"]
                method = "ocr_layout"
                ocr_confidence = ocr_result["mean_confidence"]

        if len(words) < self.config.min_text_words:
            return None

        columns = (
            list(column_anchors)
            if column_anchors
            else self._infer_columns(words)
        )
        if len(columns) < self.config.min_columns:
            return None

        visual_lines = self._group_visual_lines(words)
        if not visual_lines:
            return None

        rows = self._group_logical_rows(
            visual_lines,
            columns,
        )
        if len(rows) < self.config.min_rows:
            return None

        structured_rows = [
            self._row_to_cells(row, columns)
            for row in rows
        ]
        structured_rows = [
            row
            for row in structured_rows
            if any(cell.strip() for cell in row)
        ]

        if len(structured_rows) < self.config.min_rows:
            return None

        nonempty = sum(
            1
            for row in structured_rows
            for cell in row
            if cell.strip()
        )
        total = len(structured_rows) * len(columns)
        occupancy = nonempty / total if total else 0.0

        confidence = self._confidence(
            column_count=len(columns),
            occupancy=occupancy,
            method=method,
            ocr_confidence=ocr_confidence,
        )

        return {
            "page": page_number,
            "rows": structured_rows,
            "columns": [
                f"column_{i}"
                for i in range(1, len(columns) + 1)
            ],
            "num_rows": len(structured_rows),
            "num_cols": len(columns),
            "column_anchors": columns,
            "method": method,
            "confidence": round(confidence, 3),
            "occupancy": round(occupancy, 3),
            "ocr_mean_confidence": (
                round(ocr_confidence, 2)
                if ocr_confidence is not None
                else None
            ),
        }

    @staticmethod
    def _merge_region_results(
        regions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        max_cols = max(
            result.get("num_cols", 0)
            for result in regions
        )

        rows: list[list[str]] = []
        pages: list[int] = []
        methods: list[str] = []
        confidences: list[float] = []

        for result in regions:
            page = result.get("page")
            if isinstance(page, int):
                pages.append(page)

            method = result.get("method")
            if method:
                methods.append(str(method))

            confidence = result.get("confidence")
            if isinstance(confidence, (int, float)):
                confidences.append(float(confidence))

            for row in result.get("rows", []):
                normalized = list(row)
                if len(normalized) < max_cols:
                    normalized.extend(
                        [""] * (max_cols - len(normalized))
                    )
                rows.append(normalized[:max_cols])

        return {
            "columns": [
                f"column_{i}"
                for i in range(1, max_cols + 1)
            ],
            "rows": rows,
            "num_rows": len(rows),
            "num_cols": max_cols,
            "pages": sorted(set(pages)),
            "method": (
                "ocr_layout"
                if "ocr_layout" in methods
                else "pdf_text_layout"
            ),
            "confidence": round(
                sum(confidences) / len(confidences),
                3,
            ) if confidences else 0.0,
            "region_results": regions,
        }

    @staticmethod
    def _pdf_text_words(
        page: Any,
        rect: Any,
    ) -> list[dict[str, Any]]:
        try:
            raw_words = page.get_text(
                "words",
                clip=rect,
            )
        except Exception:
            return []

        result = []
        for word in raw_words:
            if len(word) < 5:
                continue

            text = str(word[4]).strip()
            if not text:
                continue

            result.append(
                {
                    "x0": float(word[0] - rect.x0),
                    "y0": float(word[1] - rect.y0),
                    "x1": float(word[2] - rect.x0),
                    "y1": float(word[3] - rect.y0),
                    "text": text,
                    "confidence": 100.0,
                }
            )

        return result

    def _ocr_words(
        self,
        page: Any,
        rect: Any,
    ) -> dict[str, Any] | None:
        try:
            import pytesseract
            from io import BytesIO
            from PIL import Image
            from pytesseract import Output
        except ImportError:
            return None

        self._configure_tesseract(pytesseract)

        try:
            pix = page.get_pixmap(
                matrix=self._matrix(),
                clip=rect,
                alpha=False,
            )
            image = Image.open(
                BytesIO(pix.tobytes("png"))
            )
            data = pytesseract.image_to_data(
                image,
                config="--psm 6",
                output_type=Output.DICT,
            )
        except Exception:
            return None

        words: list[dict[str, Any]] = []
        confidences: list[float] = []

        for i, raw_text in enumerate(
            data.get("text", [])
        ):
            text = str(raw_text).strip()
            if not text:
                continue

            try:
                confidence = float(
                    data["conf"][i]
                )
            except Exception:
                confidence = 0.0

            if confidence < self.config.min_ocr_confidence:
                continue

            x = float(data["left"][i])
            y = float(data["top"][i])
            width = float(data["width"][i])
            height = float(data["height"][i])

            words.append(
                {
                    "x0": x,
                    "y0": y,
                    "x1": x + width,
                    "y1": y + height,
                    "text": text,
                    "confidence": confidence,
                    "block_num": data.get(
                        "block_num", [None]
                    )[i],
                    "par_num": data.get(
                        "par_num", [None]
                    )[i],
                    "line_num": data.get(
                        "line_num", [None]
                    )[i],
                }
            )
            confidences.append(confidence)

        if not words:
            return None

        return {
            "words": words,
            "mean_confidence": (
                sum(confidences) / len(confidences)
                if confidences
                else 0.0
            ),
        }

    @staticmethod
    def _configure_tesseract(
        pytesseract: Any,
    ) -> None:
        configured = os.environ.get(
            "TESSERACT_CMD"
        )
        if configured and Path(configured).exists():
            pytesseract.pytesseract.tesseract_cmd = (
                configured
            )
            return

        # Common Windows installation paths. PATH is still preferred by
        # pytesseract when these paths are not present.
        candidates = (
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        )
        for candidate in candidates:
            if Path(candidate).exists():
                pytesseract.pytesseract.tesseract_cmd = (
                    candidate
                )
                return

    def _infer_columns(
        self,
        words: list[dict[str, Any]],
    ) -> list[float]:
        """
        Infer column anchors primarily from the table header, then
        validate them against repeated line-level x alignments.

        The previous implementation treated individual word gaps as
        column boundaries. That is unreliable for wrapped research-table
        cells because the gap between two words inside one header/cell can
        be comparable to the gap between adjacent columns.

        This implementation:
          1. builds visual lines,
          2. searches the first few lines for a header-like line,
          3. clusters nearby word starts to merge multi-word headers,
          4. validates the candidate anchors against the whole table, and
          5. falls back to repeated line-start alignment when no header is
             confidently detected.
        """

        if not words:
            return []

        ordered = sorted(
            words,
            key=lambda w: (w["y0"], w["x0"]),
        )

        lines = self._group_visual_lines(
            ordered
        )

        if not lines:
            return []

        median_height = max(
            4.0,
            self._median(
                [
                    max(
                        1.0,
                        float(w["y1"]) - float(w["y0"]),
                    )
                    for w in ordered
                ]
            ),
        )

        # Word-starts that are this close are likely part of the same
        # multi-word header/cell rather than separate columns.
        header_cluster_gap = max(
            6.0,
            min(
                16.0,
                median_height * 1.35,
            ),
        )

        # -------------------------------------------------------------
        # 1. Look for a header-like line near the top of the table.
        # -------------------------------------------------------------

        header_candidates: list[
            tuple[
                float,
                list[float],
            ]
        ] = []

        for line_index, line in enumerate(
            lines[:8]
        ):
            if not line:
                continue

            line = sorted(
                line,
                key=lambda item: item["x0"],
            )

            anchors = self._cluster_header_starts(
                line,
                gap=header_cluster_gap,
            )

            if len(anchors) < self.config.min_columns:
                continue

            if len(anchors) > 16:
                continue

            token_lengths = [
                len(str(word.get("text", "")).strip())
                for word in line
                if str(word.get("text", "")).strip()
            ]

            if not token_lengths:
                continue

            # Header lines normally have several short/medium tokens and
            # occupy a broad fraction of the table width.
            x_left = float(line[0]["x0"])
            x_right = float(line[-1]["x1"])
            spread = max(
                0.0,
                x_right - x_left,
            )

            broadness = spread / max(
                1.0,
                float(ordered[-1]["x1"]) - float(ordered[0]["x0"]),
            )

            short_token_ratio = sum(
                1
                for length in token_lengths
                if length <= 24
            ) / len(token_lengths)

            # Prefer earlier lines, broader coverage, and sensible token
            # lengths. A caption often has only one x-start, while a body
            # line may have many words but usually has weaker multi-column
            # alignment near the top.
            score = (
                min(
                    1.0,
                    len(anchors) / 8.0,
                )
                * 0.45
                + min(
                    1.0,
                    broadness,
                )
                * 0.30
                + short_token_ratio * 0.20
                + (
                    0.05
                    * (1.0 - (line_index / 8.0))
                )
            )

            header_candidates.append(
                (
                    score,
                    anchors,
                )
            )

        if header_candidates:
            header_candidates.sort(
                key=lambda item: (
                    item[0],
                    len(item[1]),
                ),
                reverse=True,
            )

            best_score, header_anchors = (
                header_candidates[0]
            )

            validated = self._validate_column_anchors(
                lines=lines,
                anchors=header_anchors,
            )

            if validated is not None:
                return validated

            # The header is still useful when validation is unavailable.
            if best_score >= 0.45:
                return header_anchors

        # -------------------------------------------------------------
        # 2. Fallback: repeated line-level x alignment.
        # -------------------------------------------------------------

        line_starts: list[float] = []

        for line in lines:
            if not line:
                continue

            line_starts.append(
                float(
                    min(
                        word["x0"]
                        for word in line
                    )
                )
            )

            # Wrapped table cells often begin each visual line at the
            # same column. Include strong secondary starts from the line.
            sorted_line = sorted(
                line,
                key=lambda item: item["x0"],
            )

            for previous, current in zip(
                sorted_line,
                sorted_line[1:],
            ):
                gap = (
                    float(current["x0"])
                    - float(previous["x1"])
                )

                if gap >= max(
                    8.0,
                    median_height * 1.75,
                ):
                    line_starts.append(
                        float(current["x0"])
                    )

        if not line_starts:
            return []

        clusters: list[list[float]] = [
            [sorted(line_starts)[0]]
        ]

        for value in sorted(line_starts)[1:]:
            if (
                value - clusters[-1][-1]
                <= max(8.0, median_height * 1.5)
            ):
                clusters[-1].append(value)
            else:
                clusters.append(
                    [value]
                )

        anchors = [
            min(cluster)
            for cluster in clusters
            if cluster
        ]

        validated = self._validate_column_anchors(
            lines=lines,
            anchors=anchors,
        )

        if validated is not None:
            return validated

        return [
            anchor
            for anchor in anchors
            if anchor >= 0
        ]

    @staticmethod
    def _cluster_header_starts(
        line: list[dict[str, Any]],
        *,
        gap: float,
    ) -> list[float]:
        """
        Cluster nearby word starts within one visual line.

        This merges headers such as:
            "Dataset used"
            "Key Hyperparameters"
            "Model Architecture"

        while keeping true column starts separated by larger horizontal
        gaps.
        """

        if not line:
            return []

        starts = sorted(
            float(word["x0"])
            for word in line
        )

        clusters: list[list[float]] = [
            [starts[0]]
        ]

        for value in starts[1:]:
            if (
                value - clusters[-1][-1]
                <= gap
            ):
                clusters[-1].append(value)
            else:
                clusters.append(
                    [value]
                )

        return [
            min(cluster)
            for cluster in clusters
        ]

    def _validate_column_anchors(
        self,
        *,
        lines: list[list[dict[str, Any]]],
        anchors: list[float],
    ) -> list[float] | None:
        """
        Validate candidate anchors by measuring whether they produce
        plausible multi-column occupancy across the table.

        This intentionally does not demand every row to fill every column:
        wrapped prose and missing values are normal in research tables.
        """

        if len(anchors) < self.config.min_columns:
            return None

        anchors = sorted(
            float(value)
            for value in anchors
        )

        # Remove anchors that are almost identical.
        cleaned: list[float] = []

        minimum_spacing = max(
            12.0,
            self._median(
                [
                    abs(
                        anchors[i + 1]
                        - anchors[i]
                    )
                    for i in range(
                        len(anchors) - 1
                    )
                ]
            )
            * 0.35
            if len(anchors) > 1
            else 12.0,
        )

        for anchor in anchors:
            if (
                not cleaned
                or anchor - cleaned[-1]
                >= minimum_spacing
            ):
                cleaned.append(anchor)

        if len(cleaned) < self.config.min_columns:
            return None

        # Limit pathological over-segmentation.
        if len(cleaned) > 12:
            return None

        assignments = [
            self._assign_line_to_columns(
                line,
                cleaned,
            )
            for line in lines
        ]

        if not assignments:
            return None

        nonempty_lines = [
            cells
            for cells in assignments
            if any(cells)
        ]

        if not nonempty_lines:
            return None

        occupancy_per_column = []

        for column_index in range(
            len(cleaned)
        ):
            occupied = sum(
                bool(
                    cells[column_index]
                )
                for cells in nonempty_lines
            )

            occupancy_per_column.append(
                occupied / len(nonempty_lines)
            )

        # The first column usually appears in many lines; columns detected
        # from pure intra-cell word gaps tend to have very low support.
        support_floor = max(
            0.10,
            min(
                0.35,
                self.config.row_start_coverage * 0.50,
            ),
        )

        supported = sum(
            support >= support_floor
            for support in occupancy_per_column
        )

        if supported < self.config.min_columns:
            return None

        return cleaned

    @staticmethod
    def _cluster_positions(
        positions: list[float],
        gap: float,
    ) -> list[float]:
        if not positions:
            return []

        clusters: list[list[float]] = [
            [positions[0]]
        ]
        for value in positions[1:]:
            if value - clusters[-1][-1] <= gap:
                clusters[-1].append(value)
            else:
                clusters.append([value])

        # Use the left edge of each cluster. This better represents the
        # start of a column than averaging words within multi-word headers.
        return [
            min(cluster)
            for cluster in clusters
        ]

    def _group_visual_lines(
        self,
        words: list[dict[str, Any]],
    ) -> list[list[dict[str, Any]]]:
        ordered = sorted(
            words,
            key=lambda w: (w["y0"], w["x0"]),
        )
        if not ordered:
            return []

        median_height = max(
            8.0,
            self._median(
                [
                    w["y1"] - w["y0"]
                    for w in ordered
                ]
            ),
        )
        tolerance = max(
            self.config.row_y_tolerance_px,
            median_height * 0.75,
        )

        lines: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        current_y = None

        for word in ordered:
            if current_y is None:
                current = [word]
                current_y = word["y0"]
                continue

            if abs(word["y0"] - current_y) <= tolerance:
                current.append(word)
                current_y = (
                    sum(
                        item["y0"]
                        for item in current
                    ) / len(current)
                )
            else:
                lines.append(
                    sorted(
                        current,
                        key=lambda item: item["x0"],
                    )
                )
                current = [word]
                current_y = word["y0"]

        if current:
            lines.append(
                sorted(
                    current,
                    key=lambda item: item["x0"],
                )
            )

        return lines

    def _group_logical_rows(
        self,
        lines: list[list[dict[str, Any]]],
        columns: list[float],
    ) -> list[list[list[dict[str, Any]]]]:
        assignments = [
            self._assign_line_to_columns(
                line,
                columns,
            )
            for line in lines
        ]

        rows: list[list[list[dict[str, Any]]]] = []
        current: list[list[dict[str, Any]]] = []

        for index, line_cells in enumerate(assignments):
            coverage = (
                sum(bool(cell) for cell in line_cells)
                / len(columns)
            )
            first_cell_text = self._cell_text(
                line_cells[0]
                if line_cells
                else []
            )

            # Strong visual lines start new logical rows. Citation-style
            # tables often wrap the reference number onto its own line;
            # those lines are continuations, not new rows.
            starts_row = (
                index > 0
                and coverage >= self.config.row_start_coverage
                and any(line_cells)
                and not self._is_reference_continuation(
                    first_cell_text
                )
            )

            if index == 0:
                current = [line_cells]
                continue

            if starts_row and current:
                rows.append(current)
                current = []

            current.append(line_cells)

        if current:
            rows.append(current)

        return [
            row
            for row in rows
            if any(
                cell
                for line in row
                for cell in line
            )
        ]

    @staticmethod
    def _is_reference_continuation(
        text: str,
    ) -> bool:
        normalized = " ".join(
            str(text or "").split()
        ).lower()
        if not normalized:
            return False

        return bool(
            re.search(
                r"(?:\[\s*\d+\s*\]|\(\s*\d+\s*\))\s*$",
                normalized,
            )
        )

    @staticmethod
    def _assign_line_to_columns(
        line: list[dict[str, Any]],
        columns: list[float],
    ) -> list[list[dict[str, Any]]]:
        cells: list[list[dict[str, Any]]] = [
            [] for _ in columns
        ]
        if not columns:
            return cells

        for word in line:
            index = 0
            for i in range(1, len(columns)):
                midpoint = (
                    columns[i - 1] + columns[i]
                ) / 2
                if word["x0"] >= midpoint:
                    index = i
                else:
                    break
            cells[index].append(word)

        return cells

    @staticmethod
    def _row_to_cells(
        row: list[list[list[dict[str, Any]]]],
        columns: list[float],
    ) -> list[str]:
        output: list[str] = []
        for column_index in range(len(columns)):
            parts: list[str] = []
            for line in row:
                if column_index >= len(line):
                    continue
                words = sorted(
                    line[column_index],
                    key=lambda item: item["x0"],
                )
                text = " ".join(
                    str(word["text"]).strip()
                    for word in words
                    if str(word["text"]).strip()
                )
                if text:
                    parts.append(text)
            output.append(" ".join(parts).strip())
        return output

    @staticmethod
    def _cell_text(
        cell: list[dict[str, Any]],
    ) -> str:
        return " ".join(
            str(word.get("text", "")).strip()
            for word in cell
            if str(word.get("text", "")).strip()
        )

    def _confidence(
        self,
        *,
        column_count: int,
        occupancy: float,
        method: str,
        ocr_confidence: float | None,
    ) -> float:
        # Do not penalize legitimate wide research tables.
        # A multi-column structure is sufficient for the column component;
        # correctness is validated separately by the reconstruction logic.
        column_score = (
            1.0
            if column_count >= self.config.min_columns
            else 0.0
        )
        occupancy_score = min(
            1.0,
            occupancy / 0.45,
        )

        if method == "ocr_layout":
            ocr_score = min(
                1.0,
                max(
                    0.0,
                    (ocr_confidence or 0.0) / 100.0,
                ),
            )
            return (
                0.35 * column_score
                + 0.35 * occupancy_score
                + 0.30 * ocr_score
            )

        return (
            0.45 * column_score
            + 0.55 * occupancy_score
        )

    @staticmethod
    def _page_number(
        region: dict[str, Any],
    ) -> int | None:
        value = region.get("page")
        if isinstance(value, int):
            return value

        value = region.get("page_no")
        if isinstance(value, int):
            return value

        value = region.get("page_number")
        if isinstance(value, int):
            return value

        return None

    @staticmethod
    def _to_pdf_rect(
        page: Any,
        region: dict[str, Any],
    ) -> Any | None:
        try:
            import fitz
        except ImportError:
            return None

        source = region
        if isinstance(region.get("bbox"), dict):
            source = dict(region)
            source.update(region.get("bbox", {}))

        try:
            x = float(
                source.get(
                    "x",
                    source.get("l", source.get("x0", 0.0)),
                )
            )
            y = float(
                source.get(
                    "y",
                    source.get("b", source.get("y0", 0.0)),
                )
            )
            width = float(
                source.get(
                    "w",
                    source.get(
                        "width",
                        float(source.get("r", 0.0))
                        - x,
                    ),
                )
            )
            height = float(
                source.get(
                    "h",
                    source.get(
                        "height",
                        float(source.get("t", 0.0))
                        - y,
                    ),
                )
            )
        except (TypeError, ValueError):
            return None

        origin = str(
            source.get(
                "coord_origin",
                "BOTTOMLEFT",
            )
        ).upper()

        if width <= 0 or height <= 0:
            return None

        if origin == "BOTTOMLEFT":
            y0 = page.rect.height - (y + height)
            y1 = page.rect.height - y
        else:
            y0 = y
            y1 = y + height

        rect = fitz.Rect(
            x,
            y0,
            x + width,
            y1,
        )
        rect.intersect(page.rect)
        return rect

    def _matrix(self) -> Any:
        import fitz
        return fitz.Matrix(
            self.config.render_scale,
            self.config.render_scale,
        )

    @staticmethod
    def _median(values: list[float]) -> float:
        if not values:
            return 0.0

        ordered = sorted(values)
        middle = len(ordered) // 2

        if len(ordered) % 2:
            return ordered[middle]

        return (
            ordered[middle - 1]
            + ordered[middle]
        ) / 2
