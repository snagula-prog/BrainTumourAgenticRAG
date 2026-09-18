from __future__ import annotations

from typing import Any


class ArtifactQualityEvaluator:
    """
    Evaluate artifact integrity on the FINAL canonical document.

    This evaluator measures preservation/structure/provenance health.
    It does NOT claim extraction accuracy against ground truth.

    Ground-truth accuracy belongs to extraction_benchmark.py.
    """

    def evaluate(
        self,
        canonical: dict[str, Any],
    ) -> dict[str, Any]:

        tables = self._records(
            canonical.get("tables")
        )
        figures = self._records(
            canonical.get("figures")
        )
        formulas = self._records(
            canonical.get("formulas")
        )

        structured_tables = sum(
            1
            for table in tables
            if table.get(
                "structure_status"
            ) == "structured"
            and self._has_table_structure(table)
        )

        captioned_figures = sum(
            1
            for figure in figures
            if (
                figure.get(
                    "caption_status"
                ) == "available"
                or self._clean(
                    figure.get(
                        "caption"
                    )
                )
            )
        )

        formula_provenance = sum(
            1
            for formula in formulas
            if self._has_provenance(
                formula
            )
        )

        table_ratio = (
            structured_tables / len(tables)
            if tables
            else 1.0
        )

        caption_ratio = (
            captioned_figures / len(figures)
            if figures
            else 1.0
        )

        formula_ratio = (
            formula_provenance / len(formulas)
            if formulas
            else 1.0
        )

        # Provenance coverage is useful independently of captions.
        figure_provenance = sum(
            1
            for figure in figures
            if self._has_provenance(
                figure
            )
        )

        figure_provenance_ratio = (
            figure_provenance / len(figures)
            if figures
            else 1.0
        )

        table_provenance = sum(
            1
            for table in tables
            if self._has_provenance(
                table
            )
        )

        table_provenance_ratio = (
            table_provenance / len(tables)
            if tables
            else 1.0
        )

        issues: list[str] = []

        if tables and table_ratio < 1.0:
            issues.append(
                "table_structure_missing"
            )

        if tables and table_provenance_ratio < 1.0:
            issues.append(
                "table_provenance_missing"
            )

        if figures and figure_provenance_ratio < 1.0:
            issues.append(
                "figure_provenance_missing"
            )

        if figures and caption_ratio < 1.0:
            issues.append(
                "figure_captions_missing"
            )

        if formulas and formula_ratio < 1.0:
            issues.append(
                "formula_provenance_missing"
            )

        # Equal importance is deliberate here: this report is a
        # structural audit, not a model-performance score.
        components = [
            table_ratio,
            table_provenance_ratio,
            figure_provenance_ratio,
            caption_ratio,
            formula_ratio,
        ]

        integrity_score = (
            sum(components)
            / len(components)
            if components
            else 1.0
        )

        status = (
            "pass"
            if not issues
            else "review"
        )

        return {
            "evaluation_type": (
                "canonical_artifact_integrity"
            ),
            "integrity_score": round(
                integrity_score,
                4,
            ),
            "status": status,
            "tables": {
                "count": len(tables),
                "structured_count": (
                    structured_tables
                ),
                "structured_ratio": round(
                    table_ratio,
                    4,
                ),
                "provenance_ratio": round(
                    table_provenance_ratio,
                    4,
                ),
            },
            "figures": {
                "count": len(figures),
                "captioned_count": (
                    captioned_figures
                ),
                "caption_coverage": round(
                    caption_ratio,
                    4,
                ),
                "provenance_ratio": round(
                    figure_provenance_ratio,
                    4,
                ),
            },
            "formulas": {
                "count": len(formulas),
                "provenance_count": (
                    formula_provenance
                ),
                "provenance_coverage": round(
                    formula_ratio,
                    4,
                ),
            },
            "issues": issues,
        }

    # =========================================================
    # HELPERS
    # =========================================================

    @staticmethod
    def _records(
        value: Any,
    ) -> list[dict[str, Any]]:

        if not isinstance(
            value,
            list,
        ):
            return []

        return [
            item
            for item in value
            if isinstance(
                item,
                dict,
            )
        ]

    @classmethod
    def _has_table_structure(
        cls,
        table: dict[str, Any],
    ) -> bool:

        rows = table.get(
            "rows"
        )
        cells = table.get(
            "cells"
        )

        return bool(
            (
                isinstance(
                    rows,
                    list,
                )
                and rows
            )
            or (
                isinstance(
                    cells,
                    list,
                )
                and cells
            )
            or cls._clean(
                table.get(
                    "content"
                )
            )
        )

    @classmethod
    def _has_provenance(
        cls,
        item: dict[str, Any],
    ) -> bool:

        page = item.get(
            "page"
        )

        if page is not None:
            return True

        coords = item.get(
            "coords"
        )

        if isinstance(
            coords,
            list,
        ) and any(
            cls._valid_coord(
                coord
            )
            for coord in coords
        ):
            return True

        provenance = item.get(
            "prov"
        )

        if isinstance(
            provenance,
            list,
        ) and any(
            cls._valid_prov(
                record
            )
            for record in provenance
        ):
            return True

        return False

    @staticmethod
    def _valid_coord(
        coord: Any,
    ) -> bool:

        if not isinstance(
            coord,
            dict,
        ):
            return False

        page = coord.get(
            "page",
            coord.get(
                "page_no"
            ),
        )

        if page is None:
            return False

        try:
            x = float(
                coord.get(
                    "x"
                )
            )
            y = float(
                coord.get(
                    "y"
                )
            )
            w = float(
                coord.get(
                    "w",
                    coord.get(
                        "width"
                    ),
                )
            )
            h = float(
                coord.get(
                    "h",
                    coord.get(
                        "height"
                    ),
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            return False

        return (
            x >= 0
            and y >= 0
            and w > 0
            and h > 0
        )

    @classmethod
    def _valid_prov(
        cls,
        record: Any,
    ) -> bool:

        if not isinstance(
            record,
            dict,
        ):
            return False

        page = record.get(
            "page_no",
            record.get(
                "page"
            ),
        )

        bbox = record.get(
            "bbox"
        )

        if page is None or not isinstance(
            bbox,
            dict,
        ):
            return False

        try:
            left = float(
                bbox["l"]
            )
            right = float(
                bbox["r"]
            )
            top = float(
                bbox["t"]
            )
            bottom = float(
                bbox["b"]
            )
        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            return False

        return (
            right > left
            and top != bottom
        )

    @staticmethod
    def _clean(
        value: Any,
    ) -> str:

        return str(
            value or ""
        ).strip()
