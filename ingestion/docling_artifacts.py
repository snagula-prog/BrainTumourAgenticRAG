from __future__ import annotations

from typing import Any


class DoclingArtifactExtractor:

    def extract(
        self,
        document_dict: dict[str, Any],
    ) -> dict[str, Any]:

        figures: list[dict[str, Any]] = []
        tables: list[dict[str, Any]] = []
        formulas: list[dict[str, Any]] = []
        text_blocks: list[dict[str, Any]] = []

        # -----------------------------------------------------
        # TEXT BLOCKS
        # -----------------------------------------------------

        for item in document_dict.get(
            "texts",
            [],
        ):

            text = str(
                item.get(
                    "text",
                    ""
                )
            ).strip()

            if not text:
                continue

            label = str(
                item.get(
                    "label",
                    ""
                )
            ).strip()

            provenance = []

            for prov in item.get(
                "prov",
                [],
            ):

                bbox = prov.get(
                    "bbox"
                )

                if not bbox:
                    continue

                page = prov.get(
                    "page_no"
                )

                if page is None:
                    continue

                provenance.append(
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

            text_blocks.append(
                {
                    "text": text,
                    "label": label,
                    "coords": provenance,
                }
            )

        # -----------------------------------------------------
        # PICTURES
        # -----------------------------------------------------

        for item in document_dict.get(
            "pictures",
            [],
        ):

            caption = ""

            for child in item.get(
                "children",
                [],
            ):
                if isinstance(
                    child,
                    dict,
                ):
                    caption = str(
                        child.get(
                            "text",
                            ""
                        )
                    ).strip()

                    if caption:
                        break

            coords = []

            for prov in item.get(
                "prov",
                [],
            ):

                bbox = prov.get(
                    "bbox"
                )

                if not bbox:
                    continue

                page = prov.get(
                    "page_no"
                )

                if page is None:
                    continue

                coords.append(
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

            figures.append(
                {
                    "label": "figure",
                    "caption": caption,
                    "coords": coords,
                }
            )

        # -----------------------------------------------------
        # TABLES
        # -----------------------------------------------------

        for item in document_dict.get(
            "tables",
            [],
        ):

            coords = []

            for prov in item.get(
                "prov",
                [],
            ):

                bbox = prov.get(
                    "bbox"
                )

                if not bbox:
                    continue

                page = prov.get(
                    "page_no"
                )

                if page is None:
                    continue

                coords.append(
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

            tables.append(
                {
                    "label": "table",
                    "coords": coords,
                }
            )

        # -----------------------------------------------------
        # FORMULAS
        # -----------------------------------------------------

        for item in document_dict.get(
            "texts",
            [],
        ):

            label = str(
                item.get(
                    "label",
                    ""
                )
            ).lower()

            if "formula" not in label:
                continue

            text = str(
                item.get(
                    "text",
                    ""
                )
            ).strip()

            if not text:
                continue

            formulas.append(
                {
                    "text": text,
                }
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
                "text_blocks": len(
                    text_blocks
                ),
            },
        }