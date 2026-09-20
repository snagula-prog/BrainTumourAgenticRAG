import unittest

from extraction.artifact_reconstruction import LogicalArtifactReconstructor
from evaluation.canonical_artifact_integrity import CanonicalArtifactIntegrity


class ArtifactReconstructionTests(unittest.TestCase):
    def setUp(self):
        self.reconstructor = LogicalArtifactReconstructor()

    @staticmethod
    def coord(page, x, y, w, h):
        return {
            "page": page,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "coord_origin": "BOTTOMLEFT",
        }

    def test_continued_table_is_one_logical_table(self):
        grobid = {
            "tables": [
                {
                    "type": "table",
                    "label": "Table 1",
                    "caption": "Table 1 Model performance.",
                    "coords": [self.coord(3, 50, 200, 500, 250)],
                },
                {
                    "type": "table",
                    "label": "Table 1",
                    "caption": "Table 1 (Continued.)",
                    "coords": [self.coord(4, 50, 180, 500, 300)],
                },
            ]
        }
        result = self.reconstructor.reconstruct(
            grobid=grobid,
            artifacts={"tables": [], "figures": [], "text_blocks": []},
        )
        self.assertEqual(len(result["tables"]), 1)
        table = result["tables"][0]
        self.assertEqual(table["label"], "Table 1")
        self.assertEqual(table["pages"], [3, 4])
        self.assertEqual(table["continuation_status"], "continued")
        self.assertEqual(len(table["source_refs"]), 2)

    def test_unstructured_table_does_not_invent_cells(self):
        result = self.reconstructor.reconstruct(
            grobid={},
            artifacts={
                "tables": [
                    {
                        "label": "Table 2",
                        "content": "Model Accuracy\nCNN 94.2",
                        "coords": [self.coord(7, 50, 100, 500, 180)],
                    }
                ],
                "figures": [],
                "text_blocks": [],
            },
        )
        table = result["tables"][0]
        self.assertEqual(table["structure_status"], "text_only")
        self.assertIsNone(table["structured_data"])
        self.assertEqual(table["content"], "Model Accuracy CNN 94.2")

    def test_nearby_figure_caption_is_associated(self):
        result = self.reconstructor.reconstruct(
            grobid={},
            artifacts={
                "tables": [],
                "figures": [
                    {
                        "label": "",
                        "caption": "",
                        "coords": [self.coord(5, 80, 220, 460, 220)],
                    }
                ],
                "text_blocks": [
                    {
                        "label": "caption",
                        "text": "Figure 3 Proposed architecture.",
                        "coords": [self.coord(5, 80, 150, 460, 40)],
                    }
                ],
            },
        )
        self.assertEqual(len(result["figures"]), 1)
        figure = result["figures"][0]
        self.assertEqual(figure["label"], "Figure 3")
        self.assertEqual(figure["caption_status"], "available")

    def test_weak_visual_is_preserved_but_not_called_a_figure(self):
        result = self.reconstructor.reconstruct(
            grobid={},
            artifacts={
                "tables": [],
                "figures": [
                    {
                        "label": "",
                        "caption": "",
                        "coords": [self.coord(1, 10, 10, 20, 20)],
                    }
                ],
                "text_blocks": [],
            },
        )
        self.assertEqual(len(result["figures"]), 0)
        self.assertEqual(len(result["unclassified_visuals"]), 1)


    def test_empty_docling_data_is_not_structured(self):
        result = self.reconstructor.reconstruct(
            grobid={},
            artifacts={
                "tables": [
                    {
                        "label": "table",
                        "caption": "TABLE 2. Example.",
                        "coords": [self.coord(6, 50, 100, 500, 180)],
                        "structured_data": {
                            "num_rows": 0,
                            "num_cols": 0,
                            "grid": [],
                            "table_cells": [],
                        },
                    }
                ],
                "figures": [],
                "text_blocks": [],
            },
        )
        table = result["tables"][0]
        self.assertEqual(table["label"], "Table 2")
        self.assertEqual(table["structure_status"], "text_only")
        self.assertIsNone(table["structured_data"])

    def test_unlabeled_region_bridges_continued_table(self):
        result = self.reconstructor.reconstruct(
            grobid={},
            artifacts={
                "tables": [
                    {
                        "label": "table",
                        "caption": "TABLE 4. Gap analysis.",
                        "coords": [self.coord(8, 40, 100, 500, 500)],
                    },
                    {
                        "label": "table",
                        "caption": "",
                        "coords": [self.coord(9, 40, 100, 500, 500)],
                    },
                    {
                        "label": "table",
                        "caption": "TABLE 4. (Continued.) Gap analysis.",
                        "coords": [self.coord(10, 40, 100, 500, 250)],
                    },
                ],
                "figures": [],
                "text_blocks": [],
            },
        )
        self.assertEqual(len(result["tables"]), 1)
        table = result["tables"][0]
        self.assertEqual(table["label"], "Table 4")
        self.assertEqual(table["pages"], [8, 9, 10])
        self.assertEqual(table["continuation_status"], "continued")
        self.assertEqual(table["source_region_count"], 3)

    def test_logical_figure_ids_are_sequential(self):
        result = self.reconstructor.reconstruct(
            grobid={},
            artifacts={
                "tables": [],
                "figures": [
                    {"label": "picture", "caption": "FIGURE 7. First.", "coords": [self.coord(11, 40, 200, 200, 150)]},
                    {"label": "picture", "caption": "FIGURE 9. Second.", "coords": [self.coord(11, 300, 200, 200, 150)]},
                ],
                "text_blocks": [],
            },
        )
        self.assertEqual(
            [f["figure_id"] for f in result["figures"]],
            ["figure_001", "figure_002"],
        )

    def test_integrity_evaluator_is_not_accuracy_evaluator(self):
        canonical = {
            "tables": [
                {
                    "table_id": "table_001",
                    "structure_status": "text_only",
                    "regions": [{"page": 2}],
                    "source_refs": [{"parser": "docling", "kind": "table", "index": 0}],
                }
            ],
            "figures": [
                {
                    "figure_id": "figure_001",
                    "caption_status": "missing",
                    "regions": [{"page": 3}],
                    "source_refs": [{"parser": "grobid", "kind": "figure", "index": 0}],
                }
            ],
            "formulas": [
                {
                    "formula_id": "formula_001",
                    "source": "docling",
                }
            ],
        }
        result = CanonicalArtifactIntegrity().evaluate(canonical)
        self.assertEqual(result["evaluation_type"], "canonical_artifact_integrity")
        self.assertNotIn("accuracy", result["evaluation_type"])
        self.assertIn("integrity_score", result)


if __name__ == "__main__":
    unittest.main()
