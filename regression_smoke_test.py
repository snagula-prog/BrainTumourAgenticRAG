from pathlib import Path
import json

from evaluation.extraction_metrics.extraction_benchmark import ExtractionBenchmark
from extraction.grobid_parser import parse_grobid_tei
from extraction.docling_artifacts import DoclingArtifactExtractor
from extraction.artifact_reconstruction import LogicalArtifactReconstructor
from extraction.canonical_builder import CanonicalBuilder


# =========================================================
# 1. Ground-truth evaluator regression
# =========================================================

benchmark = ExtractionBenchmark()

canonical = {
    "metadata": {
        "title": "Test",
        "authors": ["A Author"],
        "year": 2024,
        "doi": None,
    },
    "abstract": "A sufficiently long abstract for the benchmark schema regression test.",
    "sections": [{"heading": "Introduction"}],
    "paragraphs": [{"text": "A paragraph for testing."}],
    "formulas": [],
    "references": [],
    "tables": [],
    "figures": [],
}

ground_truth = {
    "metadata": {
        "title": "Test",
        "authors": ["A Author"],
        "year": 2024,
        "doi": None,
        "abstract": "A sufficiently long abstract for the benchmark schema regression test.",
    },
    "sections": ["Introduction"],
    "paragraphs": ["A paragraph for testing."],
    "formulas": [],
    "references": [],
    "table_count": 0,
    "figure_count": 0,
}

result = benchmark.evaluate(
    canonical=canonical,
    ground_truth=ground_truth,
)

assert result["accuracy_score"] == 1.0, result


# =========================================================
# 2. Locate local parser outputs
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

candidate_dirs = [
    PROJECT_ROOT,
    PROJECT_ROOT / "storage",
    PROJECT_ROOT / "storage" / "parsed",
    PROJECT_ROOT / "storage" / "cache",
    PROJECT_ROOT / "storage" / "grobid",
    PROJECT_ROOT / "storage" / "docling",
]

def find_file(filename: str) -> Path | None:
    for directory in candidate_dirs:
        candidate = directory / filename
        if candidate.exists():
            return candidate

    # Last-resort recursive search inside project
    matches = list(PROJECT_ROOT.rglob(filename))
    return matches[0] if matches else None


# =========================================================
# 3. Real parser-output regression checks
# =========================================================

papers = {
    8: "efficientnet.pdf",
    9: "lightweightnn.pdf",
    10: "segmentation.pdf",
    12: "verydeeplearning.pdf",
}

for paper_number, pdf_name in papers.items():

    tei_path = find_file(f"paper_{paper_number:03d}.tei.xml")
    docling_path = find_file(f"paper_{paper_number:03d}_docling.json")

    if tei_path is None:
        raise FileNotFoundError(
            f"Could not locate paper_{paper_number:03d}.tei.xml "
            f"inside {PROJECT_ROOT}"
        )

    if docling_path is None:
        raise FileNotFoundError(
            f"Could not locate paper_{paper_number:03d}_docling.json "
            f"inside {PROJECT_ROOT}"
        )

    grobid = parse_grobid_tei(
        tei_path.read_text(encoding="utf-8")
    )

    docling_json = json.loads(
        docling_path.read_text(encoding="utf-8")
    )

    document = docling_json["docling_document"]

    artifacts = DoclingArtifactExtractor().extract(
        document
    )

    reconstruction = LogicalArtifactReconstructor().reconstruct(
        grobid=grobid,
        artifacts=artifacts,
    )

    print(
        f"paper_{paper_number:03d}: "
        f"{len(reconstruction['figures'])} figures, "
        f"{len(reconstruction['tables'])} tables"
    )

    if paper_number == 8:
        assert len(reconstruction["figures"]) == 8, (
            f"paper_008 figures: "
            f"{len(reconstruction['figures'])}"
        )
        assert len(reconstruction["tables"]) == 6, (
            f"paper_008 tables: "
            f"{len(reconstruction['tables'])}"
        )

    elif paper_number == 10:
        assert len(reconstruction["figures"]) == 4, (
            f"paper_010 figures: "
            f"{len(reconstruction['figures'])}"
        )

        abstract = CanonicalBuilder._extract_abstract(
            grobid=grobid,
            pymupdf={},
        )

        assert len(abstract) >= 800, (
            f"paper_010 abstract too short: {len(abstract)}"
        )

    elif paper_number == 9:
        assert len(reconstruction["figures"]) == 16, (
            f"paper_009 figures: "
            f"{len(reconstruction['figures'])}"
        )

        numbered_tables = sum(
            1
            for table in reconstruction["tables"]
            if table.get("label")
        )

        assert numbered_tables == 4, (
            f"paper_009 numbered tables: {numbered_tables}"
        )

    elif paper_number == 12:
        assert len(reconstruction["tables"]) == 12, (
            f"paper_012 tables: "
            f"{len(reconstruction['tables'])}"
        )


print()
print("REGRESSION_SMOKE_TEST_OK")
