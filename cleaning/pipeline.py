# cleaning/pipeline.py

from __future__ import annotations

import json
from pathlib import Path

from config.settings import settings
from cleaning.pdf_text_extractor import (
    PDFTextExtractor,
)


def clean_paper(
    *,
    parsed_path: Path,
    output_path: Path,
) -> dict:

    with parsed_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        parsed = json.load(file)

    paper_id = parsed[
        "paper_id"
    ]

    pdf_filename = parsed[
        "filename"
    ]

    pdf_path = (
        Path(settings.papers_dir)
        / pdf_filename
    )

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF for {paper_id} not found: "
            f"{pdf_path}"
        )

    extractor = PDFTextExtractor(
        pdf_path
    )

    canonical = extractor.extract()

    output = {
        "paper_id": paper_id,
        "filename": pdf_filename,

        "source_parser": (
            "pymupdf_canonical"
        ),

        "title": parsed.get(
            "title"
        ),

        "authors": parsed.get(
            "authors",
            [],
        ),

        "year": parsed.get(
            "year"
        ),

        "num_pages": canonical[
            "num_pages"
        ],

        "canonical_text": canonical[
            "text"
        ],

        "pages": canonical[
            "pages"
        ],

        # Keep Docling structure untouched.
        "docling_document": parsed[
            "docling_document"
        ],

        "docling_metadata": parsed.get(
            "docling_metadata",
            {},
        ),
    }

    # ---------------------------------------------------------
    # METADATA FALLBACKS
    # ---------------------------------------------------------

    pdf_metadata = parsed.get(
        "docling_metadata",
        {},
    )

    if not output["authors"]:

        output["authors"] = (
            _parse_authors(
                pdf_metadata.get(
                    "pdf_author"
                )
            )
        )

    if output["year"] is None:

        output["year"] = (
            _extract_year(
                canonical["text"]
            )
        )

    # ---------------------------------------------------------
    # TITLE
    # ---------------------------------------------------------

    if not output["title"]:

        output["title"] = (
            _extract_title(
                canonical
            )
        )

    # ---------------------------------------------------------
    # ABSTRACT
    # ---------------------------------------------------------

    output["abstract"] = (
        _extract_abstract(
            canonical
        )
    )

    # ---------------------------------------------------------
    # REFERENCES
    # ---------------------------------------------------------

    output["references"] = (
        _extract_references(
            canonical["text"]
        )
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"[CLEANED] {paper_id}"
    )

    return output


def clean_all_papers() -> None:

    parsed_dir = Path(
        settings.parsed_dir
    )

    cleaned_dir = Path(
        settings.cleaned_dir
    )

    cleaned_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = sorted(
        parsed_dir.glob(
            "paper_*.json"
        )
    )

    for parsed_path in files:

        try:

            with parsed_path.open(
                "r",
                encoding="utf-8",
            ) as file:

                parsed = json.load(file)

            paper_id = parsed[
                "paper_id"
            ]

            output_path = (
                cleaned_dir
                / f"{paper_id}_clean.json"
            )

            clean_paper(
                parsed_path=parsed_path,
                output_path=output_path,
            )

        except Exception as exc:

            print(
                f"[FAILED] "
                f"{parsed_path.name}: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )


def _parse_authors(
    value,
) -> list[str]:

    if not value:
        return []

    text = str(
        value
    ).strip()

    # Scientific Reports style:
    # Author1 & Author2
    parts = []

    if " & " in text:

        parts = text.split(
            " & "
        )

    elif " and " in text.lower():

        parts = text.split(
            " and "
        )

    else:

        parts = [
            text
        ]

    authors = []

    for part in parts:

        part = part.strip()

        part = __import__(
            "re"
        ).sub(
            r"\d+[\*\u2020\u2021]*",
            "",
            part,
        )

        part = part.strip(
            " ,.*"
        )

        if part:
            authors.append(
                part
            )

    return authors


def _extract_title(
    canonical: dict,
) -> str | None:

    for block in canonical[
        "pages"
    ][0]["blocks"]:

        text = block["text"].strip()

        if (
            20 <= len(text) <= 250
            and not _looks_like_metadata(
                text
            )
        ):
            return text

    return None


def _extract_abstract(
    canonical: dict,
) -> str:

    page = canonical[
        "pages"
    ][0]

    blocks = page[
        "blocks"
    ]

    title_seen = False
    abstract_parts = []

    for block in blocks:

        text = block["text"].strip()

        if not text:
            continue

        if not title_seen:

            if _is_title_candidate(
                text
            ):

                title_seen = True

            continue

        if _looks_like_author_block(
            text
        ):
            continue

        if _looks_like_affiliation(
            text
        ):
            continue

        # Keywords terminate the abstract.
        if text.lower().startswith(
            "keywords"
        ):
            break

        # Section heading means abstract is over.
        if __import__(
            "re"
        ).match(
            r"^\d+(?:\.\d+)*\.?\s+",
            text,
        ):
            break

        if len(text) >= 60:

            abstract_parts.append(
                text
            )

    return " ".join(
        abstract_parts
    ).strip()


def _extract_references(
    text: str,
) -> list[str]:

    import re

    lines = text.splitlines()

    references = []

    in_references = False
    current = ""

    for line in lines:

        line = line.strip()

        if not line:
            continue

        if re.fullmatch(
            r"References",
            line,
            re.IGNORECASE,
        ):

            in_references = True
            continue

        if not in_references:
            continue

        # Stop at publisher disclaimer.
        if (
            "disclaimer" in line.lower()
            or "publisher's note" in line.lower()
        ):
            break

        match = re.match(
            r"^(\d+)[.)]\s+(.*)$",
            line,
        )

        if match:

            if current:
                references.append(
                    current
                )

            current = (
                f"{match.group(1)}. "
                f"{match.group(2)}"
            )

        else:

            if current:
                current += " " + line

    if current:
        references.append(
            current
        )

    return references


def _extract_year(
    text: str,
) -> int | None:

    import re

    match = re.search(
        r"\b(?:Published|Accepted|Received)"
        r"[^0-9]{0,30}"
        r"((?:19|20)\d{2})",
        text,
        re.IGNORECASE,
    )

    if match:
        return int(
            match.group(1)
        )

    return None


def _is_title_candidate(
    text: str,
) -> bool:

    return (
        20 <= len(text) <= 250
        and not _looks_like_metadata(
            text
        )
    )


def _looks_like_metadata(
    text: str,
) -> bool:

    lower = text.lower()

    patterns = (
        "scientific reports",
        "www.nature.com",
        "doi.org",
        "open access",
        "eng. proc.",
    )

    return any(
        p in lower
        for p in patterns
    )


def _looks_like_author_block(
    text: str,
) -> bool:

    if len(text) > 300:
        return False

    return bool(
        __import__(
            "re"
        ).search(
            r"[A-Z][a-z]+\d",
            text,
        )
    )


def _looks_like_affiliation(
    text: str,
) -> bool:

    lower = text.lower()

    return any(
        word in lower
        for word in (
            "department",
            "university",
            "school of",
            "faculty",
            "institute",
            "college",
            "email",
            "@",
        )
    )


if __name__ == "__main__":
    clean_all_papers()