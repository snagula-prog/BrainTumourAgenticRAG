from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import requests


class GrobidError(RuntimeError):
    pass


class GrobidParser:
    def __init__(
        self,
        base_url: str,
        timeout_seconds: int = 120,
        coordinate_elements: list[str] | None = None,
        segment_sentences: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.coordinate_elements = coordinate_elements or [
            "head",
            "p",
            "figure",
            "table",
        ]
        self.segment_sentences = segment_sentences

    def health_check(self) -> bool:
        try:
            response = requests.get(
                f"{self.base_url}/api/isalive",
                timeout=15,
            )
            response.raise_for_status()
            return True
        except requests.RequestException:
            return False

    def process_pdf(
        self,
        file_path: Path,
    ) -> str:
        if not file_path.exists():
            raise GrobidError(
                f"PDF not found: {file_path}"
            )

        try:
            with file_path.open(
                "rb"
            ) as file:
                response = requests.post(
                    f"{self.base_url}/api/processFulltextDocument",
                    files={
                        "input": (
                            file_path.name,
                            file,
                            "application/pdf",
                        )
                    },
                    data={
                        "consolidateHeader": "0",
                        "consolidateCitations": "0",
                        "includeRawCitations": "1",
                        "segmentSentences": (
                            "1"
                            if self.segment_sentences
                            else "0"
                        ),
                        "teiCoordinates": ",".join(
                            self.coordinate_elements
                        ),
                    },
                    timeout=self.timeout_seconds,
                )

            response.raise_for_status()

            if not response.text.strip():
                raise GrobidError(
                    "GROBID returned an empty TEI response."
                )

            return response.text

        except requests.RequestException as exc:
            raise GrobidError(
                f"GROBID request failed: {exc}"
            ) from exc


# =============================================================
# TEI PARSER
# =============================================================

def parse_grobid_tei(
    tei_xml: str,
) -> dict[str, Any]:

    NS = {
        "tei": "http://www.tei-c.org/ns/1.0"
    }

    root = ET.fromstring(
        tei_xml
    )

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------

    def clean_text(
        text: str | None,
    ) -> str:
        if not text:
            return ""

        text = text.replace(
            "\xa0",
            " ",
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    def text_content(
        element: ET.Element | None,
    ) -> str:
        if element is None:
            return ""

        return clean_text(
            "".join(
                element.itertext()
            )
        )

    def paragraph_content(
        element: ET.Element,
    ) -> str:
        parts: list[str] = []

        for node in element.iter():
            if node.tag.endswith("formula"):
                value = text_content(node)
                if value:
                    parts.append(value)
                continue

            if node.text:
                parts.append(node.text)

        return clean_text(
            " ".join(parts)
        )

    def parse_coords(
        element: ET.Element,
    ) -> list[dict[str, float | int]]:
        coords: list[
            dict[str, float | int]
        ] = []

        for coord in element.findall(
            ".//tei:coordinates",
            NS,
        ):
            try:
                page = int(
                    coord.attrib.get(
                        "page",
                        "0",
                    )
                )

                x = float(
                    coord.attrib.get(
                        "x",
                        "0",
                    )
                )

                y = float(
                    coord.attrib.get(
                        "y",
                        "0",
                    )
                )

                w = float(
                    coord.attrib.get(
                        "width",
                        "0",
                    )
                )

                h = float(
                    coord.attrib.get(
                        "height",
                        "0",
                    )
                )

                coords.append(
                    {
                        "page": page,
                        "x": x,
                        "y": y,
                        "w": w,
                        "h": h,
                    }
                )

            except (
                TypeError,
                ValueError,
            ):
                continue

        return coords

    def split_sentences(
        text: str,
    ) -> list[str]:
        if not text:
            return []

        return [
            item.strip()
            for item in re.split(
                r"(?<=[.!?])\s+",
                text,
            )
            if item.strip()
        ]

    def deduplicate_records(
        records: list[dict[str, Any]],
        *fields: str,
    ) -> list[dict[str, Any]]:

        seen: set[tuple[str, ...]] = set()
        result: list[
            dict[str, Any]
        ] = []

        for record in records:
            key = tuple(
                clean_text(
                    str(
                        record.get(
                            field,
                            "",
                        )
                    )
                ).lower()
                for field in fields
            )

            if key in seen:
                continue

            seen.add(key)
            result.append(record)

        return result

    # ---------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------

    title = ""

    title_el = root.find(
        ".//tei:teiHeader/tei:fileDesc/tei:titleStmt/tei:title",
        NS,
    )

    if title_el is not None:
        title = text_content(
            title_el
        )

    # ---------------------------------------------------------
    # AUTHORS
    # ---------------------------------------------------------

    authors: list[str] = []

    author_elements = root.findall(
        ".//tei:teiHeader/tei:fileDesc/tei:sourceDesc/"
        "tei:biblStruct/tei:analytic/tei:author",
        NS,
    )

    for author_el in author_elements:

        pers_name = author_el.find(
            "./tei:persName",
            NS,
        )

        if pers_name is None:
            continue

        name_parts: list[str] = []

        for forename in pers_name.findall(
            "./tei:forename",
            NS,
        ):
            value = text_content(
                forename
            )

            if value:
                name_parts.append(
                    value
                )

        surname = pers_name.find(
            "./tei:surname",
            NS,
        )

        if surname is not None:
            value = text_content(
                surname
            )

            if value:
                name_parts.append(
                    value
                )

        full_name = clean_text(
            " ".join(
                name_parts
            )
        )

        if full_name:
            authors.append(
                full_name
            )


    # ---------------------------------------------------------
    # AUTHOR DEDUPLICATION
    # ---------------------------------------------------------

    deduplicated_authors: list[str] = []
    seen_authors: set[str] = set()

    for author in authors:

        author = clean_text(
            author
        )

        if not author:
            continue

        key = author.lower()

        if key in seen_authors:
            continue

        seen_authors.add(
            key
        )

        deduplicated_authors.append(
            author
        )

    authors = deduplicated_authors
    
    # ---------------------------------------------------------
    # DOI
    # ---------------------------------------------------------

    doi: str | None = None

    doi_elements = root.findall(
        ".//tei:teiHeader/tei:fileDesc/tei:sourceDesc/"
        "tei:biblStruct/tei:idno",
        NS,
    )

    for idno in doi_elements:

        id_type = (
            idno.attrib.get(
                "type",
                ""
            )
            .strip()
            .lower()
        )

        value = clean_text(
            text_content(
                idno
            )
        )

        if (
            id_type == "doi"
            and value
        ):
            doi = value
            break

    year: int | None = None

    date_candidates = root.findall(
        ".//tei:teiHeader//tei:date",
        NS,
    )

    for date in date_candidates:
        value = (
            date.attrib.get(
                "when"
            )
            or text_content(date)
        )

        match = re.search(
            r"(19|20)\d{2}",
            value,
        )

        if match:
            year = int(
                match.group(0)
            )
            break

    # ---------------------------------------------------------
    # ABSTRACT
    # ---------------------------------------------------------

    abstract = ""

    abstract_elements = root.findall(
        ".//tei:teiHeader/tei:profileDesc/tei:abstract",
        NS,
    )

    for abstract_el in abstract_elements:

        parts: list[str] = []

        paragraphs_in_abstract = abstract_el.findall(
            ".//tei:p",
            NS,
        )

        if paragraphs_in_abstract:

            for paragraph in paragraphs_in_abstract:

                value = paragraph_content(
                    paragraph
                )

                if value:
                    parts.append(
                        value
                    )

        else:

            value = text_content(
                abstract_el
            )

            if value:
                parts.append(
                    value
                )

        if parts:
            abstract = clean_text(
                " ".join(parts)
            )

            break

    # ---------------------------------------------------------
    # Sections / Paragraphs
    # ---------------------------------------------------------

    sections: list[dict[str, Any]] = []
    paragraphs: list[dict[str, Any]] = []

    body = root.find(
        ".//tei:text/tei:body",
        NS,
    )

    def walk_div(
        div: ET.Element,
        parent_path: list[str],
        level: int,
    ) -> None:

        heading_el = div.find(
            "./tei:head",
            NS,
        )

        heading = ""

        if heading_el is not None:
            heading = text_content(
                heading_el
            )

        current_path = list(
            parent_path
        )

        if heading:
            current_path.append(
                heading
            )

            sections.append(
                {
                    "heading": heading,
                    "level": level,
                    "path": current_path.copy(),
                    "coords": parse_coords(
                        heading_el
                    ),
                }
            )

        current_section = (
            heading
            if heading
            else (
                parent_path[-1]
                if parent_path
                else None
            )
        )

        # Direct paragraphs only. Nested divs are processed separately.
        for paragraph in div.findall(
            "./tei:p",
            NS,
        ):
            text = paragraph_content(
                paragraph
            )

            if not text:
                continue

            paragraphs.append(
                {
                    "text": text,
                    "sentences": split_sentences(
                        text
                    ),
                    "section": current_section,
                    "section_path": current_path.copy(),
                    "coords": parse_coords(
                        paragraph
                    ),
                    "page": (
                        parse_coords(
                            paragraph
                        )[0]["page"]
                        if parse_coords(
                            paragraph
                        )
                        else None
                    ),
                }
            )

        for child_div in div.findall(
            "./tei:div",
            NS,
        ):
            walk_div(
                child_div,
                current_path,
                level + 1,
            )

    if body is not None:

        # Direct body paragraphs without a section.
        for paragraph in body.findall(
            "./tei:p",
            NS,
        ):
            text = paragraph_content(
                paragraph
            )

            if not text:
                continue

            coords = parse_coords(
                paragraph
            )

            paragraphs.append(
                {
                    "text": text,
                    "sentences": split_sentences(
                        text
                    ),
                    "section": None,
                    "section_path": [],
                    "coords": coords,
                    "page": (
                        coords[0]["page"]
                        if coords
                        else None
                    ),
                }
            )

        for div in body.findall(
            "./tei:div",
            NS,
        ):
            walk_div(
                div,
                [],
                1,
            )

    sections = deduplicate_records(
        sections,
        "heading",
        "path",
    )

    paragraphs = deduplicate_records(
        paragraphs,
        "text",
        "page",
        "section",
    )

    # ---------------------------------------------------------
    # Figures / Tables
    # ---------------------------------------------------------

    figures: list[
        dict[str, Any]
    ] = []

    tables: list[
        dict[str, Any]
    ] = []

    for figure in root.findall(
        ".//tei:figure",
        NS,
    ):
        figure_type = figure.attrib.get(
            "type",
            "",
        )

        label = text_content(
            figure.find(
                "./tei:label",
                NS,
            )
        )

        caption = text_content(
            figure.find(
                "./tei:figDesc",
                NS,
            )
        )

        coords = parse_coords(
            figure
        )

        record = {
            "type": figure_type or "figure",
            "label": label,
            "caption": caption,
            "section_path": [],
            "coords": coords,
            "page": (
                coords[0]["page"]
                if coords
                else None
            ),
        }

        if figure_type.lower() == "table":
            tables.append(
                record
            )
        else:
            figures.append(
                record
            )

    figures = deduplicate_records(
        figures,
        "type",
        "label",
        "caption",
        "page",
    )

    tables = deduplicate_records(
        tables,
        "type",
        "label",
        "caption",
        "page",
    )

    # ---------------------------------------------------------
    # References
    # ---------------------------------------------------------

    references: list[
        dict[str, Any]
    ] = []

    for bibl in root.findall(
        ".//tei:listBibl/tei:biblStruct",
        NS,
    ):
        title_el = bibl.find(
            ".//tei:analytic/tei:title",
            NS,
        )

        if title_el is None:
            title_el = bibl.find(
                ".//tei:monogr/tei:title",
                NS,
            )

        ref_title = (
            text_content(title_el)
            if title_el is not None
            else ""
        )

        ref_authors: list[str] = []

        for author in bibl.findall(
            ".//tei:author",
            NS,
        ):
            value = text_content(
                author
            )

            if value:
                ref_authors.append(
                    value
                )

        raw = text_content(
            bibl
        )

        references.append(
            {
                "title": ref_title,
                "authors": ref_authors,
                "raw": raw,
            }
        )

    references = deduplicate_records(
        references,
        "title",
        "raw",
    )

    # ---------------------------------------------------------
    # Return
    # ---------------------------------------------------------

    return {
        "source": "grobid",
        "title": title,
        "authors": authors,
        "year": year,
        "doi": doi,
        "abstract": abstract,
        "sections": sections,
        "paragraphs": paragraphs,
        "figures": figures,
        "tables": tables,
        "references": references,
    }