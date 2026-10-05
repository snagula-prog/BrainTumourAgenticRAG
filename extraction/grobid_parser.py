from __future__ import annotations

import re
from difflib import SequenceMatcher
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

    def _iter_text_preserving_tails(element: ET.Element):
        """Yield element text in document order, including child tails."""
        if element.text:
            yield element.text
        for child in list(element):
            yield from _iter_text_preserving_tails(child)
            if child.tail:
                yield child.tail

    def paragraph_content(
        element: ET.Element | None,
    ) -> str:
        if element is None:
            return ""

        # Build the paragraph from the full XML stream instead of reconstructing
        # each sentence with a separate itertext() call. This preserves text that
        # occurs after inline nodes such as <ref>, <hi>, and <formula> (their
        # child.tail values) and avoids citation-adjacent truncation.
        return clean_text("".join(_iter_text_preserving_tails(element)))

        
    def parse_tei_table_structure(
        figure_element,
    ) -> dict[str, Any]:
        """
        Extract actual TEI table structure from a GROBID table figure.

        This preserves the cells GROBID actually found. It does not infer
        missing cells or use PDF-specific coordinates/dimensions.
        """
        table_el = figure_element.find(
            "tei:table",
            NS,
        )

        if table_el is None:
            return {}

        rows: list[list[dict[str, Any]]] = []

        for row_el in table_el.findall(
            "tei:row",
            NS,
        ):
            row: list[dict[str, Any]] = []

            for cell_el in row_el.findall(
                "tei:cell",
                NS,
            ):

                def parse_span(*names: str) -> int:
                    for name in names:
                        raw = cell_el.attrib.get(name)

                        if raw is None:
                            continue

                        try:
                            return max(
                                1,
                                int(raw),
                            )
                        except (
                            TypeError,
                            ValueError,
                        ):
                            pass

                    return 1

                row.append(
                    {
                        "text": text_content(
                            cell_el
                        ),
                        "row_span": parse_span(
                            "rows",
                            "rowspan",
                        ),
                        "col_span": parse_span(
                            "cols",
                            "colspan",
                        ),
                        "column_header": (
                            cell_el.attrib.get(
                                "role"
                            )
                            in {
                                "head",
                                "header",
                            }
                        ),
                    }
                )

            if row:
                rows.append(row)

        if not rows:
            return {}

        num_cols = max(
            (
                sum(
                    max(
                        1,
                        int(
                            cell.get(
                                "col_span",
                                1,
                            )
                        ),
                    )
                    for cell in row
                )
                for row in rows
            ),
            default=0,
        )

        return {
            "num_rows": len(rows),
            "num_cols": num_cols,
            "grid": rows,
            "source": "grobid_tei",
        }

    def parse_coords(
        element: ET.Element | None,
    ) -> list[dict[str, float | int]]:
        coords: list[
            dict[str, float | int]
        ] = []

        if element is None:
            return coords

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
                        # GROBID TEI coordinates use a top-left origin.
                        # Preserve this provenance so downstream PDF crop
                        # conversion does not incorrectly assume bottom-left.
                        "coord_origin": "TOPLEFT",
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

        # GROBID may promote list items or grading labels to <head>. Treat
        # standalone list markers/grade labels as paragraph prefixes. If a
        # real heading has an accidental trailing list marker (for example a
        # section title followed by ``1)``), keep the title and remove only the
        # marker. No document-specific heading names are required.
        heading = re.sub(r"\s+\d+[.)]\s*$", "", heading).strip()
        is_false_heading = bool(
            heading
            and (
                re.match(r"^Grade\s+(?:[IVX]+|\d+)\s*: ?$", heading, re.IGNORECASE)
                or re.fullmatch(r"(?:\(?\d+\)?[.)]|[A-Za-z]\.)", heading.strip())
            )
        )
        false_heading_prefix = ""
        if is_false_heading:
            false_heading_prefix = heading.strip()
            heading = ""

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

        # Preserve the TEI child order exactly. GROBID can emit a mixture of
        # paragraphs and nested divs; processing all paragraphs first would
        # reorder the document stream.
        for child in list(div):
            tag = child.tag.rsplit("}", 1)[-1]

            if tag == "head":
                continue

            if tag == "p":
                paragraph = child
                text = paragraph_content(
                    paragraph
                )

                if not text:
                    continue

                if false_heading_prefix and not text.startswith(false_heading_prefix):
                    sep = " " if false_heading_prefix.endswith(":") or false_heading_prefix.endswith(")") else ": "
                    text = f"{false_heading_prefix}{sep}{text}"
                    false_heading_prefix = ""

                s_elems = paragraph.findall(".//tei:s", NS)
                if s_elems:
                    sentences = [clean_text("".join(s.itertext())) for s in s_elems]
                    sentences = [st for st in sentences if st]
                else:
                    sentences = split_sentences(
                        text
                    )

                coords = parse_coords(
                    paragraph
                )

                paragraphs.append(
                    {
                        "text": text,
                        "sentences": sentences,
                        "section": current_section,
                        "section_path": current_path.copy(),
                        "coords": coords,
                        "page": (
                            coords[0]["page"]
                            if coords
                            else None
                        ),
                    }
                )

            elif tag == "div":
                walk_div(
                    child,
                    current_path,
                    level + 1,
                )

    if body is not None:

        # Preserve exact body child order for top-level p/div elements.
        for child in list(body):
            tag = child.tag.rsplit("}", 1)[-1]

            if tag == "p":
                paragraph = child
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

            elif tag == "div":
                walk_div(
                    child,
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

    def _infer_section_for_recovered_text(text: str) -> tuple[str | None, list[str]]:
        """Infer a section using evidence already present in the document."""
        target = clean_text(text).lower()
        if not target:
            return None, []

        target_tokens = set(re.findall(r"[a-z0-9]+", target))
        target_refs = set(re.findall(r"\[(\d+)\]", target))

        structural_heading_terms = {
            "introduction", "review", "literature", "techniques", "methods",
            "method", "models", "model", "approaches", "approach", "analysis",
            "findings", "parameters", "limitations", "future", "research",
            "directions", "conclusion", "conclusions", "proposed", "work",
        }

        profiles: dict[str, dict[str, Any]] = {}
        for section in sections:
            heading = section.get("heading") or ""
            if not heading:
                continue
            heading_tokens = set(re.findall(r"[a-z0-9]+", heading.lower()))
            heading_terms = {
                token
                for token in heading_tokens
                if token not in structural_heading_terms
                and not token.isdigit()
                and len(token) > 1
            }
            profiles.setdefault(
                heading,
                {"body_tokens": set(), "refs": set(), "heading_terms": heading_terms},
            )

        for paragraph in paragraphs:
            sec = paragraph.get("section")
            if not sec or sec not in profiles:
                continue
            ptext = clean_text(paragraph.get("text"))
            profiles[sec]["body_tokens"].update(re.findall(r"[a-z0-9]+", ptext.lower()))
            profiles[sec]["refs"].update(re.findall(r"\[(\d+)\]", ptext))

        best: tuple[float, str | None, list[str]] = (0.0, None, [])
        for section_name, profile in profiles.items():
            body_tokens = profile["body_tokens"]
            shared_body = target_tokens & body_tokens
            body_score = len(shared_body) / max(len(target_tokens), 1)

            ref_score = 0.0
            if target_refs and profile["refs"]:
                ref_score = len(target_refs & profile["refs"]) / len(target_refs)

            heading_terms = profile["heading_terms"]
            heading_score = (
                len(target_tokens & heading_terms) / len(heading_terms)
                if heading_terms else 0.0
            )

            # Distinctive heading terms are strongest, references are next, and
            # broad body vocabulary is supporting evidence only.
            score = max(
                2.0 * heading_score,
                1.35 * ref_score + 0.35 * body_score,
                0.55 * body_score,
            )

            if score > best[0]:
                path = next(
                    (s.get("path", [section_name]) for s in sections if s.get("heading") == section_name),
                    [section_name],
                )
                best = (score, section_name, list(path) if isinstance(path, list) else [])

        if best[0] < 0.20:
            return None, []
        return best[1], best[2]

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

        head_el = figure.find(
            "./tei:head",
            NS,
        )

        is_figure = bool(
            label
            or (head_el is not None and re.match(r"^figure\b", text_content(head_el), re.IGNORECASE))
            or (caption and re.match(r"^figure\b", caption, re.IGNORECASE))
        )

        if not is_figure and figure_type.lower() != "table":
            # Recover misclassified body paragraphs from figure elements (e.g. fig_0, fig_4)
            desc_div = figure.find("./tei:figDesc", NS)
            target_el = desc_div if desc_div is not None else figure
            prose_paragraphs = target_el.findall(".//tei:p", NS)
            if not prose_paragraphs:
                prose_paragraphs = [target_el]

            for p_node in prose_paragraphs:
                ptxt = paragraph_content(p_node)
                if not ptxt:
                    continue

                target_sec, sec_path = _infer_section_for_recovered_text(ptxt)

                p_coords = parse_coords(p_node) or coords
                p_s_elems = p_node.findall(".//tei:s", NS)
                if p_s_elems:
                    p_sentences = [clean_text("".join(s.itertext())) for s in p_s_elems]
                    p_sentences = [st for st in p_sentences if st]
                else:
                    p_sentences = split_sentences(ptxt)

                recovered_para = {
                    "text": ptxt,
                    "sentences": p_sentences,
                    "section": target_sec,
                    "section_path": sec_path,
                    "coords": p_coords,
                    "page": (
                        p_coords[0]["page"]
                        if p_coords
                        else None
                    ),
                }

                # Insert right after the last paragraph of target_sec if found
                inserted = False
                if target_sec:
                    for idx in range(len(paragraphs) - 1, -1, -1):
                        if paragraphs[idx].get("section") == target_sec:
                            paragraphs.insert(idx + 1, recovered_para)
                            inserted = True
                            break
                if not inserted:
                    paragraphs.append(recovered_para)
            continue

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

    paragraphs = deduplicate_records(
        paragraphs,
        "text",
        "page",
        "section",
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
    # References / author biographies
    # ---------------------------------------------------------

    def _bibl_raw_text(bibl: ET.Element) -> str:
        raw_el = bibl.find(
            './tei:note[@type="raw_reference"]',
            NS,
        )
        if raw_el is not None:
            raw_value = text_content(raw_el)
            if raw_value:
                return raw_value
        return text_content(bibl)

    def _bibl_is_biography(
        bibl: ET.Element,
        raw: str,
        candidate_name: str,
        header_authors: list[str],
    ) -> bool:
        """Classify biography-like back-matter using document evidence only."""
        lower = raw.lower()
        biography_score = 0
        biography_patterns = (
            r"\breceived (?:the|a|an)\b",
            r"\bwas born\b",
            r"\bis currently\b",
            r"\bcurrently pursuing\b",
            r"\bcurrently working\b",
            r"\bresearch interests?\b",
            r"\bhas published\b",
            r"\bhas authored\b",
            r"\bdegrees? in\b",
        )
        for pattern in biography_patterns:
            if re.search(pattern, lower):
                biography_score += 1

        publication_score = 0
        if bibl.findall('.//tei:idno', NS):
            publication_score += 2
        if bibl.findall('.//tei:biblScope', NS):
            publication_score += 2
        if bibl.find('.//tei:analytic/tei:title', NS) is not None and text_content(
            bibl.find('.//tei:analytic/tei:title', NS)
        ):
            publication_score += 1
        if bibl.find('.//tei:monogr/tei:title', NS) is not None and text_content(
            bibl.find('.//tei:monogr/tei:title', NS)
        ):
            publication_score += 1
        if bibl.findall('.//tei:date', NS):
            publication_score += 1

        if not candidate_name:
            return False

        def name_tokens(value: str) -> set[str]:
            return set(re.findall(r"[a-z]+", value.casefold()))

        candidate_tokens = name_tokens(candidate_name)
        if not candidate_tokens:
            return False

        matched_header_author = False
        for header_author in header_authors:
            author_tokens = name_tokens(header_author)
            shared = candidate_tokens & author_tokens
            if len(shared) >= 2 and len(shared) / max(len(author_tokens), 1) >= 0.66:
                matched_header_author = True
                break
            similarity = SequenceMatcher(
                None,
                " ".join(sorted(candidate_tokens)),
                " ".join(sorted(author_tokens)),
            ).ratio()
            if similarity >= 0.82:
                matched_header_author = True
                break

        if not matched_header_author:
            return False

        word_count = len(raw.split())
        if word_count < 15 or word_count > 300:
            return False

        return biography_score >= 3 and publication_score <= 3

    def _extract_biography_name(raw: str) -> str:
        # Biography names are commonly printed in uppercase at the start of
        # the block, optionally followed by a role such as (Member, IEEE).
        match = re.match(
            r"^\s*([A-Z][A-Z .,'’\-]{2,})(?:\s+\([^)]*\))?\s+"
            r"(?:received|was born|is currently|has published|has authored|earned|obtained)\b",
            raw,
            re.IGNORECASE,
        )
        return clean_text(match.group(1)) if match else ""

    references: list[dict[str, Any]] = []
    author_biographies: list[dict[str, Any]] = []

    for bibl in root.findall(
        ".//tei:listBibl/tei:biblStruct",
        NS,
    ):
        raw = _bibl_raw_text(bibl)

        biography_name = _extract_biography_name(raw)
        if _bibl_is_biography(bibl, raw, biography_name, authors):
            author_biographies.append(
                {
                    "author": biography_name or None,
                    "text": clean_text(raw),
                    "source": "grobid",
                }
            )
            continue

        title_el = bibl.find(
            ".//tei:analytic/tei:title",
            NS,
        )

        if title_el is None:
            title_el = bibl.find(
                ".//tei:monogr/tei:title",
                NS,
            )

        ref_title = text_content(title_el) if title_el is not None else ""

        ref_authors: list[str] = []
        for author in bibl.findall(
            ".//tei:author",
            NS,
        ):
            value = text_content(author)
            if value:
                ref_authors.append(value)

        if not ref_title and not raw:
            continue

        references.append(
            {
                "title": ref_title,
                "authors": ref_authors,
                "raw": raw,
                "ref_id": bibl.attrib.get("xml:id") or bibl.attrib.get("id"),
            }
        )

    references = deduplicate_records(
        references,
        "title",
        "raw",
    )

    # Reconcile header names with the document's own biography blocks. This
    # corrects partial/misparsed names using evidence from the same paper, not
    # a paper-specific lookup table.
    biography_names = [
        item.get("author")
        for item in author_biographies
        if isinstance(item.get("author"), str) and item.get("author")
    ]

    if biography_names:
        def name_tokens(value: str) -> set[str]:
            return set(re.findall(r"[a-z]+", value.lower()))

        reconciled: list[str] = []
        for author in authors:
            original = clean_text(author)
            original_tokens = name_tokens(original)
            best = original
            best_score = 0.0
            for candidate in biography_names:
                candidate_tokens = name_tokens(candidate)
                shared = original_tokens & candidate_tokens
                if len(shared) < 2 or not original_tokens:
                    continue
                coverage = len(shared) / len(original_tokens)
                similarity = SequenceMatcher(None, original.lower(), candidate.lower()).ratio()
                score = max(coverage, similarity)
                if coverage >= 0.66 and score > best_score:
                    best_score = score
                    best = candidate
            reconciled.append(best)
        authors = reconciled

    keywords: list[str] = []
    for term in root.findall(".//tei:profileDesc//tei:keywords//tei:term", NS):
        v = text_content(term)
        if v:
            keywords.append(v)

    return {
        "source": "grobid",
        "title": title,
        "authors": authors,
        "year": year,
        "doi": doi,
        "abstract": abstract,
        "keywords": keywords,
        "sections": sections,
        "paragraphs": paragraphs,
        "figures": figures,
        "tables": tables,
        "references": references,
        "author_biographies": author_biographies,
    }