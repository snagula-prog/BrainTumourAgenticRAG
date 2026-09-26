import hashlib
import re
from typing import Any, Dict, List


class SectionAwareChunker():
    """
    Transforms a cleaned canonical document into structured,
    section-aware chunks for vector ingestion.

    Design:
      - body content is section-aware and windowed
      - tables/figures/formulas remain atomic
      - references preserve explicit canonical reference numbers
      - every chunk receives a deterministic order_index
      - provenance is retained wherever available
    """

    def __init__(self, max_words: int = 300):
        self.max_words = max_words

    def chunk(
        self,
        paper_id: str,
        cleaned_doc: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
        chunk_index = 0

        def add_chunk(
            chunk_type: str,
            text: str,
            metadata: Dict[str, Any],
        ) -> None:
            nonlocal chunk_index

            text = text.strip()

            chunk_id = self._generate_chunk_id(
                paper_id,
                chunk_type,
                chunk_index,
                text,
            )

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "paper_id": paper_id,
                    "chunk_type": chunk_type,
                    "order_index": chunk_index,
                    "text": text,
                    "word_count": len(text.split()),
                    "metadata": metadata,
                }
            )

            chunk_index += 1

        def copy_provenance(
            source: Dict[str, Any],
            metadata: Dict[str, Any],
        ) -> Dict[str, Any]:
            """
            Copy provenance fields without assuming every artifact has
            every field.
            """
            field_map = {
                "page": "page",
                "pages": "pages",
                "coords": "coordinates",
                "coordinates": "coordinates",
                "section": "section",
                "section_path": "section_path",
                "subsection_id": "subsection_id",
                "paragraph_id": "paragraph_id",
                "paragraph_ids": "paragraph_ids",
                "source": "source",
                "source_refs": "source_refs",
            }

            for source_key, metadata_key in field_map.items():
                value = source.get(source_key)

                if value is None:
                    continue

                if isinstance(value, list):
                    metadata[metadata_key] = list(value)
                else:
                    metadata[metadata_key] = value

            return metadata

        def get_section_path(
            source: Dict[str, Any],
        ) -> tuple[str, ...]:
            path = source.get("section_path")

            if isinstance(path, list):
                cleaned = tuple(
                    str(item).strip()
                    for item in path
                    if str(item).strip()
                )
                if cleaned:
                    return cleaned

            section = str(
                source.get("section")
                or ""
            ).strip()

            return (section,) if section else ()

        def make_subsection_id(
            path: tuple[str, ...],
        ) -> str | None:
            if not path:
                return None

            payload = "\x1f".join(path)
            digest = hashlib.sha256(
                payload.encode("utf-8")
            ).hexdigest()[:12]

            return f"{paper_id}_subsection_{digest}"

        # =========================================================
        # 1. Abstract + Metadata
        # =========================================================

        abstract_text = cleaned_doc.get("abstract")

        if abstract_text:
            meta = cleaned_doc.get("metadata", {})

            title = str(
                meta.get("title")
                or ""
            ).strip()

            authors = (
                meta.get("authors", [])
                if isinstance(meta.get("authors"), list)
                else []
            )

            index_terms = cleaned_doc.get(
                "index_terms"
            )

            if not isinstance(index_terms, list):
                index_terms = (
                    meta.get("index_terms")
                    or meta.get("keywords")
                    or []
                )

            terms = [
                str(term).strip()
                for term in index_terms
                if str(term).strip()
            ]

            add_chunk(
                chunk_type="abstract",
                text=str(abstract_text).strip(),
                metadata={
                    "section": "Abstract",
                    "title": title or None,
                    "authors": authors,
                    "index_terms": terms,
                },
            )
        # =========================================================
        # 2. Index Terms
        # =========================================================

        index_terms = cleaned_doc.get("index_terms")

        if not isinstance(index_terms, list):
            metadata = cleaned_doc.get(
                "metadata",
                {},
            )

            index_terms = (
                metadata.get("index_terms")
                or metadata.get("keywords")
                or []
            )

        if isinstance(index_terms, list):
            terms = [
                str(term).strip()
                for term in index_terms
                if str(term).strip()
            ]

            if terms:
                add_chunk(
                    chunk_type="index_terms",
                    text="[Index Terms] " + ", ".join(terms),
                    metadata={
                        "section": "Index Terms",
                        "terms": terms,
                    },
                )

        # =========================================================
        # 3. Author Biographies
        # =========================================================

        biographies = cleaned_doc.get(
            "author_biographies",
            [],
        )

        if not isinstance(
            biographies,
            list,
        ):
            biographies = []

        if not biographies:
            biographies = [
                paragraph
                for paragraph in cleaned_doc.get(
                    "paragraphs",
                    [],
                )
                if (
                    isinstance(paragraph, dict)
                    and str(
                        paragraph.get("section")
                        or ""
                    ).strip().casefold()
                    == "author biographies"
                )
            ]

        for biography in biographies:
            if not isinstance(
                biography,
                dict,
            ):
                continue

            bio_text = str(
                biography.get("text")
                or ""
            ).strip()

            if not bio_text:
                continue

            author = str(
                biography.get("author")
                or ""
            ).strip()

            prefix = (
                f"[Author Biography: {author}] "
                if author
                else "[Author Biography] "
            )

            metadata = {
                "section": "Author Biographies",
                "author": author or None,
            }

            metadata = copy_provenance(
                biography,
                metadata,
            )

            add_chunk(
                chunk_type="author_bio",
                text=bio_text,
                metadata=metadata,
            )

        # =========================================================
        # 4. Body Paragraphs
        # =========================================================

        paragraphs = cleaned_doc.get(
            "paragraphs",
            [],
        )

        if not isinstance(
            paragraphs,
            list,
        ):
            paragraphs = []

        current_section = None
        current_buffer: List[str] = []
        current_word_count = 0
        current_pages: set = set()

        current_paragraph_ids: List[str] = []
        current_coords: List[Dict[str, Any]] = []
        current_section_path: List[str] = []

        def emit_headings_for_path(
            path: tuple[str, ...],
        ) -> None:
            if not path:
                return

            for depth in range(
                1,
                len(path) + 1,
            ):
                heading_path = path[:depth]

                if heading_path in emitted_heading_paths:
                    continue

                heading = heading_by_path.get(
                    heading_path
                )

                if not heading:
                    continue

                add_chunk(
                    chunk_type="heading",
                    text=heading,
                    metadata={
                        "section": path[0],
                        "section_path": list(
                            heading_path
                        ),
                        "subsection_id": (
                            make_subsection_id(
                                heading_path
                            )
                        ),
                    },
                )

                emitted_heading_paths.add(
                    heading_path
                )

        def flush_buffer() -> None:
            nonlocal current_word_count

            if not current_buffer:
                return

            text = "\n".join(
                current_buffer
            )

            unique_para_ids = list(
                dict.fromkeys(
                    current_paragraph_ids
                )
            )

            unique_coords: List[Dict[str, Any]] = []

            for coord in current_coords:
                if coord not in unique_coords:
                    unique_coords.append(coord)

            section_path = tuple(current_section_path)

            metadata = {
                "section": current_section,
                "section_path": list(section_path),
                "subsection_id": make_subsection_id(section_path),
                "pages": sorted(current_pages),
                "paragraph_ids": unique_para_ids,
                "coordinates": unique_coords,
            }

            add_chunk(
                chunk_type="body",
                text=text,
                metadata=metadata,
            )

            current_buffer.clear()
            current_pages.clear()
            current_paragraph_ids.clear()
            current_coords.clear()

            current_section_path.clear()

            current_word_count = 0

        sections = cleaned_doc.get(
            "sections",
            [],
        )

        heading_by_path: Dict[
            tuple[str, ...],
            str,
        ] = {}

        if isinstance(sections, list):
            for section in sections:
                if not isinstance(section, dict):
                    continue

                heading = str(
                    section.get("heading")
                    or ""
                ).strip()

                path = section.get("path")

                if isinstance(path, list):
                    key = tuple(
                        str(item).strip()
                        for item in path
                        if str(item).strip()
                    )
                elif heading:
                    key = (heading,)
                else:
                    continue

                if heading:
                    heading_by_path[key] = heading

        emitted_heading_paths: set[
            tuple[str, ...]
        ] = set()

        current_section_key: tuple[str, ...] = ()

        for para in paragraphs:
            if not isinstance(
                para,
                dict,
            ):
                continue

            para_text = str(
                para.get("text")
                or ""
            ).strip()

            if not para_text:
                continue

            if (
                str(
                    para.get("section")
                    or ""
                ).strip().casefold()
                == "author biographies"
            ):
                continue

            section_path = get_section_path(
                para
            )

            para_section = (
                str(
                    para.get("section")
                    or (
                        section_path[-1]
                        if section_path
                        else "Body"
                    )
                ).strip()
            )

            # Hard structural boundary:
            # section OR subsection changes.
            if (
                current_section_key
                != section_path
            ):
                flush_buffer()

                current_section_key = (
                    section_path
                )

                current_section = (
                    para_section
                )

                current_section_path = list(
                    section_path
                )

                emit_headings_for_path(
                    section_path
                )

            para_words = len(
                para_text.split()
            )

            paragraph_id = para.get(
                "paragraph_id"
            )

            para_page = para.get(
                "page"
            )

            # -------------------------------------------------
            # Oversized paragraph:
            # NEVER split it.
            # -------------------------------------------------

            if para_words > self.max_words:
                flush_buffer()

                metadata = {
                    "section": para_section,
                    "section_path": list(
                        section_path
                    ),
                    "subsection_id": (
                        make_subsection_id(
                            section_path
                        )
                    ),
                    "pages": (
                        [para_page]
                        if para_page is not None
                        else []
                    ),
                    "paragraph_ids": (
                        [paragraph_id]
                        if paragraph_id
                        else []
                    ),
                    "coordinates": list(
                        para.get("coords")
                        or []
                    ),
                    "oversized": True,
                }

                add_chunk(
                    chunk_type="body",
                    text=para_text,
                    metadata=metadata,
                )

                continue

            # -------------------------------------------------
            # Normal paragraph
            # -------------------------------------------------

            if (
                current_buffer
                and (
                    current_word_count
                    + para_words
                    > self.max_words
                )
            ):
                flush_buffer()

            current_section = (
                para_section
            )
            current_section_path = list(
                section_path
            )

            current_buffer.append(
                para_text
            )

            if (
                paragraph_id
                and paragraph_id
                not in current_paragraph_ids
            ):
                current_paragraph_ids.append(
                    paragraph_id
                )

            for coord in (
                para.get("coords")
                or []
            ):
                if coord not in current_coords:
                    current_coords.append(
                        coord
                    )

            if para_page is not None:
                current_pages.add(
                    para_page
                )

            current_word_count += (
                para_words
            )

        flush_buffer()


        # =========================================================
        # 5. Table captions
        # =========================================================

        tables = cleaned_doc.get(
            "tables",
            [],
        )

        if not isinstance(
            tables,
            list,
        ):
            tables = []

        for table in tables:
            if not isinstance(
                table,
                dict,
            ):
                continue

            caption = str(
                table.get("caption")
                or ""
            ).strip()

            # No caption -> no embeddable table-caption chunk.
            # The structured table remains in cleaned_doc.
            if not caption:
                continue

            label = str(
                table.get("label")
                or "Table"
            ).strip()

            metadata = {
                "table_id": table.get(
                    "table_id"
                ),
                "label": label,
                "structure_status": (
                    table.get(
                        "structure_status"
                    )
                ),
            }

            metadata = copy_provenance(
                table,
                metadata,
            )

            section_path = get_section_path(
                table
            )

            metadata["subsection_id"] = (
                make_subsection_id(
                    section_path
                )
            )

            add_chunk(
                chunk_type="table_caption",
                text=caption,
                metadata=metadata,
            )

        # =========================================================
        # 6. Figure captions
        # =========================================================

        figures = cleaned_doc.get(
            "figures",
            [],
        )

        if not isinstance(
            figures,
            list,
        ):
            figures = []

        for figure in figures:
            if not isinstance(
                figure,
                dict,
            ):
                continue

            caption = str(
                figure.get("caption")
                or ""
            ).strip()

            if not caption:
                continue

            label = str(
                figure.get("label")
                or "Figure"
            ).strip()

            metadata = {
                "figure_id": figure.get(
                    "figure_id"
                ),
                "label": label,
            }

            metadata = copy_provenance(
                figure,
                metadata,
            )

            section_path = get_section_path(
                figure
            )

            metadata["subsection_id"] = (
                make_subsection_id(
                    section_path
                )
            )

            add_chunk(
                chunk_type="figure_caption",
                text=caption,
                metadata=metadata,
            )

        # =========================================================
        # 7. Formulas — atomic
        # =========================================================

        formulas = cleaned_doc.get(
            "formulas",
            [],
        )

        if not isinstance(
            formulas,
            list,
        ):
            formulas = []

        for formula in formulas:
            if not isinstance(
                formula,
                dict,
            ):
                continue

            fid = formula.get(
                "formula_id",
                "Eq",
            )

            # Support the current artifact representation.
            source = str(
                formula.get("source")
                or formula.get("text")
                or ""
            ).strip()

            text = (
                f"[Formula: {fid}]"
                f"{' ' + source if source else ''}"
            )

            metadata = {
                "formula_id": fid,
            }

            metadata = copy_provenance(
                formula,
                metadata,
            )

            add_chunk(
                chunk_type="formula",
                text=text,
                metadata=metadata,
            )

        # =========================================================
        # 8. References — explicit canonical numbers
        # =========================================================

        references = cleaned_doc.get(
            "references",
            [],
        )

        if not isinstance(references, list):
            references = []

        if references:
            filtered_references: List[Dict[str, Any]] = []

            # Defensive filter against malformed biography-like records.
            for ref in references:
                if not isinstance(ref, dict):
                    continue

                raw = str(ref.get("raw") or "")
                lower = raw.casefold()

                bio_signals = sum(
                    bool(
                        re.search(
                            pattern,
                            lower,
                        )
                    )
                    for pattern in (
                        r"\breceived (?:the|a|an)\b",
                        r"\bwas born\b",
                        r"\bis currently\b",
                        r"\bcurrently pursuing\b",
                        r"\bcurrently working\b",
                        r"\bresearch interests?\b",
                        r"\bhas published\b",
                        r"\bhas authored\b",
                    )
                )

                if bio_signals >= 3:
                    continue

                filtered_references.append(ref)

            references = filtered_references

            ref_buffer: List[str] = []
            ref_numbers_buffer: List[int] = []

            ref_start_number: int | None = None
            previous_number: int | None = None
            current_ref_words = 0

            def flush_reference_buffer() -> None:
                nonlocal ref_buffer
                nonlocal ref_numbers_buffer
                nonlocal ref_start_number
                nonlocal previous_number
                nonlocal current_ref_words

                if not ref_buffer:
                    return

                start_ref = (
                    ref_start_number
                    if ref_start_number is not None
                    else ref_numbers_buffer[0]
                )

                end_ref = (
                    previous_number
                    if previous_number is not None
                    else ref_numbers_buffer[-1]
                )

                add_chunk(
                    chunk_type="reference",
                    text=(
                        f"[References [{start_ref}]-[{end_ref}]]\n"
                        + "\n".join(ref_buffer)
                    ),
                    metadata={
                        "section": "References",
                        "start_ref": start_ref,
                        "end_ref": end_ref,
                        "reference_numbers": list(
                            ref_numbers_buffer
                        ),
                    },
                )

                ref_buffer = []
                ref_numbers_buffer = []
                ref_start_number = None
                previous_number = None
                current_ref_words = 0

            for position, ref in enumerate(
                references,
                start=1,
            ):
                raw = str(
                    ref.get("raw") or ""
                ).strip()

                if not raw:
                    continue

                explicit_number = ref.get("number")

                if isinstance(explicit_number, int):
                    ref_number = explicit_number
                else:
                    # Compatibility fallback for older canonical artifacts.
                    ref_number = position

                ref_line = (
                    f"[{ref_number}] {raw}"
                )

                line_words = len(
                    ref_line.split()
                )

                if ref_start_number is None:
                    ref_start_number = ref_number

                if (
                    current_ref_words + line_words
                    > self.max_words
                    and ref_buffer
                ):
                    flush_reference_buffer()
                    ref_start_number = ref_number

                ref_buffer.append(ref_line)
                ref_numbers_buffer.append(ref_number)

                current_ref_words += line_words
                previous_number = ref_number

            flush_reference_buffer()

        return chunks

    @staticmethod
    def _generate_chunk_id(
        paper_id: str,
        chunk_type: str,
        index: int,
        text: str,
    ) -> str:
        """
        Generates a stable, reproducible ID based on content and position.
        """
        content_hash = hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()[:8]

        return (
            f"{paper_id}_{chunk_type}_"
            f"{index:04d}_{content_hash}"
        )