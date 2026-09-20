from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from typing import Any, Iterable


class LogicalArtifactReconstructor:
    """Reconstruct logical research artifacts from parser-level evidence.

    This layer deliberately does not attempt to "correct" parser output. It
    creates logical objects only when the available evidence supports them and
    keeps source references back to the parser-level records.
    """

    TABLE_LABEL_RE = re.compile(
        r"^\s*(?:table|tab\.?)\b\s*([A-Za-z0-9IVXivx]+)\b",
        re.IGNORECASE,
    )
    FIGURE_LABEL_RE = re.compile(
        r"^\s*(?:figure|fig\.?)\b\s*([A-Za-z0-9IVXivx]+)\b",
        re.IGNORECASE,
    )
    CONTINUED_RE = re.compile(r"\b(?:continued|continu(?:ed)?|cont\.?)\b", re.IGNORECASE)

    def reconstruct(
        self,
        *,
        grobid: dict[str, Any],
        artifacts: dict[str, Any],
    ) -> dict[str, Any]:
        text_blocks = self._safe_list(artifacts.get("text_blocks"))

        tables = self._reconstruct_tables(
            grobid_tables=self._safe_list(grobid.get("tables")),
            docling_tables=self._safe_list(artifacts.get("tables")),
            text_blocks=text_blocks,
        )
        figures, unclassified_visuals = self._reconstruct_figures(
            grobid_figures=self._safe_list(grobid.get("figures")),
            docling_figures=self._safe_list(artifacts.get("figures")),
            text_blocks=text_blocks,
        )

        return {
            "tables": tables,
            "figures": figures,
            "unclassified_visuals": unclassified_visuals,
            "reconstruction": {
                "raw_table_regions": len(self._safe_list(artifacts.get("tables"))),
                "logical_tables": len(tables),
                "raw_visual_regions": len(self._safe_list(artifacts.get("figures"))),
                "logical_figures": len(figures),
                "unclassified_visuals": len(unclassified_visuals),
            },
        }

    # ------------------------------------------------------------------
    # TABLES
    # ------------------------------------------------------------------

    def _reconstruct_tables(
        self,
        *,
        grobid_tables: list[dict[str, Any]],
        docling_tables: list[dict[str, Any]],
        text_blocks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []

        for index, record in enumerate(grobid_tables):
            if not isinstance(record, dict):
                continue
            candidates.append(
                self._table_candidate(
                    record,
                    parser="grobid",
                    index=index,
                    text_blocks=text_blocks,
                )
            )

        for index, record in enumerate(docling_tables):
            if not isinstance(record, dict):
                continue
            candidates.append(
                self._table_candidate(
                    record,
                    parser="docling",
                    index=index,
                    text_blocks=text_blocks,
                )
            )

        if not candidates:
            return []

        # Work in document order. Explicit labels are strong identity evidence,
        # but unlabeled regions can sit between a table's first page and its
        # explicit ``continued`` page. Build labeled groups first, then bridge
        # those unlabeled regions using page/continuation evidence.
        ordered = sorted(candidates, key=self._candidate_sort_key)
        groups: list[list[dict[str, Any]]] = []

        explicit_by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
        unlabeled: list[dict[str, Any]] = []
        for candidate in ordered:
            if candidate["label_key"]:
                explicit_by_label[candidate["label_key"]].append(candidate)
            else:
                unlabeled.append(candidate)

        consumed: set[int] = set()

        for label_key, members in explicit_by_label.items():
            members = sorted(members, key=self._candidate_sort_key)
            current: list[dict[str, Any]] = []

            for candidate in members:
                if not current:
                    current = [candidate]
                    continue

                previous = current[-1]
                prev_end = self._last_page(previous)
                current_page = self._first_page(candidate)
                gap = (
                    current_page - prev_end
                    if prev_end is not None and current_page is not None
                    else None
                )

                # A page-less explicit label from GROBID is strong identity
                # evidence even though there is no page adjacency. Join it to
                # the same labeled region, while still requiring continuation
                # or adjacency for two spatially located regions.
                same_label = bool(
                    candidate.get("label_key")
                    and candidate.get("label_key") == previous.get("label_key")
                )
                page_less_identity = same_label and (
                    not candidate.get("coords")
                    or not previous.get("coords")
                )
                should_join = bool(
                    page_less_identity
                    or candidate["continued"]
                    or previous["continued"]
                    or (gap is not None and gap <= 1)
                )

                if should_join:
                    current.append(candidate)
                else:
                    groups.append(current)
                    current = [candidate]

            if current:
                groups.append(current)

        # Attach parser regions with no label when they fall inside or directly
        # between a known logical table's page span. This is what handles cases
        # such as: Table 4 (p8) -> unlabeled region (p9) -> Table 4 Continued (p10).
        # Attach unlabeled parser evidence to an existing logical table when
        # page/continuation evidence or caption identity supports the match.
        for candidate in unlabeled:

            # GROBID can emit a page-less continuation caption fragment such as:
            # "(Continued.) Brain tumor detection using deep learning techniques."
            # Do not promote such a fragment into a new logical table when its
            # caption matches an already reconstructed table.
            if (
                candidate["parser"] == "grobid"
                and not candidate["coords"]
                and candidate["caption"]
                and candidate["continued"]
            ):
                candidate_identity = self._table_caption_identity(
                    candidate["caption"]
                )

                if candidate_identity:
                    best_match = None
                    best_similarity = 0.0

                    for group in groups:
                        group_captions = [
                            g["caption"]
                            for g in group
                            if g["caption"]
                        ]

                        for group_caption in group_captions:
                            group_identity = self._table_caption_identity(
                                group_caption
                            )

                            if not group_identity:
                                continue

                            similarity = SequenceMatcher(
                                None,
                                candidate_identity,
                                group_identity,
                            ).ratio()

                            if similarity > best_similarity:
                                best_similarity = similarity
                                best_match = group

                    if best_match is not None and best_similarity >= 0.80:
                        best_match.append(candidate)
                        continue

            candidate_page = self._first_page(candidate)

            if candidate_page is None:
                groups.append([candidate])
                continue

            matching_groups: list[
                tuple[int, list[dict[str, Any]]]
            ] = []

            for group_index, group in enumerate(groups):
                first_page = self._first_page(group[0])
                last_page = self._last_page(group[-1])

                if first_page is None or last_page is None:
                    continue

                if first_page <= candidate_page <= last_page:
                    matching_groups.append((0, group))
                    continue

                if candidate_page == last_page + 1:
                    matching_groups.append((1, group))
                    continue

            if matching_groups:
                _, best_group = min(
                    matching_groups,
                    key=lambda item: (
                        abs(
                            candidate_page
                            - (
                                self._last_page(item[1][-1])
                                if item[1]
                                else candidate_page
                            )
                        ),
                        self._first_page(item[1][0])
                        if item[1]
                        else candidate_page,
                    ),
                )
                best_group.append(candidate)
            else:
                groups.append([candidate])

        # Re-sort regions inside each group so provenance follows page order.
        groups = [
            sorted(group, key=self._candidate_sort_key)
            for group in groups
        ]

        result: list[dict[str, Any]] = []
        for number, group in enumerate(groups, start=1):
            group = self._dedupe_equivalent_candidates(group)
            result.append(self._build_table(group, number))

        return result

    @classmethod
    def _label_from_structured_data(cls, structure: Any) -> str:
        """Extract an explicit Table N identity embedded in native cell data."""
        if isinstance(structure, dict):
            rows = structure.get("rows") or structure.get("grid")
        elif isinstance(structure, list):
            rows = structure
        else:
            rows = None

        if not isinstance(rows, list):
            return ""

        # Restrict identity detection to the header area so a data cell
        # mentioning "Table N" cannot create a false table identity.
        for row in rows[:3]:
            if not isinstance(row, list):
                continue
            for cell in row[:8]:
                if not isinstance(cell, dict):
                    continue
                cell_text = cls._clean_text(cell.get("text"))
                if not cell_text:
                    continue
                match = cls.TABLE_LABEL_RE.search(cell_text)
                if match:
                    return f"Table {match.group(1)}"

        return ""

    def _table_candidate(
        self,
        record: dict[str, Any],
        *,
        parser: str,
        index: int,
        text_blocks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        label = self._clean_text(record.get("label"))
        caption = self._clean_text(record.get("caption"))
        content = self._first_text(
            record,
            ("content", "text", "markdown", "md"),
        )

        coords = self._normalize_coords(record.get("coords"))
        label_key, continued = self._table_label_key(label)

        # GROBID may store a table label as a bare number (for example "6")
        # while <head> carries the full "Table 6" identity. Treat the numeric
        # label as an explicit table identity at reconstruction time.
        if (
            parser == "grobid"
            and not label_key
            and label
            and re.fullmatch(r"\d+", label)
        ):
            label = f"Table {label}"
            label_key, grobid_continued = self._table_label_key(label)
            continued = continued or grobid_continued

        # Native Docling cell data can contain the real scientific identity
        # even when the parser-level label is the generic ``table``.
        structured_data = self._structured_data(record)
        structure_evidence = structured_data
        if structure_evidence is None:
            # DoclingArtifactExtractor preserves native table structure under
            # ``structure``. Use that parser-level evidence for identity even
            # when the older ``structured_data`` key is not present.
            structure_evidence = record.get("structure")

        if not label_key:
            structured_label = self._label_from_structured_data(structure_evidence)
            if structured_label:
                label = structured_label
                label_key, structured_continued = self._table_label_key(
                    structured_label
                )
                continued = continued or structured_continued

        # Docling commonly labels every detected table simply as ``table``.
        # Prefer a scientific identity extracted from the caption instead of
        # treating the generic parser label as the logical table identity.
        if not label_key and caption:
            caption_label_key, caption_continued = self._table_label_key(caption)
            if caption_label_key:
                label_key = caption_label_key
                continued = continued or caption_continued
                if not label or self._normalize_generic_label(label, kind="table"):
                    label = self._extract_label(caption, kind="table")

        nearby_caption = ""
        if not caption and coords:
            nearby_caption = self._find_nearby_caption(
                coords[0],
                text_blocks,
                kind="table",
            )
            if nearby_caption:
                caption = nearby_caption
                if not label_key:
                    label_key, continued = self._table_label_key(caption)
                if not label:
                    label = self._extract_label(caption, kind="table")

        return {
            "record": record,
            "parser": parser,
            "index": index,
            "label": label,
            "caption": caption,
            "content": content,
            "coords": coords,
            "label_key": label_key,
            "continued": continued or self._is_continued(label) or self._is_continued(caption),
            "structured_data": structured_data,
            "source_ref": {
                "parser": parser,
                "kind": "table",
                "index": index,
            },
        }

    def _build_table(self, group: list[dict[str, Any]], number: int) -> dict[str, Any]:
        labels = [g["label"] for g in group if g["label"]]
        captions = [g["caption"] for g in group if g["caption"]]
        contents = [g["content"] for g in group if g["content"]]
        coords = [coord for g in group for coord in g["coords"]]
        pages = sorted({int(c["page"]) for c in coords if c.get("page") is not None})
        source_refs = [g["source_ref"] for g in group]

        structured = next(
            (g["structured_data"] for g in group if g["structured_data"] is not None),
            None,
        )
        if structured is not None:
            structure_status = "structured"
        elif contents or captions:
            structure_status = "text_only"
        elif coords or source_refs:
            structure_status = "detected_no_content"
        else:
            structure_status = "unavailable"

        label = self._preferred_label(labels)
        caption = self._preferred_caption(captions)

        continuation = (
            "continued"
            if len(pages) > 1 or any(g["continued"] for g in group)
            else "single"
        )

        confidence = self._table_confidence(group, structure_status, label, caption)

        return {
            "table_id": f"table_{number:03d}",
            "label": label,
            "caption": caption,
            "pages": pages,
            "regions": coords,
            "content": "\n".join(self._unique_preserve_order(contents)),
            "structure_status": structure_status,
            "structured_data": structured,
            "continuation_status": continuation,
            "source_refs": source_refs,
            "source_region_count": len(group),
            "confidence": round(confidence, 4),
        }

    # ------------------------------------------------------------------
    # FIGURES
    # ------------------------------------------------------------------

    def _reconstruct_figures(
        self,
        *,
        grobid_figures: list[dict[str, Any]],
        docling_figures: list[dict[str, Any]],
        text_blocks: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        candidates: list[dict[str, Any]] = []

        for index, record in enumerate(grobid_figures):
            if not isinstance(record, dict):
                continue
            candidates.append(
                self._figure_candidate(
                    record,
                    parser="grobid",
                    index=index,
                    text_blocks=text_blocks,
                )
            )

        for index, record in enumerate(docling_figures):
            if not isinstance(record, dict):
                continue
            candidates.append(
                self._figure_candidate(
                    record,
                    parser="docling",
                    index=index,
                    text_blocks=text_blocks,
                )
            )

        if not candidates:
            return [], []

        groups: list[list[dict[str, Any]]] = []
        used: set[int] = set()

        explicit_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for candidate in candidates:
            if candidate["label_key"]:
                explicit_groups[candidate["label_key"]].append(candidate)

        for members in explicit_groups.values():
            group = sorted(members, key=self._candidate_sort_key)
            groups.append(group)
            used.update(id(x) for x in group)

        for candidate in sorted(candidates, key=self._candidate_sort_key):
            if id(candidate) in used:
                continue
            matched_group = None
            for group in groups:
                if self._same_figure_region(candidate, group[0]):
                    matched_group = group
                    break
            if matched_group is not None:
                matched_group.append(candidate)
            else:
                groups.append([candidate])

        logical_figures: list[dict[str, Any]] = []
        unclassified: list[dict[str, Any]] = []
        next_id = 1

        for group in groups:
            group = self._dedupe_equivalent_candidates(group)
            label = self._preferred_label([g["label"] for g in group if g["label"]])
            caption = self._preferred_caption([g["caption"] for g in group if g["caption"]])
            coords = [coord for g in group for coord in g["coords"]]
            pages = sorted({int(c["page"]) for c in coords if c.get("page") is not None})
            confidence, signals = self._figure_confidence(group, label, caption)

            has_scientific_signal = confidence >= 0.45
            source_refs = [g["source_ref"] for g in group]

            if has_scientific_signal:
                logical_figures.append(
                    {
                        "figure_id": f"figure_{next_id:03d}",
                        "label": label,
                        "caption": caption,
                        "pages": pages,
                        "regions": coords,
                        "caption_status": "available" if caption else "missing",
                        "source_refs": source_refs,
                        "confidence": round(confidence, 4),
                        "signals": signals,
                    }
                )
                next_id += 1
            else:
                unclassified.append(
                    {
                        "visual_id": f"visual_{len(unclassified) + 1:03d}",
                        "pages": pages,
                        "regions": coords,
                        "source_refs": source_refs,
                        "confidence": round(confidence, 4),
                        "reason": self._unclassified_reason(signals),
                    }
                )

        return logical_figures, unclassified

    def _figure_candidate(
        self,
        record: dict[str, Any],
        *,
        parser: str,
        index: int,
        text_blocks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        label = self._clean_text(record.get("label"))
        caption = self._clean_text(record.get("caption"))
        coords = self._normalize_coords(record.get("coords"))

        label_key = self._figure_label_key(label)

        # Docling commonly labels visual regions as ``picture``. The scientific
        # figure identity is usually present in the associated caption.
        if not label_key and caption:
            caption_label_key = self._figure_label_key(caption)
            if caption_label_key:
                label_key = caption_label_key
                if not label or self._normalize_generic_label(label, kind="figure"):
                    label = self._extract_label(caption, kind="figure")

        nearby_caption = ""
        if not caption and coords:
            nearby_caption = self._find_nearby_caption(
                coords[0],
                text_blocks,
                kind="figure",
            )
            if nearby_caption:
                caption = nearby_caption
                if not label_key:
                    label_key = self._figure_label_key(caption)
                if not label:
                    label = self._extract_label(caption, kind="figure")

        return {
            "parser": parser,
            "index": index,
            "label": label,
            "caption": caption,
            "coords": coords,
            "label_key": label_key,
            "grobid": parser == "grobid",
            "source_ref": {
                "parser": parser,
                "kind": "figure",
                "index": index,
            },
        }

    # ------------------------------------------------------------------
    # Matching / evidence
    # ------------------------------------------------------------------

    @classmethod
    def _same_table_region(cls, left: dict[str, Any], right: dict[str, Any]) -> bool:
        if left["label_key"] and right["label_key"]:
            return left["label_key"] == right["label_key"]
        return cls._spatial_match(left["coords"], right["coords"], max_page_gap=0)

    @classmethod
    def _same_figure_region(cls, left: dict[str, Any], right: dict[str, Any]) -> bool:
        if left["label_key"] and right["label_key"]:
            return left["label_key"] == right["label_key"]
        if cls._clean_text(left.get("caption")) and cls._clean_text(right.get("caption")):
            return cls._text_similarity(left["caption"], right["caption"]) >= 0.88
        return cls._spatial_match(left["coords"], right["coords"], max_page_gap=0)

    @classmethod
    def _spatial_match(
        cls,
        left_coords: list[dict[str, Any]],
        right_coords: list[dict[str, Any]],
        *,
        max_page_gap: int,
    ) -> bool:
        if not left_coords or not right_coords:
            return False
        for left in left_coords:
            for right in right_coords:
                lp = left.get("page")
                rp = right.get("page")
                if lp is None or rp is None or abs(int(lp) - int(rp)) > max_page_gap:
                    continue
                if cls._iou(left, right) >= 0.15:
                    return True
                if cls._bbox_distance(left, right) <= 36.0:
                    return True
        return False

    @classmethod
    def _figure_confidence(
        cls,
        group: list[dict[str, Any]],
        label: str,
        caption: str,
    ) -> tuple[float, list[str]]:
        score = 0.0
        signals: list[str] = []

        has_figure_identity = bool(
            label
            and cls._figure_label_key(label)
        )

        has_caption_identity = bool(
            cls._figure_label_key(caption)
        )

        has_grobid = any(
            g["grobid"]
            for g in group
        )

        # GROBID itself is NOT sufficient evidence.
        # GROBID can emit false-positive <figure> elements.
        if has_grobid:
            if (
                has_figure_identity
                or has_caption_identity
            ):
                score += 0.45
                signals.append(
                    "grobid_figure_with_identity"
                )
            else:
                signals.append(
                    "grobid_figure_without_identity"
                )

        # Explicit Figure N / Fig. N identity is strong evidence.
        if has_figure_identity:
            score += 0.30
            signals.append(
                "figure_label"
            )

        # A caption explicitly beginning with Figure N is strong evidence.
        if has_caption_identity:
            score += 0.20
            signals.append(
                "figure_caption_identity"
            )
        elif caption:
            # Plain text alone is weak evidence.
            score += 0.05
            signals.append(
                "caption"
            )

        # Independent Docling caption evidence gets a small boost.
        if any(
            g["parser"] == "docling"
            and g["caption"]
            for g in group
        ):
            score += 0.05
            signals.append(
                "docling_caption"
            )

        if cls._repeated_small_region(
            group
        ):
            score -= 0.35
            signals.append(
                "repeated_small_region"
            )

        return (
            max(
                0.0,
                min(
                    1.0,
                    score,
                ),
            ),
            signals,
        )

    @classmethod
    def _table_confidence(
        cls,
        group: list[dict[str, Any]],
        structure_status: str,
        label: str,
        caption: str,
    ) -> float:
        score = 0.0
        if any(g["parser"] == "grobid" for g in group):
            score += 0.45
        if label and cls._table_label_key(label)[0]:
            score += 0.25
        if caption:
            score += 0.15
        if structure_status == "structured":
            score += 0.10
        elif structure_status == "text_only":
            score += 0.05
        return max(0.0, min(1.0, score))

    @classmethod
    def _repeated_small_region(cls, group: list[dict[str, Any]]) -> bool:
        coords = [c for g in group for c in g["coords"]]
        if len(coords) < 3:
            return False
        signatures = [
            (
                round(float(c.get("x", 0.0)), 1),
                round(float(c.get("y", 0.0)), 1),
                round(float(c.get("w", 0.0)), 1),
                round(float(c.get("h", 0.0)), 1),
            )
            for c in coords
        ]
        repeated = Counter(signatures).most_common(1)[0][1]
        if repeated < 3:
            return False
        area = max(float(coords[0].get("w", 0.0)) * float(coords[0].get("h", 0.0)), 0.0)
        return area < 8000.0

    # ------------------------------------------------------------------
    # Captions and labels
    # ------------------------------------------------------------------

    @classmethod
    def _find_nearby_caption(
        cls,
        region: dict[str, Any],
        text_blocks: list[dict[str, Any]],
        *,
        kind: str,
    ) -> str:
        label_re = cls.TABLE_LABEL_RE if kind == "table" else cls.FIGURE_LABEL_RE
        page = region.get("page")
        if page is None:
            return ""

        best: tuple[float, str] | None = None
        for block in text_blocks:
            if not isinstance(block, dict):
                continue
            text = cls._clean_text(block.get("text"))
            if not text or not label_re.search(text):
                continue
            block_coords = cls._normalize_coords(block.get("coords"))
            for candidate in block_coords:
                if candidate.get("page") != page:
                    continue
                x_overlap = cls._x_overlap_ratio(region, candidate)
                distance = cls._vertical_gap(region, candidate)
                if x_overlap <= 0.0:
                    continue
                # Nearby captions are preferable to distant matching labels.
                score = x_overlap * 2.0 - min(distance / 100.0, 2.0)
                if best is None or score > best[0]:
                    best = (score, text)

        return best[1] if best is not None else ""

    @classmethod
    def _extract_label(cls, text: str, *, kind: str) -> str:
        pattern = cls.TABLE_LABEL_RE if kind == "table" else cls.FIGURE_LABEL_RE
        match = pattern.search(cls._clean_text(text))
        if not match:
            return ""
        prefix = "Table" if kind == "table" else "Figure"
        return f"{prefix} {match.group(1)}"

    @classmethod
    def _table_label_key(cls, text: str) -> tuple[str, bool]:
        match = cls.TABLE_LABEL_RE.search(cls._clean_text(text))
        if not match:
            return "", False
        token = match.group(1).lower()
        continued = cls._is_continued(text)
        return f"table {token}", continued
    
    @classmethod
    def _table_caption_identity(cls, text: str) -> str:
        """Normalize a table caption for continuation/deduplication matching."""

        text = cls._clean_text(text).lower()

        # Remove table label.
        text = re.sub(
            r"^\s*table\s+[a-z0-9ivx]+\s*[\.\-:]?\s*",
            "",
            text,
        )

        # Remove continuation markers.
        text = re.sub(
            r"\(\s*continued\.?\s*\)",
            "",
            text,
        )
        text = re.sub(
            r"\bcontinued\b",
            "",
            text,
        )

        text = re.sub(r"\s+", " ", text).strip(" .:-")

        return text

    @classmethod
    def _figure_label_key(cls, text: str) -> str:
        match = cls.FIGURE_LABEL_RE.search(cls._clean_text(text))
        if not match:
            return ""
        token = match.group(1).lower()
        return f"figure {token}"

    @classmethod
    def _is_continued(cls, text: str) -> bool:
        return bool(cls.CONTINUED_RE.search(cls._clean_text(text)))

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_list(value: Any) -> list[dict[str, Any]]:
        return value if isinstance(value, list) else []

    @staticmethod
    def _clean_text(value: Any) -> str:
        if value is None:
            return ""
        text = re.sub(r"\s+", " ", str(value).replace("\xa0", " ")).strip()
        return text

    @classmethod
    def _first_text(cls, record: dict[str, Any], keys: Iterable[str]) -> str:
        for key in keys:
            value = cls._clean_text(record.get(key))
            if value:
                return value
        return ""

    @staticmethod
    def _structured_data(record: dict[str, Any]) -> Any:
        """Return structured table data only when it actually contains structure.

        Docling can legitimately emit a ``data`` object whose grid/cell lists are
        empty and whose row/column counts are zero. That is detection evidence,
        not structured cell data.
        """

        for key in ("structured_data", "table_data", "data"):
            value = record.get(key)
            if not isinstance(value, (list, dict)) or not value:
                continue

            if isinstance(value, dict):
                table_cells = value.get("table_cells")
                grid = value.get("grid")
                rows = value.get("rows")
                cells = value.get("cells")
                num_rows = value.get("num_rows")
                num_cols = value.get("num_cols")

                if isinstance(table_cells, list) and table_cells:
                    return value
                if isinstance(grid, list) and grid and any(grid):
                    return value
                if isinstance(rows, list) and rows:
                    return value
                if isinstance(cells, list) and cells:
                    return value
                if (
                    isinstance(num_rows, int)
                    and isinstance(num_cols, int)
                    and num_rows > 0
                    and num_cols > 0
                ):
                    return value

                continue

            if isinstance(value, list) and value:
                return value

        for key in ("grid", "rows", "cells"):
            value = record.get(key)
            if isinstance(value, list) and value and any(value):
                return value

        return None

    @classmethod
    def _normalize_coords(cls, coords: Any) -> list[dict[str, Any]]:
        if not isinstance(coords, list):
            return []
        result: list[dict[str, Any]] = []
        for coord in coords:
            if not isinstance(coord, dict):
                continue
            try:
                page = int(coord.get("page")) if coord.get("page") is not None else None
                x = float(coord.get("x", 0.0))
                y = float(coord.get("y", 0.0))
                w = float(coord.get("w", coord.get("width", 0.0)))
                h = float(coord.get("h", coord.get("height", 0.0)))
            except (TypeError, ValueError):
                continue
            if page is None or w <= 0.0 or h <= 0.0:
                continue
            result.append(
                {
                    "page": page,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "coord_origin": coord.get("coord_origin", "BOTTOMLEFT"),
                }
            )
        return result

    @staticmethod
    def _iou(left: dict[str, Any], right: dict[str, Any]) -> float:
        lx1, ly1 = float(left["x"]), float(left["y"])
        lx2, ly2 = lx1 + float(left["w"]), ly1 + float(left["h"])
        rx1, ry1 = float(right["x"]), float(right["y"])
        rx2, ry2 = rx1 + float(right["w"]), ry1 + float(right["h"])
        ix1, iy1 = max(lx1, rx1), max(ly1, ry1)
        ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        intersection = iw * ih
        union = float(left["w"]) * float(left["h"]) + float(right["w"]) * float(right["h"]) - intersection
        return intersection / union if union > 0.0 else 0.0

    @staticmethod
    def _bbox_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
        lx2 = float(left["x"]) + float(left["w"])
        ly2 = float(left["y"]) + float(left["h"])
        rx2 = float(right["x"]) + float(right["w"])
        ry2 = float(right["y"]) + float(right["h"])
        dx = max(float(right["x"]) - lx2, float(left["x"]) - rx2, 0.0)
        dy = max(float(right["y"]) - ly2, float(left["y"]) - ry2, 0.0)
        return math.hypot(dx, dy)

    @classmethod
    def _x_overlap_ratio(cls, left: dict[str, Any], right: dict[str, Any]) -> float:
        lx1, lx2 = float(left["x"]), float(left["x"]) + float(left["w"])
        rx1, rx2 = float(right["x"]), float(right["x"]) + float(right["w"])
        overlap = max(0.0, min(lx2, rx2) - max(lx1, rx1))
        base = min(float(left["w"]), float(right["w"]))
        return overlap / base if base > 0.0 else 0.0

    @classmethod
    def _vertical_gap(cls, left: dict[str, Any], right: dict[str, Any]) -> float:
        l_bottom, l_top = float(left["y"]), float(left["y"]) + float(left["h"])
        r_bottom, r_top = float(right["y"]), float(right["y"]) + float(right["h"])
        if r_bottom > l_top:
            return r_bottom - l_top
        if l_bottom > r_top:
            return l_bottom - r_top
        return 0.0

    @staticmethod
    def _first_page(candidate: dict[str, Any]) -> int | None:
        coords = candidate.get("coords") or []
        if not coords:
            return None
        pages = [int(c["page"]) for c in coords if c.get("page") is not None]
        return min(pages) if pages else None

    @staticmethod
    def _last_page(candidate: dict[str, Any]) -> int | None:
        coords = candidate.get("coords") or []
        if not coords:
            return None
        pages = [int(c["page"]) for c in coords if c.get("page") is not None]
        return max(pages) if pages else None

    @staticmethod
    def _candidate_sort_key(candidate: dict[str, Any]) -> tuple[int, float, int]:
        first = candidate["coords"][0] if candidate["coords"] else {}
        page = int(first.get("page", 10**9))
        y = -float(first.get("y", 0.0))
        return page, y, 0 if candidate["parser"] == "grobid" else 1

    @classmethod
    def _dedupe_equivalent_candidates(cls, group: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        signatures: set[tuple[Any, ...]] = set()
        for candidate in group:
            coords = candidate["coords"]
            coord = coords[0] if coords else {}
            signature = (
                candidate["label_key"],
                candidate["parser"],
                coord.get("page"),
                round(float(coord.get("x", 0.0)), 1),
                round(float(coord.get("y", 0.0)), 1),
                round(float(coord.get("w", 0.0)), 1),
                round(float(coord.get("h", 0.0)), 1),
                cls._clean_text(candidate.get("caption")),
            )
            if signature in signatures:
                continue
            signatures.add(signature)
            result.append(candidate)
        return result

    @staticmethod
    def _normalize_generic_label(value: str, *, kind: str) -> bool:
        normalized = re.sub(r"[^a-z]+", "", value.lower())
        if kind == "table":
            return normalized in {"", "table", "tab"}
        return normalized in {"", "picture", "figure", "fig"}

    @classmethod
    def _preferred_label(cls, labels: list[str]) -> str:
        if not labels:
            return ""

        for label in labels:
            table_match = cls.TABLE_LABEL_RE.search(label)
            if table_match:
                return f"Table {table_match.group(1)}"

            figure_match = cls.FIGURE_LABEL_RE.search(label)
            if figure_match:
                return f"Figure {figure_match.group(1)}"

        for label in labels:
            if not cls._normalize_generic_label(label, kind="table"):
                return label

        return ""

    @classmethod
    def _preferred_caption(cls, captions: list[str]) -> str:
        unique = cls._unique_preserve_order(captions)
        if not unique:
            return ""
        return max(unique, key=len)

    @staticmethod
    def _unique_preserve_order(values: Iterable[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            if not value:
                continue
            key = value.lower()
            if key in seen:
                continue
            seen.add(key)
            result.append(value)
        return result

    @classmethod
    def _text_similarity(cls, left: str, right: str) -> float:
        a = re.sub(r"\W+", " ", cls._clean_text(left).lower()).strip()
        b = re.sub(r"\W+", " ", cls._clean_text(right).lower()).strip()
        if not a or not b:
            return 0.0
        if a == b:
            return 1.0
        return SequenceMatcher(None, a, b).ratio()

    @staticmethod
    def _unclassified_reason(signals: list[str]) -> str:
        if "repeated_small_region" in signals:
            return "repeated_small_visual_without_strong_figure_evidence"
        return "insufficient_figure_identity_evidence"
    
    def reconstruct_artifacts(
    *args,
    **kwargs,
) -> dict[str, Any]:
        """
        Compatibility entry point for canonical_builder.py.

        Keeps the existing LogicalArtifactReconstructor as the actual
        implementation while exposing the function name expected by
        the existing canonicalization layer.
        """

        grobid = kwargs.get("grobid")
        artifacts = kwargs.get("artifacts")

        # Support positional calls regardless of whether the caller
        # passes (grobid, artifacts) or (artifacts, grobid).
        if args:
            dictionaries = [
                value
                for value in args
                if isinstance(value, dict)
            ]

            if artifacts is None:
                artifacts = next(
                    (
                        value
                        for value in dictionaries
                        if (
                            "text_blocks" in value
                            or "counts" in value
                            or "formulas" in value
                        )
                    ),
                    None,
                )

            if grobid is None:
                grobid = next(
                    (
                        value
                        for value in dictionaries
                        if value is not artifacts
                    ),
                    None,
                )

        if not isinstance(grobid, dict):
            grobid = {}

        if not isinstance(artifacts, dict):
            artifacts = {}

        reconstructor = LogicalArtifactReconstructor()

        result = reconstructor.reconstruct(
            grobid=grobid,
            artifacts=artifacts,
        )

        # Preserve formulas from the parser artifact bundle.
        result["formulas"] = artifacts.get(
            "formulas",
            [],
        )

        # Compatibility aliases used by different
        # parts of the canonical layer.
        result["visual_artifacts"] = result.get(
            "unclassified_visuals",
            [],
        )

        result["stats"] = result.get(
            "reconstruction",
            {},
        )

        return result
['LogicalArtifactReconstructor']

def reconstruct_artifacts(
    *args,
    **kwargs,
) -> dict[str, Any]:
    """
    Compatibility wrapper for the existing canonical builder.
    """

    grobid = kwargs.get("grobid")
    artifacts = kwargs.get("artifacts")

    dictionaries = [
        value
        for value in args
        if isinstance(value, dict)
    ]

    if artifacts is None:
        artifacts = next(
            (
                value
                for value in dictionaries
                if (
                    "text_blocks" in value
                    or "counts" in value
                    or "formulas" in value
                )
            ),
            None,
        )

    if grobid is None:
        grobid = next(
            (
                value
                for value in dictionaries
                if value is not artifacts
            ),
            None,
        )

    if not isinstance(grobid, dict):
        grobid = {}

    if not isinstance(artifacts, dict):
        artifacts = {}

    reconstructor = LogicalArtifactReconstructor()

    result = reconstructor.reconstruct(
        grobid=grobid,
        artifacts=artifacts,
    )

    result["formulas"] = artifacts.get(
        "formulas",
        [],
    )

    result["visual_artifacts"] = result.get(
        "unclassified_visuals",
        [],
    )

    result["stats"] = result.get(
        "reconstruction",
        {},
    )

    return result

