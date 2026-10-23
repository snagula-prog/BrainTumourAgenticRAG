"""Exhaustively check requested metric labels in a paper's indexed Chroma text.

This tool reads the text payloads already stored in the existing Chroma collection.
It does not embed a query, run similarity top-k, read canonical/cleaned/chunk JSON files,
or create another index. A negative result is only conclusive for the indexed segments
when their coverage/integrity checks pass; it is never proof that the original PDF omits
that metric.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from typing import Any


FieldMatcher = Callable[[str, str], bool]
MetricPatternFactory = Callable[[str], re.Pattern[str] | None]
FieldTermsFactory = Callable[[str], list[str]]
PaperTitleResolver = Callable[[str], str | None]


class DocumentMetricCheckTool:
    """Scan every indexed text segment for metrics within explicitly named papers."""

    name = "check_document_metrics"
    description = (
        "Exhaustively scan all stored Chroma text segments for requested metric labels "
        "within named papers. Reports indexed-coverage status and returns matched segments "
        "as evidence; absence claims must be qualified to the indexed text."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "paper_ids": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "minItems": 1,
                "description": "Explicitly scoped paper IDs to scan.",
            },
            "fields": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "minItems": 1,
                "description": "Requested metric labels/field names.",
            },
        },
        "required": ["paper_ids", "fields"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        retrieval_tool: Any,
        *,
        field_matcher: FieldMatcher,
        metric_pattern_factory: MetricPatternFactory,
        field_terms_factory: FieldTermsFactory,
        paper_title_resolver: PaperTitleResolver | None = None,
        max_evidence_per_paper: int = 12,
    ) -> None:
        if not callable(getattr(retrieval_tool, "get_indexed_segments", None)):
            # Construction remains possible for test doubles or an older adapter. The
            # execution result will explicitly say that a full scan was not performed.
            self._retrieval_tool = retrieval_tool
        else:
            self._retrieval_tool = retrieval_tool
        if max_evidence_per_paper < 1:
            raise ValueError("max_evidence_per_paper must be positive")
        self._field_matcher = field_matcher
        self._metric_pattern_factory = metric_pattern_factory
        self._field_terms_factory = field_terms_factory
        self._paper_title_resolver = paper_title_resolver or (lambda _paper_id: None)
        self._max_evidence_per_paper = max_evidence_per_paper

    @staticmethod
    def _validate_arguments(arguments: Any) -> tuple[list[str], list[str]]:
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be an object")
        unexpected = set(arguments) - {"paper_ids", "fields"}
        if unexpected:
            raise ValueError(f"Unexpected arguments: {', '.join(sorted(unexpected))}")

        paper_ids = arguments.get("paper_ids")
        fields = arguments.get("fields")
        if not isinstance(paper_ids, list) or not paper_ids:
            raise ValueError("paper_ids must be a non-empty list")
        if not isinstance(fields, list) or not fields:
            raise ValueError("fields must be a non-empty list")
        if any(not isinstance(value, str) or not value.strip() for value in paper_ids):
            raise ValueError("Every paper_id must be a non-empty string")
        if any(not isinstance(value, str) or not value.strip() for value in fields):
            raise ValueError("Every field must be a non-empty string")
        return (
            list(dict.fromkeys(value.strip() for value in paper_ids)),
            list(dict.fromkeys(value.strip() for value in fields)),
        )

    @staticmethod
    def _availability_id(paper_id: str, field: str) -> str:
        suffix = hashlib.sha256(f"{paper_id}|{field.casefold()}".encode("utf-8")).hexdigest()[:16]
        return f"av_{suffix}"

    @staticmethod
    def _source_metadata(segment: Mapping[str, Any]) -> dict[str, Any]:
        metadata = segment.get("metadata")
        if not isinstance(metadata, Mapping):
            return {}
        raw = metadata.get("source_metadata_json")
        if isinstance(raw, str) and raw.strip():
            try:
                decoded = json.loads(raw)
                if isinstance(decoded, dict):
                    return decoded
            except json.JSONDecodeError:
                pass
        # Accept normalized records from test doubles and future adapter versions.
        source_metadata = metadata.get("source_metadata")
        return dict(source_metadata) if isinstance(source_metadata, Mapping) else {}

    @classmethod
    def _page(cls, segment: Mapping[str, Any]) -> Any:
        source = cls._source_metadata(segment)
        metadata = segment.get("metadata")
        candidates = [
            source.get(key)
            for key in ("page", "page_number", "page_start", "start_page")
        ]
        if isinstance(metadata, Mapping):
            candidates.extend(
                metadata.get(key)
                for key in ("page", "page_number", "page_start", "start_page")
            )
        for value in candidates:
            if value is not None:
                return value
        pages = source.get("pages")
        return pages[0] if isinstance(pages, list) and pages else None

    def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        paper_ids, fields = self._validate_arguments(arguments)
        get_segments = getattr(self._retrieval_tool, "get_indexed_segments", None)
        paper_results: list[dict[str, Any]] = []
        appended_evidence: dict[str, dict[str, Any]] = {}

        for paper_id in paper_ids:
            if not callable(get_segments):
                scan = {
                    "paper_id": paper_id,
                    "segments": [],
                    "coverage": {
                        "status": "not_checked",
                        "expected_segment_count": None,
                        "segments_scanned": 0,
                        "unique_parent_chunks_scanned": 0,
                        "issues": ["RetrievalTool does not expose get_indexed_segments."],
                    },
                }
            else:
                try:
                    scan = get_segments(paper_id)
                    if not isinstance(scan, Mapping):
                        raise TypeError("get_indexed_segments returned a non-object")
                except Exception as exc:
                    scan = {
                        "paper_id": paper_id,
                        "segments": [],
                        "coverage": {
                            "status": "incomplete_indexed_coverage",
                            "expected_segment_count": None,
                            "segments_scanned": 0,
                            "unique_parent_chunks_scanned": 0,
                            "issues": [f"Index scan failed: {type(exc).__name__}: {str(exc)[:240]}"],
                        },
                    }

            segments = scan.get("segments", [])
            if not isinstance(segments, list):
                segments = []
            raw_coverage = scan.get("coverage", {})
            coverage = dict(raw_coverage) if isinstance(raw_coverage, Mapping) else {
                "status": "incomplete_indexed_coverage",
                "expected_segment_count": None,
                "segments_scanned": len(segments),
                "unique_parent_chunks_scanned": len({str(s.get("chunk_id")) for s in segments if isinstance(s, Mapping)}),
                "issues": ["The retrieval adapter returned malformed coverage metadata."],
            }
            coverage_status = str(coverage.get("status", "incomplete_indexed_coverage"))
            title = self._paper_title_resolver(paper_id)

            by_field: dict[str, dict[str, Any]] = {}
            ranked_matches: list[tuple[int, int, str, Mapping[str, Any]]] = []
            for field in fields:
                pattern = self._metric_pattern_factory(field)
                mention_matches: list[Mapping[str, Any]] = []
                numeric_matches: list[Mapping[str, Any]] = []
                terms_found: set[str] = set()
                for segment in segments:
                    if not isinstance(segment, Mapping):
                        continue
                    text = segment.get("text")
                    segment_id = segment.get("segment_id")
                    if not isinstance(text, str) or not text.strip() or not isinstance(segment_id, str):
                        continue
                    if self._field_matcher(text, field):
                        mention_matches.append(segment)
                        normalized_text = " " + re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip() + " "
                        for term in self._field_terms_factory(field):
                            normalized_term = " " + re.sub(r"[^a-z0-9]+", " ", term.casefold()).strip() + " "
                            if normalized_term.strip() and normalized_term in normalized_text:
                                terms_found.add(term)
                    if pattern is not None and pattern.search(text):
                        numeric_matches.append(segment)

                if coverage_status != "complete_indexed_coverage":
                    status = "unknown_incomplete_index"
                elif numeric_matches:
                    status = "numeric_value_evidence_found"
                elif mention_matches:
                    status = "metric_mentioned_no_explicit_value_found"
                else:
                    status = "not_found_in_indexed_text"

                by_field[field] = {
                    "availability_id": self._availability_id(paper_id, field),
                    "field": field,
                    "status": status,
                    "coverage_status": coverage_status,
                    "mention_match_count": len(mention_matches),
                    "numeric_value_match_count": len(numeric_matches),
                    "matched_terms": sorted(terms_found, key=str.casefold),
                    "matched_evidence_ids": [],
                }
                # Prefer directly labelled number/value passages. If no explicit
                # numeric pair exists, retain a small sample of term-mention passages.
                selected = numeric_matches or mention_matches
                for segment in selected:
                    segment_id = str(segment.get("segment_id"))
                    chunk_id = str(segment.get("chunk_id") or segment_id)
                    priority = 0 if segment in numeric_matches else 1
                    ranked_matches.append((priority, len(str(segment.get("text", ""))), segment_id, segment))

            # Rank direct label/value matches first, then keep one evidence record per
            # Chroma segment. The scan still counts all rows even though the evidence
            # ledger only retains the most useful matching passages.
            ranked_matches.sort(key=lambda item: (item[0], -item[1], item[2]))
            unique_ranked: list[Mapping[str, Any]] = []
            seen_segments: set[str] = set()
            for _priority, _length, segment_id, segment in ranked_matches:
                if segment_id in seen_segments:
                    continue
                seen_segments.add(segment_id)
                unique_ranked.append(segment)
            for segment in unique_ranked[: self._max_evidence_per_paper]:
                segment_id = str(segment.get("segment_id"))
                chunk_id = str(segment.get("chunk_id") or segment_id)
                evidence_id = "ev_indexseg_" + hashlib.sha256(
                    f"{paper_id}|{segment_id}".encode("utf-8")
                ).hexdigest()[:16]
                source_metadata = self._source_metadata(segment)
                appended_evidence[evidence_id] = {
                    "evidence_id": evidence_id,
                    "paper_id": paper_id,
                    "paper_title": title,
                    "chunk_id": chunk_id,
                    "segment_id": segment_id,
                    "page": self._page(segment),
                    "text": str(segment.get("text", "")).strip(),
                    "metadata": source_metadata,
                    "source": "chroma_index_scan",
                }

            for field, field_result in by_field.items():
                pattern = self._metric_pattern_factory(field)
                candidate_segments = []
                for segment in segments:
                    if not isinstance(segment, Mapping):
                        continue
                    text = segment.get("text")
                    segment_id = segment.get("segment_id")
                    if not isinstance(text, str) or not isinstance(segment_id, str):
                        continue
                    is_numeric = bool(pattern and pattern.search(text))
                    if is_numeric or self._field_matcher(text, field):
                        candidate_segments.append(segment)
                matched_ids = []
                for segment in candidate_segments:
                    segment_id = str(segment.get("segment_id"))
                    evidence_id = "ev_indexseg_" + hashlib.sha256(
                        f"{paper_id}|{segment_id}".encode("utf-8")
                    ).hexdigest()[:16]
                    if evidence_id in appended_evidence:
                        matched_ids.append(evidence_id)
                field_result["matched_evidence_ids"] = matched_ids[: self._max_evidence_per_paper]

            paper_results.append({
                "paper_id": paper_id,
                "paper_title": title,
                "coverage": coverage,
                "fields": by_field,
            })

        coverage_values = [str(item.get("coverage", {}).get("status", "unknown")) for item in paper_results]
        if coverage_values and all(value == "complete_indexed_coverage" for value in coverage_values):
            overall_status = "complete_indexed_coverage"
        elif coverage_values and all(value == "not_checked" for value in coverage_values):
            overall_status = "not_checked"
        else:
            overall_status = "incomplete_indexed_coverage"

        availability = {
            "status": overall_status,
            "scope": "indexed_chroma_segments_only",
            "fields_requested": fields,
            "paper_ids": paper_ids,
            "paper_results": paper_results,
            "limitation": (
                "This scan covers stored Chroma text segments only. It does not prove that "
                "the original PDF contains or omits a metric if content was lost before indexing."
            ),
        }
        return {
            "document_metric_availability": availability,
            "evidence": list(appended_evidence.values()),
        }
