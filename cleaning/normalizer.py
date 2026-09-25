from __future__ import annotations

import copy
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Any


@dataclass
class CleaningStats:
    input_text_fields: int = 0
    changed_text_fields: int = 0
    removed_empty_items: int = 0
    duplicate_paragraphs_removed: int = 0
    numeric_guard_failures: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "input_text_fields": self.input_text_fields,
            "changed_text_fields": self.changed_text_fields,
            "removed_empty_items": self.removed_empty_items,
            "duplicate_paragraphs_removed": self.duplicate_paragraphs_removed,
            "numeric_guard_failures": self.numeric_guard_failures,
        }


class CanonicalNormalizer:
    """Phase 6 normalization. No logical reconstruction or semantic rewriting."""

    VERSION = "phase6-normalizer-v1"

    def clean(
            self,
            canonical: dict[str, Any],
            input_file_hash: str | None = None,
        ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not isinstance(canonical, dict):
            raise TypeError("canonical must be a dict")

        result = copy.deepcopy(canonical)
        stats = CleaningStats()
        input_hash = input_file_hash

        if isinstance(result.get("metadata"), dict):
            result["metadata"] = self._clean_metadata(result["metadata"], stats)

        for key in ("abstract",):
            if isinstance(result.get(key), str):
                result[key] = self._normalize_text(result[key], stats)

        self._clean_sections(result.get("sections"), stats)
        result["paragraphs"] = self._clean_paragraphs(result.get("paragraphs"), stats)
        self._clean_tables(result.get("tables"), stats)
        self._clean_figures(result.get("figures"), stats)
        self._clean_formulas(result.get("formulas"), stats)
        result["references"] = self._clean_references(result.get("references"), stats)

        report = {
            "version": self.VERSION,
            "input_file_hash": input_hash,
            "transformations": [
                "unicode_nfkc",
                "soft_hyphen_removal",
                "zero_width_character_removal",
                "pdf_line_break_hyphen_repair",
                "newline_tab_normalization",
                "whitespace_normalization",
                "safe_punctuation_spacing_normalization",
                "empty_paragraph_removal",
                "exact_duplicate_paragraph_removal",
            ],
            "stats": stats.as_dict(),
            "notes": [
                "Logical IDs and provenance are preserved.",
                "Formula text is not passed through generic prose cleanup.",
                "No missing structure or content is invented.",
            ],
        }
        result["cleaning"] = report
        return result, report

    def _clean_metadata(self, metadata: dict[str, Any], stats: CleaningStats) -> dict[str, Any]:
        metadata = copy.deepcopy(metadata)
        for key in ("title", "subject", "keywords"):
            if isinstance(metadata.get(key), str):
                metadata[key] = self._normalize_text(metadata[key], stats)
        if isinstance(metadata.get("authors"), list):
            metadata["authors"] = [
                self._normalize_text(x, stats) if isinstance(x, str) else x
                for x in metadata["authors"]
            ]
        return metadata

    def _clean_sections(self, sections: Any, stats: CleaningStats) -> None:
        if not isinstance(sections, list):
            return
        for section in sections:
            if not isinstance(section, dict):
                continue
            if isinstance(section.get("heading"), str):
                section["heading"] = self._normalize_text(section["heading"], stats)
            if isinstance(section.get("path"), list):
                section["path"] = [
                    self._normalize_text(x, stats) if isinstance(x, str) else x
                    for x in section["path"]
                ]

    def _clean_paragraphs(self, paragraphs: Any, stats: CleaningStats) -> list[dict[str, Any]]:
        if not isinstance(paragraphs, list):
            return []

        output = []
        seen: set[tuple[str, Any, str]] = set()

        for paragraph in paragraphs:
            if not isinstance(paragraph, dict):
                continue

            if isinstance(paragraph.get("text"), str):
                text = self._normalize_text(paragraph["text"], stats)
                paragraph["text"] = text

                if isinstance(paragraph.get("sentences"), list):
                    normalized_sentences = []

                    for sentence in paragraph["sentences"]:
                        if not isinstance(sentence, str):
                            continue

                        normalized_sentence = self._normalize_text(sentence, stats)

                        if normalized_sentence:
                            normalized_sentences.append(normalized_sentence)

                    paragraph["sentences"] = normalized_sentences
                if not text:
                    stats.removed_empty_items += 1
                    continue

                key = (text, paragraph.get("page"), str(paragraph.get("section") or ""))
                if key in seen:
                    stats.duplicate_paragraphs_removed += 1
                    continue
                seen.add(key)

            output.append(paragraph)

        return output

    def _clean_tables(self, tables: Any, stats: CleaningStats) -> None:
        if not isinstance(tables, list):
            return
        for table in tables:
            if not isinstance(table, dict):
                continue
            for key in ("label", "caption", "content"):
                if isinstance(table.get(key), str):
                    table[key] = self._normalize_text(table[key], stats)
            for key in ("structured_data", "structure"):
                if isinstance(table.get(key), dict):
                    self._clean_nested_structure(table[key], stats)

    def _clean_nested_structure(self, obj: dict[str, Any], stats: CleaningStats) -> None:
        for key in ("grid", "rows", "table_cells"):
            value = obj.get(key)
            if not isinstance(value, list):
                continue
            self._clean_nested(value, stats)

    def _clean_nested(self, value: list[Any], stats: CleaningStats) -> None:
        for item in value:
            if isinstance(item, list):
                self._clean_nested(item, stats)
            elif isinstance(item, dict):
                for key in ("text", "label"):
                    if isinstance(item.get(key), str):
                        item[key] = self._normalize_text(item[key], stats)

    def _clean_figures(self, figures: Any, stats: CleaningStats) -> None:
        if not isinstance(figures, list):
            return
        for figure in figures:
            if not isinstance(figure, dict):
                continue
            for key in ("label", "caption"):
                if isinstance(figure.get(key), str):
                    figure[key] = self._normalize_text(figure[key], stats)

    def _clean_formulas(self, formulas: Any, stats: CleaningStats) -> None:
        if not isinstance(formulas, list):
            return
        # Only metadata/IDs are normalized. Formula semantics are untouched.
        for formula in formulas:
            if not isinstance(formula, dict):
                continue
            for key in ("formula_id", "equation_number", "source"):
                if isinstance(formula.get(key), str):
                    formula[key] = self._normalize_text(formula[key], stats)

    def _clean_references(self, references: Any, stats: CleaningStats) -> list[dict[str, Any]]:
        if not isinstance(references, list):
            return []
        output = []
        for ref in references:
            if not isinstance(ref, dict):
                continue
            for key in ("title", "raw"):
                if isinstance(ref.get(key), str):
                    ref[key] = self._normalize_text(ref[key], stats)
            if isinstance(ref.get("authors"), list):
                ref["authors"] = [
                    self._normalize_text(x, stats) if isinstance(x, str) else x
                    for x in ref["authors"]
                ]
            output.append(ref)
        return output

    @staticmethod
    def _numeric_tokens(text: str) -> Counter[str]:
        return Counter(re.findall(r"(?<![\w])(?:\d+(?:\.\d+)?%?)(?![\w])", text))

    def _normalize_text(self, value: str, stats: CleaningStats) -> str:
        stats.input_text_fields += 1
        original = value
        text = unicodedata.normalize("NFKC", value)
        text = text.replace("\u00ad", "").replace("\u00a0", " ")
        for char in ("\u200b", "\u200c", "\u200d", "\ufeff"):
            text = text.replace(char, "")

        # Only repair a hyphen when it is actually splitting a word at a line break.
        text = re.sub(r"(?<=[A-Za-z0-9])-\s*\n\s*(?=[A-Za-z0-9])", "", text)
        text = re.sub(r"[\r\n\t]+", " ", text)
        text = re.sub(r" {2,}", " ", text).strip()
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)

        if text != original:
            stats.changed_text_fields += 1
            if self._numeric_tokens(original) != self._numeric_tokens(text):
                stats.numeric_guard_failures += 1
                return original

        return text
