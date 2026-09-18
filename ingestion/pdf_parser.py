"""
PDF parsing via PyMuPDF, with reading-order correction and line-level
font metadata.

Three things this handles, based on real problems found in our actual
papers via layout_diagnostics.py and debug_blocks.py:

1. READING ORDER: blocks are sorted by (column, vertical position) so
   two-column pages don't get scrambled.

2. LINE-LEVEL GRANULARITY (not block-level): PyMuPDF groups text into
   "blocks" based on whitespace gaps, and some journal templates (e.g.
   Nature/Scientific Reports) place a heading directly above its
   paragraph with no gap — so heading + paragraph become ONE block.
   If we only inspect whole blocks, headings glued to body text are
   invisible to any heading detector. So parse_pdf exposes a flat,
   ordered list of individual LINES (each with its own font size),
   in addition to blocks, specifically so section_detector.py can
   examine one line at a time.

3. FONT SIZE per line (not just per block) — needed for the same
   reason: a heading line can have a different size than the body
   line sitting directly beneath it in the same block.
"""
import fitz  # provided by the pymupdf package
from pathlib import Path
from statistics import median
import re
import unicodedata

BOLD_FLAG = 1 << 4  # PyMuPDF span flags bit for bold


def _normalize_text(text: str) -> str:
    """
    Fix invisible-character issues found in real papers before anyone
    downstream (section detection, chunking, BM25, citations) ever
    sees this text.

    NFKC ("compatibility composition") folds visually-identical but
    distinct Unicode characters to a canonical form — critically for
    us, it turns non-breaking spaces, thin spaces, and em-spaces (all
    of which publishers use inside phrases like "Results and<NBSP>
    discussion" so they don't line-wrap awkwardly) into normal spaces.
    Without this, "Results and\xa0discussion" silently fails to match
    "results and discussion" in any plain string comparison.

    We also collapse any remaining runs of whitespace to a single
    space, since multiple space-like characters can still appear
    adjacent after normalization.
    """
    normalized = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", normalized).strip()


def _sort_entries_reading_order(entries: list[dict], page_width: float) -> list[dict]:
    """Left column (top-to-bottom) before right column (top-to-bottom).
    Works on anything with a 'bbox' key."""
    midpoint = page_width / 2

    def column_key(entry):
        x0 = entry["bbox"][0]
        column = 0 if x0 < midpoint else 1
        return (column, entry["bbox"][1])

    return sorted(entries, key=column_key)


def _extract_page_blocks_and_lines(page) -> tuple[list[dict], list[dict]]:
    """
    Returns (blocks, flat_lines), both already in reading order.

    blocks: [{"text": full block text (joined lines), "bbox": ...,
              "font_size": max size in block, "is_bold": bool}]
    flat_lines: [{"text": ..., "font_size": ..., "is_bold": ...}]
                one entry per line, in reading order across the WHOLE
                page (block order, then line order within each block)
    """
    raw = page.get_text("dict")
    block_entries = []

    for block in raw.get("blocks", []):
        if block.get("type") != 0:  # 0 = text block; skip images
            continue

        lines_for_block = []
        for line in block.get("lines", []):
            span_texts = []
            max_size = 0.0
            is_bold = False
            for span in line.get("spans", []):
                text = span.get("text", "")
                if not text.strip():
                    continue
                span_texts.append(text)
                size = span.get("size", 0.0)
                if size > max_size:
                    max_size = size
                if span.get("flags", 0) & BOLD_FLAG:
                    is_bold = True
            line_text = "".join(span_texts).strip()
            line_text = _normalize_text(line_text)
            if line_text:
                lines_for_block.append({
                    "text": line_text,
                    "font_size": round(max_size, 1),
                    "is_bold": is_bold,
                })

        if lines_for_block:
            block_entries.append({"bbox": block["bbox"], "lines": lines_for_block})

    # Sort blocks into reading order first (column, then vertical position)
    ordered_block_entries = _sort_entries_reading_order(block_entries, page.rect.width)

    blocks = []
    flat_lines = []
    for entry in ordered_block_entries:
        block_text = "\n".join(l["text"] for l in entry["lines"])
        block_font_size = max(l["font_size"] for l in entry["lines"])
        block_is_bold = any(l["is_bold"] for l in entry["lines"])

        blocks.append({
            "text": block_text,
            "bbox": entry["bbox"],
            "font_size": block_font_size,
            "is_bold": block_is_bold,
        })
        flat_lines.extend(entry["lines"])  # preserves reading order

    return blocks, flat_lines


def parse_pdf(file_path: Path) -> dict:
    """
    Returns:
        {
            "num_pages": int,
            "doc_metadata": {...},
            "body_font_size": float,   # median line font size across
                                        # the doc; this paper's own
                                        # baseline for "normal" text
            "pages": [
                {
                    "page_number": 1,
                    "text": "...",        # reading-order-corrected
                    "blocks": [ {text, bbox, font_size, is_bold}, ... ],
                    "lines": [ {text, font_size, is_bold}, ... ],  # NEW
                },
                ...
            ]
        }
    """
    doc = fitz.open(file_path)

    pages = []
    all_line_font_sizes = []

    for i, page in enumerate(doc):
        blocks, flat_lines = _extract_page_blocks_and_lines(page)
        page_text = "\n\n".join(b["text"] for b in blocks)
        all_line_font_sizes.extend(l["font_size"] for l in flat_lines)

        pages.append({
            "page_number": i + 1,
            "text": page_text,
            "blocks": blocks,
            "lines": flat_lines,
        })

    doc_metadata = doc.metadata
    num_pages = doc.page_count
    doc.close()

    body_font_size = median(all_line_font_sizes) if all_line_font_sizes else 10.0

    return {
        "num_pages": num_pages,
        "doc_metadata": doc_metadata,
        "pages": pages,
        "body_font_size": body_font_size,
    }