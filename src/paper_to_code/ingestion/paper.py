"""Paper ingestion: PDF -> Docling -> section/page-aware chunks.

Docling parses the PDF (layout-aware) and the HybridChunker produces token-sized,
heading-aware chunks. For each chunk we capture metadata used for both citations and
cross-reference following:

- ``section`` / ``headings``: the heading path the chunk lives under.
- ``page``: the first page the chunk appears on (from Docling provenance).
- ``equations`` / ``figures`` / ``section_refs``: cross-reference labels mentioned in
  the text (e.g. "Equation 3", "Figure 2", "Section 4.2"), which drive reference-following.

Docling is imported lazily inside :func:`ingest_paper` so importing this module (and
unit-testing the pure metadata helpers) does not require the heavy Docling stack.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from paper_to_code.models import Chunk

if TYPE_CHECKING:  # pragma: no cover - typing only
    from docling.chunking import HybridChunker
    from docling.document_converter import DocumentConverter

# Cross-reference labels surfaced from chunk text.
_EQUATION_RE = re.compile(r"\bEq(?:uation|\.)?\s*\(?(\d+(?:\.\d+)?)\)?", re.IGNORECASE)
_FIGURE_RE = re.compile(r"\b(?:Fig(?:ure|\.)?|Table)\s*(\d+(?:\.\d+)?)", re.IGNORECASE)
_SECTION_RE = re.compile(r"\bSection\s*(\d+(?:\.\d+)*)", re.IGNORECASE)


def _find_labels(pattern: re.Pattern[str], text: str) -> list[str]:
    """Return the unique, sorted set of label strings matched by ``pattern`` in ``text``."""
    return sorted({m.group(0).strip() for m in pattern.finditer(text)})


def _page_from_doc_items(doc_items: Iterable[Any]) -> int | None:
    """Smallest page number across a chunk's doc items' provenance, or None."""
    pages = [prov.page_no for item in doc_items for prov in getattr(item, "prov", [])]
    return min(pages) if pages else None


def chunk_metadata(
    paper_id: str, index: int, text: str, headings: list[str], page: int | None
) -> dict[str, str | int | float | bool]:
    """Build Chroma-compatible (scalar-only) metadata for a paper chunk."""
    section = headings[-1] if headings else ""
    return {
        "paper_id": paper_id,
        "source": "paper",
        "chunk_index": index,
        "section": section,
        "headings": " > ".join(headings),
        "page": page if page is not None else -1,
        "equations": ", ".join(_find_labels(_EQUATION_RE, text)),
        "figures": ", ".join(_find_labels(_FIGURE_RE, text)),
        "section_refs": ", ".join(_find_labels(_SECTION_RE, text)),
    }


@dataclass
class PaperIngestResult:
    paper_id: str
    chunks: list[Chunk]


def ingest_paper(
    pdf_path: str | Path,
    paper_id: str | None = None,
    *,
    converter: "DocumentConverter | None" = None,
    chunker: "HybridChunker | None" = None,
) -> PaperIngestResult:
    """Parse and chunk a paper PDF into a list of paper :class:`Chunk` objects.

    First conversion downloads Docling's layout models (a one-time cost).
    """
    from docling.chunking import HybridChunker
    from docling.document_converter import DocumentConverter

    pdf_path = Path(pdf_path)
    paper_id = paper_id or pdf_path.stem
    converter = converter or DocumentConverter()
    chunker = chunker or HybridChunker()

    document = converter.convert(str(pdf_path)).document
    chunks: list[Chunk] = []
    for i, dc in enumerate(chunker.chunk(dl_doc=document)):
        headings = list(dc.meta.headings or [])
        page = _page_from_doc_items(dc.meta.doc_items)
        metadata = chunk_metadata(paper_id, i, dc.text, headings, page)
        chunks.append(Chunk(id=f"{paper_id}::p{i}", text=dc.text, source="paper", metadata=metadata))

    return PaperIngestResult(paper_id=paper_id, chunks=chunks)
