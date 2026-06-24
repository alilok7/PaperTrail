"""Stage tests for paper-ingestion metadata helpers (no Docling / PDF needed)."""

from __future__ import annotations

from paper_to_code.ingestion.paper import (
    _EQUATION_RE,
    _FIGURE_RE,
    _SECTION_RE,
    _find_labels,
    _page_from_doc_items,
    chunk_metadata,
)


class _Prov:
    def __init__(self, page_no: int) -> None:
        self.page_no = page_no


class _Item:
    def __init__(self, pages: list[int]) -> None:
        self.prov = [_Prov(p) for p in pages]


def test_find_equation_labels():
    text = "As shown in Equation 3 and Eq. 4, and again Eq 5."
    assert _find_labels(_EQUATION_RE, text) == ["Eq 5", "Eq. 4", "Equation 3"]


def test_find_figure_and_section_labels():
    assert _find_labels(_FIGURE_RE, "See Figure 2 and Table 1.") == ["Figure 2", "Table 1"]
    assert _find_labels(_SECTION_RE, "described in Section 4.2") == ["Section 4.2"]


def test_page_from_doc_items_takes_minimum():
    assert _page_from_doc_items([_Item([5, 6]), _Item([4])]) == 4
    assert _page_from_doc_items([]) is None


def test_chunk_metadata_is_scalar_only():
    meta = chunk_metadata(
        "attention",
        2,
        "See Figure 2 and Section 3.1 and Equation 1",
        ["3 Model Architecture", "3.1 Encoder"],
        3,
    )
    assert meta["paper_id"] == "attention"
    assert meta["source"] == "paper"
    assert meta["section"] == "3.1 Encoder"
    assert meta["headings"] == "3 Model Architecture > 3.1 Encoder"
    assert meta["page"] == 3
    assert "Figure 2" in meta["figures"]
    assert "Section 3.1" in meta["section_refs"]
    assert "Equation 1" in meta["equations"]
    # Chroma only accepts scalar metadata values.
    assert all(isinstance(v, (str, int, float, bool)) for v in meta.values())


def test_chunk_metadata_handles_no_headings():
    meta = chunk_metadata("p", 0, "plain text", [], None)
    assert meta["section"] == ""
    assert meta["headings"] == ""
    assert meta["page"] == -1
