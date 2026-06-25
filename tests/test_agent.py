"""Stage tests for reference-following and the agent-loop orchestration (no API)."""

from __future__ import annotations

from paper_to_code.config import Settings
from paper_to_code.models import Chunk, Direction, RetrievedChunk, Verdict, VerdictType
from paper_to_code.retrieval.reference import collect_reference_labels, follow_references


def _rc(cid: str, text: str = "", source: str = "paper", **meta) -> RetrievedChunk:
    return RetrievedChunk(chunk=Chunk(id=cid, text=text, source=source, metadata=meta), score=1.0)


class _FakeRetriever:
    def __init__(self, results: list[RetrievedChunk]) -> None:
        self.results = results
        self.queries: list[str] = []

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        self.queries.append(query)
        return self.results[:k]


def test_collect_reference_labels_dedupes():
    hits = [
        _rc("p1", section_refs="Section 4.2, Section 3.1", equations="Eq 3"),
        _rc("p2", figures="Figure 2"),
    ]
    labels = collect_reference_labels(hits)
    assert "Section 4.2" in labels and "Eq 3" in labels and "Figure 2" in labels


def test_follow_references_adds_new_chunks():
    hits = [_rc("p1", section_refs="Section 4.2")]
    ref_retriever = _FakeRetriever([_rc("p9", "the encoder section")])
    augmented = follow_references(hits, ref_retriever, max_refs=1)
    assert [h.chunk.id for h in augmented] == ["p1", "p9"]
    assert ref_retriever.queries == ["Section 4.2"]


def test_follow_references_skips_duplicates():
    hits = [_rc("p1", section_refs="Section 4.2")]
    ref_retriever = _FakeRetriever([_rc("p1")])  # returns an already-present chunk
    assert [h.chunk.id for h in follow_references(hits, ref_retriever)] == ["p1"]


def test_run_agent_paper_to_code_flow(monkeypatch):
    from paper_to_code.agent import loop

    paper_hit = _rc("p1", "scaled dot product attention", source="paper", section="3.2")
    code_hit = _rc("c1", "class CausalSelfAttention", source="code", file="model.py", symbol="CausalSelfAttention")
    paper_r = _FakeRetriever([paper_hit])
    code_r = _FakeRetriever([code_hit])

    monkeypatch.setattr(loop, "follow_references", lambda hits, r, **kw: hits)
    monkeypatch.setattr(loop, "reformulate", lambda *a, **kw: "attention head qkv softmax")
    monkeypatch.setattr(
        loop,
        "reconcile",
        lambda q, d, ph, ch, s=None: Verdict(
            verdict=VerdictType.MATCH, confidence=0.9, direction=d, explanation="match"
        ),
    )

    result = loop.run_agent(
        "How is multi-head attention implemented?",
        paper_retriever=paper_r,
        code_retriever=code_r,
        settings=Settings(_env_file=None),
        direction=Direction.PAPER_TO_CODE,
    )

    assert result.direction == Direction.PAPER_TO_CODE
    assert result.verdict.verdict == VerdictType.MATCH
    assert result.reformulated_query == "attention head qkv softmax"
    # The reformulated query is what gets sent to the code side.
    assert code_r.queries == ["attention head qkv softmax"]
    assert result.code_evidence[0].chunk.id == "c1"
