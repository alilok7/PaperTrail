"""Stage tests for the vector store + BM25 + RRF hybrid retrieval (no API; real Chroma)."""

from __future__ import annotations

from paper_to_code.models import Chunk
from paper_to_code.retrieval.hybrid import (
    BM25Index,
    HybridRetriever,
    reciprocal_rank_fusion,
    tokenize,
)
from paper_to_code.store.vector_store import VectorStore


def test_tokenize_splits_identifiers():
    assert tokenize("CausalSelfAttention") == ["causal", "self", "attention"]
    assert tokenize("multi_head attention") == ["multi", "head", "attention"]


def test_bm25_ranks_relevant_chunk_first():
    chunks = [
        Chunk(id="1", text="multi head self attention mechanism", source="code", metadata={}),
        Chunk(id="2", text="dataloader reads files from disk", source="code", metadata={}),
    ]
    res = BM25Index(chunks).search("attention", 2)
    assert res[0][0].id == "1"


def test_rrf_prefers_consistently_top_ranked():
    a = Chunk(id="a", text="", source="code", metadata={})
    b = Chunk(id="b", text="", source="code", metadata={})
    fused = reciprocal_rank_fusion([[(a, 1.0), (b, 0.5)], [(a, 0.9), (b, 0.4)]], top_n=2)
    assert [r.chunk.id for r in fused] == ["a", "b"]


def test_vector_store_add_search_roundtrip(tmp_path):
    vs = VectorStore("test", tmp_path / "chroma")
    chunks = [
        Chunk(id="x", text="alpha", source="code", metadata={"source": "code", "file": "a.py"}),
        Chunk(id="y", text="beta", source="code", metadata={"source": "code", "file": "b.py"}),
    ]
    vs.add(chunks, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert vs.count() == 2
    assert {c.id for c in vs.all_chunks()} == {"x", "y"}
    hits = vs.vector_search([1.0, 0.0, 0.0], k=2)
    assert hits[0].chunk.id == "x"
    assert hits[0].score > hits[1].score


class _FakeEmbedder:
    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]


def test_hybrid_retriever_combines_signals(tmp_path):
    vs = VectorStore("hybrid", tmp_path / "chroma")
    chunks = [
        Chunk(id="x", text="attention mechanism", source="code", metadata={"source": "code"}),
        Chunk(id="y", text="data loader utility", source="code", metadata={"source": "code"}),
    ]
    vs.add(chunks, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    retriever = HybridRetriever(vs, _FakeEmbedder())
    results = retriever.search("attention", k=2)
    assert "x" in [r.chunk.id for r in results]
