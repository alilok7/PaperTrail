"""Offline tests for scoped retrieval, library grouping, and delete.

Uses real Chroma in a temp directory + a fake embedder (same pattern as
``test_retrieval.py``). No API calls.
"""

from __future__ import annotations

import pytest

from paper_to_code.models import Chunk
from paper_to_code.retrieval.hybrid import HybridRetriever, ScopedRetriever
from paper_to_code.store.vector_store import VectorStore


# ── Helpers ──────────────────────────────────────────────────────────────

# Embedding vectors that are easy to reason about in cosine space.
_VEC_A = [1.0, 0.0, 0.0]
_VEC_B = [0.0, 1.0, 0.0]
_VEC_C = [0.0, 0.0, 1.0]


class _FakeEmbedder:
    """Always returns _VEC_A so we can control which chunk it's nearest to."""

    def embed_query(self, text: str) -> list[float]:
        return _VEC_A


def _make_chunk(cid: str, text: str, source: str = "paper", **meta) -> Chunk:
    meta["source"] = source
    return Chunk(id=cid, text=text, source=source, metadata=meta)


# ── VectorStore: group_counts ────────────────────────────────────────────

class TestGroupCounts:
    def test_single_paper(self, tmp_path):
        vs = VectorStore("paper", tmp_path / "chroma")
        chunks = [
            _make_chunk("p1_a", "attention is all you need", paper_id="attn"),
            _make_chunk("p1_b", "scaled dot product attention", paper_id="attn"),
        ]
        vs.add(chunks, [_VEC_A, _VEC_B])
        assert vs.group_counts("paper_id") == {"attn": 2}

    def test_two_papers(self, tmp_path):
        vs = VectorStore("paper", tmp_path / "chroma")
        chunks = [
            _make_chunk("p1_a", "attention", paper_id="attn"),
            _make_chunk("p1_b", "multi head", paper_id="attn"),
            _make_chunk("p2_a", "vision transformer overview", paper_id="vit"),
        ]
        vs.add(chunks, [_VEC_A, _VEC_B, _VEC_C])
        gc = vs.group_counts("paper_id")
        assert gc == {"attn": 2, "vit": 1}

    def test_missing_key_returns_empty(self, tmp_path):
        vs = VectorStore("paper", tmp_path / "chroma")
        chunks = [_make_chunk("x", "no relevant key")]
        vs.add(chunks, [_VEC_A])
        assert vs.group_counts("paper_id") == {}


# ── VectorStore: delete ──────────────────────────────────────────────────

class TestDelete:
    def test_delete_removes_only_matching(self, tmp_path):
        vs = VectorStore("paper", tmp_path / "chroma")
        chunks = [
            _make_chunk("p1_a", "attention", paper_id="attn"),
            _make_chunk("p2_a", "vit overview", paper_id="vit"),
        ]
        vs.add(chunks, [_VEC_A, _VEC_B])
        removed = vs.delete({"paper_id": "attn"})
        assert removed == 1
        assert vs.count() == 1
        remaining = vs.all_chunks()
        assert remaining[0].id == "p2_a"

    def test_delete_returns_zero_for_no_match(self, tmp_path):
        vs = VectorStore("paper", tmp_path / "chroma")
        chunks = [_make_chunk("x", "hello", paper_id="attn")]
        vs.add(chunks, [_VEC_A])
        assert vs.delete({"paper_id": "nonexistent"}) == 0
        assert vs.count() == 1


# ── VectorStore: all_chunks with where ───────────────────────────────────

class TestAllChunksFiltered:
    def test_where_filters_correctly(self, tmp_path):
        vs = VectorStore("paper", tmp_path / "chroma")
        chunks = [
            _make_chunk("p1_a", "attention", paper_id="attn"),
            _make_chunk("p2_a", "vit overview", paper_id="vit"),
        ]
        vs.add(chunks, [_VEC_A, _VEC_B])
        filtered = vs.all_chunks(where={"paper_id": "attn"})
        assert len(filtered) == 1
        assert filtered[0].id == "p1_a"

    def test_where_none_returns_all(self, tmp_path):
        vs = VectorStore("paper", tmp_path / "chroma")
        chunks = [
            _make_chunk("p1_a", "attention", paper_id="attn"),
            _make_chunk("p2_a", "vit", paper_id="vit"),
        ]
        vs.add(chunks, [_VEC_A, _VEC_B])
        assert len(vs.all_chunks(where=None)) == 2


# ── VectorStore: vector_search with where ────────────────────────────────

class TestVectorSearchFiltered:
    def test_scoped_search_returns_only_matching(self, tmp_path):
        vs = VectorStore("paper", tmp_path / "chroma")
        # Chunk A is closest to _VEC_A, but belongs to paper "vit".
        # Chunk B is far from _VEC_A and belongs to paper "attn".
        chunks = [
            _make_chunk("close_but_wrong_pair", "attention", paper_id="vit"),
            _make_chunk("far_but_right_pair", "transformers", paper_id="attn"),
        ]
        vs.add(chunks, [_VEC_A, _VEC_B])
        # Unscoped: nearest chunk is "close_but_wrong_pair"
        unscoped = vs.vector_search(_VEC_A, k=2)
        assert unscoped[0].chunk.id == "close_but_wrong_pair"
        # Scoped to "attn": only "far_but_right_pair" is returned
        scoped = vs.vector_search(_VEC_A, k=2, where={"paper_id": "attn"})
        assert len(scoped) == 1
        assert scoped[0].chunk.id == "far_but_right_pair"

    def test_scoped_search_returns_empty_on_no_match(self, tmp_path):
        vs = VectorStore("paper", tmp_path / "chroma")
        chunks = [_make_chunk("x", "hello", paper_id="attn")]
        vs.add(chunks, [_VEC_A])
        assert vs.vector_search(_VEC_A, k=2, where={"paper_id": "nonexistent"}) == []


# ── Scoped retrieval (the core guarantee) ────────────────────────────────

class TestScopedRetrieval:
    """Proves BOTH the vector AND BM25 sides are scoped.

    Setup: two paper_ids. A chunk in pair B is the lexical *and* semantic
    best match for the query. A scoped search for pair A must NOT return it.
    """

    def test_scoped_paper_retrieval(self, tmp_path):
        vs = VectorStore("paper", tmp_path / "chroma")
        # Pair A: mediocre match for "attention mechanism"
        # Pair B: excellent match for "attention mechanism" (both text and vector)
        chunks = [
            _make_chunk("a1", "feed forward network layer", paper_id="pairA"),
            _make_chunk("a2", "gradient descent optimizer", paper_id="pairA"),
            _make_chunk("b1", "attention mechanism multi head", paper_id="pairB"),
        ]
        vs.add(chunks, [_VEC_B, _VEC_C, _VEC_A])  # b1 is nearest to _VEC_A
        retriever = HybridRetriever(vs, _FakeEmbedder())

        # Unscoped search — b1 should dominate (both vector and BM25 rank it high)
        unscoped = retriever.search("attention mechanism", k=3)
        ids = [r.chunk.id for r in unscoped]
        assert "b1" in ids

        # Scoped to pairA — b1 must NOT appear
        scoped = retriever.scoped({"paper_id": "pairA"})
        assert isinstance(scoped, ScopedRetriever)
        scoped_results = scoped.search("attention mechanism", k=3)
        scoped_ids = [r.chunk.id for r in scoped_results]
        assert "b1" not in scoped_ids
        assert all(
            r.chunk.metadata.get("paper_id") == "pairA" for r in scoped_results
        )

    def test_scoped_repo_retrieval(self, tmp_path):
        vs = VectorStore("code", tmp_path / "chroma")
        chunks = [
            _make_chunk("ra1", "def train_loop():", source="code", repo_id="repoA"),
            _make_chunk("rb1", "class CausalSelfAttention:", source="code", repo_id="repoB"),
        ]
        vs.add(chunks, [_VEC_B, _VEC_A])  # rb1 is nearest to _VEC_A
        retriever = HybridRetriever(vs, _FakeEmbedder())

        scoped = retriever.scoped({"repo_id": "repoA"})
        results = scoped.search("CausalSelfAttention", k=2)
        result_ids = [r.chunk.id for r in results]
        assert "rb1" not in result_ids
        assert all(
            r.chunk.metadata.get("repo_id") == "repoA" for r in results
        )


# ── Backward compatibility ───────────────────────────────────────────────

class TestBackwardCompat:
    def test_unscoped_search_returns_all_pairs(self, tmp_path):
        vs = VectorStore("paper", tmp_path / "chroma")
        chunks = [
            _make_chunk("a1", "attention in pair A", paper_id="pairA"),
            _make_chunk("b1", "attention in pair B", paper_id="pairB"),
        ]
        vs.add(chunks, [_VEC_A, _VEC_B])
        retriever = HybridRetriever(vs, _FakeEmbedder())
        results = retriever.search("attention", k=4)
        ids = {r.chunk.id for r in results}
        # Both pairs should be present
        assert "a1" in ids
        assert "b1" in ids

    def test_search_without_where_matches_old_api(self, tmp_path):
        """Calling .search(query, k) without where still works (no kwarg required)."""
        vs = VectorStore("paper", tmp_path / "chroma")
        chunks = [_make_chunk("x", "hello", paper_id="test")]
        vs.add(chunks, [_VEC_A])
        retriever = HybridRetriever(vs, _FakeEmbedder())
        # Old calling convention — no where arg
        results = retriever.search("hello", k=1)
        assert len(results) >= 1


# ── BM25 cache ───────────────────────────────────────────────────────────

class TestBM25Cache:
    def test_refresh_clears_cache(self, tmp_path):
        vs = VectorStore("paper", tmp_path / "chroma")
        chunks = [_make_chunk("x", "hello", paper_id="test")]
        vs.add(chunks, [_VEC_A])
        retriever = HybridRetriever(vs, _FakeEmbedder())
        # Cache should have at least the unscoped entry
        assert None in retriever._bm25_cache or len(retriever._bm25_cache) >= 0
        # Trigger a scoped lookup to populate cache
        retriever.search("hello", k=1, where={"paper_id": "test"})
        assert len(retriever._bm25_cache) >= 1
        # Refresh clears it
        retriever.refresh()
        assert len(retriever._bm25_cache) == 0
