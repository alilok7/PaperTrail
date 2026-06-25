"""Hybrid retrieval = semantic (vector) + lexical (BM25), fused with Reciprocal Rank Fusion.

Vector search captures meaning; BM25 captures exact tokens (symbol names, equation labels).
RRF combines them by *rank* (not raw score), which avoids having to calibrate two
incomparable score scales.
"""

from __future__ import annotations

import re
from typing import Protocol

from rank_bm25 import BM25Okapi

from paper_to_code.models import Chunk, RetrievedChunk

_WORD_RE = re.compile(r"[A-Za-z0-9]+")
# Split identifiers into sub-tokens: CamelCase, PascalCase, ALLCAPS, digits.
_SUBTOKEN_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, splitting identifiers (e.g. ``CausalSelfAttention`` ->
    ``[causal, self, attention]``, ``multi_head`` -> ``[multi, head]``)."""
    tokens: list[str] = []
    for word in _WORD_RE.findall(text):
        parts = _SUBTOKEN_RE.findall(word) or [word]
        tokens.extend(p.lower() for p in parts)
    return tokens


class BM25Index:
    """In-memory BM25 over a list of chunks (rebuilt from the vector store on load)."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = list(chunks)
        self._bm25 = BM25Okapi([tokenize(c.text) for c in self.chunks]) if self.chunks else None

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self.chunks, scores), key=lambda pair: pair[1], reverse=True)
        return [(chunk, float(score)) for chunk, score in ranked[:k]]


def reciprocal_rank_fusion(
    ranked_lists: list[list], *, k: int = 60, top_n: int = 8
) -> list[RetrievedChunk]:
    """Fuse several ranked lists of chunks. Accepts lists of RetrievedChunk or (Chunk, score)."""
    scores: dict[str, float] = {}
    by_id: dict[str, Chunk] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            chunk = item.chunk if isinstance(item, RetrievedChunk) else item[0]
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (k + rank + 1)
            by_id[chunk.id] = chunk
    fused = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:top_n]
    return [RetrievedChunk(chunk=by_id[cid], score=score) for cid, score in fused]


class _Embedder(Protocol):
    def embed_query(self, text: str) -> list[float]: ...


class _Store(Protocol):
    def vector_search(self, query_embedding: list[float], k: int) -> list[RetrievedChunk]: ...
    def all_chunks(self) -> list[Chunk]: ...


class HybridRetriever:
    """Combines a vector store and a BM25 index over the same chunks."""

    def __init__(self, store: _Store, embedder: _Embedder, *, rrf_k: int = 60) -> None:
        self.store = store
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.bm25 = BM25Index(store.all_chunks())

    def refresh(self) -> None:
        """Rebuild the BM25 index from the store (call after adding chunks)."""
        self.bm25 = BM25Index(self.store.all_chunks())

    def search(self, query: str, k: int = 8) -> list[RetrievedChunk]:
        query_vec = self.embedder.embed_query(query)
        vector_hits = self.store.vector_search(query_vec, k)
        bm25_hits = self.bm25.search(query, k)
        return reciprocal_rank_fusion([vector_hits, bm25_hits], k=self.rrf_k, top_n=k)
