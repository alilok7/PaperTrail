"""Chroma-backed vector store (one collection per side).

We precompute embeddings with Voyage and pass the vectors to Chroma directly (no
``embedding_function``), because voyage-context-3 embeds chunks in document-context
groups, which Chroma's per-text embedding function can't express. Cosine space matches
Voyage's normalized embeddings.
"""

from __future__ import annotations

from pathlib import Path

from paper_to_code.models import Chunk, RetrievedChunk, SourceType


def _chunk_from_record(cid: str, document: str, metadata: dict) -> Chunk:
    source: SourceType = metadata.get("source", "paper")  # type: ignore[assignment]
    return Chunk(id=cid, text=document, source=source, metadata=metadata)


class VectorStore:
    """A single persistent Chroma collection of precomputed embeddings."""

    def __init__(self, name: str, persist_dir: str | Path | None = None, *, client=None) -> None:
        import chromadb

        if client is None:
            if persist_dir is None:
                raise ValueError("Provide either persist_dir or an existing client.")
            Path(persist_dir).mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(persist_dir))
        self.client = client
        self.collection = self.client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must be the same length")
        self.collection.upsert(
            ids=[c.id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            # Chroma rejects empty metadata dicts; always keep at least the source.
            metadatas=[dict(c.metadata) or {"source": c.source} for c in chunks],
        )

    def count(self) -> int:
        return self.collection.count()

    def all_chunks(self) -> list[Chunk]:
        got = self.collection.get(include=["documents", "metadatas"])
        return [
            _chunk_from_record(cid, doc, meta)
            for cid, doc, meta in zip(got["ids"], got["documents"], got["metadatas"])
        ]

    def vector_search(self, query_embedding: list[float], k: int) -> list[RetrievedChunk]:
        if self.collection.count() == 0:
            return []
        res = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        out: list[RetrievedChunk] = []
        for cid, doc, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            chunk = _chunk_from_record(cid, doc, meta)
            out.append(RetrievedChunk(chunk=chunk, score=1.0 - float(dist)))  # cosine sim
        return out
