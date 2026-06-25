"""High-level PaperTrail service: ingest a paper and a repo, then ask questions.

Wires the two ingestion pipelines, the two Voyage embedders, the two Chroma collections
(sharing one client), and the two hybrid retrievers into the agent loop. This is the
single object the CLI and the Streamlit UI talk to.
"""

from __future__ import annotations

from pathlib import Path

from paper_to_code.agent.loop import AgentResult, run_agent
from paper_to_code.config import Settings, get_settings
from paper_to_code.embeddings.code_embedder import CodeEmbedder
from paper_to_code.embeddings.paper_embedder import PaperEmbedder
from paper_to_code.ingestion.code import CodeIngestResult, ingest_code
from paper_to_code.ingestion.paper import PaperIngestResult, ingest_paper
from paper_to_code.models import Direction
from paper_to_code.retrieval.hybrid import HybridRetriever
from paper_to_code.store.vector_store import VectorStore


class PaperTrail:
    def __init__(self, settings: Settings | None = None) -> None:
        import chromadb

        self.settings = settings or get_settings()
        self.settings.ensure_dirs()
        client = chromadb.PersistentClient(path=str(self.settings.chroma_dir))

        self.paper_store = VectorStore("paper", client=client)
        self.code_store = VectorStore("code", client=client)
        self.paper_embedder = PaperEmbedder()
        self.code_embedder = CodeEmbedder()
        self.paper_retriever = HybridRetriever(self.paper_store, self.paper_embedder, rrf_k=self.settings.rrf_k)
        self.code_retriever = HybridRetriever(self.code_store, self.code_embedder, rrf_k=self.settings.rrf_k)

    def ingest_paper(self, pdf_path: str | Path, paper_id: str | None = None) -> PaperIngestResult:
        result = ingest_paper(pdf_path, paper_id)
        if result.chunks:
            embeddings = self.paper_embedder.embed_documents([c.text for c in result.chunks])
            self.paper_store.add(result.chunks, embeddings)
            self.paper_retriever.refresh()
        return result

    def ingest_code(self, source: str | Path, repo_id: str | None = None) -> CodeIngestResult:
        result = ingest_code(source, repo_id, repos_dir=self.settings.repos_dir)
        if result.chunks:
            embeddings = self.code_embedder.embed_documents([c.text for c in result.chunks])
            self.code_store.add(result.chunks, embeddings)
            self.code_retriever.refresh()
        return result

    def ask(
        self,
        question: str,
        direction: Direction | None = None,
        k: int | None = None,
        follow_refs: bool = True,
    ) -> AgentResult:
        return run_agent(
            question,
            paper_retriever=self.paper_retriever,
            code_retriever=self.code_retriever,
            settings=self.settings,
            direction=direction,
            k=k or self.settings.retrieval_top_k,
            follow_refs=follow_refs,
        )

    def counts(self) -> dict[str, int]:
        return {"paper_chunks": self.paper_store.count(), "code_chunks": self.code_store.count()}
