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
from paper_to_code.progress import ProgressFn, report
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

    def ingest_paper(
        self,
        pdf_path: str | Path,
        paper_id: str | None = None,
        *,
        on_progress: ProgressFn | None = None,
    ) -> PaperIngestResult:
        result = ingest_paper(pdf_path, paper_id, on_progress=on_progress)
        if result.chunks:
            embeddings = self.paper_embedder.embed_documents(
                [c.text for c in result.chunks], on_progress=on_progress
            )
            report(on_progress, "Storing in the vector index …", None)
            self.paper_store.add(result.chunks, embeddings)
            self.paper_retriever.refresh()
            report(on_progress, f"Indexed {len(result.chunks)} paper chunks.", 1.0)
        return result

    def ingest_code(
        self,
        source: str | Path,
        repo_id: str | None = None,
        *,
        on_progress: ProgressFn | None = None,
    ) -> CodeIngestResult:
        result = ingest_code(
            source, repo_id, repos_dir=self.settings.repos_dir, on_progress=on_progress
        )
        if result.chunks:
            embeddings = self.code_embedder.embed_documents(
                [c.text for c in result.chunks], on_progress=on_progress
            )
            report(on_progress, "Storing in the vector index …", None)
            self.code_store.add(result.chunks, embeddings)
            self.code_retriever.refresh()
            report(on_progress, f"Indexed {len(result.chunks)} code chunks.", 1.0)
        return result

    # -- Library management ------------------------------------------------

    def list_library(self) -> dict[str, dict[str, int]]:
        """Return ``{papers: {id: count}, repos: {id: count}}``."""
        return {
            "papers": self.paper_store.group_counts("paper_id"),
            "repos": self.code_store.group_counts("repo_id"),
        }

    def remove_paper(self, paper_id: str) -> int:
        """Delete all chunks for *paper_id* and return the count removed."""
        n = self.paper_store.delete({"paper_id": paper_id})
        self.paper_retriever.refresh()
        return n

    def remove_repo(self, repo_id: str) -> int:
        """Delete all chunks for *repo_id* and return the count removed."""
        n = self.code_store.delete({"repo_id": repo_id})
        self.code_retriever.refresh()
        return n

    # -- Ask ---------------------------------------------------------------

    def ask(
        self,
        question: str,
        direction: Direction | None = None,
        k: int | None = None,
        follow_refs: bool = True,
        paper_id: str | None = None,
        repo_id: str | None = None,
    ) -> AgentResult:
        # Scope retrievers when a specific paper/repo is selected.
        paper_r = (
            self.paper_retriever.scoped({"paper_id": paper_id})
            if paper_id
            else self.paper_retriever
        )
        code_r = (
            self.code_retriever.scoped({"repo_id": repo_id})
            if repo_id
            else self.code_retriever
        )
        return run_agent(
            question,
            paper_retriever=paper_r,
            code_retriever=code_r,
            settings=self.settings,
            direction=direction,
            k=k or self.settings.retrieval_top_k,
            follow_refs=follow_refs,
        )

    def counts(self) -> dict[str, int]:
        return {"paper_chunks": self.paper_store.count(), "code_chunks": self.code_store.count()}
