"""Code embeddings via Voyage ``voyage-code-3`` (code-specialized)."""

from __future__ import annotations

import voyageai

from paper_to_code.config import get_settings
from paper_to_code.embeddings.client import (
    batch_by_budget,
    call_with_retry,
    make_voyage_client,
    truncate,
)


class CodeEmbedder:
    """Embeds source-code chunks. ``input_type`` distinguishes documents from queries."""

    def __init__(
        self,
        client: voyageai.Client | None = None,
        *,
        model: str | None = None,
        max_items: int = 128,
        max_tokens: int = 8_000,  # stay under Voyage's free-tier 10K tokens/min per request
        max_chars: int = 120_000,
    ) -> None:
        settings = get_settings()
        self.client = client or make_voyage_client(settings)
        self.model = model or settings.code_embedding_model
        self.max_items = max_items
        self.max_tokens = max_tokens
        self.max_chars = max_chars

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prepared = [truncate(t, self.max_chars) for t in texts]
        out: list[list[float]] = []
        for batch in batch_by_budget(prepared, self.max_items, self.max_tokens):
            result = call_with_retry(
                lambda b=batch: self.client.embed(b, model=self.model, input_type="document")
            )
            out.extend(result.embeddings)
        return out

    def embed_query(self, text: str) -> list[float]:
        prepared = truncate(text, self.max_chars)
        result = call_with_retry(
            lambda: self.client.embed([prepared], model=self.model, input_type="query")
        )
        return result.embeddings[0]
