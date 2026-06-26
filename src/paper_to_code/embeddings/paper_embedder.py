"""Paper embeddings via Voyage ``voyage-context-3`` (contextualized chunk embeddings).

Each request takes a nested input where one inner list is embedded as a group, so each
chunk is encoded *in the context of its sibling chunks*. We batch the paper's chunks into
groups (by count + token budget); chunks within a group share context.
"""

from __future__ import annotations

import voyageai

from paper_to_code.config import get_settings
from paper_to_code.embeddings.client import (
    batch_by_budget,
    call_with_retry,
    make_voyage_client,
    truncate,
)
from paper_to_code.progress import ProgressFn, report


class PaperEmbedder:
    """Embeds paper chunks with contextualized embeddings."""

    def __init__(
        self,
        client: voyageai.Client | None = None,
        *,
        model: str | None = None,
        group_size: int = 64,
        max_tokens: int = 8_000,  # stay under Voyage's free-tier 10K tokens/min per request
        max_chars: int = 120_000,
    ) -> None:
        settings = get_settings()
        self.client = client or make_voyage_client(settings)
        self.model = model or settings.paper_embedding_model
        self.group_size = group_size
        self.max_tokens = max_tokens
        self.max_chars = max_chars

    def embed_documents(
        self, texts: list[str], *, on_progress: ProgressFn | None = None
    ) -> list[list[float]]:
        prepared = [truncate(t, self.max_chars) for t in texts]
        groups = list(batch_by_budget(prepared, self.group_size, self.max_tokens))
        total = len(groups)
        out: list[list[float]] = []
        for i, group in enumerate(groups, 1):
            report(on_progress, f"Embedding paper … group {i}/{total}", (i - 1) / total)

            def _on_retry(exc, attempt, wait, i=i, total=total):
                report(
                    on_progress,
                    f"Voyage rate limit — waiting {wait:.0f}s, then retrying "
                    f"(group {i}/{total}, attempt {attempt})",
                    (i - 1) / total,
                )

            result = call_with_retry(
                lambda g=group: self.client.contextualized_embed(
                    inputs=[g], model=self.model, input_type="document"
                ),
                on_retry=_on_retry,
            )
            out.extend(result.results[0].embeddings)
        report(on_progress, "Embedding complete.", 1.0)
        return out

    def embed_query(self, text: str) -> list[float]:
        prepared = truncate(text, self.max_chars)
        result = call_with_retry(
            lambda: self.client.contextualized_embed(
                inputs=[[prepared]], model=self.model, input_type="query"
            )
        )
        return result.results[0].embeddings[0]
