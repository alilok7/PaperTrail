"""Shared Voyage client factory and batching helpers.

Voyage caps both the number of inputs and the total tokens per request, so we batch by
*both* a max item count and an approximate-token budget before each API call.
"""

from __future__ import annotations

from typing import Iterator

import voyageai

from paper_to_code.config import Settings, get_settings


def make_voyage_client(settings: Settings | None = None) -> voyageai.Client:
    settings = settings or get_settings()
    if not settings.has_voyage:
        raise RuntimeError("VOYAGE_API_KEY is not set; add it to .env to use embeddings.")
    return voyageai.Client(api_key=settings.voyage_api_key.get_secret_value())  # type: ignore[union-attr]


def approx_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token) for budgeting batches."""
    return len(text) // 4 + 1


def truncate(text: str, max_chars: int) -> str:
    """Safety net so a single oversized chunk can't exceed the per-input token limit."""
    return text if len(text) <= max_chars else text[:max_chars]


def batch_by_budget(texts: list[str], max_items: int, max_tokens: int) -> Iterator[list[str]]:
    """Yield batches respecting both a max item count and an approximate token budget."""
    batch: list[str] = []
    tokens = 0
    for text in texts:
        cost = approx_tokens(text)
        if batch and (len(batch) >= max_items or tokens + cost > max_tokens):
            yield batch
            batch, tokens = [], 0
        batch.append(text)
        tokens += cost
    if batch:
        yield batch


def call_with_retry(fn, *, max_retries: int = 5, wait_seconds: float = 21.0):
    """Call ``fn`` retrying on Voyage rate-limit errors with a fixed wait.

    Free-tier accounts without a payment method are capped at ~3 requests/min; adding a
    payment method (free tokens still apply) removes the need for this backoff.
    """
    import time

    from voyageai.error import RateLimitError

    for attempt in range(max_retries + 1):
        try:
            return fn()
        except RateLimitError:
            if attempt >= max_retries:
                raise
            time.sleep(wait_seconds)
