"""Stage tests for embedding batching helpers (no API calls)."""

from __future__ import annotations

from paper_to_code.embeddings.client import approx_tokens, batch_by_budget, truncate


def test_batch_by_item_count():
    batches = list(batch_by_budget(["a"] * 5, max_items=2, max_tokens=10_000))
    assert [len(b) for b in batches] == [2, 2, 1]


def test_batch_by_token_budget_splits_and_preserves_all():
    items = ["x" * 40] * 5  # ~11 approx tokens each
    batches = list(batch_by_budget(items, max_items=100, max_tokens=25))
    assert sum(len(b) for b in batches) == 5
    # No batch should exceed the token budget once it holds more than one item.
    for b in batches:
        assert sum(approx_tokens(t) for t in b) <= 25 or len(b) == 1


def test_batch_empty():
    assert list(batch_by_budget([], max_items=10, max_tokens=10)) == []


def test_truncate():
    assert truncate("abcdef", 3) == "abc"
    assert truncate("ab", 5) == "ab"
