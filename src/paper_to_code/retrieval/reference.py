"""Single-hop reference following.

When a top retrieved paper chunk references another element ("as in Equation 3",
"see Section 4.2", "Figure 2"), we do one extra retrieval for that element and add it to
the evidence before answering. This stays a single hop (no full reference graph), matching
the project's scope.
"""

from __future__ import annotations

from typing import Protocol

from paper_to_code.models import RetrievedChunk


class _Retriever(Protocol):
    def search(self, query: str, k: int) -> list[RetrievedChunk]: ...


_REFERENCE_KEYS = ("section_refs", "equations", "figures")


def _split_labels(value: object) -> list[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def collect_reference_labels(hits: list[RetrievedChunk], scan_top: int = 2) -> list[str]:
    """Gather unique cross-reference labels from the top ``scan_top`` chunks' metadata."""
    labels: list[str] = []
    for hit in hits[:scan_top]:
        for key in _REFERENCE_KEYS:
            for label in _split_labels(hit.chunk.metadata.get(key)):
                if label not in labels:
                    labels.append(label)
    return labels


def follow_references(
    hits: list[RetrievedChunk],
    retriever: _Retriever,
    *,
    max_refs: int = 2,
    per_ref_k: int = 1,
    scan_top: int = 2,
) -> list[RetrievedChunk]:
    """Return ``hits`` augmented with chunks fetched for referenced elements (deduped)."""
    seen = {h.chunk.id for h in hits}
    added: list[RetrievedChunk] = []
    for label in collect_reference_labels(hits, scan_top=scan_top)[:max_refs]:
        for found in retriever.search(label, per_ref_k):
            if found.chunk.id not in seen:
                seen.add(found.chunk.id)
                added.append(found)
    return hits + added
