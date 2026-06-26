"""Offline tests for progress reporting and retry notification (no Voyage API calls).

Both embedders accept a fake client, so we can assert that ``embed_documents`` reports
batch progress and a final 100%, and that ``call_with_retry`` notifies ``on_retry`` and
clears transient errors without ever sleeping for real.
"""

from __future__ import annotations

from paper_to_code.embeddings.client import call_with_retry
from paper_to_code.embeddings.code_embedder import CodeEmbedder
from paper_to_code.embeddings.paper_embedder import PaperEmbedder
from paper_to_code.progress import report


# ── progress.report is best-effort ────────────────────────────────────────

def test_report_swallows_callback_errors():
    def boom(_m, _f):
        raise ValueError("ui blew up")

    report(boom, "msg", 0.5)  # must not raise
    report(None, "msg", 0.5)  # no-op when no sink


# ── code embedder progress ────────────────────────────────────────────────

class _Embeddings:
    def __init__(self, vecs):
        self.embeddings = vecs


class _FakeCodeClient:
    def embed(self, batch, model, input_type):
        return _Embeddings([[0.1, 0.2] for _ in batch])


def test_code_embedder_reports_progress():
    emb = CodeEmbedder(client=_FakeCodeClient())
    events: list[tuple[str, object]] = []
    out = emb.embed_documents(["a", "b", "c"], on_progress=lambda m, f: events.append((m, f)))
    assert len(out) == 3
    assert any("batch" in m.lower() for m, _ in events)
    assert events[-1][1] == 1.0  # ends at 100%


# ── paper embedder progress ───────────────────────────────────────────────

class _Group:
    def __init__(self, vecs):
        self.embeddings = vecs


class _Results:
    def __init__(self, vecs):
        self.results = [_Group(vecs)]


class _FakePaperClient:
    def contextualized_embed(self, inputs, model, input_type):
        group = inputs[0]
        return _Results([[0.1] for _ in group])


def test_paper_embedder_reports_progress():
    emb = PaperEmbedder(client=_FakePaperClient())
    events: list[tuple[str, object]] = []
    out = emb.embed_documents(["x", "y"], on_progress=lambda m, f: events.append((m, f)))
    assert len(out) == 2
    assert events[-1][1] == 1.0


# ── retry notification ────────────────────────────────────────────────────

def test_call_with_retry_notifies_on_rate_limit(monkeypatch):
    from voyageai.error import RateLimitError

    monkeypatch.setattr("time.sleep", lambda _s: None)  # no real waiting

    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RateLimitError("rate limited")
        return "ok"

    seen: list[tuple[int, float]] = []
    out = call_with_retry(flaky, on_retry=lambda exc, attempt, wait: seen.append((attempt, wait)))

    assert out == "ok"
    assert calls["n"] == 2
    assert seen and seen[0][0] == 1  # reported attempt #1 before retrying
