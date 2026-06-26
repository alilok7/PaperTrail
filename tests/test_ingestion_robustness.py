"""Offline tests for ingestion validation and graceful failure handling.

No network, no clone, no Voyage: we exercise URL validation, git-error mapping, the
clone reuse/cleanup logic (with a monkeypatched ``Repo.clone_from``), and the local-path
and PDF guards.
"""

from __future__ import annotations

import pytest

from paper_to_code.errors import IngestionError
from paper_to_code.ingestion.code import (
    _derive_repo_id,
    _friendly_clone_error,
    _validate_git_url,
    clone_repo,
    ingest_code,
)
from paper_to_code.ingestion.paper import ingest_paper


# ── URL validation ────────────────────────────────────────────────────────

class TestValidateGitUrl:
    @pytest.mark.parametrize(
        "url",
        ["https://github.com/owner/repo", "http://x/y", "git@github.com:owner/repo.git"],
    )
    def test_accepts_valid(self, url):
        _validate_git_url(url)  # should not raise

    @pytest.mark.parametrize("url", ["not a url", "ftp://x/y", "://nope", "git@nohost"])
    def test_rejects_invalid(self, url):
        with pytest.raises(IngestionError):
            _validate_git_url(url)

    def test_derive_repo_id(self):
        assert _derive_repo_id("https://github.com/karpathy/nanoGPT.git") == "nanoGPT"
        assert _derive_repo_id("https://github.com/owner/repo/") == "repo"


# ── git error mapping ─────────────────────────────────────────────────────

class TestFriendlyCloneError:
    def test_not_found(self):
        msg = _friendly_clone_error("u", Exception("fatal: repository not found"))
        assert "not found" in msg.lower()

    def test_auth(self):
        msg = _friendly_clone_error("u", Exception("fatal: Authentication failed"))
        assert "private" in msg.lower() or "denied" in msg.lower()

    def test_network(self):
        msg = _friendly_clone_error("u", Exception("Could not resolve host: github.com"))
        assert "reach" in msg.lower() or "connection" in msg.lower()


# ── clone_repo reuse / cleanup ────────────────────────────────────────────

class TestCloneRepo:
    def test_reuses_healthy_clone_without_cloning(self, tmp_path, monkeypatch):
        repos = tmp_path / "repos"
        dest = repos / "repo"
        (dest / ".git").mkdir(parents=True)
        (dest / "main.py").write_text("x", encoding="utf-8")

        def _boom(*a, **k):
            raise AssertionError("clone_from should not be called for a healthy clone")

        monkeypatch.setattr("git.Repo.clone_from", staticmethod(_boom))
        repo_id, path = clone_repo("https://github.com/owner/repo", repos)
        assert repo_id == "repo"
        assert path == dest

    def test_cleans_up_partial_clone_on_failure(self, tmp_path, monkeypatch):
        repos = tmp_path / "repos"

        def _fail(url, to_path, **k):
            # Simulate a clone that wrote a partial directory and then failed.
            from pathlib import Path

            Path(to_path).mkdir(parents=True, exist_ok=True)
            (Path(to_path) / "partial").write_text("x", encoding="utf-8")
            raise RuntimeError("fatal: repository not found")

        monkeypatch.setattr("git.Repo.clone_from", staticmethod(_fail))
        with pytest.raises(IngestionError) as exc:
            clone_repo("https://github.com/owner/missing", repos)
        assert "not found" in str(exc.value).lower()
        assert not (repos / "missing").exists()  # the half-clone was removed


# ── ingest_code local-path guards ─────────────────────────────────────────

class TestIngestCodeLocalPath:
    def test_missing_path_raises(self, tmp_path):
        with pytest.raises(IngestionError):
            ingest_code(tmp_path / "does_not_exist")

    def test_file_instead_of_dir_raises(self, tmp_path):
        f = tmp_path / "a.py"
        f.write_text("def a():\n    return 1\n", encoding="utf-8")
        with pytest.raises(IngestionError):
            ingest_code(f)

    def test_local_dir_scans_and_reports_progress(self, tmp_path):
        (tmp_path / "m.py").write_text("def a():\n    return 1\n", encoding="utf-8")
        events: list[tuple[str, object]] = []
        result = ingest_code(tmp_path, on_progress=lambda m, f: events.append((m, f)))
        assert result.chunks  # at least the function chunk
        assert events  # progress was reported


# ── ingest_paper guards (no Docling needed — fails before convert) ─────────

class TestIngestPaperGuards:
    def test_missing_pdf_raises(self, tmp_path):
        with pytest.raises(IngestionError):
            ingest_paper(tmp_path / "nope.pdf")

    def test_non_pdf_raises(self, tmp_path):
        f = tmp_path / "notes.txt"
        f.write_text("hello", encoding="utf-8")
        with pytest.raises(IngestionError):
            ingest_paper(f)
