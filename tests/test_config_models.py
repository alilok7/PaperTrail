"""Stage tests for the config and model layer (no credentials required)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from paper_to_code.config import Settings
from paper_to_code.models import (
    CodeCitation,
    Direction,
    PaperCitation,
    Verdict,
    VerdictType,
)


def test_settings_defaults_without_env():
    # _env_file=None ignores the local .env so the test is deterministic.
    s = Settings(_env_file=None)
    assert s.aws_region == "us-east-1"
    assert s.bedrock_model_id
    assert s.paper_embedding_model == "voyage-context-3"
    assert s.code_embedding_model == "voyage-code-3"
    assert s.temperature == 0.0
    # No keys provided -> nothing is "ready".
    assert s.has_bedrock is False
    assert s.has_voyage is False
    assert s.has_langsmith is False


def test_empty_secret_counts_as_unset():
    s = Settings(_env_file=None, voyage_api_key="")
    assert s.has_voyage is False


def test_paths_derive_from_data_dir(tmp_path):
    s = Settings(_env_file=None, data_dir=tmp_path)
    assert s.chroma_dir == tmp_path / "chroma"
    assert s.papers_dir == tmp_path / "papers"
    assert s.repos_dir == tmp_path / "repos"
    s.ensure_dirs()
    assert s.chroma_dir.is_dir()


def test_verdict_roundtrip():
    v = Verdict(
        verdict=VerdictType.MATCH,
        confidence=0.9,
        direction=Direction.PAPER_TO_CODE,
        explanation="Multi-head attention maps to CausalSelfAttention.",
        paper_citation=PaperCitation(section="3.2", page=4, quote="Scaled dot-product attention..."),
        code_citation=CodeCitation(file="model.py", symbol="CausalSelfAttention", start_line=29, end_line=58),
    )
    dumped = v.model_dump()
    assert dumped["verdict"] == "MATCH"
    assert dumped["direction"] == "paper_to_code"
    assert dumped["code_citation"]["symbol"] == "CausalSelfAttention"


def test_confidence_is_bounded():
    with pytest.raises(ValidationError):
        Verdict(
            verdict=VerdictType.UNCERTAIN,
            confidence=1.5,
            direction=Direction.CODE_TO_PAPER,
            explanation="x",
        )
