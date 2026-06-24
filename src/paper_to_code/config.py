"""Application configuration, loaded from the environment / `.env`.

All secrets live here and nowhere else. Secret values use Pydantic's ``SecretStr`` so
they are never accidentally printed or logged. Nothing in this module writes a secret
to stdout or to a file.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


def _is_set(secret: SecretStr | None) -> bool:
    """True only if a secret is present *and* non-empty (env vars can be set to '')."""
    return secret is not None and bool(secret.get_secret_value())


class Settings(BaseSettings):
    """Strongly-typed settings sourced from environment variables / `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- AWS Bedrock (LLM) ---
    aws_region: str = "us-east-1"
    # Auth option A: a Bedrock API key (bearer token). Auth option B: an IAM access-key pair.
    aws_bearer_token_bedrock: SecretStr | None = None
    aws_access_key_id: SecretStr | None = None
    aws_secret_access_key: SecretStr | None = None
    bedrock_model_id: str = "us.anthropic.claude-sonnet-4-6"

    # --- Voyage AI (embeddings) ---
    voyage_api_key: SecretStr | None = None
    paper_embedding_model: str = "voyage-context-3"
    code_embedding_model: str = "voyage-code-3"

    # --- LangSmith (observability — required for real runs) ---
    langsmith_api_key: SecretStr | None = None
    langsmith_tracing: bool = True
    langsmith_project: str = "paper-trail"

    # --- LLM parameters ---
    temperature: float = 0.0
    max_tokens: int = 4096

    # --- Retrieval parameters ---
    retrieval_top_k: int = 8
    rrf_k: int = 60

    # --- Storage paths (everything runtime lives under data/, which is gitignored) ---
    data_dir: Path = Path("data")

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def papers_dir(self) -> Path:
        return self.data_dir / "papers"

    @property
    def repos_dir(self) -> Path:
        return self.data_dir / "repos"

    # --- Readiness flags (used by `p2c check`) ---
    @property
    def has_bedrock(self) -> bool:
        has_auth = _is_set(self.aws_bearer_token_bedrock) or (
            _is_set(self.aws_access_key_id) and _is_set(self.aws_secret_access_key)
        )
        return has_auth and bool(self.bedrock_model_id)

    @property
    def has_voyage(self) -> bool:
        return _is_set(self.voyage_api_key)

    @property
    def has_langsmith(self) -> bool:
        return _is_set(self.langsmith_api_key)

    def ensure_dirs(self) -> None:
        """Create the runtime directories if they don't exist."""
        for path in (self.data_dir, self.chroma_dir, self.papers_dir, self.repos_dir):
            path.mkdir(parents=True, exist_ok=True)

    def apply_bedrock_env(self) -> None:
        """Export AWS/Bedrock auth to the process environment so boto3 picks it up.

        Supports both a Bedrock API key (``AWS_BEARER_TOKEN_BEDROCK``) and a classic
        IAM access-key pair. Region is always set.
        """
        os.environ.setdefault("AWS_REGION", self.aws_region)
        os.environ.setdefault("AWS_DEFAULT_REGION", self.aws_region)
        if _is_set(self.aws_bearer_token_bedrock):
            os.environ["AWS_BEARER_TOKEN_BEDROCK"] = self.aws_bearer_token_bedrock.get_secret_value()  # type: ignore[union-attr]
        if _is_set(self.aws_access_key_id):
            os.environ["AWS_ACCESS_KEY_ID"] = self.aws_access_key_id.get_secret_value()  # type: ignore[union-attr]
        if _is_set(self.aws_secret_access_key):
            os.environ["AWS_SECRET_ACCESS_KEY"] = self.aws_secret_access_key.get_secret_value()  # type: ignore[union-attr]

    def apply_langsmith_env(self) -> None:
        """Export LangSmith settings so LangChain emits traces.

        Sets both the new ``LANGSMITH_*`` and legacy ``LANGCHAIN_*`` variable names.
        """
        if not (self.has_langsmith and self.langsmith_tracing):
            return
        key = self.langsmith_api_key.get_secret_value()  # type: ignore[union-attr]
        os.environ.setdefault("LANGSMITH_TRACING", "true")
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ["LANGSMITH_API_KEY"] = key
        os.environ["LANGCHAIN_API_KEY"] = key
        os.environ["LANGSMITH_PROJECT"] = self.langsmith_project
        os.environ["LANGCHAIN_PROJECT"] = self.langsmith_project


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
