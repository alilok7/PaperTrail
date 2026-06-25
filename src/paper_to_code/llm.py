"""Bedrock Claude chat model (temperature 0) + structured output + LangSmith wiring.

The verdict is produced via ``with_structured_output`` so the LLM must return a validated
Pydantic object. ``include_raw=True`` keeps the raw response and any parsing error so the
agent can fall back gracefully instead of crashing.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain_aws import ChatBedrockConverse

from paper_to_code.config import Settings, get_settings


def get_chat_model(settings: Settings | None = None) -> ChatBedrockConverse:
    """Construct the temperature-0 Bedrock Claude model, wiring auth + tracing first."""
    settings = settings or get_settings()
    if not settings.has_bedrock:
        raise RuntimeError(
            "Bedrock is not configured. Set AWS_BEARER_TOKEN_BEDROCK (or AWS access keys) "
            "and BEDROCK_MODEL_ID in .env."
        )
    settings.apply_bedrock_env()
    if settings.langsmith_tracing and _langsmith_enabled():
        settings.apply_langsmith_env()
    return ChatBedrockConverse(
        model=settings.bedrock_model_id,
        region_name=settings.aws_region,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )


def structured_model(schema: type, settings: Settings | None = None) -> Any:
    """Return a model that emits a validated instance of ``schema`` (with raw + errors)."""
    return get_chat_model(settings).with_structured_output(schema, include_raw=True)


def validate_langsmith(settings: Settings | None = None) -> tuple[bool, str]:
    """Live-check the LangSmith key against the configured endpoint."""
    settings = settings or get_settings()
    if not settings.has_langsmith:
        return False, "no key set"
    try:
        from langsmith import Client

        client = Client(
            api_key=settings.langsmith_api_key.get_secret_value(),  # type: ignore[union-attr]
            api_url=settings.langsmith_endpoint,
        )
        list(client.list_projects(limit=1))
        return True, "reachable"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {str(exc)[:120]}"


@lru_cache(maxsize=1)
def _langsmith_enabled() -> bool:
    """Validate the LangSmith key once per process; tracing is gated on this so an
    invalid key silently disables tracing instead of spamming 403s on every call."""
    ok, _ = validate_langsmith()
    return ok
