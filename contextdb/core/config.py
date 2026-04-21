"""Configuration for ContextDB.

:class:`ContextDBConfig` is the single source of truth for runtime settings.
It is a :class:`pydantic_settings.BaseSettings` subclass, so fields may be
populated from environment variables prefixed with ``CONTEXTDB_`` in addition
to keyword arguments.

The one exception is :attr:`ContextDBConfig.llm_api_key`, which falls back to
the standard ``OPENAI_API_KEY`` environment variable when not supplied
explicitly. This mirrors the behavior of the OpenAI SDK and lets users point
ContextDB at their existing key without renaming it.
"""

from __future__ import annotations

import os
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PIIAction = Literal["redact", "encrypt", "flag", "allow"]


class ContextDBConfig(BaseSettings):
    """Runtime configuration for a :class:`ContextDB` instance."""

    model_config = SettingsConfigDict(
        env_prefix="CONTEXTDB_",
        env_file=None,
        extra="ignore",
        case_sensitive=False,
    )

    storage_url: str = Field(
        default="sqlite:///contextdb.db",
        description="Storage backend URL. SQLite for local dev, Postgres for production.",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Embedding model name (OpenAI by default).",
    )
    embedding_dim: int = Field(
        default=1536,
        description="Embedding vector dimensionality; must match embedding_model.",
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="LLM used for extraction, compression, and reasoning steps.",
    )
    llm_api_key: str | None = Field(
        default=None,
        description="API key for the LLM provider. Falls back to OPENAI_API_KEY env var.",
    )
    pii_action: PIIAction = Field(
        default="redact",
        description="How detected PII should be handled before storage.",
    )
    retention_ttl_days: int | None = Field(
        default=730,
        description="Default retention horizon in days. None disables TTL enforcement.",
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )

    # Tier / feature flags. Single-graph semantic memory is always on; every
    # richer pathway is explicit so operators can cleanly A/B free vs paid.
    enable_entity_graph: bool = Field(default=True)
    enable_multi_graph: bool = Field(default=False)
    enable_rl_manager: bool = Field(default=False)
    enable_audit: bool = Field(default=True)
    enable_auto_link: bool = Field(default=True)

    @field_validator("llm_api_key", mode="before")
    @classmethod
    def _default_api_key_from_env(cls, value: str | None) -> str | None:
        if value:
            return value
        return os.environ.get("OPENAI_API_KEY")
