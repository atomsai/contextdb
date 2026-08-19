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
    llm_base_url: str | None = Field(
        default=None,
        description=(
            "Base URL of an OpenAI-compatible chat endpoint (Groq, Ollama, "
            "vLLM, Together, ...). When set, ANY model name is accepted and "
            "the API key may be omitted for keyless local servers."
        ),
    )
    embedding_base_url: str | None = Field(
        default=None,
        description=(
            "Base URL of an OpenAI-compatible embeddings endpoint. When set, "
            "any embedding model name is routed there instead of OpenAI."
        ),
    )
    pii_action: PIIAction = Field(
        default="redact",
        description="How detected PII should be handled before storage.",
    )
    pii_encryption_key: str | None = Field(
        default=None,
        description=(
            "Secret used when pii_action='encrypt'. Falls back to "
            "CONTEXTDB_PII_KEY env var. Without either, initialization "
            "raises ConfigError — encrypt fails closed rather than "
            "storing plaintext annotation originals."
        ),
    )
    retention_ttl_days: int | None = Field(
        default=730,
        description="Default retention horizon in days. None disables TTL enforcement.",
    )
    relevance_floor: float = Field(
        default=-1.0,
        description=(
            "Minimum cosine for a recall hit. -1 disables the floor. "
            "0 is NOT off (it drops negative cosines). Production "
            "embedders should use 0.15–0.25 — below that, top-1 "
            "similarity is a guess, not a memory."
        ),
    )
    enable_salience: bool = Field(
        default=True,
        description=(
            "Multiply fused retrieval scores by salience (recency x "
            "recurrence x criticality). Disable to A/B against pure RRF."
        ),
    )
    salience_half_life_days: float = Field(
        default=90.0,
        description="Half-life of the recency decay term in salience scoring.",
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )

    # Optional local pathways in the open SDK. No feature in this package
    # is gated behind a hosted offering.
    enable_entity_graph: bool = Field(default=True)
    enable_multi_graph: bool = Field(default=False)
    enable_rl_manager: bool = Field(default=False)
    enable_audit: bool = Field(default=True)
    enable_read_audit: bool = Field(
        default=True,
        description=(
            "Append SEARCH events to the synchronous SDK audit chain. "
            "High-throughput hosts may disable this only when they durably "
            "consume the read_audit hook's redacted audit_details payload. "
            "Write and lifecycle audit events remain enabled."
        ),
    )
    enable_auto_link: bool = Field(default=True)
    require_source: bool = Field(
        default=False,
        description=(
            "If true, factual.add / remember raise when epistemic source "
            "is omitted. HTTP and MCP remember always require it. The "
            "Python SDK defaults to a warning so existing callers keep working."
        ),
    )
    embedding_cache_size: int = Field(
        default=2048,
        ge=0,
        description="LRU size for query/write embedding strings. 0 disables.",
    )
    embed_timeout_seconds: float | None = Field(
        default=None,
        description="If set, embedding calls abort after this many seconds.",
    )
    lexical_on_embed_failure: bool = Field(
        default=True,
        description=(
            "If an embedding call fails or times out, recall falls back to "
            "token overlap instead of raising. Writes still fail — a write "
            "without an embedding cannot be retrieved later."
        ),
    )

    @field_validator("llm_api_key", mode="before")
    @classmethod
    def _default_api_key_from_env(cls, value: str | None) -> str | None:
        if value:
            return value
        return os.environ.get("OPENAI_API_KEY")
