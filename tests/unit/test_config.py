"""Tests for ContextDBConfig."""

from __future__ import annotations

from typing import TYPE_CHECKING

from contextdb import ContextDBConfig

if TYPE_CHECKING:
    import pytest
from contextdb.core.exceptions import (
    ConfigError,
    ContextDBError,
    MemoryNotFoundError,
    PrivacyError,
    StorageError,
)


def test_defaults() -> None:
    cfg = ContextDBConfig(llm_api_key="explicit-key")
    assert cfg.storage_url == "sqlite:///contextdb.db"
    assert cfg.embedding_model == "text-embedding-3-small"
    assert cfg.embedding_dim == 1536
    assert cfg.llm_model == "gpt-4o-mini"
    assert cfg.pii_action == "redact"
    assert cfg.retention_ttl_days == 730
    assert cfg.log_level == "INFO"
    assert cfg.llm_api_key == "explicit-key"


def test_api_key_falls_back_to_openai_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    monkeypatch.delenv("CONTEXTDB_LLM_API_KEY", raising=False)
    cfg = ContextDBConfig()
    assert cfg.llm_api_key == "sk-from-env"


def test_api_key_none_when_nothing_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CONTEXTDB_LLM_API_KEY", raising=False)
    cfg = ContextDBConfig()
    assert cfg.llm_api_key is None


def test_reads_from_contextdb_env_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTEXTDB_STORAGE_URL", "postgresql://localhost/ctx")
    monkeypatch.setenv("CONTEXTDB_LOG_LEVEL", "DEBUG")
    cfg = ContextDBConfig(llm_api_key="x")
    assert cfg.storage_url == "postgresql://localhost/ctx"
    assert cfg.log_level == "DEBUG"


def test_exception_hierarchy() -> None:
    for exc in (MemoryNotFoundError, StorageError, PrivacyError, ConfigError):
        assert issubclass(exc, ContextDBError)
        assert issubclass(exc, Exception)
