"""Tests for the top-level ``contextdb`` package."""

from __future__ import annotations

import contextdb


def test_version_exported() -> None:
    assert contextdb.__version__ == "0.1.0"


def test_init_returns_client() -> None:
    client = contextdb.init(user_id="u1", llm_api_key="x", storage_url="sqlite:///:memory:")
    assert isinstance(client, contextdb.ContextDB)
    assert client.user_id == "u1"


def test_public_exports_present() -> None:
    expected = {
        "ConfigError",
        "ContextDB",
        "ContextDBConfig",
        "ContextDBError",
        "MemoryNotFoundError",
        "PrivacyError",
        "StorageError",
        "__version__",
        "init",
    }
    assert expected.issubset(set(contextdb.__all__))
