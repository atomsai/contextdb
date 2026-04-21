"""ContextDB — The unified context layer for AI agents."""

from __future__ import annotations

from typing import Any

from contextdb.client import ContextDB
from contextdb.core.config import ContextDBConfig
from contextdb.core.exceptions import (
    ConfigError,
    ContextDBError,
    MemoryNotFoundError,
    PrivacyError,
    StorageError,
)
from contextdb.core.models import (
    Edge,
    Entity,
    MemoryItem,
    MemoryStatus,
    MemoryType,
    PIIAnnotation,
    PIIType,
    RetentionPolicy,
)

__version__ = "0.1.0"

__all__ = [
    "ConfigError",
    "ContextDB",
    "ContextDBConfig",
    "ContextDBError",
    "Edge",
    "Entity",
    "MemoryItem",
    "MemoryNotFoundError",
    "MemoryStatus",
    "MemoryType",
    "PIIAnnotation",
    "PIIType",
    "PrivacyError",
    "RetentionPolicy",
    "StorageError",
    "__version__",
    "init",
]


def init(
    user_id: str | None = None,
    config: ContextDBConfig | None = None,
    **kwargs: Any,
) -> ContextDB:
    """Create a :class:`ContextDB` client.

    The client is lazy — resources are provisioned on the first ``await`` on
    any I/O method, so ``init()`` itself does not touch the disk or network.

    Args:
        user_id: Optional user scope. Every write carries this as a filter.
        config: Pre-built configuration. When ``None`` one is constructed
            from ``kwargs`` and environment variables prefixed with
            ``CONTEXTDB_``.
        **kwargs: Forwarded to :class:`ContextDBConfig` if ``config`` is
            not provided.

    Returns:
        A fully-configured :class:`ContextDB` client.
    """
    resolved = config or ContextDBConfig(**kwargs)
    return ContextDB(resolved, user_id=user_id)
