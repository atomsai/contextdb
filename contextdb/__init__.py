"""ContextDB — The unified context layer for AI agents."""

from __future__ import annotations

from typing import Any

from contextdb.client import ContextDB
from contextdb.core.clock import Clock, FrozenClock, utc_now
from contextdb.core.config import ContextDBConfig
from contextdb.core.exceptions import (
    ConfigError,
    ContextDBError,
    MemoryNotFoundError,
    PrivacyError,
    ScopeConflictError,
    SourceRequiredError,
    StaleReadError,
    StorageError,
    UnauthorizedError,
)
from contextdb.core.models import (
    Edge,
    Entity,
    MemoryConsistencyToken,
    MemoryItem,
    MemoryStatus,
    MemoryType,
    PIIAnnotation,
    PIIType,
    RetentionPolicy,
)
from contextdb.core.policy import TrustPolicy
from contextdb.pool import ContextDBPool
from contextdb.utils.embeddings import EmbeddingProvider

__version__ = "0.3.2"

__all__ = [
    "Clock",
    "ConfigError",
    "ContextDB",
    "ContextDBConfig",
    "ContextDBError",
    "ContextDBPool",
    "Edge",
    "EmbeddingProvider",
    "Entity",
    "FrozenClock",
    "MemoryItem",
    "MemoryConsistencyToken",
    "MemoryNotFoundError",
    "MemoryStatus",
    "MemoryType",
    "PIIAnnotation",
    "PIIType",
    "PrivacyError",
    "RetentionPolicy",
    "ScopeConflictError",
    "SourceRequiredError",
    "StaleReadError",
    "StorageError",
    "TrustPolicy",
    "UnauthorizedError",
    "__version__",
    "init",
    "utc_now",
]


def init(
    user_id: str | None = None,
    config: ContextDBConfig | None = None,
    *,
    tenant_id: str | None = None,
    agent_id: str | None = None,
    session_id: str | None = None,
    clock: Clock | None = None,
    trust_policy: TrustPolicy | None = None,
    postgres_pool: Any | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    **kwargs: Any,
) -> ContextDB:
    """Create a :class:`ContextDB` client.

    The client is lazy — resources are provisioned on the first ``await`` on
    any I/O method, so ``init()`` itself does not touch the disk or network.

    Args:
        user_id: Default user scope. Omit this and pass ``user_id=`` on
            each call for a shared multi-tenant client. A scoped client
            cannot be widened to another user.
        config: Pre-built configuration. When ``None`` one is constructed
            from ``kwargs`` and environment variables prefixed with
            ``CONTEXTDB_``.
        postgres_pool: Optional externally owned asyncpg pool. Valid only with
            a PostgreSQL ``storage_url``. ContextDB never closes this pool.
        embedding_provider: Optional externally owned provider shared by the
            host. ContextDB uses but never closes this provider.
        **kwargs: Forwarded to :class:`ContextDBConfig` if ``config`` is
            not provided.

    Returns:
        A fully-configured :class:`ContextDB` client.
    """
    resolved = config or ContextDBConfig(**kwargs)
    return ContextDB(
        resolved,
        user_id=user_id,
        tenant_id=tenant_id,
        agent_id=agent_id,
        session_id=session_id,
        clock=clock,
        trust_policy=trust_policy,
        postgres_pool=postgres_pool,
        embedding_provider=embedding_provider,
    )
