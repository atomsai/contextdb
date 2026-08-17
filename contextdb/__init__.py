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
from contextdb.core.policy import TrustPolicy

__version__ = "0.2.0"

__all__ = [
    "Clock",
    "ConfigError",
    "ContextDB",
    "ContextDBConfig",
    "ContextDBError",
    "Edge",
    "Entity",
    "FrozenClock",
    "MemoryItem",
    "MemoryNotFoundError",
    "MemoryStatus",
    "MemoryType",
    "PIIAnnotation",
    "PIIType",
    "PrivacyError",
    "RetentionPolicy",
    "StorageError",
    "TrustPolicy",
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
    return ContextDB(
        resolved,
        user_id=user_id,
        tenant_id=tenant_id,
        agent_id=agent_id,
        session_id=session_id,
        clock=clock,
        trust_policy=trust_policy,
    )
