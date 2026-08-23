"""Core primitives: configuration, exception hierarchy, and data models."""

from __future__ import annotations

from contextdb.core.config import ContextDBConfig
from contextdb.core.exceptions import (
    ConfigError,
    ContextDBError,
    EvolutionConflictError,
    EvolutionOperationConflictError,
    EvolutionTargetNotFoundError,
    EvolutionTargetRequiredError,
    MemoryEvolutionError,
    MemoryNotFoundError,
    PrivacyError,
    StorageError,
)
from contextdb.core.models import (
    Edge,
    Entity,
    EvolutionOperation,
    EvolutionOutcome,
    GraphType,
    MemoryConsistencyToken,
    MemoryEvolutionResult,
    MemoryItem,
    MemoryStatus,
    MemoryType,
    PIIAnnotation,
    PIIType,
    RetentionPolicy,
)

__all__ = [
    "ConfigError",
    "ContextDBConfig",
    "ContextDBError",
    "Edge",
    "Entity",
    "EvolutionConflictError",
    "EvolutionOperationConflictError",
    "EvolutionOperation",
    "EvolutionOutcome",
    "EvolutionTargetNotFoundError",
    "EvolutionTargetRequiredError",
    "GraphType",
    "MemoryConsistencyToken",
    "MemoryEvolutionError",
    "MemoryEvolutionResult",
    "MemoryItem",
    "MemoryNotFoundError",
    "MemoryStatus",
    "MemoryType",
    "PIIAnnotation",
    "PIIType",
    "PrivacyError",
    "RetentionPolicy",
    "StorageError",
]
