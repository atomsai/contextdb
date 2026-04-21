"""Core data models for ContextDB.

Every persisted object in ContextDB is modeled here. These Pydantic v2 types
are the canonical representation used across storage, graphs, and the public
API — so please keep them backwards-compatible when amending.

Design notes:

* All timestamps are timezone-aware UTC (``datetime.now(tz=timezone.utc)``).
  Storing naive times is a recipe for silent drift between nodes.
* Enums inherit from ``str`` so that ``model_dump()`` and SQL round-trips
  produce human-readable values.
* Default-factory is used for all mutable defaults (dict / list) — never bare
  ``= {}`` or ``= []`` — so shared-state bugs cannot appear.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

GraphType = Literal["semantic", "temporal", "causal", "entity"]


def _utcnow() -> datetime:
    """Return the current UTC time as a timezone-aware :class:`datetime`."""
    return datetime.now(tz=timezone.utc)


class MemoryType(str, Enum):
    """High-level category a :class:`MemoryItem` belongs to."""

    FACTUAL = "FACTUAL"
    EXPERIENTIAL = "EXPERIENTIAL"
    WORKING = "WORKING"


class MemoryStatus(str, Enum):
    """Lifecycle state of a memory item."""

    ACTIVE = "ACTIVE"
    ARCHIVED = "ARCHIVED"
    DELETED = "DELETED"


class PIIType(str, Enum):
    """Recognized PII categories. ``CUSTOM`` is an escape hatch for users."""

    NAME = "NAME"
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    ADDRESS = "ADDRESS"
    SSN = "SSN"
    CREDIT_CARD = "CREDIT_CARD"
    CUSTOM = "CUSTOM"


class PIIAnnotation(BaseModel):
    """A single PII span detected within ``MemoryItem.content``.

    ``start`` and ``end`` are character offsets (half-open, Python slice
    semantics) into the **original** content — not the redacted form.
    """

    pii_type: PIIType
    start: int = Field(ge=0, description="Character offset (inclusive).")
    end: int = Field(ge=0, description="Character offset (exclusive).")
    original: str = Field(description="Original text that was flagged.")
    redacted: str = Field(description="Replacement text (e.g., '[NAME]').")


class Edge(BaseModel):
    """A directed edge between two memory items in one of four graphs."""

    source_id: str
    target_id: str
    graph_type: GraphType
    weight: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utcnow)


class Entity(BaseModel):
    """A named entity extracted from one or more memories."""

    name: str
    entity_type: str = Field(description="e.g., PERSON, ORG, PRODUCT, LOCATION.")
    attributes: dict[str, Any] = Field(default_factory=dict)
    memory_ids: list[str] = Field(
        default_factory=list,
        description="IDs of memories that mention this entity.",
    )


class RetentionPolicy(BaseModel):
    """Declarative retention rules applied by the retention enforcer.

    Each ``*_ttl`` field may be ``None`` to disable expiry for that class of
    memory. The default policy is chosen to match common privacy expectations:
    long for factual, unbounded for experiential, short for working.
    """

    default_ttl: timedelta | None = timedelta(days=730)
    factual_ttl: timedelta | None = timedelta(days=1825)
    experiential_ttl: timedelta | None = None
    working_ttl: timedelta | None = timedelta(hours=24)
    right_to_erasure: bool = True


class MemoryItem(BaseModel):
    """The canonical unit of memory in ContextDB.

    A :class:`MemoryItem` carries its content, vector embedding, lifecycle
    metadata, privacy annotations, and back-references to entities and tags.
    Graph relationships live in :class:`Edge` objects, not on the item
    directly, so a memory can participate in multiple graphs without
    schema churn.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    embedding: list[float] | None = None
    memory_type: MemoryType = MemoryType.FACTUAL
    source: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    event_time: datetime | None = Field(
        default=None,
        description="When the event occurred (valid-time). Distinct from ingestion_time.",
    )
    ingestion_time: datetime = Field(
        default_factory=_utcnow,
        description="When ContextDB stored the memory (system-time).",
    )

    pii_annotations: list[PIIAnnotation] = Field(default_factory=list)
    retention_policy: RetentionPolicy | None = None

    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    access_count: int = 0
    last_accessed: datetime | None = None
    confidence: float = 1.0
    status: MemoryStatus = MemoryStatus.ACTIVE

    entity_mentions: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
