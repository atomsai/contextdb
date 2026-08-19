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
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, computed_field

if TYPE_CHECKING:
    from contextdb.core.policy import TrustPolicy

GraphType = Literal["semantic", "temporal", "causal", "entity"]

# Epistemic provenance of a stored claim:
# * ``user_stated`` — the user asserted it themselves.
# * ``agent_inferred`` — the agent derived it; never first-party evidence.
# * ``third_party`` — hearsay, forwarded claims, or content of uncertain origin.
EpistemicSource = Literal["user_stated", "agent_inferred", "third_party"]

# A user-stated fact is trusted for action only at or above this confidence.
# Below it, action-gating facts require corroboration (count >= 2). Wishes and
# speculation are extracted at 0.5, which is exactly what keeps an
# uncorroborated "I'd like to come in Thursday" out of recall_for_action().
ACTION_CONFIDENCE_THRESHOLD = 0.7

# Corroboration count at which any fact — regardless of source — is trusted.
ACTION_CORROBORATION_THRESHOLD = 2


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


class MemoryConsistencyToken(BaseModel):
    """A lower bound that a later read must observe."""

    memory_version: int = Field(ge=0)
    primary_wal_lsn: str | None = Field(
        default=None,
        pattern=r"^[0-9A-F]+/[0-9A-F]+$",
    )


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


class MemoryExplanation(BaseModel):
    """Reconstruction of a memory's formation and recall history (Epic 6).

    Answers "why did the agent say that?" from the audit log alone:
    which writes formed the memory, what it superseded / was superseded by,
    and which queries surfaced it (with their logged score components).
    """

    memory_id: str
    memory: dict[str, Any] | None = Field(
        default=None, description="model_dump of the memory, if it still exists."
    )
    writes: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Audit entries (CREATE/CORROBORATE/SUPERSEDE/UPDATE/...) for this id.",
    )
    supersede_chain: list[str] = Field(
        default_factory=list,
        description="Ordered memory ids: oldest ancestor … this memory … latest successor.",
    )
    surfaced_by: list[dict[str, Any]] = Field(
        default_factory=list,
        description="SEARCH audit entries whose returned_ids include this memory.",
    )


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

    # ------------------------------------------------------------------ #
    # Trust model (Epics 1-3). All fields have backward-compatible
    # defaults; the storage layer migrates old rows with these defaults.
    # ------------------------------------------------------------------ #

    epistemic_source: EpistemicSource = Field(
        default="user_stated",
        description=(
            "Who vouches for this claim. Distinct from ``source``, which "
            "remains the free-form provenance string (e.g. 'consolidator')."
        ),
    )
    corroboration_count: int = Field(
        default=1,
        ge=0,
        description="Independent writes of the same entity+attribute value.",
    )
    action_relevant: bool = Field(
        default=False,
        description=(
            "True for facts that gate actions: bookings, prices, schedules, "
            "contact details, identity, health/finance/legal attributes."
        ),
    )
    entity_key: str | None = Field(
        default=None,
        description="Stable slot key (entity) for dedupe/contradiction matching.",
    )
    attribute_key: str | None = Field(
        default=None,
        description="Stable slot key (attribute) for dedupe/contradiction matching.",
    )
    valid_from: datetime | None = Field(
        default=None,
        description="Start of validity (system-time). None means 'always was'.",
    )
    valid_until: datetime | None = Field(
        default=None,
        description="End of validity. Set when superseded; None means current.",
    )
    superseded_by: str | None = Field(
        default=None,
        description="ID of the memory that replaced this one, if any.",
    )
    pending_consolidation: bool = Field(
        default=False,
        description="Written via add_fast; extraction/dedupe still outstanding.",
    )
    injection_suspect: bool = Field(
        default=False,
        description="Write-time screening flagged instruction-shaped content.",
    )
    corroborated_by: list[str] = Field(
        default_factory=list,
        description=(
            "Distinct speaker/session ids that independently restated this "
            "value. Length is the independent-corroboration count; repeats "
            "from the same speaker do not graduate a fact."
        ),
    )
    confirmed: bool = Field(
        default=False,
        description="Explicit user confirmation via factual.confirm().",
    )
    confirmed_at: datetime | None = Field(
        default=None,
        description="When the fact was explicitly confirmed, if ever.",
    )
    write_generation: int = Field(
        default=0,
        description=(
            "Monotonic per-slot generation. A pending raw must not "
            "supersede a typed fact with a higher generation."
        ),
    )
    slot_class: str | None = Field(
        default=None,
        description="Action class from the versioned slot vocabulary (health, booking, ...).",
    )
    slot_value: str | None = Field(
        default=None,
        description="Canonical value inside the slot (e.g. 'thursday', '4:00pm').",
    )
    negated: bool = Field(
        default=False,
        description="True when the utterance denies the slot value.",
    )
    user_id: str | None = Field(
        default=None,
        description="User scope stamped on the row. Distinct from tenant_id.",
    )
    tenant_id: str | None = Field(default=None)
    agent_id: str | None = Field(default=None)
    session_id: str | None = Field(default=None)
    pii_shadow: str | None = Field(
        default=None,
        description=(
            "Typed stand-in embedded alongside redacted content so a query "
            "containing a real email can still retrieve '[EMAIL]'."
        ),
    )
    contested: bool = Field(
        default=False,
        description=(
            "True when an independent speaker asserted a different value "
            "in this slot. Neither side may gate an action until confirm()."
        ),
    )

    @property
    def independent_corroboration(self) -> int:
        """Distinct speakers, falling back to the raw count for pre-migration rows."""
        if self.corroborated_by:
            return len(self.corroborated_by)
        return self.corroboration_count

    def requires_confirmation_under(self, policy: TrustPolicy) -> bool:
        return policy.requires_confirmation(self)

    def action_trusted_under(self, policy: TrustPolicy) -> bool:
        return policy.is_trusted(self)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def requires_confirmation(self) -> bool:
        """Default-policy view — kept so existing callers and evals stay valid.

        Prefer :meth:`requires_confirmation_under` when a custom
        :class:`TrustPolicy` is in play.
        """
        from contextdb.core.policy import DEFAULT_TRUST_POLICY

        return DEFAULT_TRUST_POLICY.requires_confirmation(self)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def action_trusted(self) -> bool:
        """Whether this fact may gate an action under the default policy."""
        from contextdb.core.policy import DEFAULT_TRUST_POLICY

        return DEFAULT_TRUST_POLICY.is_trusted(self)

    def is_valid_at(self, moment: datetime) -> bool:
        """Temporal validity check (Epic 2). ``valid_until`` is exclusive."""
        if self.valid_from is not None and moment < self.valid_from:
            return False
        return not (self.valid_until is not None and moment >= self.valid_until)
