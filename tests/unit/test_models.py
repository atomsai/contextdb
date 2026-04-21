"""Tests for core data models."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from contextdb.core.models import (
    Edge,
    Entity,
    GraphType,
    MemoryItem,
    MemoryStatus,
    MemoryType,
    PIIAnnotation,
    PIIType,
    RetentionPolicy,
)

# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #


def test_memory_type_values() -> None:
    assert MemoryType.FACTUAL == "FACTUAL"
    assert MemoryType.EXPERIENTIAL == "EXPERIENTIAL"
    assert MemoryType.WORKING == "WORKING"
    assert {m.value for m in MemoryType} == {"FACTUAL", "EXPERIENTIAL", "WORKING"}


def test_memory_status_values() -> None:
    assert {s.value for s in MemoryStatus} == {"ACTIVE", "ARCHIVED", "DELETED"}


def test_pii_type_values() -> None:
    assert {p.value for p in PIIType} == {
        "NAME",
        "EMAIL",
        "PHONE",
        "ADDRESS",
        "SSN",
        "CREDIT_CARD",
        "CUSTOM",
    }


# --------------------------------------------------------------------------- #
# PIIAnnotation
# --------------------------------------------------------------------------- #


def test_pii_annotation_email() -> None:
    content = "Email me at alice@example.com tomorrow."
    start, end = content.index("alice@example.com"), content.index("alice@example.com") + len(
        "alice@example.com"
    )
    ann = PIIAnnotation(
        pii_type=PIIType.EMAIL,
        start=start,
        end=end,
        original="alice@example.com",
        redacted="[EMAIL]",
    )
    assert content[ann.start : ann.end] == ann.original
    assert ann.redacted == "[EMAIL]"


def test_pii_annotation_credit_card() -> None:
    ann = PIIAnnotation(
        pii_type=PIIType.CREDIT_CARD,
        start=10,
        end=26,
        original="4111111111111111",
        redacted="[CREDIT_CARD]",
    )
    assert ann.pii_type == PIIType.CREDIT_CARD


# --------------------------------------------------------------------------- #
# Edge
# --------------------------------------------------------------------------- #


def test_edge_defaults() -> None:
    edge = Edge(source_id="a", target_id="b", graph_type="semantic")
    assert edge.weight == 1.0
    assert edge.metadata == {}
    assert edge.created_at.tzinfo is not None


def test_edge_each_graph_type() -> None:
    types: list[GraphType] = ["semantic", "temporal", "causal", "entity"]
    for t in types:
        edge = Edge(source_id="x", target_id="y", graph_type=t, weight=0.5)
        assert edge.graph_type == t
        assert edge.weight == 0.5


def test_edge_metadata_is_independent_between_instances() -> None:
    e1 = Edge(source_id="a", target_id="b", graph_type="semantic")
    e2 = Edge(source_id="c", target_id="d", graph_type="semantic")
    e1.metadata["key"] = "value"
    assert "key" not in e2.metadata


# --------------------------------------------------------------------------- #
# Entity
# --------------------------------------------------------------------------- #


def test_entity_defaults() -> None:
    e = Entity(name="Acme Corp", entity_type="ORG")
    assert e.attributes == {}
    assert e.memory_ids == []


def test_entity_with_memory_ids() -> None:
    e = Entity(
        name="Priya",
        entity_type="PERSON",
        attributes={"plan": "pro"},
        memory_ids=["m1", "m2"],
    )
    assert e.memory_ids == ["m1", "m2"]
    assert e.attributes["plan"] == "pro"


# --------------------------------------------------------------------------- #
# RetentionPolicy
# --------------------------------------------------------------------------- #


def test_retention_policy_defaults() -> None:
    p = RetentionPolicy()
    assert p.default_ttl == timedelta(days=730)
    assert p.factual_ttl == timedelta(days=1825)
    assert p.experiential_ttl is None
    assert p.working_ttl == timedelta(hours=24)
    assert p.right_to_erasure is True


def test_retention_policy_overrides() -> None:
    p = RetentionPolicy(default_ttl=None, right_to_erasure=False)
    assert p.default_ttl is None
    assert p.right_to_erasure is False


# --------------------------------------------------------------------------- #
# MemoryItem — defaults and full construction
# --------------------------------------------------------------------------- #


def test_memory_item_defaults() -> None:
    m = MemoryItem(content="hello world")
    assert m.id  # auto-generated uuid
    assert len(m.id) == 36
    assert m.content == "hello world"
    assert m.embedding is None
    assert m.memory_type == MemoryType.FACTUAL
    assert m.source == ""
    assert m.metadata == {}
    assert m.event_time is None
    assert m.ingestion_time.tzinfo is not None
    assert m.pii_annotations == []
    assert m.retention_policy is None
    assert m.access_count == 0
    assert m.last_accessed is None
    assert m.confidence == 1.0
    assert m.status == MemoryStatus.ACTIVE
    assert m.entity_mentions == []
    assert m.tags == []


def test_memory_item_unique_ids() -> None:
    a = MemoryItem(content="one")
    b = MemoryItem(content="two")
    assert a.id != b.id


def test_memory_item_all_fields() -> None:
    event = datetime(2026, 4, 1, 12, 0, tzinfo=timezone.utc)
    pii = PIIAnnotation(
        pii_type=PIIType.EMAIL,
        start=0,
        end=17,
        original="alice@example.com",
        redacted="[EMAIL]",
    )
    policy = RetentionPolicy(default_ttl=timedelta(days=30))
    m = MemoryItem(
        id="mem-1",
        content="alice@example.com signed up",
        embedding=[0.1, 0.2, 0.3],
        memory_type=MemoryType.EXPERIENTIAL,
        source="helpwise",
        metadata={"channel": "email"},
        event_time=event,
        pii_annotations=[pii],
        retention_policy=policy,
        access_count=3,
        confidence=0.9,
        status=MemoryStatus.ARCHIVED,
        entity_mentions=["Alice"],
        tags=["signup"],
    )
    assert m.id == "mem-1"
    assert m.embedding == [0.1, 0.2, 0.3]
    assert m.memory_type == MemoryType.EXPERIENTIAL
    assert m.event_time == event
    assert m.pii_annotations[0].pii_type == PIIType.EMAIL
    assert m.retention_policy is not None
    assert m.retention_policy.default_ttl == timedelta(days=30)
    assert m.status == MemoryStatus.ARCHIVED


# --------------------------------------------------------------------------- #
# Serialization
# --------------------------------------------------------------------------- #


def test_memory_item_roundtrip() -> None:
    original = MemoryItem(
        content="round-trip me",
        memory_type=MemoryType.WORKING,
        embedding=[0.0, 1.0],
        tags=["x", "y"],
        metadata={"k": 1},
    )
    data = original.model_dump(mode="json")
    restored = MemoryItem.model_validate(data)
    assert restored.id == original.id
    assert restored.content == original.content
    assert restored.memory_type == MemoryType.WORKING
    assert restored.embedding == [0.0, 1.0]
    assert restored.tags == ["x", "y"]
    assert restored.metadata == {"k": 1}
    assert restored.status == MemoryStatus.ACTIVE


def test_enum_serializes_as_string() -> None:
    m = MemoryItem(content="x", memory_type=MemoryType.EXPERIENTIAL)
    data = m.model_dump(mode="json")
    assert data["memory_type"] == "EXPERIENTIAL"
    assert data["status"] == "ACTIVE"


def test_edge_roundtrip() -> None:
    e = Edge(
        source_id="m1",
        target_id="m2",
        graph_type="causal",
        weight=0.75,
        metadata={"reason": "refund -> credit"},
    )
    restored = Edge.model_validate(e.model_dump(mode="json"))
    assert restored.source_id == "m1"
    assert restored.graph_type == "causal"
    assert restored.metadata == {"reason": "refund -> credit"}


def test_retention_policy_roundtrip() -> None:
    p = RetentionPolicy(default_ttl=timedelta(days=7), experiential_ttl=timedelta(days=365))
    restored = RetentionPolicy.model_validate(p.model_dump(mode="json"))
    assert restored.default_ttl == timedelta(days=7)
    assert restored.experiential_ttl == timedelta(days=365)
    assert restored.working_ttl == timedelta(hours=24)
