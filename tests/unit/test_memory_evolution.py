"""Reference-contract tests for explicit deterministic memory evolution."""

from __future__ import annotations

import pytest

import contextdb
from contextdb import (
    ContextDB,
    EvolutionOperation,
    EvolutionOperationConflictError,
    EvolutionOutcome,
    EvolutionTargetNotFoundError,
    MemoryEvolutionResult,
    MemoryItem,
)


async def test_explicit_add_and_update_return_lineage(
    client: ContextDB,
) -> None:
    added = await client.factual.evolve(
        EvolutionOperation.ADD,
        "The profile color is blue",
        source="user_stated",
        entity="profile",
        attribute="color",
        user_id="alice",
    )

    assert isinstance(added, MemoryEvolutionResult)
    assert added.applied_operation == EvolutionOperation.ADD
    assert added.outcome == EvolutionOutcome.ADDED
    assert added.memory is not None
    assert added.previous_memory_ids == []
    assert added.deleted_memory_ids == []
    first = added.memory

    updated = await client.factual.evolve(
        "update",
        "The profile color is green",
        source="user_stated",
        target_memory_id=first.id,
        user_id="alice",
    )

    assert updated.applied_operation == EvolutionOperation.UPDATE
    assert updated.outcome == EvolutionOutcome.UPDATED
    assert updated.memory is not None
    assert updated.memory.id != first.id
    assert updated.previous_memory_ids == [first.id]
    predecessor = await client._require_store().get_raw(first.id)
    assert predecessor is not None
    assert predecessor.valid_until is not None
    assert predecessor.superseded_by == updated.memory.id
    # Historical/as-of recall depends on keeping the predecessor vector.
    assert first.id in await client._require_store().index_ids()


async def test_update_rejects_superseded_target_without_mutation(
    client: ContextDB,
) -> None:
    added = await client.factual.evolve(
        "add",
        "The profile color is blue",
        source="user_stated",
        entity="profile",
        attribute="color",
        user_id="alice",
    )
    assert added.memory is not None
    updated = await client.factual.evolve(
        "update",
        "The profile color is green",
        source="user_stated",
        target_memory_id=added.memory.id,
        user_id="alice",
    )
    assert updated.memory is not None
    version = updated.consistency_token.memory_version

    with pytest.raises(EvolutionTargetNotFoundError):
        await client.factual.evolve(
            "update",
            "The profile color is red",
            source="user_stated",
            target_memory_id=added.memory.id,
            user_id="alice",
        )

    assert (await client.consistency_token()).memory_version == version
    rows = await client._require_store().list_by_slot(
        "profile",
        "color",
        user_id="alice",
    )
    current = [row for row in rows if row.is_valid_at(client.clock())]
    assert [row.id for row in current] == [updated.memory.id]
    predecessor = next(row for row in rows if row.id == added.memory.id)
    assert predecessor.superseded_by == updated.memory.id


async def test_explicit_delete_can_erase_superseded_history(
    client: ContextDB,
) -> None:
    added = await client.factual.evolve(
        "add",
        "The profile color is blue",
        source="user_stated",
        entity="profile",
        attribute="color",
        user_id="alice",
    )
    assert added.memory is not None
    updated = await client.factual.evolve(
        "update",
        "The profile color is green",
        source="user_stated",
        target_memory_id=added.memory.id,
        user_id="alice",
    )
    assert updated.memory is not None

    deleted = await client.factual.evolve(
        "delete",
        target_memory_id=added.memory.id,
        user_id="alice",
    )

    assert deleted.deleted_memory_ids == [added.memory.id]
    assert (
        deleted.consistency_token.memory_version
        > updated.consistency_token.memory_version
    )
    store = client._require_store()
    assert await store.get_raw(added.memory.id) is None
    current = await store.get_raw(updated.memory.id)
    assert current is not None
    assert current.valid_until is None


async def test_explicit_add_same_speaker_is_true_noop(
    client: ContextDB,
) -> None:
    first = await client.factual.evolve(
        "add",
        "The profile color is blue",
        source="user_stated",
        entity="profile",
        attribute="color",
        user_id="alice",
    )
    assert first.memory is not None
    version = first.consistency_token.memory_version
    rows_before = await client._require_store().list_by_slot(
        "profile",
        "color",
        user_id="alice",
    )

    duplicate = await client.factual.evolve(
        "add",
        "The profile color is blue",
        source="user_stated",
        entity="profile",
        attribute="color",
        user_id="alice",
    )

    assert duplicate.applied_operation == EvolutionOperation.NOOP
    assert duplicate.outcome == EvolutionOutcome.NOOP
    assert duplicate.memory is not None
    assert duplicate.memory.id == first.memory.id
    assert duplicate.noop_reason == "same_speaker_same_value"
    assert duplicate.consistency_token.memory_version == version
    rows_after = await client._require_store().list_by_slot(
        "profile",
        "color",
        user_id="alice",
    )
    assert [row.id for row in rows_after] == [row.id for row in rows_before]
    assert client.audit is not None
    noop = [entry for entry in await client.audit.get_history() if entry.operation == "NOOP"][-1]
    assert noop.memory_id == first.memory.id
    assert noop.details == {
        "operation": "noop",
        "reason": "same_speaker_same_value",
        "entity": "profile",
        "attribute": "color",
    }


async def test_explicit_add_independent_corroboration_applies_update(
    client: ContextDB,
) -> None:
    first = await client.factual.evolve(
        "add",
        "The office is in Denver",
        source="third_party",
        entity="office",
        attribute="location",
        user_id="alice",
    )
    assert first.memory is not None
    other = contextdb.init(
        config=client.config,
        session_id="independent-session",
    )
    try:
        corroborated = await other.factual.evolve(
            "add",
            "The office is in Denver",
            source="third_party",
            entity="office",
            attribute="location",
            user_id="alice",
        )
    finally:
        await other.close()

    assert corroborated.applied_operation == EvolutionOperation.UPDATE
    assert corroborated.outcome == EvolutionOutcome.UPDATED
    assert corroborated.memory is not None
    assert corroborated.memory.id == first.memory.id
    assert corroborated.previous_memory_ids == [first.memory.id]
    assert corroborated.memory.independent_corroboration == 2


async def test_explicit_add_conflict_fails_without_superseding(
    client: ContextDB,
) -> None:
    first = await client.factual.evolve(
        "add",
        "The profile color is blue",
        source="user_stated",
        entity="profile",
        attribute="color",
        user_id="alice",
    )
    assert first.memory is not None
    version = first.consistency_token.memory_version

    with pytest.raises(EvolutionOperationConflictError):
        await client.factual.evolve(
            "add",
            "The profile color is green",
            source="user_stated",
            entity="profile",
            attribute="color",
            user_id="alice",
        )

    current = await client._require_store().list_by_slot(
        "profile",
        "color",
        user_id="alice",
    )
    assert [row.id for row in current if row.valid_until is None] == [first.memory.id]
    assert (await client.consistency_token()).memory_version == version


async def test_same_value_update_is_noop_then_unambiguous_slot_delete_succeeds(
    client: ContextDB,
) -> None:
    added = await client.factual.evolve(
        "add",
        "The profile color is blue",
        source="user_stated",
        entity="profile",
        attribute="color",
        user_id="alice",
    )
    assert added.memory is not None

    unchanged = await client.factual.evolve(
        "update",
        "The profile color is blue",
        source="user_stated",
        entity="profile",
        attribute="color",
        user_id="alice",
    )
    assert unchanged.applied_operation == EvolutionOperation.NOOP
    assert (
        unchanged.consistency_token.memory_version
        == added.consistency_token.memory_version
    )

    deleted = await client.factual.evolve(
        "delete",
        entity="profile",
        attribute="color",
        user_id="alice",
    )
    assert deleted.deleted_memory_ids == [added.memory.id]
    assert (
        deleted.consistency_token.memory_version
        > unchanged.consistency_token.memory_version
    )


async def test_independent_update_contests_and_exact_delete_is_verified(
    client: ContextDB,
) -> None:
    first = await client.factual.evolve(
        "add",
        "The meeting is at 3pm",
        source="user_stated",
        entity="meeting",
        attribute="time",
        user_id="alice",
    )
    assert first.memory is not None
    other = contextdb.init(
        config=client.config,
        session_id="independent-session",
    )
    try:
        contested = await other.factual.evolve(
            "update",
            "The meeting is at 4pm",
            source="user_stated",
            entity="meeting",
            attribute="time",
            user_id="alice",
        )
        assert contested.outcome == EvolutionOutcome.CONTESTED
        assert contested.applied_operation == EvolutionOperation.UPDATE
        assert contested.memory is not None
        assert contested.previous_memory_ids == [first.memory.id]

        with pytest.raises(EvolutionOperationConflictError):
            await other.factual.evolve(
                "delete",
                entity="meeting",
                attribute="time",
                user_id="alice",
            )

        before_delete = contested.consistency_token.memory_version
        deleted = await other.factual.evolve(
            "delete",
            target_memory_id=contested.memory.id,
            user_id="alice",
        )
        assert deleted.deleted_memory_ids == [contested.memory.id]
        assert deleted.memory is None
        assert deleted.consistency_token.memory_version > before_delete
        store = other._require_store()
        assert await store.get_raw(contested.memory.id) is None
        assert contested.memory.id not in await store.index_ids()
        assert await store.get_raw(first.memory.id) is not None
    finally:
        await other.close()


async def test_target_ids_are_scoped_and_missing_is_indistinguishable(
    client: ContextDB,
) -> None:
    alice = await client.factual.evolve(
        "add",
        "Alice profile color is blue",
        source="user_stated",
        entity="profile",
        attribute="color",
        user_id="alice",
    )
    assert alice.memory is not None

    for target in (alice.memory.id, "missing-memory"):
        with pytest.raises(EvolutionTargetNotFoundError):
            await client.factual.evolve(
                "delete",
                target_memory_id=target,
                user_id="bob",
            )
    assert await client._require_store().get_raw(alice.memory.id) is not None

    version = (await client.consistency_token()).memory_version
    for target in (alice.memory.id, "missing-memory"):
        with pytest.raises(EvolutionTargetNotFoundError):
            await client.factual.evolve(
                "noop",
                target_memory_id=target,
                noop_reason="already checked",
                user_id="bob",
            )
    assert (await client.consistency_token()).memory_version == version
    assert client.audit is not None
    assert not [
        entry
        for entry in await client.audit.get_history(user_id="bob")
        if entry.operation == "NOOP"
    ]


async def test_targeted_noop_returns_lineage_without_private_audit_values(
    client: ContextDB,
) -> None:
    raw_content = "The profile color is blue for SSN 123-45-6789"
    raw_user_id = "alice-private-user"
    added = await client.factual.evolve(
        "add",
        raw_content,
        source="user_stated",
        entity="Profile",
        attribute="Color",
        user_id=raw_user_id,
    )
    assert added.memory is not None
    assert added.memory.pii_annotations
    version = added.consistency_token.memory_version

    result = await client.factual.evolve(
        "noop",
        target_memory_id=added.memory.id,
        user_id=raw_user_id,
        noop_reason="duplicate reported by reviewer@example.com",
    )

    assert result.memory is not None
    assert result.memory.id == added.memory.id
    assert result.consistency_token.memory_version == version
    assert result.noop_reason == "duplicate reported by [EMAIL]"
    assert client.audit is not None
    entry = [
        entry
        for entry in await client.audit.get_history(memory_id=added.memory.id)
        if entry.operation == "NOOP"
    ][-1]
    assert entry.memory_id == added.memory.id
    assert entry.details == {
        "operation": "noop",
        "reason": "duplicate reported by [EMAIL]",
        "entity": "profile",
        "attribute": "color",
    }
    serialized_details = str(entry.details)
    for private_value in (
        raw_content,
        "123-45-6789",
        "reviewer@example.com",
        raw_user_id,
        added.memory.content,
        added.memory.pii_annotations[0].original,
    ):
        assert private_value not in serialized_details
    assert {"content", "old_value", "new_value", "user_id", "pii_annotations"}.isdisjoint(
        entry.details
    )


async def test_slot_noop_returns_unambiguous_current_memory(
    client: ContextDB,
) -> None:
    added = await client.factual.evolve(
        "add",
        "The user has a peanut allergy",
        source="user_stated",
        entity="Customer",
        attribute="Allergies",
        user_id="alice",
    )
    assert added.memory is not None
    version = added.consistency_token.memory_version

    result = await client.factual.evolve(
        "noop",
        entity="CUSTOMER",
        attribute="ALLERGIES",
        user_id="alice",
        noop_reason="already verified",
    )

    assert result.memory is not None
    assert result.memory.id == added.memory.id
    assert result.consistency_token.memory_version == version
    assert client.audit is not None
    entry = [
        entry
        for entry in await client.audit.get_history(memory_id=added.memory.id)
        if entry.operation == "NOOP"
    ][-1]
    assert entry.details == {
        "operation": "noop",
        "reason": "already verified",
        "entity": "user",
        "attribute": "allergy",
    }


async def test_slot_noop_rejects_ambiguous_contested_slot(
    client: ContextDB,
) -> None:
    first = await client.factual.evolve(
        "add",
        "The meeting is at 3pm",
        source="user_stated",
        entity="meeting",
        attribute="time",
        user_id="alice",
    )
    assert first.memory is not None
    other = contextdb.init(
        config=client.config,
        session_id="independent-session",
    )
    try:
        contested = await other.factual.evolve(
            "update",
            "The meeting is at 4pm",
            source="user_stated",
            entity="meeting",
            attribute="time",
            user_id="alice",
        )
        assert contested.outcome == EvolutionOutcome.CONTESTED
        version = contested.consistency_token.memory_version
        assert other.audit is not None
        noops_before = [
            entry
            for entry in await other.audit.get_history(user_id="alice")
            if entry.operation == "NOOP"
        ]

        with pytest.raises(EvolutionOperationConflictError, match="ambiguous"):
            await other.factual.evolve(
                "noop",
                entity="meeting",
                attribute="time",
                user_id="alice",
                noop_reason="already handled",
            )

        assert (await other.consistency_token()).memory_version == version
        noops_after = [
            entry
            for entry in await other.audit.get_history(user_id="alice")
            if entry.operation == "NOOP"
        ]
        assert [entry.id for entry in noops_after] == [
            entry.id for entry in noops_before
        ]
    finally:
        await other.close()


async def test_explicit_noop_is_pii_safe_and_does_not_mutate_store(
    client: ContextDB,
) -> None:
    await client._ensure_init()
    store = client._require_store()
    before = await store.consistency_token()
    rows_before = await store.list_memories(status=None, limit=100)

    result = await client.factual.evolve(
        "noop",
        user_id="alice",
        noop_reason="duplicate reported by alice@example.com",
    )

    assert result.outcome == EvolutionOutcome.NOOP
    assert result.memory is None
    assert result.consistency_token.memory_version == before.memory_version
    assert result.noop_reason is not None
    assert "alice@example.com" not in result.noop_reason
    assert "[EMAIL]" in result.noop_reason
    rows_after = await store.list_memories(status=None, limit=100)
    assert [row.id for row in rows_after] == [row.id for row in rows_before]
    assert client.audit is not None
    entry = [entry for entry in await client.audit.get_history() if entry.operation == "NOOP"][-1]
    assert entry.memory_id is None
    assert set(entry.details) == {"operation", "reason"}
    assert "alice@example.com" not in str(entry.model_dump(mode="json"))


async def test_explicit_evolution_never_calls_optional_rl_manager(
    client: ContextDB,
) -> None:
    class FailingManager:
        async def decide(
            self,
            content: str,
            candidates: list[MemoryItem],
        ) -> dict[str, object]:
            raise AssertionError(
                f"RL manager was called for {content!r} with {len(candidates)} candidates"
            )

    await client._ensure_init()
    client._rl_manager = FailingManager()  # type: ignore[assignment]
    result = await client.factual.evolve(
        "add",
        "The profile color is blue",
        source="user_stated",
        entity="profile",
        attribute="color",
        user_id="alice",
    )
    assert result.outcome == EvolutionOutcome.ADDED


async def test_existing_add_return_type_and_duplicate_id_are_compatible(
    client: ContextDB,
) -> None:
    first = await client.factual.add(
        "The profile color is blue",
        source="user_stated",
        entity="profile",
        attribute="color",
        user_id="alice",
    )
    assert isinstance(first, MemoryItem)
    version = (await client.consistency_token()).memory_version

    duplicate = await client.factual.add(
        "The profile color is blue",
        source="user_stated",
        entity="profile",
        attribute="color",
        user_id="alice",
    )

    assert isinstance(duplicate, MemoryItem)
    assert duplicate.id == first.id
    assert (await client.consistency_token()).memory_version == version
    assert client.audit is not None
    noop = [
        entry
        for entry in await client.audit.get_history(memory_id=first.id)
        if entry.operation == "NOOP"
    ][-1]
    assert noop.memory_id == first.id
    assert noop.details == {
        "operation": "noop",
        "reason": "same_speaker_same_value",
        "entity": "profile",
        "attribute": "color",
    }
