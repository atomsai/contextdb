"""Unit tests for clock, slot vocab, TrustPolicy, and verify-before-act."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from contextdb.core.clock import FrozenClock
from contextdb.core.models import MemoryItem
from contextdb.core.policy import TrustPolicy
from contextdb.core.slots import (
    SLOT_VOCAB_VERSION,
    canonical_slot_value,
    canonicalize_slot,
    extract_party_size,
    infer_slot,
)


def test_frozen_clock_rejects_naive_datetime() -> None:
    with pytest.raises(ValueError, match="timezone"):
        FrozenClock(datetime(2026, 1, 1, 12, 0))


def test_frozen_clock_advances() -> None:
    clock = FrozenClock(datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc))
    clock.advance(hours=2)
    assert clock.now.hour == 14


def test_canonicalize_slot_aliases() -> None:
    slot = canonicalize_slot("Customer", "Allergies")
    assert slot is not None
    assert slot.entity == "user"
    assert slot.attribute == "allergy"
    assert slot.slot_class == "health"
    assert slot.vocab_version == SLOT_VOCAB_VERSION


def test_infer_slot_wish_and_party() -> None:
    wish = infer_slot("I'd like to come in Thursday afternoon")
    assert wish is not None
    assert (wish.entity, wish.attribute) == ("caller", "preferred_visit_day")
    assert canonical_slot_value("I'd like to come in Thursday afternoon", wish) == "thursday"

    party = infer_slot("Make it four seats")
    assert party is not None
    assert party.attribute == "party_size"
    assert extract_party_size("Make it four seats") == "4"
    assert canonical_slot_value("Make it four seats", party) == "4"


def test_hospital_policy_excludes_first_party_health() -> None:
    item = MemoryItem(
        content="severe peanut allergy",
        epistemic_source="user_stated",
        confidence=0.99,
        action_relevant=True,
        entity_key="user",
        attribute_key="allergy",
        slot_class="health",
        corroborated_by=["user:a"],
    )
    assert TrustPolicy().is_trusted(item) is True
    assert TrustPolicy.hospital().is_trusted(item) is False
    item.confirmed = True
    assert TrustPolicy.hospital().is_trusted(item) is True


def test_unknown_slot_is_untrusted_until_corroborated() -> None:
    item = MemoryItem(
        content="the flux capacitor is set to 1.21",
        epistemic_source="user_stated",
        confidence=0.99,
        action_relevant=True,
        entity_key="flux",
        attribute_key="setting",
        slot_class=None,
        corroborated_by=["user:a"],
    )
    policy = TrustPolicy()
    assert policy.is_trusted(item) is False
    item.corroborated_by = ["user:a", "session:b"]
    assert policy.is_trusted(item) is True
