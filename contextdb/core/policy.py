"""Trust policy — the action bar as data, not a hardcoded ``if``.

A hospital wants "health facts need 3 independent sources." A restaurant
wants "a wish confirmed in-session is enough." Both are the same product
with different constants. Encoding the bar as :class:`TrustPolicy` means
``confirm()``'s semantics stay "whatever the active policy says," not
"whatever 0.7/2 said that week."
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from contextdb.core.models import ACTION_CONFIDENCE_THRESHOLD, ACTION_CORROBORATION_THRESHOLD

if TYPE_CHECKING:
    from contextdb.core.models import MemoryItem


class TrustPolicy(BaseModel):
    """Declarative verify-before-act bar.

    A fact is trusted for action when it is action-relevant AND any of:

    * independent corroboration count >= :attr:`corroboration_threshold`
    * first-party (``user_stated``) at confidence >= :attr:`confidence_threshold`
    * it has been explicitly confirmed via :meth:`FactualMemory.confirm`

    Per-class overrides (``health``, ``legal``, ``identity``, ``money``,
    ``booking``, ``contact``) raise the corroboration bar for that class
    only. Unknown classes use the default.
    """

    confidence_threshold: float = Field(default=ACTION_CONFIDENCE_THRESHOLD, ge=0.0, le=1.0)
    corroboration_threshold: int = Field(default=ACTION_CORROBORATION_THRESHOLD, ge=1)
    class_corroboration: dict[str, int] = Field(
        default_factory=lambda: {
            "health": 3,
            "legal": 3,
            "identity": 3,
            "money": 2,
            "booking": 2,
            "contact": 2,
        }
    )
    # Similarity below this is abstention: "nothing on file," not a guess.
    # -1.0 disables the floor. 0.0 is NOT off — it drops the negative
    # half of cosine space, which a mock (and some real) embedder uses.
    relevance_floor: float = Field(
        default=-1.0,
        ge=-1.0,
        le=1.0,
        description=(
            "Minimum cosine for a recall hit. -1 disables the floor (safe "
            "default for small stores / mock embedders). With a production "
            "embedder, 0.15–0.25 is the right band — below that, top-1 "
            "similarity is a guess, not a memory."
        ),
    )
    # Unknown (non-vocabulary) slots cannot gate actions via the
    # first-party shortcut; they still can via corroboration or confirm().
    unknown_slots_untrusted: bool = True
    # Classes that may never skip corroboration just because the speaker
    # is the user. A hospital does not plate a drug because the patient
    # once said the dose; they confirm or they wait for a second source.
    first_party_excluded_classes: list[str] = Field(default_factory=list)

    def corroboration_needed(self, slot_class: str | None) -> int:
        if slot_class and slot_class in self.class_corroboration:
            return self.class_corroboration[slot_class]
        return self.corroboration_threshold

    def is_trusted(self, item: MemoryItem) -> bool:
        """Whether ``item`` may gate an action under this policy."""
        if not item.action_relevant:
            return False
        if item.injection_suspect:
            return False
        if item.confirmed:
            return True
        needed = self.corroboration_needed(item.slot_class)
        if item.independent_corroboration >= needed:
            return True
        if (
            self.unknown_slots_untrusted
            and item.entity_key
            and item.attribute_key
            and item.slot_class is None
        ):
            return False
        if item.slot_class in self.first_party_excluded_classes:
            return False
        return (
            item.epistemic_source == "user_stated"
            and item.confidence >= self.confidence_threshold
        )

    def requires_confirmation(self, item: MemoryItem) -> bool:
        return item.action_relevant and not self.is_trusted(item)

    @classmethod
    def hospital(cls) -> TrustPolicy:
        """Health / legal / identity never skip corroboration."""
        return cls(
            confidence_threshold=0.95,
            corroboration_threshold=2,
            class_corroboration={
                "health": 3,
                "legal": 3,
                "identity": 3,
                "money": 3,
                "booking": 2,
                "contact": 2,
            },
            first_party_excluded_classes=["health", "legal", "identity"],
        )

    @classmethod
    def restaurant(cls) -> TrustPolicy:
        """Bookings may be first-party; allergies still need a second source."""
        return cls(
            class_corroboration={
                "health": 3,
                "legal": 3,
                "identity": 2,
                "money": 2,
                "booking": 2,
                "contact": 2,
            },
            first_party_excluded_classes=["health", "legal"],
        )


DEFAULT_TRUST_POLICY = TrustPolicy()
