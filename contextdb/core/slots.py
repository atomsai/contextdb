"""Versioned slot vocabulary + deterministic slotter.

Supersede is exactly as good as the entity+attribute keys. Free strings
mean "meeting"/"time", "Meeting"/"Time", and "appt"/"when" are three
slots and contradictions silently coexist. A published, versioned
vocabulary for the action-relevant classes — booking, contact, health,
identity, money, legal — plus a regex slotter that needs no LLM, is
what makes supersede work on a messy corpus, and what lets ``add_fast``
carry a slot *before* consolidation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SLOT_VOCAB_VERSION = "1"

# Canonical (entity, attribute) → class. The class drives the per-class
# corroboration bar in :class:`TrustPolicy`.
SLOT_VOCAB: dict[tuple[str, str], str] = {
    ("caller", "preferred_visit_day"): "booking",
    ("caller", "preferred_visit_time"): "booking",
    ("appointment", "day"): "booking",
    ("appointment", "time"): "booking",
    ("appointment", "status"): "booking",
    ("meeting", "time"): "booking",
    ("meeting", "day"): "booking",
    ("meeting", "location"): "booking",
    ("reservation", "party_size"): "booking",
    ("reservation", "time"): "booking",
    ("reservation", "day"): "booking",
    ("user", "email"): "contact",
    ("user", "phone"): "contact",
    ("user", "address"): "contact",
    ("user", "name"): "identity",
    ("account", "number"): "identity",
    ("account", "pin"): "identity",
    ("account", "tier"): "money",
    ("invoice", "number"): "money",
    ("invoice", "amount"): "money",
    ("user", "allergy"): "health",
    ("user", "medication"): "health",
    ("user", "peanut_allergy"): "health",
    ("office", "location"): "booking",
    ("deploy", "window"): "booking",
    ("user", "garage_code"): "identity",
    ("caller", "balance"): "money",
}

# Aliases fold messy extractor / human input onto the vocab.
_ENTITY_ALIASES: dict[str, str] = {
    "appt": "appointment",
    "booking": "appointment",
    "reservation": "reservation",
    "customer": "user",
    "client": "user",
    "patient": "user",
    "person": "user",
    "caller": "caller",
    "guest": "user",
}

_ATTRIBUTE_ALIASES: dict[str, str] = {
    "when": "time",
    "datetime": "time",
    "date": "day",
    "preferred_visit_day": "preferred_visit_day",
    "preferred_visit_time": "preferred_visit_time",
    "party": "party_size",
    "seats": "party_size",
    "covers": "party_size",
    "headcount": "party_size",
    "e-mail": "email",
    "mail": "email",
    "telephone": "phone",
    "mobile": "phone",
    "allergies": "allergy",
    "allergic": "allergy",
}


@dataclass(frozen=True)
class Slot:
    """A canonical (entity, attribute) pair plus its action class."""

    entity: str
    attribute: str
    slot_class: str | None
    vocab_version: str = SLOT_VOCAB_VERSION

    @property
    def known(self) -> bool:
        return self.slot_class is not None


def canonicalize_slot(entity: str | None, attribute: str | None) -> Slot | None:
    """Fold aliases and case onto the published vocabulary."""
    if not entity or not attribute:
        return None
    ent = _ENTITY_ALIASES.get(entity.casefold().strip(), entity.casefold().strip())
    attr = _ATTRIBUTE_ALIASES.get(
        attribute.casefold().strip(), attribute.casefold().strip()
    )
    # snake_case only
    ent = re.sub(r"[^a-z0-9]+", "_", ent).strip("_")
    attr = re.sub(r"[^a-z0-9]+", "_", attr).strip("_")
    if not ent or not attr:
        return None
    slot_class = SLOT_VOCAB.get((ent, attr))
    return Slot(entity=ent, attribute=attr, slot_class=slot_class)


_WEEKDAYS = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
)
_WEEKDAY_RE = re.compile(r"\b(" + "|".join(_WEEKDAYS) + r")\b", re.IGNORECASE)
_TIME_RE = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", re.IGNORECASE)
_WORD_NUMBERS: dict[str, str] = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
_PARTY_RE = re.compile(
    r"\b(?:make it|party of|table for|for)\s+"
    r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
    r"\s*(?:seats?|people|guests?)?\b",
    re.IGNORECASE,
)
_NEGATION_RE = re.compile(
    r"\b(?:don't|do not|doesn't|does not|never|no longer|not)\b",
    re.IGNORECASE,
)
_ALLERGY_RE = re.compile(
    r"\b(?:allerg(?:y|ic|ies)|anaphylax\w*)\b.*\b(peanuts?|tree nuts?|shellfish|"
    r"gluten|dairy|eggs?|soy|sesame|latex)\b|"
    r"\b(peanuts?|tree nuts?|shellfish|gluten|dairy|eggs?|soy|sesame|latex)\b.*"
    r"\b(?:allerg(?:y|ic|ies)|anaphylax\w*)\b",
    re.IGNORECASE,
)
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_ACCOUNT_NUM_RE = re.compile(
    r"\b(?:account(?:\s+number)?(?:\s+ends\s+in|\s+ending\s+in|\s+is)?|"
    r"ends\s+in|ending\s+in)\s+(\d{3,})\b",
    re.IGNORECASE,
)
_WISH_RE = re.compile(
    r"\b(?:i(?:'d| would) like|i want|can i|could i|hoping to|prefer)\b",
    re.IGNORECASE,
)


def infer_slot(content: str) -> Slot | None:
    """Deterministic, LLM-free slot assignment for the action-relevant classes.

    Used on ``add_fast`` (no LLM on the turn path) and as a bound on
    extractor output: an extracted slot the raw text does not support is
    discarded. False negatives are safe — the fact is still stored.
    """
    text = content.casefold()
    if _ALLERGY_RE.search(content):
        return canonicalize_slot("user", "allergy")
    if _EMAIL_RE.search(content) or "[email]" in text:
        return canonicalize_slot("user", "email")
    if _ACCOUNT_NUM_RE.search(content):
        return canonicalize_slot("account", "number")
    if "meeting" in text and _TIME_RE.search(content):
        return canonicalize_slot("meeting", "time")
    if "meeting" in text and _WEEKDAY_RE.search(content):
        return canonicalize_slot("meeting", "day")
    if _PARTY_RE.search(content):
        return canonicalize_slot("reservation", "party_size")
    if _WISH_RE.search(content) and _WEEKDAY_RE.search(content):
        return canonicalize_slot("caller", "preferred_visit_day")
    if _WISH_RE.search(content) and _TIME_RE.search(content):
        return canonicalize_slot("caller", "preferred_visit_time")
    if ("come in" in text or "stop by" in text or "appointment" in text) and _WEEKDAY_RE.search(
        content
    ):
        return canonicalize_slot("appointment", "day")
    if ("come in" in text or "appointment" in text) and _TIME_RE.search(content):
        return canonicalize_slot("appointment", "time")
    return None


def infer_negation(content: str) -> bool:
    """True when the utterance denies rather than asserts the slot value."""
    return bool(_NEGATION_RE.search(content))


def extract_weekday(content: str) -> str | None:
    match = _WEEKDAY_RE.search(content)
    return match.group(1).casefold() if match else None


def extract_time(content: str) -> str | None:
    match = _TIME_RE.search(content)
    if not match:
        return None
    hour = int(match.group(1))
    minute = match.group(2) or "00"
    meridiem = match.group(3).casefold()
    return f"{hour}:{minute}{meridiem}"


def extract_party_size(content: str) -> str | None:
    match = _PARTY_RE.search(content)
    if not match:
        return None
    raw = match.group(1).casefold()
    return _WORD_NUMBERS.get(raw, raw)


def canonical_slot_value(content: str, slot: Slot | None) -> str:
    """Normalize the *value* inside a slot so 'Thursday' == 'thursday afternoon'."""
    if slot is None:
        return " ".join(content.casefold().split()).strip(" .!?")
    if slot.attribute in {"day", "preferred_visit_day"}:
        day = extract_weekday(content)
        if day:
            return day
    if slot.attribute in {"time", "preferred_visit_time"}:
        t = extract_time(content)
        if t:
            return t
    if slot.attribute == "party_size":
        n = extract_party_size(content)
        if n:
            return n
    if slot.entity == "account" and slot.attribute == "number":
        match = _ACCOUNT_NUM_RE.search(content)
        if match:
            return match.group(1)
    if slot.attribute == "allergy":
        match = _ALLERGY_RE.search(content)
        if match:
            allergen = (match.group(1) or match.group(2)).casefold()
            if allergen.endswith("s") and allergen not in {"shellfish"}:
                allergen = allergen[:-1]
            denied = infer_negation(content)
            return f"{'no-' if denied else ''}{allergen}"
    return " ".join(content.casefold().split()).strip(" .!?")
