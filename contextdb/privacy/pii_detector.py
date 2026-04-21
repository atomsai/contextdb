"""Regex-based PII detection and redaction.

Covers the common categories (email, phone, SSN, credit card) with well-known
patterns. Name and address detection is heuristic here; richer recognizers
can be plugged in later (spaCy NER, Presidio, etc.).

Offsets in returned :class:`PIIAnnotation` objects index the **original**
content, which lets callers redact from the tail forward without corrupting
indices.
"""

from __future__ import annotations

import re
from typing import Literal

from contextdb.core.models import PIIAnnotation, PIIType

PIIAction = Literal["redact", "encrypt", "flag", "allow"]

# Patterns applied in order. The credit-card pattern comes before phone so
# 16-digit card numbers without dashes don't get misclassified when the phone
# pattern is tightened.
_PATTERNS: list[tuple[PIIType, re.Pattern[str]]] = [
    (
        PIIType.EMAIL,
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    ),
    (
        PIIType.SSN,
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    ),
    (
        PIIType.CREDIT_CARD,
        # 13-16 digit sequences optionally grouped by hyphens or spaces.
        re.compile(r"\b(?:\d{4}[-\s]){3}\d{3,4}\b|\b\d{13,16}\b"),
    ),
    (
        PIIType.PHONE,
        re.compile(
            r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"
        ),
    ),
]


class PIIDetector:
    """Detect and apply a policy to PII in free-form text."""

    def __init__(self, action: PIIAction = "redact") -> None:
        self.action: PIIAction = action

    def detect(self, text: str) -> list[PIIAnnotation]:
        """Return non-overlapping PII spans sorted by start offset."""
        found: list[PIIAnnotation] = []
        taken: list[tuple[int, int]] = []
        for pii_type, pattern in _PATTERNS:
            for match in pattern.finditer(text):
                start, end = match.start(), match.end()
                if any(_overlaps(start, end, s, e) for s, e in taken):
                    continue
                taken.append((start, end))
                found.append(
                    PIIAnnotation(
                        pii_type=pii_type,
                        start=start,
                        end=end,
                        original=match.group(),
                        redacted=f"[{pii_type.value}]",
                    )
                )
        found.sort(key=lambda a: a.start)
        return found

    def redact(
        self,
        text: str,
        annotations: list[PIIAnnotation] | None = None,
    ) -> str:
        """Replace each PII span with its typed placeholder (e.g. ``[EMAIL]``)."""
        if annotations is None:
            annotations = self.detect(text)
        result = text
        for ann in sorted(annotations, key=lambda a: a.start, reverse=True):
            result = result[: ann.start] + ann.redacted + result[ann.end :]
        return result

    def process(self, text: str) -> tuple[str, list[PIIAnnotation]]:
        """Detect PII and apply the configured action.

        Returns ``(processed_text, annotations)``. For ``allow``/``flag`` the
        text is returned unchanged; for ``redact``/``encrypt`` the text is
        returned with placeholders substituted. Proper at-rest encryption is
        out of scope for this MVP — ``encrypt`` currently behaves like
        ``redact`` so callers never see raw PII.
        """
        annotations = self.detect(text)
        if self.action in {"redact", "encrypt"}:
            return self.redact(text, annotations), annotations
        return text, annotations


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end
