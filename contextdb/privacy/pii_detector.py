"""Regex-based PII detection, redaction, and encryption.

Covers the common categories (email, phone, SSN, credit card) with well-known
patterns. Name and address detection is heuristic here; richer recognizers
can be plugged in later (spaCy NER, Presidio, etc.).

Offsets in returned :class:`PIIAnnotation` objects index the **original**
content, which lets callers redact from the tail forward without corrupting
indices.

Three actions are supported:

* ``redact`` — replace each PII span with ``[<TYPE>]`` and keep plaintext
  in :attr:`PIIAnnotation.original` (useful for operator audit on a trusted
  store).
* ``encrypt`` — replace each PII span with ``[<TYPE>]`` but store a Fernet
  ciphertext in :attr:`PIIAnnotation.original`. Reversible via :meth:`decrypt`
  when the caller holds the key.
* ``flag`` / ``allow`` — leave text intact; only annotations are produced.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import re
from typing import Literal

from cryptography.fernet import Fernet, InvalidToken

from contextdb.core.models import PIIAnnotation, PIIType

PIIAction = Literal["redact", "encrypt", "flag", "allow"]

_logger = logging.getLogger(__name__)

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


def _derive_fernet_key(raw: str) -> bytes:
    """Derive a 32-byte urlsafe-base64 Fernet key from any string.

    We SHA-256 the input so users can supply a human-readable secret without
    having to know Fernet's key format. This is intentionally not PBKDF2 —
    the input is already expected to be a high-entropy operator secret, not
    a user-memorable password.
    """
    digest = hashlib.sha256(raw.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)


class PIIDetector:
    """Detect and apply a policy to PII in free-form text."""

    def __init__(
        self,
        action: PIIAction = "redact",
        encryption_key: str | None = None,
    ) -> None:
        self.action: PIIAction = action
        self._fernet: Fernet | None = None
        if action == "encrypt":
            key = encryption_key or os.environ.get("CONTEXTDB_PII_KEY")
            if key:
                self._fernet = Fernet(_derive_fernet_key(key))
            else:
                _logger.warning(
                    "PII action is 'encrypt' but no key is configured "
                    "(pass encryption_key= or set CONTEXTDB_PII_KEY). "
                    "Falling back to redact; originals will NOT be recoverable."
                )

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
                original = match.group()
                stored_original = (
                    self._fernet.encrypt(original.encode("utf-8")).decode("ascii")
                    if self._fernet is not None
                    else original
                )
                found.append(
                    PIIAnnotation(
                        pii_type=pii_type,
                        start=start,
                        end=end,
                        original=stored_original,
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
        returned with placeholders substituted. When action is ``encrypt``
        the :attr:`PIIAnnotation.original` field holds a Fernet ciphertext
        that can be round-tripped via :meth:`decrypt`.
        """
        annotations = self.detect(text)
        if self.action in {"redact", "encrypt"}:
            return self.redact(text, annotations), annotations
        return text, annotations

    def decrypt(self, annotation: PIIAnnotation) -> str:
        """Recover the plaintext original from an encrypted annotation.

        Raises :class:`ValueError` if the detector is not configured for
        encryption or if the ciphertext is tampered / wrong key.
        """
        if self._fernet is None:
            raise ValueError(
                "PIIDetector is not configured for encryption. Pass "
                "encryption_key= or set CONTEXTDB_PII_KEY when constructing."
            )
        try:
            return self._fernet.decrypt(annotation.original.encode("ascii")).decode(
                "utf-8"
            )
        except InvalidToken as exc:
            raise ValueError("Invalid or tampered PII ciphertext.") from exc


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end
