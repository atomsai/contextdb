"""Tests for the PII detector."""

from __future__ import annotations

import pytest

from contextdb.core.models import PIIType
from contextdb.privacy.pii_detector import PIIDetector


def test_detects_email() -> None:
    det = PIIDetector()
    spans = det.detect("Contact me at foo@bar.com please.")
    assert any(s.pii_type == PIIType.EMAIL for s in spans)


def test_detects_ssn_and_phone() -> None:
    det = PIIDetector()
    spans = det.detect("SSN 123-45-6789 phone 415-555-1212")
    kinds = {s.pii_type for s in spans}
    assert PIIType.SSN in kinds
    assert PIIType.PHONE in kinds


def test_redact_replaces_spans() -> None:
    det = PIIDetector()
    out, anns = det.process("email me at foo@bar.com")
    assert "[EMAIL]" in out
    assert "foo@bar.com" not in out
    assert len(anns) == 1


def test_allow_action_leaves_text_intact() -> None:
    det = PIIDetector(action="allow")
    out, anns = det.process("email foo@bar.com")
    assert "foo@bar.com" in out
    assert len(anns) == 1


def test_pii_encrypt_decrypt_roundtrip() -> None:
    """encrypt action must replace text with placeholders and let decrypt() recover originals."""
    det = PIIDetector(action="encrypt", encryption_key="test-secret-abc123")
    out, anns = det.process("Reach Alex at alex@example.com or 415-555-1212.")
    assert "[EMAIL]" in out and "[PHONE]" in out
    assert "alex@example.com" not in out
    assert "415-555-1212" not in out
    by_type = {a.pii_type: a for a in anns}
    assert det.decrypt(by_type[PIIType.EMAIL]) == "alex@example.com"
    assert det.decrypt(by_type[PIIType.PHONE]) == "415-555-1212"


def test_pii_decrypt_requires_key() -> None:
    det = PIIDetector(action="redact")
    _, anns = det.process("email foo@bar.com")
    with pytest.raises(ValueError):
        det.decrypt(anns[0])


def test_pii_encrypt_without_key_falls_back_to_redact(caplog: pytest.LogCaptureFixture) -> None:
    """Missing key: warn and degrade to plain redact (originals not recoverable)."""
    import logging

    with caplog.at_level(logging.WARNING):
        det = PIIDetector(action="encrypt")
    assert any("encrypt" in r.message.lower() for r in caplog.records)
    out, anns = det.process("ping foo@bar.com")
    assert "[EMAIL]" in out
    # Plaintext original is preserved since no key was available.
    assert anns[0].original == "foo@bar.com"
