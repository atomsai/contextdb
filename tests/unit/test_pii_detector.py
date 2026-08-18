"""Tests for the PII detector."""

from __future__ import annotations

import pytest

from contextdb.core.exceptions import ConfigError
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


def test_pii_encrypt_annotations_never_hold_plaintext() -> None:
    """In encryption mode the annotation originals must be ciphertext —
    the persisted store row must not carry the plaintext span."""
    det = PIIDetector(action="encrypt", encryption_key="test-secret-abc123")
    _, anns = det.process("SSN 123-45-6789, email alex@example.com, card 4111 1111 1111 1111")
    assert len(anns) == 3
    for ann in anns:
        assert ann.original not in {"123-45-6789", "alex@example.com", "4111 1111 1111 1111"}
        assert "123-45-6789" not in ann.original
        assert "alex@example.com" not in ann.original
        assert "4111" not in ann.original


def test_pii_decrypt_requires_key() -> None:
    det = PIIDetector(action="redact")
    _, anns = det.process("email foo@bar.com")
    with pytest.raises(ValueError):
        det.decrypt(anns[0])


def test_pii_encrypt_without_key_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing key: encrypt must refuse to construct, not silently degrade
    to redact (which would store plaintext originals)."""
    monkeypatch.delenv("CONTEXTDB_PII_KEY", raising=False)
    with pytest.raises(ConfigError, match="encrypt"):
        PIIDetector(action="encrypt")


def test_pii_encrypt_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTEXTDB_PII_KEY", "env-secret")
    det = PIIDetector(action="encrypt")
    out, anns = det.process("ping foo@bar.com")
    assert "[EMAIL]" in out
    assert anns[0].original != "foo@bar.com"
    assert det.decrypt(anns[0]) == "foo@bar.com"
