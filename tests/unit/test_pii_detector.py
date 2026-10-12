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


def test_names_and_addresses_are_opt_in() -> None:
    text = "My name is Priya Sharma and I live at 12 Elm Street, Springfield."
    assert PIIDetector().process(text)[0] == text

    out, anns = PIIDetector(detect_names_and_addresses=True).process(text)
    assert out == "My name is [NAME] and I live at [ADDRESS], Springfield."
    assert {a.pii_type for a in anns} == {PIIType.NAME, PIIType.ADDRESS}
    assert {a.original for a in anns} == {"Priya Sharma", "12 Elm Street"}


@pytest.mark.parametrize(
    "text",
    [
        "Call me tomorrow about the order",
        "This is Monday's schedule",
        "My name is not important",
        "Order 12 shipped on Main",
        "We need 3 more chairs",
    ],
)
def test_name_and_address_patterns_need_strong_cues(text: str) -> None:
    assert PIIDetector(detect_names_and_addresses=True).detect(text) == []


def test_cued_names_and_suffixed_streets_are_found() -> None:
    det = PIIDetector(detect_names_and_addresses=True)
    found = {
        (a.pii_type, a.original)
        for a in det.detect("Hi, this is Jordan Rivera. Ship to 4410 Lake Shore Dr. please.")
    }
    assert found == {(PIIType.NAME, "Jordan Rivera"), (PIIType.ADDRESS, "4410 Lake Shore Dr.")}
