"""Tests for the PII detector."""

from __future__ import annotations

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
