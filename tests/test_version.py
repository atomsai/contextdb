"""Basic version test."""

from __future__ import annotations


def test_version() -> None:
    import contextdb

    assert contextdb.__version__ == "0.1.0"
