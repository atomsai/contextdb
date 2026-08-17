"""Basic version test."""

from __future__ import annotations

from pathlib import Path


def _pyproject_version() -> str:
    for line in Path("pyproject.toml").read_text(encoding="utf-8").splitlines():
        if line.startswith("version = "):
            return line.split("=", 1)[1].strip().strip('"')
    raise AssertionError("version not found in pyproject.toml")


def test_version() -> None:
    import contextdb

    assert contextdb.__version__ == _pyproject_version()
