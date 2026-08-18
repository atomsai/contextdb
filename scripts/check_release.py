#!/usr/bin/env python3
"""Release-safety checks for the ContextDB package.

Run before publishing (the publish workflow runs it on every release):

* the release tag must match the version in ``pyproject.toml``;
* the sdist and wheel must contain ``LICENSE`` and ``NOTICE``;
* the archives must not contain secrets, key material, virtualenvs, or
  other scratch files (see the denylist below).
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_ARCHIVE_PARTS = (
    "/.env",
    ".env.",
    "secrets/",
    ".venv/",
)

FORBIDDEN_ARCHIVE_SUFFIXES = (".pem", ".p12", ".rtf", ".key")


def _archive_members(archive: Path) -> list[str]:
    if archive.suffixes[-2:] == [".tar", ".gz"] or archive.suffix == ".tar":
        with tarfile.open(archive) as tf:
            return [m.name for m in tf.getmembers() if m.isfile()]
    if archive.suffix == ".whl":
        with zipfile.ZipFile(archive) as zf:
            return zf.namelist()
    raise ValueError(f"unsupported archive: {archive}")


def _forbidden_member(name: str) -> str | None:
    normalized = name.replace("\\", "/")
    lowered = normalized.lower()
    for part in FORBIDDEN_ARCHIVE_PARTS:
        if part in lowered:
            return f"contains denied path {part!r}: {normalized}"
    for suffix in FORBIDDEN_ARCHIVE_SUFFIXES:
        if lowered.endswith(suffix):
            return f"contains denied suffix {suffix!r}: {normalized}"
    return None


def check_dist(dist_dir: Path) -> list[str]:
    errors: list[str] = []
    if not dist_dir.is_dir():
        return [f"dist directory missing: {dist_dir}"]
    archives = sorted(dist_dir.glob("*.whl")) + sorted(dist_dir.glob("*.tar.gz"))
    if not archives:
        return [f"no wheel/sdist in {dist_dir}"]
    wheels = [p for p in archives if p.suffix == ".whl"]
    if not wheels:
        errors.append("wheel missing from dist/")
    for archive in archives:
        try:
            members = _archive_members(archive)
        except (OSError, tarfile.TarError, zipfile.BadZipFile, ValueError) as exc:
            errors.append(f"{archive.name}: {exc}")
            continue
        if not any(m.endswith("LICENSE") or m.endswith("LICENSE.txt") for m in members):
            errors.append(f"{archive.name}: LICENSE not in archive")
        if not any(m.endswith("NOTICE") for m in members):
            errors.append(f"{archive.name}: NOTICE not in archive")
        for name in members:
            hit = _forbidden_member(name)
            if hit:
                errors.append(f"{archive.name}: {hit}")
        if archive.suffix == ".whl":
            for name in members:
                if name.endswith("/"):
                    continue
                if name.startswith("contextdb/") or ".dist-info/" in name:
                    continue
                errors.append(f"{archive.name}: unexpected wheel member {name}")
    return errors


def _pyproject_version() -> str:
    for line in (ROOT / "pyproject.toml").read_text(encoding="utf-8").splitlines():
        if line.startswith("version = "):
            return line.split("=", 1)[1].strip().strip('"')
    raise RuntimeError("version not found in pyproject.toml")


def check_release_tag(tag: str) -> list[str]:
    version = _pyproject_version()
    expected = f"v{version}"
    if tag != expected:
        return [f"release tag {tag!r} does not match pyproject version {expected!r}"]
    return []


def run(dist: Path | None, release_tag: str | None) -> int:
    errors: list[str] = []
    if dist is not None:
        errors.extend(check_dist(dist))
    if release_tag:
        errors.extend(check_release_tag(release_tag))
    if dist is None and not release_tag:
        errors.append("nothing to check: pass --dist and/or --release-tag")
    if errors:
        print("release check FAILED:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1
    print("release check passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dist",
        type=Path,
        default=None,
        help="Inspect built wheel/sdist in this directory",
    )
    parser.add_argument(
        "--release-tag",
        default=None,
        help="Require this git tag (e.g. v0.2.0) to match pyproject version",
    )
    args = parser.parse_args()
    return run(args.dist, args.release_tag)


if __name__ == "__main__":
    raise SystemExit(main())
