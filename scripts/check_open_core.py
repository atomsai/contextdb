#!/usr/bin/env python3
"""Open-core boundary and accidental-publication checks.

Cloud may depend on this SDK. This SDK must never depend on cloud.
Entitlements are server-side. Premium code does not live in Apache
modules behind a flag. See OPEN_CORE.md.
"""

from __future__ import annotations

import argparse
import re
import sys
import tarfile
import zipfile
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SDK_DIR = ROOT / "contextdb"

FORBIDDEN_DIRS = (
    "contextdb_cloud",
    "contextdb/cloud",
    "contextdb/entitlements",
    "contextdb/billing",
)

# Operator/bootstrap files and pre-release planning docs stay out of
# the public Apache tree.
FORBIDDEN_FILES = (
    "scripts/setup-contextdb-cloud.sh",
    "TASKS.md",
    "PRD.md",
)

FORBIDDEN_IMPORT = re.compile(
    r"^\s*(?:from|import)\s+"
    r"(contextdb_cloud|contextdb\.cloud|contextdb\.entitlements|contextdb\.billing)\b",
    re.MULTILINE,
)

# Local entitlement / license gates — not optional runtime pathways.
FORBIDDEN_IDENT = re.compile(
    r"\b("
    r"license_key|entitlement_token|entitlement_id|is_entitled|"
    r"paid_tier|cloud_entitlement|requires_entitlement"
    r")\b"
)

PAID_TIER_PHRASE = re.compile(r"paid\s+tier", re.IGNORECASE)

FORBIDDEN_ARCHIVE_PARTS = (
    "contextdb_cloud",
    "contextdb/cloud",
    "contextdb/entitlements",
    "contextdb/billing",
    "/.env",
    ".env.",
    "secrets/",
    "setup-contextdb-cloud.sh",
    "TASKS.md",
    "PRD.md",
)

FORBIDDEN_ARCHIVE_SUFFIXES = (".pem", ".p12", ".rtf", ".key")

GOVERNANCE_FILES = (
    "OPEN_CORE.md",
    "COMMERCIAL.md",
    "NOTICE",
    "CONTRIBUTING.md",
    "LICENSE",
    "legal/CLA-individual.md",
    "legal/CLA-entity.md",
    ".github/CODEOWNERS",
)


def _iter_sdk_python() -> Iterable[Path]:
    yield from sorted(SDK_DIR.rglob("*.py"))


def check_governance_docs() -> list[str]:
    errors: list[str] = []
    for rel in GOVERNANCE_FILES:
        path = ROOT / rel
        if not path.is_file():
            errors.append(f"missing governance file: {rel}")
    open_core = ROOT / "OPEN_CORE.md"
    if open_core.is_file():
        text = open_core.read_text(encoding="utf-8")
        for needle in ("2554dae", "Apache-2.0", "contextdb-cloud", "server-side"):
            if needle not in text:
                errors.append(f"OPEN_CORE.md must document {needle!r}")
    return errors


def check_forbidden_paths() -> list[str]:
    errors: list[str] = []
    for rel in FORBIDDEN_DIRS:
        path = ROOT / rel
        if path.exists():
            errors.append(f"premium/cloud path must not exist in the Apache tree: {rel}")
    for rel in FORBIDDEN_FILES:
        path = ROOT / rel
        if path.exists():
            errors.append(f"internal/bootstrap file must not exist in the Apache tree: {rel}")
    return errors


def check_sdk_source() -> list[str]:
    errors: list[str] = []
    if not SDK_DIR.is_dir():
        return ["missing contextdb/ package"]
    for path in _iter_sdk_python():
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(ROOT).as_posix()
        if FORBIDDEN_IMPORT.search(text):
            errors.append(f"{rel}: imports a cloud/entitlement package")
        if FORBIDDEN_IDENT.search(text):
            errors.append(f"{rel}: local entitlement/license identifier")
        if PAID_TIER_PHRASE.search(text):
            errors.append(f"{rel}: 'paid tier' does not belong in the Apache SDK")
    return errors


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
            return f"contains forbidden path {part!r}: {normalized}"
    for suffix in FORBIDDEN_ARCHIVE_SUFFIXES:
        if lowered.endswith(suffix):
            return f"contains forbidden suffix {suffix}: {normalized}"
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
            # wheels put LICENSE in dist-info; sdists at the root
            if archive.suffix == ".whl":
                if not any("LICENSE" in m for m in members):
                    errors.append(f"{archive.name}: LICENSE not in wheel")
            else:
                errors.append(f"{archive.name}: LICENSE not in sdist")
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
    errors.extend(check_governance_docs())
    errors.extend(check_forbidden_paths())
    errors.extend(check_sdk_source())
    if dist is not None:
        errors.extend(check_dist(dist))
    if release_tag:
        errors.extend(check_release_tag(release_tag))
    if errors:
        print("open-core boundary check FAILED:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1
    print("open-core boundary check passed")
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
