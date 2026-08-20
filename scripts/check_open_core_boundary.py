#!/usr/bin/env python3
"""Mechanical checks for the binding ContextDB open-core constitution."""

from __future__ import annotations

import ast
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANONICAL = (
    "Open the single-node semantics and reference engine. Keep multi-tenant "
    "coordination, operated guarantees, governance, and proof private."
)
CHANGE_CLASSES = (
    "No capability change",
    "Semantic",
    "Contract",
    "Reference",
    "Operated",
)
FORBIDDEN_SOURCE_PARTS = {
    "billing",
    "cloud",
    "console",
    "control_plane",
    "enterprise",
    "entitlements",
    "private_link",
    "residency",
    "scim",
    "sso",
}
FORBIDDEN_RUNTIME_TOKENS = {
    "CONTEXTDB_LICENSE_KEY",
    "enterprise_only",
    "if_paid",
    "requires_entitlement",
    "subscription_tier",
}


def _normalized(path: Path) -> str:
    return " ".join(path.read_text(encoding="utf-8").split())


def check_pr_classification(body: str) -> list[str]:
    selected = [
        change_class
        for change_class in CHANGE_CLASSES
        if re.search(
            rf"-\s*\[[xX]\]\s*{re.escape(change_class)}\b",
            body,
        )
    ]
    if len(selected) != 1:
        return [
            "select exactly one open-core classification in the PR body; "
            f"found {len(selected)}"
        ]
    if selected[0] == "Operated":
        return ["Operated changes belong in private Cloud, not the public SDK"]
    return []


def check_source_tree(source: Path) -> list[str]:
    errors: list[str] = []
    for path in source.rglob("*.py"):
        relative = path.relative_to(source)
        lowered_parts = {part.lower() for part in relative.parts}
        denied_parts = sorted(lowered_parts & FORBIDDEN_SOURCE_PARTS)
        if denied_parts:
            errors.append(
                f"private capability path in SDK: {relative} "
                f"({', '.join(denied_parts)})"
            )
        text = path.read_text(encoding="utf-8")
        for token in sorted(FORBIDDEN_RUNTIME_TOKENS):
            if token in text:
                errors.append(f"commercial gate token {token!r} in {relative}")
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError as exc:
            errors.append(f"cannot parse {relative}: {exc}")
            continue
        for node in ast.walk(tree):
            names: list[str]
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            else:
                continue
            if any(
                name == "contextdb_cloud"
                or name.startswith("contextdb_cloud.")
                for name in names
            ):
                errors.append(f"SDK imports private Cloud code: {relative}")
    return errors


def check_root(root: Path) -> list[str]:
    errors: list[str] = []
    constitution = root / "OPEN_CORE.md"
    rule = root / ".cursor" / "rules" / "open-core-boundary.mdc"
    template = root / ".github" / "pull_request_template.md"
    required = (constitution, rule, template)
    for path in required:
        if not path.is_file():
            errors.append(f"boundary artifact missing: {path.relative_to(root)}")
    if constitution.is_file() and CANONICAL not in _normalized(constitution):
        errors.append("OPEN_CORE.md is missing the canonical boundary sentence")
    if rule.is_file() and "alwaysApply: true" not in rule.read_text(
        encoding="utf-8"
    ):
        errors.append("open-core Cursor rule must always apply")
    if template.is_file():
        text = template.read_text(encoding="utf-8")
        for change_class in ("Semantic", "Contract", "Reference", "Operated"):
            if change_class not in text:
                errors.append(
                    f"pull-request template omits {change_class} classification"
                )
    pyproject = root / "pyproject.toml"
    if pyproject.is_file() and 'license = "Apache-2.0"' not in pyproject.read_text(
        encoding="utf-8"
    ):
        errors.append("public SDK project license must remain Apache-2.0")
    errors.extend(check_source_tree(root / "contextdb"))
    return errors


def main() -> int:
    errors = check_root(ROOT)
    if os.environ.get("CONTEXTDB_REQUIRE_PR_CLASSIFICATION") == "true":
        errors.extend(
            check_pr_classification(
                os.environ.get("CONTEXTDB_PR_BODY", "")
            )
        )
    if errors:
        print("open-core boundary check FAILED", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print("open-core boundary check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
