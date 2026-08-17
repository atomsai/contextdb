"""Open-core boundary — the SDK must not grow a cloud dependency."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check_open_core.py"


def test_open_core_boundary_script_passes() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_sdk_has_no_cloud_import() -> None:
    for path in (ROOT / "contextdb").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "contextdb_cloud" not in text
        assert "contextdb.cloud" not in text
