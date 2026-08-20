"""Open-core boundary checks stay executable and detect private imports."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
CHECK = ROOT / "scripts" / "check_open_core_boundary.py"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("open_core_check", CHECK)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_open_core_boundary_script() -> None:
    result = subprocess.run(
        [sys.executable, str(CHECK)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_private_cloud_import_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "contextdb"
    source.mkdir()
    (source / "leak.py").write_text(
        "from contextdb_cloud.gateway import app\n",
        encoding="utf-8",
    )
    errors = _module().check_source_tree(source)
    assert any("imports private Cloud code" in error for error in errors)


def test_commercial_gate_path_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "contextdb"
    (source / "billing").mkdir(parents=True)
    (source / "billing" / "plans.py").write_text(
        "PLAN = 'enterprise'\n",
        encoding="utf-8",
    )
    errors = _module().check_source_tree(source)
    assert any("private capability path" in error for error in errors)


def test_sdk_pr_requires_one_non_operated_classification() -> None:
    check = _module().check_pr_classification
    assert check("- [x] Semantic — offline correctness") == []
    assert check("- [X] Contract — interface") == []
    assert check("- [x] Reference — single node") == []
    assert check("- [x] No capability change — docs") == []
    assert check("") != []
    assert check("- [x] Semantic\n- [x] Contract") != []
    assert check("- [x] Operated — hosted behavior") != []
