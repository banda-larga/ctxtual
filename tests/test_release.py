"""The version in pyproject, __init__ and CHANGELOG must agree (see scripts/release_check.py)."""

import subprocess
import sys
import tomllib
from pathlib import Path

import ctxtual

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "release_check.py"


def run(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], capture_output=True, text=True
    )


def test_versions_and_changelog_agree():
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    assert ctxtual.__version__ == version
    check = run("version", f"v{version}")
    assert check.returncode == 0, check.stderr
    notes = run("notes", version)
    assert notes.returncode == 0 and f"ctxtual/{version}/" in notes.stdout


def test_mismatched_or_malformed_tags_are_rejected():
    assert "does not match" in run("version", "v0.0.1").stderr
    assert "not of the form" in run("version", "0.0.1").stderr
