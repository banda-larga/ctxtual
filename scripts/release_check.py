"""
Release helpers used by .github/workflows/release.yml.

    python scripts/release_check.py version v0.3.1   # tag must match pyproject + __init__
    python scripts/release_check.py notes 0.3.1      # print that CHANGELOG section
"""

import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def check_version(tag: str) -> str:
    if not re.fullmatch(r"v\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?", tag):
        sys.exit(f"tag {tag!r} is not of the form vX.Y.Z")
    version = tag[1:]
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    init = re.search(
        r'__version__ = "([^"]+)"', (ROOT / "src/ctxtual/__init__.py").read_text()
    )
    found = {
        "pyproject.toml": project,
        "src/ctxtual/__init__.py": init.group(1) if init else None,
    }
    wrong = {where: v for where, v in found.items() if v != version}
    if wrong:
        sys.exit(f"tag {tag} does not match the package version: {wrong}")
    if not changelog_section(version):
        sys.exit(f"CHANGELOG.md has no '## {version}' section")
    return version


def changelog_section(version: str) -> str:
    text = (ROOT / "CHANGELOG.md").read_text()
    match = re.search(
        rf"^## {re.escape(version)}\s*$(.*?)(?=^## |\Z)", text, re.M | re.S
    )
    return match.group(1).strip() if match else ""


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("version", "notes"):
        sys.exit(__doc__)
    if sys.argv[1] == "version":
        print(check_version(sys.argv[2]))
    else:
        version = sys.argv[2].removeprefix("v")
        body = changelog_section(version)
        if not body:
            sys.exit(f"CHANGELOG.md has no '## {version}' section")
        print(
            f'{body}\n\n**Install:** `pip install "ctxtual[mcp]=={version}"` · '
            f"[PyPI](https://pypi.org/project/ctxtual/{version}/)"
        )
