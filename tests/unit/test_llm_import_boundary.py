"""The OpenAI client libraries stay behind the LLM provider seam."""

from __future__ import annotations

import ast
import pathlib
import subprocess

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SEAM_DIR = pathlib.PurePosixPath("gubbi/extraction/llm")
_GUARDED_PACKAGES = frozenset({"openai", "pydantic_ai", "httpx2"})
# Adapter tests drive a real client over an httpx2 transport, so they import
# the client libraries directly. Nothing else outside the seam may.
_EXEMPT_FILES = frozenset(
    {
        pathlib.PurePosixPath("tests/unit/test_openai_compatible_provider.py"),
        pathlib.PurePosixPath("tests/extraction/test_openai_compatible_live.py"),
    }
)


def _imported_top_level_packages(source: str) -> set[str]:
    packages: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            packages.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            packages.add(node.module.split(".")[0])
    return packages


def _tracked_python_files() -> list[pathlib.PurePosixPath]:
    listed = subprocess.run(
        ["git", "ls-files", "-z", "--", "*.py"],  # noqa: S607 -- git from PATH
        cwd=_REPO_ROOT,
        capture_output=True,
        check=True,
    ).stdout.decode()
    return [pathlib.PurePosixPath(name) for name in listed.split("\0") if name]


def _files_importing_guarded_packages() -> set[pathlib.PurePosixPath]:
    return {
        path
        for path in _tracked_python_files()
        if _imported_top_level_packages((_REPO_ROOT / path).read_text(encoding="utf-8"))
        & _GUARDED_PACKAGES
    }


def test_openai_libraries_imported_only_inside_llm_package() -> None:
    importers = _files_importing_guarded_packages()

    outside = sorted(
        str(p) for p in importers if _SEAM_DIR not in p.parents and p not in _EXEMPT_FILES
    )
    assert outside == []


def test_scan_covers_every_top_level_tree() -> None:
    roots = {p.parts[0] for p in _tracked_python_files() if len(p.parts) > 1}

    assert {"gubbi", "tests", "tools", "scripts"} <= roots


def test_scan_detects_the_adapter_and_each_exempt_test() -> None:
    importers = _files_importing_guarded_packages()

    assert _SEAM_DIR / "openai_compatible.py" in importers
    assert importers >= _EXEMPT_FILES


def test_scan_detects_each_import_spelling() -> None:
    source = "import openai.types\nfrom pydantic_ai.direct import model_request\nimport httpx2\n"

    assert _imported_top_level_packages(source) == {"openai", "pydantic_ai", "httpx2"}
