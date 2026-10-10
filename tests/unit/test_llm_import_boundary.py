"""The OpenAI client libraries stay behind the LLM provider seam."""

from __future__ import annotations

import ast
import pathlib

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SCANNED_ROOTS = ("gubbi", "tools")
_SEAM_DIR = _REPO_ROOT / "gubbi" / "extraction" / "llm"
_GUARDED_PACKAGES = frozenset({"openai", "pydantic_ai"})


def _imported_top_level_packages(source: str) -> set[str]:
    packages: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            packages.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            packages.add(node.module.split(".")[0])
    return packages


def _files_importing_guarded_packages() -> set[pathlib.Path]:
    return {
        path
        for root in _SCANNED_ROOTS
        for path in (_REPO_ROOT / root).rglob("*.py")
        if _imported_top_level_packages(path.read_text(encoding="utf-8")) & _GUARDED_PACKAGES
    }


def test_openai_libraries_imported_only_inside_llm_package() -> None:
    importers = _files_importing_guarded_packages()

    outside = sorted(
        str(p.relative_to(_REPO_ROOT)) for p in importers if _SEAM_DIR not in p.parents
    )
    assert outside == []


def test_scan_detects_the_adapter_inside_llm_package() -> None:
    importers = _files_importing_guarded_packages()

    assert _SEAM_DIR / "openai_compatible.py" in importers


def test_scan_detects_each_import_spelling() -> None:
    source = "import openai.types\nfrom pydantic_ai.direct import model_request\n"

    assert _imported_top_level_packages(source) == {"openai", "pydantic_ai"}
