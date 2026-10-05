"""Load ``tools/testdb/testdb.py`` as a module; ``tools/`` is not a package."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

TESTDB_TOOL = Path(__file__).resolve().parents[2] / "tools" / "testdb" / "testdb.py"
_MODULE_NAME = "testdb_tool"


def load_testdb_tool() -> ModuleType:
    """Return the controller module, loading it once per process."""
    if _MODULE_NAME in sys.modules:
        return sys.modules[_MODULE_NAME]
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, TESTDB_TOOL)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {TESTDB_TOOL}")
    module = importlib.util.module_from_spec(spec)
    # Registered before exec: dataclasses resolve their module through sys.modules.
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module
