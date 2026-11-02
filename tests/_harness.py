"""Import the standalone harness for structural-rule regression tests."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HARNESS_PATH = Path(__file__).resolve().parents[1] / "tools" / "harness.py"


def _load_harness():
    spec = importlib.util.spec_from_file_location("asymmetry_harness", HARNESS_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
