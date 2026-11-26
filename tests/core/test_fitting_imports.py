"""The public fitting package and empty plugin scan stay lightweight."""

import subprocess
import sys


def test_fitting_api_and_empty_plugin_scan_load_owners_only_on_access(tmp_path):
    script = """
import sys
from asymmetry.core import fitting, plugins

plugins.importlib.metadata.entry_points = lambda **kwargs: ()
assert plugins.load_user_functions(sys.argv[1]).sources == []
assert "ParameterSet" in dir(fitting)
assert {"numpy", "scipy", "iminuit"}.isdisjoint(sys.modules)

from asymmetry.core.fitting import ParameterSet
from asymmetry.core.fitting.parameters import ParameterSet as Owner

assert ParameterSet is Owner
assert fitting.ParameterSet is Owner
assert "asymmetry.core.fitting.composite" not in sys.modules
assert "asymmetry.core.fitting.global_fit_wizard" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
