"""The fast harness selects synthetic math without optional corpus I/O."""

import os
import subprocess
import sys
from pathlib import Path

from tests._harness import _load_harness


def test_fast_tier_excludes_corpus_cases_but_keeps_mixed_module_mathematics():
    root = Path(__file__).resolve().parents[2]
    environment = dict(os.environ)
    environment.pop("ASYMMETRY_CORPUS_ROOT", None)
    result = subprocess.run(
        [
            sys.executable,
            "tools/harness.py",
            "test",
            "--tier",
            "fast",
            "--",
            "tests/io/test_cds_affine_ties_corpus.py",
            "tests/io/test_mu_kinetics.py",
            "tests/io/test_period_selection.py",
            "tests/io/test_period_mapping.py",
            "tests/io/test_rf_scan_builder.py",
            "tests/gui/test_fft_axis_corpus.py",
            "-m",
            _load_harness()._TIER_MARKER["fast"],
            "--collect-only",
            "-n",
            "0",
            "-q",
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    selected = [line for line in result.stdout.splitlines() if line.startswith("tests/")]
    assert any("test_mu_kinetics.py::Test" in line for line in selected)
    assert any(
        "test_rf_scan_builder.py::test_fit_recovers_input_couplings" in line for line in selected
    )
    assert any("test_period_selection.py::" in line for line in selected)
    assert all("TestRealCorpusSweep" not in line for line in selected)
    assert all("corpus" not in line.lower() for line in selected)
    assert all("benzene_rf_count_integrals_match_workflow" not in line for line in selected)
