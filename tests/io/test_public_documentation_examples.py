"""Execute selected public examples rather than only checking their syntax."""

import runpy
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def documentation_block(path, heading):
    section = (ROOT / path).read_text().split(heading, 1)[1]
    block = section.split(".. code-block:: python\n\n", 1)[1]
    lines = []
    for line in block.splitlines():
        if line and not line.startswith("   "):
            break
        lines.append(line[3:] if line else "")
    return "\n".join(lines)


def test_readme_fit_example():
    code = (ROOT / "README.md").read_text().split("```python\n", 1)[1].split("```", 1)[0]
    exec(code, namespace := {})
    result = namespace["result"]
    assert result.success, result.message
    assert result.parameters["Lambda"].value == pytest.approx(0.3, abs=0.02)
    assert result.parameters["A0"].value == pytest.approx(24.0, abs=0.5)
    assert np.all(np.isfinite(result.residuals))


def test_grouping_and_alpha_examples():
    namespace = {}
    for heading in ("Grouping histograms\n", "Alpha estimation and asymmetry\n"):
        exec(documentation_block("docs/reference/data_processing.rst", heading), namespace)
    assert namespace["forward_counts"].shape == (200,)
    assert namespace["alpha"] > 0
    assert np.all(np.isfinite(namespace["asymmetry"]))
    assert np.all(namespace["error"] > 0)


def test_fourier_window_example():
    exec(
        documentation_block(
            "docs/reference/data_processing.rst", "Windowing for Fourier analysis\n"
        ),
        namespace := {},
    )
    np.testing.assert_allclose(namespace["windowed_data"], namespace["signal"] * np.hanning(200))


@pytest.mark.parametrize(
    ("path", "heading"),
    [
        ("docs/getting_started/installation.rst", "Verifying installation\n"),
        ("docs/reference/cookbook.rst", "Select a logged thermometer\n"),
        ("docs/reference/cookbook.rst", "Parameter trends across runs\n"),
    ],
)
def test_self_contained_examples(path, heading):
    exec(documentation_block(path, heading), {})


def test_transform_script(capsys):
    runpy.run_path(str(ROOT / "examples/transform_workflow.py"), run_name="__main__")
    assert "alpha = " in capsys.readouterr().out


def test_file_calibration_example(tmp_path, monkeypatch):
    pytest.importorskip("h5py")
    from asymmetry.core.data.dataset import Histogram, Run
    from asymmetry.core.io.nexus_writer import write_nexus_v1

    source = Run(
        run_number=707,
        histograms=[
            Histogram(np.full(200, value), 0.01, good_bin_start=5, good_bin_end=150)
            for value in (1200.0, 1000.0)
        ],
        grouping={"groups": {1: [1], 2: [2]}, "forward_group": 1, "backward_group": 2},
    )
    write_nexus_v1(source, tmp_path / "calibration.nxs")
    monkeypatch.chdir(tmp_path)
    exec(
        documentation_block("docs/reference/cookbook.rst", "Load a run, calibrate α, and reduce\n"),
        namespace := {},
    )
    assert namespace["calibration"].alpha == pytest.approx(1.2)
    assert namespace["reduced"].n_points > 0
    np.testing.assert_allclose(namespace["asymmetry"], 0.0, atol=1e-12)
    assert np.all(namespace["error"] > 0)
