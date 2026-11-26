"""Executable examples and numerical conventions in the fitting API reference."""

from inspect import signature
from itertools import takewhile
from pathlib import Path
from textwrap import dedent

import pytest

from asymmetry.core.fitting.composite import COMPONENTS
from asymmetry.core.fitting.models import MODELS


def test_damped_precession_example_executes():
    page = Path(__file__).parents[2] / "docs/api/fitting/models.rst"
    marker = ".. code-block:: python\n   :name: python-damped-precession-example\n"
    source = page.read_text(encoding="utf-8").split(marker, 1)[1]
    lines = takewhile(lambda line: not line or line.startswith("   "), source.splitlines())
    namespace = {}
    exec(compile(dedent("\n".join(lines)), str(page), "exec"), namespace)
    assert namespace["model"].param_names == ["A_1", "frequency", "phase", "Lambda", "A_bg"]


def test_standalone_model_signatures_match_registry_parameter_order():
    for model in MODELS.values():
        assert list(signature(model.function).parameters) == ["t", *model.param_names], model.name

    assert MODELS["Oscillatory"].param_names == ["A0", "frequency", "phase", "Lambda", "baseline"]
    component = COMPONENTS["Oscillatory"]
    assert component.param_names == ["A", "frequency", "phase"]
    assert list(signature(component.function).parameters) == ["t", *component.param_names]


def test_longitudinal_field_kt_decouples_to_amplitude_plus_baseline():
    model = MODELS["LFKuboToyabe"].function
    polarisation = model(1.0, A0=1.0, Delta=0.3, B_L=10_000.0)
    asymmetry = model(1.0, A0=20.0, Delta=0.3, B_L=10_000.0, baseline=3.0)
    assert polarisation == pytest.approx(1.0, abs=1e-6, rel=0)
    assert asymmetry == pytest.approx(23.0, abs=2e-5, rel=0)
    assert asymmetry == 20.0 * polarisation + 3.0
