"""Tests for the synthetic YbZn2GaO5 example generator.

Two tiers:

* **Fast** (standard tier): a small 2-temperature x 5-field dataset - files load
  through the core loader registry, a mid-field run's exponential fit recovers
  its known relaxation rate, the generation is deterministic, and the manifest
  is internally consistent.
* **Recovery gate** (``slow`` + ``integration``, full tier only): the full
  8-temperature x 20-field dataset is generated, each spectrum is fitted for its
  relaxation rate (level-1 batch), the rates are assembled into per-temperature
  lambda(B) groups, and a cross-temperature global fit recovers every Table I
  global within tolerance. This is the identifiability proof for the whole
  YbZn2GaO5 documentation programme.
"""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.fitting.engine import FitEngine
from asymmetry.core.fitting.models import MODELS
from asymmetry.core.fitting.parameter_models import (
    ErrorMode,
    ParameterCompositeModel,
    ParameterGroupData,
    global_fit_parameter_model,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.io import load as io_load
from asymmetry.examples import ybzn2gao5
from asymmetry.examples.ybzn2gao5 import (
    FIXED_PARAMS,
    GLOBAL_PARAMS,
    LOCAL_PARAMS,
    MODEL_COMPONENTS,
    PARAM_BOUNDS,
    TRUTH,
    Manifest,
    generate_ybzn2gao5_runs,
)


def _load_dataset(path: str):
    result = io_load(path)
    return result[0] if isinstance(result, list) else result


def _fit_relaxation_rate(dataset, lambda_seed: float) -> tuple[float, float]:
    """Fit a(t) = A0 exp(-Lambda t) + baseline; return (Lambda, sigma_Lambda)."""
    engine = FitEngine()
    model = MODELS["ExponentialRelaxation"]
    params = ParameterSet()
    params.add(Parameter("A0", 20.0, 0.0, 100.0))
    params.add(Parameter("Lambda", max(float(lambda_seed), 0.05), 0.0, 100.0))
    params.add(Parameter("baseline", 3.0, -20.0, 20.0))
    result = engine.fit(dataset, model.function, params)
    lam = float(result.parameters["Lambda"].value)
    err = float(result.uncertainties.get("Lambda", 0.0) or 0.01 * abs(lam))
    return lam, err


# ---------------------------------------------------------------------------
# Fast tier
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_manifest(tmp_path_factory) -> Manifest:
    """A small 2-temperature x 5-field dataset, generated once per module."""
    out = tmp_path_factory.mktemp("ybzn2gao5_small")
    return generate_ybzn2gao5_runs(
        out,
        seed=123,
        fields_per_temperature=5,
        temperatures=(0.05, 12.0),
    )


def test_manifest_sanity(small_manifest: Manifest) -> None:
    man = small_manifest
    assert man.n_runs == 2 * 5
    assert man.temperatures == (0.05, 12.0)
    assert len(man.fields_gauss) == 5
    # Run numbers are contiguous in the fictitious 90xxx range.
    numbers = [r.run_number for r in man.runs]
    assert numbers == list(range(ybzn2gao5.FIRST_RUN_NUMBER, ybzn2gao5.FIRST_RUN_NUMBER + 10))
    # Every run records a positive truth relaxation rate and a real file.
    for spec in man.runs:
        assert spec.lambda_truth > 0.0
        assert spec.temperature in (0.05, 12.0)
        assert ybzn2gao5.FIELD_MIN_GAUSS * (1 - 1e-9) <= spec.field_gauss
        assert spec.field_gauss <= ybzn2gao5.FIELD_MAX_GAUSS * (1 + 1e-9)
    assert man.truth_global == TRUTH.global_params


def test_runs_load_through_registry(small_manifest: Manifest) -> None:
    for spec in small_manifest.runs:
        dataset = _load_dataset(spec.path)
        assert dataset.n_points > 0
        # Provenance survives the NeXus round trip.
        assert dataset.metadata.get("temperature") == pytest.approx(spec.temperature)
        assert dataset.metadata.get("field") == pytest.approx(spec.field_gauss, rel=1e-6)
        # Asymmetry is on the percent scale, starting near a0 + a_BG.
        assert dataset.asymmetry[:5].mean() == pytest.approx(
            TRUTH.a0_percent + TRUTH.a_bg_percent, abs=6.0
        )


def test_single_run_exponential_recovers_lambda(small_manifest: Manifest) -> None:
    # A mid-field run of the low-temperature group carries a large, well-defined
    # relaxation rate; its exponential fit should recover the truth closely.
    spec = small_manifest.runs[2]
    dataset = _load_dataset(spec.path)
    lam, err = _fit_relaxation_rate(dataset, spec.lambda_truth)
    assert lam == pytest.approx(spec.lambda_truth, rel=0.05)
    # The event budget is sized so the single-run rate error is a few percent.
    assert 0.0 < err / lam < 0.1


def test_generation_is_deterministic(tmp_path) -> None:
    kwargs = dict(seed=7, fields_per_temperature=4, temperatures=(0.4, 6.0))
    man_a = generate_ybzn2gao5_runs(tmp_path / "a", **kwargs)
    man_b = generate_ybzn2gao5_runs(tmp_path / "b", **kwargs)
    assert man_a.n_runs == man_b.n_runs
    # Same seed -> byte-identical files (the writer stamps no wall-clock time).
    for spec_a, spec_b in zip(man_a.runs, man_b.runs, strict=True):
        assert spec_a.lambda_truth == pytest.approx(spec_b.lambda_truth)
        bytes_a = np.fromfile(spec_a.path, dtype=np.uint8)
        bytes_b = np.fromfile(spec_b.path, dtype=np.uint8)
        assert np.array_equal(bytes_a, bytes_b)


def test_different_seed_changes_data(tmp_path) -> None:
    kwargs = dict(fields_per_temperature=4, temperatures=(0.4, 6.0))
    man_a = generate_ybzn2gao5_runs(tmp_path / "a", seed=1, **kwargs)
    man_b = generate_ybzn2gao5_runs(tmp_path / "b", seed=2, **kwargs)
    a = np.fromfile(man_a.runs[0].path, dtype=np.uint8)
    b = np.fromfile(man_b.runs[0].path, dtype=np.uint8)
    assert not np.array_equal(a, b)


# ---------------------------------------------------------------------------
# Recovery gate (full tier only)
# ---------------------------------------------------------------------------

#: Global-fit seed: within a few percent of the Table I truth (near, not at).
#: A single value per local parameter is applied to every temperature (the
#: cross-group fitter seeds all groups' local instances from one number).
_RECOVERY_SEED = {
    "A": 61.0,
    "D": 18.0,
    "lambda_BG": 0.065,
    "m": 6.9,
    "B0": 26800.0,
    "Bwid": 12900.0,
    "D_2D": 15.0e3,
    "nu": 350.0,
    "f": 0.09,
}


@pytest.mark.slow
@pytest.mark.integration
def test_recovery_gate(tmp_path) -> None:
    """Full pipeline: generate -> batch lambda -> global fit -> Table I recovery."""
    manifest = generate_ybzn2gao5_runs(
        tmp_path / "full",
        seed=ybzn2gao5.DEFAULT_SEED,
        fields_per_temperature=20,
    )
    assert manifest.n_runs == len(TRUTH.temperatures) * 20

    # Level 1: fit each spectrum for its relaxation rate.
    rows_by_temp: dict[float, list[tuple[float, float, float]]] = {}
    for spec in manifest.runs:
        dataset = _load_dataset(spec.path)
        lam, err = _fit_relaxation_rate(dataset, spec.lambda_truth)
        rows_by_temp.setdefault(spec.temperature, []).append((spec.field_gauss, lam, err))

    # Level 2: assemble per-temperature lambda(B) groups and fit jointly.
    model = ParameterCompositeModel(list(MODEL_COMPONENTS))
    groups: list[ParameterGroupData] = []
    for temperature in TRUTH.temperatures:
        rows = sorted(rows_by_temp[temperature])
        x = np.array([r[0] for r in rows], dtype=float)
        y = np.array([r[1] for r in rows], dtype=float)
        yerr = np.array([r[2] for r in rows], dtype=float)
        groups.append(
            ParameterGroupData(
                group_id=f"T={temperature:g}K",
                group_name=f"{temperature:g} K",
                x=x,
                y=y,
                yerr=yerr,
                group_variable_value=float(temperature),
            )
        )

    result = global_fit_parameter_model(
        groups,
        model,
        list(GLOBAL_PARAMS),
        list(LOCAL_PARAMS),
        dict(FIXED_PARAMS),
        initial_params=_RECOVERY_SEED,
        parameter_bounds=PARAM_BOUNDS,
        error_mode=ErrorMode.COLUMN,
    )

    assert result.reduced_chi_squared == pytest.approx(1.0, abs=0.5)
    for name in GLOBAL_PARAMS:
        truth = TRUTH.global_params[name]
        fitted = result.global_parameters[name].value
        sigma = result.global_uncertainties.get(name)
        assert sigma is not None and sigma > 0.0, f"{name}: no uncertainty"
        rel = abs(fitted - truth) / abs(truth)
        nsig = abs(fitted - truth) / sigma
        assert rel < 0.10, f"{name}: {fitted:.4g} vs {truth:.4g} ({rel:.1%} off)"
        assert nsig < 3.0, f"{name}: {fitted:.4g} vs {truth:.4g} ({nsig:.1f} sigma)"
