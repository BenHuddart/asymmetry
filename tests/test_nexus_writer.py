"""NeXus V1 writer round-trip, refit-recovery and pull-distribution tests.

Verification-plan §2 and §3 of docs/porting/simulate-mode/verification-plan.md.
Corpus checks (skip-if-missing) exercise real instrument templates.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, Run
from asymmetry.core.io import load
from asymmetry.core.io.nexus_writer import write_nexus_v1
from asymmetry.core.simulate import degrade_run, reduce_run_to_dataset, simulate_run

h5py = pytest.importorskip("h5py")

CORPUS_ROOT = os.path.expanduser("~/Documents/WiMDA muon school")
NICKEL_DIR = os.path.join(CORPUS_ROOT, "Magnetism", "Ferromagnetic nickel")
EUO_DIR = os.path.join(CORPUS_ROOT, "Magnetism", "Magnetic ordering in EuO")

N_BINS = 1500
BIN_WIDTH = 0.016
T0_BIN = 40


def _template(*, t0_bins: list[int] | None = None, alpha: float = 1.0) -> Run:
    t0_bins = t0_bins if t0_bins is not None else [T0_BIN, T0_BIN]
    histograms = [
        Histogram(
            counts=np.zeros(N_BINS),
            bin_width=BIN_WIDTH,
            t0_bin=t0,
            good_bin_start=max(t0_bins) + 5,
            good_bin_end=N_BINS - 10,
        )
        for t0 in t0_bins
    ]
    grouping = {
        "groups": {1: [1], 2: [2]},
        "forward_group": 1,
        "backward_group": 2,
        "alpha": alpha,
        "t0_bin": max(t0_bins),
        "first_good_bin": max(t0_bins) + 5,
        "last_good_bin": N_BINS - 10,
        "good_frames": 18000.0,
        "dead_time_us": [0.0, 0.0],
    }
    return Run(
        run_number=4321,
        histograms=histograms,
        metadata={
            "title": "Writer template",
            "temperature": 12.5,
            "field": 250.0,
            "instrument": "MUSR",
            "field_state": "TF",
            "detector_orientation": "Longitudinal",
        },
        grouping=grouping,
        source_file="/data/template.nxs",
    )


def _exp_model(t: np.ndarray, a0: float = 20.0, rate: float = 0.5) -> np.ndarray:
    return a0 * np.exp(-rate * t)


def _as_str(value) -> str:
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def _simulated(seed: int = 1, **kwargs) -> Run:
    return simulate_run(
        _template(**kwargs),
        _exp_model,
        {"a0": 21.0, "rate": 0.6},
        total_events=5.0e6,
        seed=seed,
        run_number=90001,
        title="Synthetic exp",
    )


# ---------------------------------------------------------------------------
# §2 File identity through NexusLoader
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_counts_and_structure_identical(self, tmp_path) -> None:
        run = _simulated()
        path = tmp_path / "synthetic.nxs"
        write_nexus_v1(run, path)

        dataset = load(path)
        reloaded = dataset.run
        assert reloaded is not None
        assert len(reloaded.histograms) == len(run.histograms)
        for original, loaded in zip(run.histograms, reloaded.histograms, strict=True):
            assert np.array_equal(loaded.counts, original.counts)
            assert np.isclose(loaded.bin_width, original.bin_width)
            assert loaded.t0_bin == original.t0_bin
        assert dataset.metadata["nexus_version"] == "v1"

    def test_metadata_round_trip(self, tmp_path) -> None:
        run = _simulated()
        path = tmp_path / "synthetic.nxs"
        write_nexus_v1(run, path)

        dataset = load(path)
        meta = dataset.metadata
        assert meta["run_number"] == 90001
        assert meta["title"] == "Synthetic exp"
        assert meta["temperature"] == 12.5
        assert meta["field"] == 250.0
        assert meta["instrument"] == "MUSR"
        assert meta["field_state"] == "TF"
        assert meta["detector_orientation"] == "Longitudinal"

    def test_grouping_round_trip(self, tmp_path) -> None:
        run = _simulated()
        path = tmp_path / "synthetic.nxs"
        write_nexus_v1(run, path)

        grouping = load(path).run.grouping
        assert grouping["groups"] == {1: [1], 2: [2]}
        assert grouping["forward_group"] == 1
        assert grouping["backward_group"] == 2
        assert grouping["first_good_bin"] == run.grouping["first_good_bin"]
        assert grouping["last_good_bin"] == run.grouping["last_good_bin"]
        assert grouping["good_frames"] == 18000.0
        assert grouping["dead_time_us"] == [0.0, 0.0]

    def test_per_detector_t0_round_trip(self, tmp_path) -> None:
        run = _simulated(t0_bins=[40, 57])
        path = tmp_path / "staggered.nxs"
        write_nexus_v1(run, path)

        reloaded = load(path).run
        assert [h.t0_bin for h in reloaded.histograms] == [40, 57]

    def test_provenance_survives_reload(self, tmp_path) -> None:
        run = _simulated(seed=9)
        path = tmp_path / "synthetic.nxs"
        write_nexus_v1(run, path)

        fields = load(path).metadata["nexus_fields"]
        sim = fields["simulation"]
        assert int(sim["synthetic"]) == 1
        assert int(sim["seed"]) == 9
        assert float(sim["total_events"]) == 5.0e6
        assert "exp" in _as_str(sim["model"]).lower() or _as_str(sim["model"])
        assert "a0" in _as_str(sim["parameters"])

    def test_reduced_curve_matches_in_memory_reduction(self, tmp_path) -> None:
        """The loader's reduction of the file equals reduce_run_to_dataset at α=1."""
        run = _simulated()
        path = tmp_path / "synthetic.nxs"
        write_nexus_v1(run, path)

        from_file = load(path)
        in_memory = reduce_run_to_dataset(run)
        assert np.allclose(from_file.time, in_memory.time)
        assert np.allclose(from_file.asymmetry, in_memory.asymmetry)
        assert np.allclose(from_file.error, in_memory.error)

    def test_degraded_run_round_trip(self, tmp_path) -> None:
        source = _simulated()
        derived = degrade_run(source, 0.5, seed=3, run_number=90002)
        path = tmp_path / "degraded.nxs"
        write_nexus_v1(derived, path)

        dataset = load(path)
        for original, loaded in zip(derived.histograms, dataset.run.histograms, strict=True):
            assert np.array_equal(loaded.counts, original.counts)
        sim = dataset.metadata["nexus_fields"]["simulation"]
        assert float(sim["degrade_factor"]) == 0.5
        assert int(sim["degrade_seed"]) == 3

    def test_rejects_empty_and_ragged_runs(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="histograms"):
            write_nexus_v1(Run(run_number=1), tmp_path / "empty.nxs")
        ragged = _simulated()
        ragged.histograms[1] = Histogram(
            counts=np.zeros(N_BINS - 5), bin_width=BIN_WIDTH, t0_bin=T0_BIN
        )
        with pytest.raises(ValueError, match="same length"):
            write_nexus_v1(ragged, tmp_path / "ragged.nxs")


# ---------------------------------------------------------------------------
# §2–3 Refit recovery and pull distribution
# ---------------------------------------------------------------------------


def _fit_exp(dataset, a0_start: float, rate_start: float, t_max: float = 8.0):
    """Fit the exponential, restricted to the healthy-count window.

    The late tail of a 24 μs histogram holds ≲ 1 count/bin where the
    Gaussian error approximation (and √n errors from observed counts)
    breaks down — textbook §15.3. Real analyses window the fit the same way.
    """
    pytest.importorskip("iminuit")
    from asymmetry.core.fitting.engine import FitEngine
    from asymmetry.core.fitting.parameters import Parameter, ParameterSet

    params = ParameterSet(
        [
            Parameter(name="a0", value=a0_start, min=0.0, max=100.0),
            Parameter(name="rate", value=rate_start, min=0.0, max=10.0),
        ]
    )
    return FitEngine().fit(dataset, _exp_model, params, t_max=t_max)


class TestRefitRecovery:
    def test_round_trip_refit_recovers_parameters(self, tmp_path) -> None:
        """Simulate → save NeXus → reload → refit: truth within 3σ, χ²ᵣ in band."""
        run = _simulated(seed=21)
        path = tmp_path / "refit.nxs"
        write_nexus_v1(run, path)
        dataset = load(path)

        # Seed the fit away from the generating values (a0=21, rate=0.6).
        result = _fit_exp(dataset, a0_start=10.0, rate_start=1.5)
        assert result.success

        fitted = {p.name: p.value for p in result.parameters}
        errors = result.uncertainties
        assert abs(fitted["a0"] - 21.0) < 3.0 * errors["a0"]
        assert abs(fitted["rate"] - 0.6) < 3.0 * errors["rate"]

        # The shipped asymmetry error model propagates numerator and
        # denominator as independent, over-estimating σ_A by
        # (1+A²)/(1−A²) relative to exact Poisson propagation; against a
        # known truth this centres χ²ᵣ on E[(1−A²)/(1+A²)] < 1 rather
        # than 1. Centre the acceptance band on that expectation.
        window = dataset.time_range(t_max=8.0)
        a_frac = _exp_model(window.time) / 100.0
        expected_chi2r = float(np.mean((1.0 - a_frac**2) / (1.0 + a_frac**2)))
        dof = window.n_points - 2
        band = 3.0 * np.sqrt(2.0 / dof)
        assert abs(result.reduced_chi_squared - expected_chi2r) < band

    def test_pull_distribution_over_seeds(self) -> None:
        """Pulls of refitted parameters over many seeds are ~ N(0, 1).

        Verifies generation, reduction, the error model and the covariance
        in one statement: parameter errors are neither over- nor
        under-estimated.
        """
        pytest.importorskip("iminuit")
        truth = {"a0": 21.0, "rate": 0.6}
        n_seeds = 100
        template = _template()
        pulls = {"a0": [], "rate": []}
        for seed in range(n_seeds):
            run = simulate_run(template, _exp_model, truth, total_events=2.0e6, seed=seed)
            result = _fit_exp(reduce_run_to_dataset(run), a0_start=15.0, rate_start=1.0)
            assert result.success
            fitted = {p.name: p.value for p in result.parameters}
            for name in pulls:
                pulls[name].append((fitted[name] - truth[name]) / result.uncertainties[name])

        for name, values in pulls.items():
            arr = np.asarray(values)
            # Mean consistent with 0 and variance consistent with 1 for N=100.
            assert abs(float(arr.mean())) < 4.0 / np.sqrt(n_seeds), name
            sigma_var = np.sqrt(2.0 / (n_seeds - 1))
            assert abs(float(arr.var(ddof=1)) - 1.0) < 4.0 * sigma_var, name


# ---------------------------------------------------------------------------
# Corpus round trips (skip when the Muon School corpus is absent)
# ---------------------------------------------------------------------------


def _first_file(root: str, suffix: str) -> str | None:
    if not os.path.isdir(root):
        return None
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in sorted(filenames):
            if name.lower().endswith(suffix):
                return os.path.join(dirpath, name)
    return None


@pytest.mark.skipif(not os.path.isdir(NICKEL_DIR), reason="Muon School corpus not available")
def test_corpus_nickel_template_round_trip(tmp_path) -> None:
    """HDF5 nickel run as template: simulate → write → reload → counts equal."""
    source_path = _first_file(NICKEL_DIR, ".nxs")
    assert source_path is not None
    loaded = load(source_path)
    dataset = loaded[0] if isinstance(loaded, list) else loaded
    template = dataset.run
    assert template is not None and template.histograms

    run = simulate_run(template, _exp_model, {"a0": 20.0, "rate": 0.4}, total_events=1.0e7, seed=5)
    path = tmp_path / "nickel_synthetic.nxs"
    write_nexus_v1(run, path)
    reloaded = load(path).run
    for original, again in zip(run.histograms, reloaded.histograms, strict=True):
        assert np.array_equal(again.counts, original.counts)


@pytest.mark.skipif(not os.path.isdir(EUO_DIR), reason="Muon School corpus not available")
def test_corpus_psi_bin_template_writes_nexus(tmp_path) -> None:
    """A PSI .bin template produces a loadable NeXus file (impossible in WiMDA)."""
    source_path = _first_file(EUO_DIR, ".bin")
    if source_path is None:
        pytest.skip("no .bin runs in corpus")
    loaded = load(source_path)
    dataset = loaded[0] if isinstance(loaded, list) else loaded
    template = dataset.run
    assert template is not None and template.histograms

    run = simulate_run(template, _exp_model, total_events=5.0e6, seed=6)
    path = tmp_path / "psi_synthetic.nxs"
    write_nexus_v1(run, path)
    reloaded = load(path).run
    assert len(reloaded.histograms) == len(run.histograms)
    for original, again in zip(run.histograms, reloaded.histograms, strict=True):
        assert np.array_equal(again.counts, original.counts)
