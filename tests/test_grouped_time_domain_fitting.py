"""Tests for grouped time-domain fitting adapter logic."""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.fitting import (
    GROUP_NUISANCE_PARAMS,
    GroupedTimeDomainFitResult,
    GroupedTimeDomainGroup,
    build_grouped_time_domain_datasets,
    fit_grouped_series,
    fit_grouped_time_domain,
)
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.engine import FitResult
from asymmetry.core.fitting.grouped_time_domain import (
    _grouped_global_is_large,
    _resolve_grouped_series_workers,
    build_grouped_count_model,
    validate_grouped_model_contract,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.utils.constants import MUON_LIFETIME_US


def _cosine_polarization(
    t: np.ndarray,
    frequency: float,
    phase: float = 0.0,
) -> np.ndarray:
    return np.cos(2.0 * np.pi * frequency * t + phase)


def _initial_group_params(frequency: float = 1.0) -> ParameterSet:
    return ParameterSet(
        [
            Parameter("N0", 100.0),
            Parameter("background", 5.0),
            Parameter("amplitude", 0.2),
            Parameter("relative_phase", 0.0),
            Parameter("frequency", frequency),
            Parameter("phase", 0.1, fixed=True),
        ]
    )


def _grouped_source_dataset() -> MuonDataset:
    run = Run(
        run_number=42,
        histograms=[
            Histogram(counts=np.array([10.0, 20.0, 30.0]), bin_width=0.1, t0_bin=0),
            Histogram(counts=np.array([5.0, 7.0, 9.0]), bin_width=0.1, t0_bin=0),
        ],
        metadata={"field": 100.0},
        grouping={
            "groups": {1: [1], 2: [2]},
            "group_names": {1: "Forward", 2: "Backward"},
            "first_good_bin": 0,
            "last_good_bin": 2,
            "bunching_factor": 1,
        },
    )
    return MuonDataset(
        time=np.array([0.0, 0.1, 0.2]),
        asymmetry=np.array([0.0, 0.0, 0.0]),
        error=np.array([1.0, 1.0, 1.0]),
        metadata={"run_number": 42},
        run=run,
    )


def test_build_grouped_time_domain_datasets_returns_lifetime_corrected_groups() -> None:
    datasets = build_grouped_time_domain_datasets(_grouped_source_dataset())

    assert [dataset.run_label for dataset in datasets] == ["Forward", "Backward"]
    assert [dataset.run_number for dataset in datasets] == [-42001, -42002]
    np.testing.assert_allclose(
        datasets[0].asymmetry,
        np.array([10.0, 20.0, 30.0]) * np.exp(np.array([0.0, 0.1, 0.2]) / MUON_LIFETIME_US),
    )
    np.testing.assert_allclose(
        datasets[1].asymmetry,
        np.array([5.0, 7.0, 9.0]) * np.exp(np.array([0.0, 0.1, 0.2]) / MUON_LIFETIME_US),
    )


def test_build_grouped_time_domain_datasets_applies_group_bunching_before_lifetime_correction() -> (
    None
):
    dataset = _grouped_source_dataset()
    assert dataset.run is not None
    dataset.run.histograms = [
        Histogram(counts=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), bin_width=0.01, t0_bin=0),
        Histogram(counts=np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0]), bin_width=0.01, t0_bin=0),
    ]
    dataset.run.grouping["last_good_bin"] = 5
    dataset.run.grouping["bunching_factor"] = 2

    datasets = build_grouped_time_domain_datasets(dataset)

    expected_time = np.array([0.005, 0.025, 0.045])
    np.testing.assert_allclose(datasets[0].time, expected_time)
    np.testing.assert_allclose(datasets[1].time, expected_time)
    np.testing.assert_allclose(
        datasets[0].asymmetry,
        np.array([3.0, 7.0, 11.0]) * np.exp(expected_time / MUON_LIFETIME_US),
    )
    np.testing.assert_allclose(
        datasets[1].asymmetry,
        np.array([11.0, 7.0, 3.0]) * np.exp(expected_time / MUON_LIFETIME_US),
    )


def test_build_grouped_time_domain_datasets_respects_time_window() -> None:
    datasets = build_grouped_time_domain_datasets(
        _grouped_source_dataset(),
        t_min=0.05,
        t_max=0.15,
    )

    assert len(datasets) == 2
    np.testing.assert_allclose(datasets[0].time, np.array([0.1]))
    np.testing.assert_allclose(datasets[1].time, np.array([0.1]))


def test_build_grouped_time_domain_datasets_skips_excluded_groups() -> None:
    dataset = _grouped_source_dataset()
    assert dataset.run is not None
    dataset.run.grouping["groups"] = {1: [1], 2: [2], 3: [1]}
    dataset.run.grouping["group_names"] = {1: "Forward", 2: "Backward", 3: "Extra"}
    dataset.run.grouping["included_groups"] = {1: True, 2: False, 3: True}

    datasets = build_grouped_time_domain_datasets(dataset)

    assert [group.run_label for group in datasets] == ["Forward", "Extra"]


def test_build_grouped_time_domain_datasets_requires_two_included_groups() -> None:
    dataset = _grouped_source_dataset()
    assert dataset.run is not None
    dataset.run.grouping["included_groups"] = {1: True, 2: False}

    with pytest.raises(ValueError, match="two included detector groups"):
        build_grouped_time_domain_datasets(dataset)


def test_fit_grouped_time_domain_maps_results_back_to_group_ids(monkeypatch) -> None:
    groups = [
        GroupedTimeDomainGroup(
            group_id="forward",
            group_name="Forward",
            time=np.array([0.0, 0.1]),
            counts=np.array([100.0, 99.0]),
            error=np.array([1.0, 1.0]),
            source_run_number=42,
        ),
        GroupedTimeDomainGroup(
            group_id="backward",
            group_name="Backward",
            time=np.array([0.0, 0.1]),
            counts=np.array([95.0, 94.0]),
            error=np.array([1.0, 1.0]),
            source_run_number=42,
        ),
    ]
    initial = {
        "forward": _initial_group_params(),
        "backward": _initial_group_params(),
    }

    captured_run_numbers: list[int] = []

    def _fake_global_fit(
        _self,
        datasets,
        _model_fn,
        global_params,
        local_params,
        initial_params,
        **_kwargs,
    ):
        captured_run_numbers.extend(int(dataset.run_number) for dataset in datasets)
        assert global_params == ["frequency"]
        assert local_params == ["N0", "background", "amplitude", "relative_phase"]
        assert set(initial_params) == set(captured_run_numbers)
        results = {
            int(dataset.run_number): FitResult(
                success=True,
                chi_squared=float(index + 1),
                reduced_chi_squared=0.1,
                parameters=initial_params[int(dataset.run_number)],
                message=str(dataset.metadata["group_id"]),
            )
            for index, dataset in enumerate(datasets)
        }
        return results, ParameterSet([Parameter("frequency", 1.0)])

    monkeypatch.setattr(
        "asymmetry.core.fitting.grouped_time_domain.FitEngine.global_fit",
        _fake_global_fit,
    )

    result = fit_grouped_time_domain(
        groups,
        _cosine_polarization,
        global_params=["frequency"],
        local_params=["N0", "background", "amplitude", "relative_phase"],
        initial_params=initial,
    )

    assert result.success is True
    assert set(result.group_results) == {"forward", "backward"}
    assert result.group_results["forward"].message == "forward"
    assert result.group_results["backward"].message == "backward"
    assert captured_run_numbers == [-1, -2]
    assert result.shared_parameters["frequency"].value == pytest.approx(1.0)


def test_fit_grouped_time_domain_preserves_internal_run_numbers_when_metadata_has_source_run(
    monkeypatch,
) -> None:
    groups = [
        GroupedTimeDomainGroup(
            group_id="forward",
            group_name="Forward",
            time=np.array([0.0, 0.1]),
            counts=np.array([100.0, 99.0]),
            error=np.array([1.0, 1.0]),
            source_run_number=1651,
            metadata={"run_number": 1651, "field": 100.0},
        ),
        GroupedTimeDomainGroup(
            group_id="backward",
            group_name="Backward",
            time=np.array([0.0, 0.1]),
            counts=np.array([95.0, 94.0]),
            error=np.array([1.0, 1.0]),
            source_run_number=1651,
            metadata={"run_number": 1651, "field": 100.0},
        ),
    ]
    initial = {
        "forward": _initial_group_params(),
        "backward": _initial_group_params(),
    }

    captured_run_numbers: list[int] = []

    def _fake_global_fit(
        _self,
        datasets,
        _model_fn,
        global_params,
        local_params,
        initial_params,
        **_kwargs,
    ):
        assert global_params == ["frequency"]
        assert local_params == ["N0", "background", "amplitude", "relative_phase"]
        captured_run_numbers.extend(int(dataset.run_number) for dataset in datasets)
        results = {
            int(dataset.run_number): FitResult(
                success=True,
                chi_squared=1.0,
                reduced_chi_squared=0.1,
                parameters=initial_params[int(dataset.run_number)],
            )
            for dataset in datasets
        }
        return results, ParameterSet([Parameter("frequency", 1.0)])

    monkeypatch.setattr(
        "asymmetry.core.fitting.grouped_time_domain.FitEngine.global_fit",
        _fake_global_fit,
    )

    result = fit_grouped_time_domain(
        groups,
        _cosine_polarization,
        global_params=["frequency"],
        local_params=["N0", "background", "amplitude", "relative_phase"],
        initial_params=initial,
    )

    assert result.success is True
    assert captured_run_numbers == [-1, -2]


def test_fit_grouped_time_domain_builds_lifetime_corrected_count_model(monkeypatch) -> None:
    groups = [
        GroupedTimeDomainGroup(
            group_id="g1",
            group_name="Group 1",
            time=np.array([0.0, 0.25]),
            counts=np.array([100.0, 100.0]),
            error=np.array([1.0, 1.0]),
        ),
        GroupedTimeDomainGroup(
            group_id="g2",
            group_name="Group 2",
            time=np.array([0.0, 0.25]),
            counts=np.array([99.0, 99.0]),
            error=np.array([1.0, 1.0]),
        ),
    ]
    initial = {"g1": _initial_group_params(), "g2": _initial_group_params()}

    def _fake_global_fit(
        _self,
        datasets,
        model_fn,
        global_params,
        local_params,
        initial_params,
        **_kwargs,
    ):
        assert global_params == ["frequency"]
        assert local_params == ["N0", "background", "amplitude", "relative_phase"]
        probe = np.array([0.25], dtype=float)
        predicted = model_fn(
            probe,
            N0=100.0,
            background=5.0,
            amplitude=0.2,
            relative_phase=np.pi / 2.0,
            frequency=1.0,
            phase=0.1,
        )
        expected = 100.0 * (
            1.0 + 0.2 * np.cos(2.0 * np.pi * probe + 0.1 + np.pi / 2.0)
        ) + 5.0 * np.exp(probe / float(MUON_LIFETIME_US))
        np.testing.assert_allclose(predicted, expected)
        results = {
            int(dataset.run_number): FitResult(
                success=True,
                chi_squared=1.0,
                reduced_chi_squared=0.1,
                parameters=initial_params[int(dataset.run_number)],
            )
            for dataset in datasets
        }
        return results, ParameterSet([Parameter("frequency", 1.0)])

    monkeypatch.setattr(
        "asymmetry.core.fitting.grouped_time_domain.FitEngine.global_fit",
        _fake_global_fit,
    )

    result = fit_grouped_time_domain(
        groups,
        _cosine_polarization,
        global_params=["frequency"],
        local_params=["N0", "background", "amplitude", "relative_phase"],
        initial_params=initial,
        # Pin the lifetime-corrected count model: that is the Gaussian path's
        # contract. The Poisson default instead passes the raw model (× e^(−t/τ))
        # — covered by test_fit_grouped_time_domain_poisson_uses_raw_count_model.
        cost="gaussian",
    )

    assert result.success is True


def test_fit_grouped_time_domain_poisson_uses_raw_count_model(monkeypatch) -> None:
    """The Poisson default fits raw counts against a raw-count (× e^(−t/τ)) model.

    Cash needs true Poisson counts, so the grouped driver inverts the lifetime
    correction on the data and multiplies the lifetime-corrected count model by
    ``e^(−t/τ_μ)`` to predict raw counts. Both pieces are pinned here.
    """
    t_probe = 0.25
    corrected_counts = np.array([99.0, 88.0])
    times = np.array([0.0, t_probe])
    groups = [
        GroupedTimeDomainGroup(
            group_id="g1",
            group_name="Group 1",
            time=times.copy(),
            counts=corrected_counts.copy(),
            error=np.array([1.0, 1.0]),
            metadata={"grouped_time_domain_lifetime_corrected": True},
        ),
        GroupedTimeDomainGroup(
            group_id="g2",
            group_name="Group 2",
            time=times.copy(),
            counts=corrected_counts.copy(),
            error=np.array([1.0, 1.0]),
            metadata={"grouped_time_domain_lifetime_corrected": True},
        ),
    ]
    initial = {"g1": _initial_group_params(), "g2": _initial_group_params()}

    def _fake_global_fit(
        _self,
        datasets,
        model_fn,
        global_params,
        local_params,
        initial_params,
        **kwargs,
    ):
        # The Poisson cost-factory must be the one routed to the engine.
        from asymmetry.core.fitting.engine import POISSON_COST

        assert kwargs.get("cost_factory") is POISSON_COST
        # The observed counts must be the raw (de-corrected) Poisson counts.
        decay = np.exp(-times / float(MUON_LIFETIME_US))
        for dataset in datasets:
            np.testing.assert_allclose(dataset.asymmetry, corrected_counts * decay)
        # The model must predict raw counts: corrected model × e^(−t/τ).
        probe = np.array([t_probe], dtype=float)
        predicted = model_fn(
            probe,
            N0=100.0,
            background=5.0,
            amplitude=0.2,
            relative_phase=np.pi / 2.0,
            frequency=1.0,
            phase=0.1,
        )
        corrected = 100.0 * (
            1.0 + 0.2 * np.cos(2.0 * np.pi * probe + 0.1 + np.pi / 2.0)
        ) + 5.0 * np.exp(probe / float(MUON_LIFETIME_US))
        expected = corrected * np.exp(-probe / float(MUON_LIFETIME_US))
        np.testing.assert_allclose(predicted, expected)
        results = {
            int(dataset.run_number): FitResult(
                success=True,
                chi_squared=1.0,
                reduced_chi_squared=0.1,
                parameters=initial_params[int(dataset.run_number)],
            )
            for dataset in datasets
        }
        return results, ParameterSet([Parameter("frequency", 1.0)])

    monkeypatch.setattr(
        "asymmetry.core.fitting.grouped_time_domain.FitEngine.global_fit",
        _fake_global_fit,
    )

    result = fit_grouped_time_domain(
        groups,
        _cosine_polarization,
        global_params=["frequency"],
        local_params=["N0", "background", "amplitude", "relative_phase"],
        initial_params=initial,
        cost="poisson",
    )
    assert result.success is True


def test_fit_grouped_series_mixed_global_local_physics() -> None:
    """A mixed grouped series fit shares one physics param across runs while
    fitting another per run (the FB-style Global/Local split).

    frequency is Global (shared across both runs); Lambda is Local (per run).
    The fit must recover one shared frequency and two distinct Lambdas.
    """

    def _relaxing_cosine(t, frequency, Lambda, phase=0.0):  # noqa: N803
        t = np.asarray(t, dtype=float)
        return np.cos(2.0 * np.pi * frequency * t + phase) * np.exp(-Lambda * t)

    time = np.linspace(0.0, 8.0, 1601)
    frequency = 3.0
    n0, background, amplitude = 5.0e4, 50.0, 0.22
    true_lambda = {10: 0.3, 11: 1.0}

    def _counts(lam: float, phase: float) -> np.ndarray:
        pol = _relaxing_cosine(time, frequency, lam, phase)
        return n0 * (1.0 + amplitude * pol) + background * np.exp(time / float(MUON_LIFETIME_US))

    members: dict[int, list[GroupedTimeDomainGroup]] = {}
    for run in (10, 11):
        groups = []
        for gid, phase in ((1, 0.2), (2, np.deg2rad(170.0))):
            counts = _counts(true_lambda[run], phase)
            groups.append(
                GroupedTimeDomainGroup(
                    group_id=gid,
                    group_name=f"g{gid}",
                    time=time.copy(),
                    counts=counts,
                    error=np.sqrt(np.clip(counts, 1.0, None)),
                    metadata={"grouped_time_domain_lifetime_corrected": True},
                )
            )
        members[run] = groups

    def _initial() -> ParameterSet:
        return ParameterSet(
            [
                Parameter("N0", 4.0e4),
                Parameter("background", 40.0),
                Parameter("amplitude", 0.2),
                Parameter("relative_phase", 0.0, min=-2.0 * np.pi, max=2.0 * np.pi),
                Parameter("frequency", 3.1),  # shared (slightly off)
                Parameter("Lambda", 0.6, min=0.0, max=10.0),  # per run
                Parameter("phase", 0.0, fixed=True),
            ]
        )

    initial = {run: {g.group_id: _initial() for g in groups} for run, groups in members.items()}

    result = fit_grouped_series(
        "global",
        members,
        _relaxing_cosine,
        global_params=["frequency", "Lambda", "phase"],
        local_params=["N0", "background", "amplitude", "relative_phase"],
        initial_params=initial,
        cost="gaussian",
        cross_run_local_params=["Lambda"],
    )

    assert result.success is True
    # frequency is shared across runs (a single fitted value).
    shared = {p.name: p.value for p in result.shared_parameters}
    assert shared["frequency"] == pytest.approx(frequency, abs=1e-2)
    assert "Lambda" not in shared  # Lambda is per-run, not a shared parameter
    # Lambda is recovered per run, distinct between runs.
    for run in (10, 11):
        member_key = min(k for k, src in result.member_source_run.items() if src == run)
        fitted = result.member_results[member_key].parameters["Lambda"].value
        assert fitted == pytest.approx(true_lambda[run], abs=2e-2)


def test_fit_grouped_time_domain_recovers_absolute_per_group_phase() -> None:
    """With the shared model phase fixed at zero, each group's free per-group
    phase nuisance recovers that group's *absolute* oscillation phase.

    This is the parameterisation the GUI seeds (model phase fixed at 0, per-group
    phase carries the full phase): the fit must converge on distinct per-group
    phases and a shared frequency from a clean two-group TF signal.
    """
    time = np.linspace(0.0, 8.0, 1601)
    frequency = 3.0
    n0, background, amplitude = 5.0e4, 50.0, 0.22
    true_phase = {"g1": np.deg2rad(40.0), "g2": np.deg2rad(-100.0)}

    def _counts(phase: float) -> np.ndarray:
        return n0 * (
            1.0 + amplitude * np.cos(2.0 * np.pi * frequency * time + phase)
        ) + background * np.exp(time / float(MUON_LIFETIME_US))

    groups = [
        GroupedTimeDomainGroup(
            group_id=gid,
            group_name=gid,
            time=time.copy(),
            counts=_counts(true_phase[gid]),
            error=np.sqrt(np.clip(_counts(true_phase[gid]), 1.0, None)),
            metadata={"grouped_time_domain_lifetime_corrected": True},
        )
        for gid in ("g1", "g2")
    ]

    def _initial(gid: str) -> ParameterSet:
        return ParameterSet(
            [
                Parameter("N0", 4.0e4),
                Parameter("background", 40.0),
                Parameter("amplitude", 0.2),
                # Seed the per-group phase near (but not at) the true value.
                Parameter("relative_phase", true_phase[gid] * 0.8, min=-np.pi, max=np.pi),
                Parameter("frequency", 3.0),
                Parameter("phase", 0.0, fixed=True),  # shared model phase held at zero
            ]
        )

    result = fit_grouped_time_domain(
        groups,
        _cosine_polarization,
        global_params=["frequency"],
        local_params=["N0", "background", "amplitude", "relative_phase"],
        initial_params={gid: _initial(gid) for gid in ("g1", "g2")},
        cost="gaussian",
    )

    assert result.success is True
    for gid in ("g1", "g2"):
        fitted = result.group_results[gid].parameters["relative_phase"].value
        assert float(np.angle(np.exp(1j * (fitted - true_phase[gid])))) == pytest.approx(
            0.0, abs=1e-2
        )


def test_fit_grouped_time_domain_rejects_unknown_cost() -> None:
    groups = [
        GroupedTimeDomainGroup(
            group_id="g1",
            group_name="Group 1",
            time=np.array([0.0, 0.25]),
            counts=np.array([99.0, 88.0]),
            error=np.array([1.0, 1.0]),
        ),
        GroupedTimeDomainGroup(
            group_id="g2",
            group_name="Group 2",
            time=np.array([0.0, 0.25]),
            counts=np.array([99.0, 88.0]),
            error=np.array([1.0, 1.0]),
        ),
    ]
    initial = {"g1": _initial_group_params(), "g2": _initial_group_params()}
    with pytest.raises(ValueError, match="Unknown grouped fit cost"):
        fit_grouped_time_domain(
            groups,
            _cosine_polarization,
            global_params=["frequency"],
            local_params=["N0", "background", "amplitude", "relative_phase"],
            initial_params=initial,
            cost="bogus",
        )


def test_grouped_count_model_applies_relative_phase_to_numbered_phase_parameters() -> None:
    def _numbered_phase_model(
        t: np.ndarray,
        frequency: float,
        phase_2: float = 0.0,
    ) -> np.ndarray:
        return np.cos(2.0 * np.pi * frequency * t + phase_2)

    model_fn = build_grouped_count_model(_numbered_phase_model)
    probe = np.array([0.25], dtype=float)
    predicted = model_fn(
        probe,
        N0=100.0,
        background=5.0,
        amplitude=0.2,
        relative_phase=np.pi / 2.0,
        frequency=1.0,
        phase_2=0.1,
    )
    expected = 100.0 * (1.0 + 0.2 * np.cos(2.0 * np.pi * probe + 0.1 + np.pi / 2.0)) + 5.0 * np.exp(
        probe / float(MUON_LIFETIME_US)
    )

    np.testing.assert_allclose(predicted, expected)


def test_validate_grouped_model_contract_accepts_unit_amplitude_and_zero_background() -> None:
    validate_grouped_model_contract(
        ["A_1", "Lambda", "A_bg"],
        model_values={"A_1": 1.0, "Lambda": 0.5, "A_bg": 0.0},
        fixed_params={"A_1", "A_bg"},
    )


def test_validate_grouped_model_contract_rejects_free_amplitude_and_background() -> None:
    with pytest.raises(ValueError, match="Fixed = 0: A_bg"):
        validate_grouped_model_contract(
            ["A_1", "Lambda", "A_bg"],
            model_values={"A_1": 0.4, "Lambda": 0.5, "A_bg": 0.02},
            fixed_params=set(),
        )


def test_fit_grouped_time_domain_rejects_non_group_local_parameters() -> None:
    groups = [
        GroupedTimeDomainGroup(
            group_id="g1",
            group_name="Group 1",
            time=np.array([0.0, 0.1]),
            counts=np.array([100.0, 100.0]),
            error=np.array([1.0, 1.0]),
        ),
        GroupedTimeDomainGroup(
            group_id="g2",
            group_name="Group 2",
            time=np.array([0.0, 0.1]),
            counts=np.array([100.0, 100.0]),
            error=np.array([1.0, 1.0]),
        ),
    ]
    initial = {"g1": _initial_group_params(), "g2": _initial_group_params()}

    with pytest.raises(ValueError, match="group block"):
        fit_grouped_time_domain(
            groups,
            _cosine_polarization,
            global_params=["N0"],
            local_params=["frequency"],
            initial_params=initial,
        )


# ── grouped-series (multi-run) relationships ─────────────────────────────────


def _two_groups(run: int) -> list[GroupedTimeDomainGroup]:
    return [
        GroupedTimeDomainGroup(
            group_id="forward",
            group_name="Forward",
            time=np.array([0.0, 0.1]),
            counts=np.array([100.0, 99.0]),
            error=np.array([1.0, 1.0]),
            source_run_number=run,
        ),
        GroupedTimeDomainGroup(
            group_id="backward",
            group_name="Backward",
            time=np.array([0.0, 0.1]),
            counts=np.array([95.0, 94.0]),
            error=np.array([1.0, 1.0]),
            source_run_number=run,
        ),
    ]


def _series_initial() -> dict:
    return {"forward": _initial_group_params(), "backward": _initial_group_params()}


def test_fit_grouped_series_individual_runs_one_fit_per_member(monkeypatch) -> None:
    members = {10: _two_groups(10), 11: _two_groups(11)}
    initial = {10: _series_initial(), 11: _series_initial()}

    fitted_runs: list[int] = []

    def _fake_fit_grouped_time_domain(groups, _model_fn, **kwargs):
        run = groups[0].source_run_number
        fitted_runs.append(int(run))
        # Physics shared across this run's groups; nuisance local.
        assert kwargs["global_params"] == ["frequency"]
        assert set(kwargs["local_params"]) == set(GROUP_NUISANCE_PARAMS)
        initial_params = kwargs["initial_params"]
        group_results = {
            group.group_id: FitResult(
                success=True,
                chi_squared=1.0,
                reduced_chi_squared=0.1,
                parameters=initial_params[group.group_id],
                message=str(group.group_id),
            )
            for group in groups
        }
        return GroupedTimeDomainFitResult(
            success=True,
            group_results=group_results,
            shared_parameters=ParameterSet(),
            message="ok",
        )

    monkeypatch.setattr(
        "asymmetry.core.fitting.grouped_time_domain.fit_grouped_time_domain",
        _fake_fit_grouped_time_domain,
    )

    result = fit_grouped_series(
        "individual",
        members,
        _cosine_polarization,
        global_params=["frequency"],
        local_params=list(GROUP_NUISANCE_PARAMS),
        initial_params=initial,
    )

    assert result.success is True
    assert result.relationship == "individual"
    assert sorted(fitted_runs) == [10, 11]  # one independent grouped fit per run
    assert set(result.member_results) == {-10001, -10002, -11001, -11002}
    assert result.member_source_run == {-10001: 10, -10002: 10, -11001: 11, -11002: 11}
    assert result.member_group_id[-10001] == "forward"
    assert result.member_group_id[-10002] == "backward"
    assert len(result.shared_parameters) == 0  # no cross-run sharing


def test_fit_grouped_series_global_shares_physics_across_runs(monkeypatch) -> None:
    members = {10: _two_groups(10), 11: _two_groups(11)}
    initial = {10: _series_initial(), 11: _series_initial()}

    captured: dict = {}

    def _fake_global_fit(
        _self,
        datasets,
        _model_fn,
        global_params,
        local_params,
        initial_params,
        **_kwargs,
    ):
        captured["run_numbers"] = [int(d.run_number) for d in datasets]
        captured["global"] = list(global_params)
        captured["local"] = list(local_params)
        captured["source_runs"] = [d.metadata["source_run_number"] for d in datasets]
        results = {
            int(d.run_number): FitResult(
                success=True,
                chi_squared=1.0,
                reduced_chi_squared=0.1,
                parameters=initial_params[int(d.run_number)],
            )
            for d in datasets
        }
        return results, ParameterSet([Parameter("frequency", 1.0)])

    monkeypatch.setattr(
        "asymmetry.core.fitting.grouped_time_domain.FitEngine.global_fit",
        _fake_global_fit,
    )

    result = fit_grouped_series(
        "global",
        members,
        _cosine_polarization,
        global_params=["frequency"],
        local_params=list(GROUP_NUISANCE_PARAMS),
        initial_params=initial,
    )

    assert result.success is True
    assert result.relationship == "global"
    # One simultaneous fit over every (run, group) pair.
    assert sorted(captured["run_numbers"]) == [-11002, -11001, -10002, -10001]
    assert captured["global"] == ["frequency"]  # physics shared across all runs
    assert set(captured["local"]) == set(GROUP_NUISANCE_PARAMS)  # nuisance per (run, group)
    assert captured["source_runs"].count(10) == 2
    assert captured["source_runs"].count(11) == 2
    assert set(result.member_results) == {-10001, -10002, -11001, -11002}
    assert result.shared_parameters["frequency"].value == pytest.approx(1.0)


def test_fit_grouped_series_rejects_unknown_relationship() -> None:
    with pytest.raises(ValueError, match="relationship"):
        fit_grouped_series(
            "bogus",
            {10: _two_groups(10)},
            _cosine_polarization,
            global_params=[],
            local_params=[],
            initial_params={10: _series_initial()},
        )


def test_fit_grouped_series_requires_members() -> None:
    with pytest.raises(ValueError, match="at least one member"):
        fit_grouped_series(
            "batch",
            {},
            _cosine_polarization,
            global_params=[],
            local_params=[],
            initial_params={},
        )


def test_fit_grouped_series_rejects_non_nuisance_local() -> None:
    with pytest.raises(ValueError, match="group block"):
        fit_grouped_series(
            "global",
            {10: _two_groups(10)},
            _cosine_polarization,
            global_params=["N0"],
            local_params=["frequency"],
            initial_params={10: _series_initial()},
        )


def test_fit_grouped_time_domain_allows_phase_less_model_when_relative_phase_is_zero(
    monkeypatch,
) -> None:
    groups = [
        GroupedTimeDomainGroup(
            group_id="g1",
            group_name="Group 1",
            time=np.array([0.0, 0.1]),
            counts=np.array([100.0, 95.0]),
            error=np.array([1.0, 1.0]),
        ),
        GroupedTimeDomainGroup(
            group_id="g2",
            group_name="Group 2",
            time=np.array([0.0, 0.1]),
            counts=np.array([110.0, 105.0]),
            error=np.array([1.0, 1.0]),
        ),
    ]
    initial = {"g1": _initial_group_params(), "g2": _initial_group_params()}

    def _exp_polarization(t: np.ndarray, Lambda: float) -> np.ndarray:
        return np.exp(-Lambda * t)

    def _fake_global_fit(
        _self,
        datasets,
        model_fn,
        global_params,
        local_params,
        initial_params,
        **_kwargs,
    ):
        probe = np.array([0.1], dtype=float)
        predicted = model_fn(
            probe,
            N0=100.0,
            background=5.0,
            amplitude=0.2,
            relative_phase=0.0,
            Lambda=0.4,
        )
        expected = 100.0 * (1.0 + 0.2 * np.exp(-0.4 * probe)) + 5.0 * np.exp(
            probe / float(MUON_LIFETIME_US)
        )
        np.testing.assert_allclose(predicted, expected)
        results = {
            int(dataset.run_number): FitResult(
                success=True,
                chi_squared=1.0,
                reduced_chi_squared=0.1,
                parameters=initial_params[int(dataset.run_number)],
            )
            for dataset in datasets
        }
        return results, ParameterSet([Parameter("Lambda", 0.4)])

    initial["g1"].add(Parameter("Lambda", 0.4))
    initial["g2"].add(Parameter("Lambda", 0.4))

    monkeypatch.setattr(
        "asymmetry.core.fitting.grouped_time_domain.FitEngine.global_fit",
        _fake_global_fit,
    )

    result = fit_grouped_time_domain(
        groups,
        _exp_polarization,
        global_params=["Lambda"],
        local_params=["N0", "background", "amplitude", "relative_phase"],
        initial_params=initial,
        # Probes the lifetime-corrected model directly → the Gaussian path.
        cost="gaussian",
    )

    assert result.success is True


def test_grouped_time_domain_available_matches_build_outcome() -> None:
    """The cheap probe agrees with the full build on the common gating cases."""
    from asymmetry.core.fitting import grouped_time_domain_available

    # A valid two-group source: probe True, build succeeds.
    dataset = _grouped_source_dataset()
    assert grouped_time_domain_available(dataset)
    assert len(build_grouped_time_domain_datasets(dataset)) == 2

    # No dataset / no run / no histograms / no grouping → probe False, build raises.
    assert not grouped_time_domain_available(None)

    bare = MuonDataset(
        time=np.array([0.0]),
        asymmetry=np.array([0.0]),
        error=np.array([1.0]),
        metadata={},
    )
    assert not grouped_time_domain_available(bare)

    no_grouping = _grouped_source_dataset()
    assert no_grouping.run is not None
    no_grouping.run.grouping = {}
    assert not grouped_time_domain_available(no_grouping)
    with pytest.raises(ValueError):
        build_grouped_time_domain_datasets(no_grouping)

    no_histograms = _grouped_source_dataset()
    assert no_histograms.run is not None
    no_histograms.run.histograms = []
    assert not grouped_time_domain_available(no_histograms)
    with pytest.raises(ValueError):
        build_grouped_time_domain_datasets(no_histograms)

    # A single included group is not enough for the joint grouped view.
    one_group = _grouped_source_dataset()
    assert one_group.run is not None
    one_group.run.grouping["included_groups"] = {2: False}
    assert not grouped_time_domain_available(one_group)
    with pytest.raises(ValueError):
        build_grouped_time_domain_datasets(one_group)


def _parallel_oscillatory_model() -> CompositeModel:
    """A real (picklable) damped-oscillation model for cross-process batch tests."""
    return CompositeModel(["Exponential", "OscillatoryField"], operators=["*"])


def _parallel_series_seed(model: CompositeModel) -> ParameterSet:
    ps = ParameterSet()
    for name in model.param_names:
        if name.startswith("A"):
            ps.add(Parameter(name, 1.0, fixed=True))  # amplitude owned by per-group block
        elif "field" in name:
            ps.add(Parameter(name, 150.0, min=-1.0e9, max=1.0e9))
        elif "phase" in name:
            ps.add(Parameter(name, 0.0, fixed=True))
        else:
            ps.add(Parameter(name, 1.0, min=0.0, max=1.0e6))
    ps.add(Parameter("N0", 120.0, min=0.0, max=1.0e9))
    ps.add(Parameter("background", 40.0, min=0.0, max=1.0e9))
    ps.add(Parameter("amplitude", 0.1, min=-1.0, max=1.0))
    ps.add(Parameter("relative_phase", 0.0, min=-7.0, max=7.0))
    return ps


def _parallel_members(model: CompositeModel, n_runs: int = 3, n_bins: int = 800):
    members: dict[int, list[GroupedTimeDomainGroup]] = {}
    initial: dict[int, dict] = {}
    rng = np.random.default_rng(7)
    t = np.linspace(0.01, 8.0, n_bins)
    for run in range(1600, 1600 + n_runs):
        groups = []
        for gi in range(2):
            mu = 120.0 * np.exp(-t / 2.197) * (1.0 + 0.15 * np.cos(2.0 * np.pi * t + gi)) + 40.0
            counts = rng.poisson(mu).astype(float)
            groups.append(
                GroupedTimeDomainGroup(
                    group_id=f"g{gi}",
                    group_name=f"G{gi}",
                    time=t,
                    counts=counts,
                    error=np.sqrt(np.clip(counts, 1.0, None)),
                    source_run_number=run,
                )
            )
        members[run] = groups
        initial[run] = {g.group_id: _parallel_series_seed(model) for g in groups}
    return members, initial


def test_resolve_grouped_series_workers_is_opt_in_and_clamped() -> None:
    # Parallelism is opt-in: None keeps the sequential path.
    assert _resolve_grouped_series_workers(None, 10) == 1
    assert _resolve_grouped_series_workers(1, 10) == 1
    # A positive count is honoured but never exceeds the run count.
    assert _resolve_grouped_series_workers(4, 10) == 4
    assert _resolve_grouped_series_workers(16, 3) == 3
    # A degenerate batch never spins up a pool.
    assert _resolve_grouped_series_workers(8, 1) == 1


def test_fit_grouped_series_parallel_matches_sequential() -> None:
    model = _parallel_oscillatory_model()
    members, initial = _parallel_members(model)
    global_params = [name for name in model.param_names if not name.startswith("A")]
    local_params = list(GROUP_NUISANCE_PARAMS)

    common = dict(
        members=members,
        polarization_model_fn=model.function,
        global_params=global_params,
        local_params=local_params,
        initial_params=initial,
        seeding="as_provided",
        cost="poisson",
    )
    serial = fit_grouped_series("batch", max_workers=1, **common)
    parallel = fit_grouped_series("batch", max_workers=4, **common)

    # Same members produced, and the per-(run,group) fitted values are bit-identical:
    # the independent path shares no state, so the worker count cannot change results.
    assert set(serial.member_results) == set(parallel.member_results)
    for key, serial_result in serial.member_results.items():
        parallel_result = parallel.member_results[key]
        for name in serial_result.parameters.names:
            assert float(serial_result.parameters[name].value) == pytest.approx(
                float(parallel_result.parameters[name].value), rel=0, abs=0
            )
    # Messages are reported in run order regardless of completion order.
    assert serial.message == parallel.message
    assert serial.success == parallel.success


def test_fit_grouped_series_parallel_falls_back_for_unpicklable_model() -> None:
    # A model captured as a local closure cannot cross a process boundary; the batch
    # must still complete by transparently falling back to the sequential path.
    base = _parallel_oscillatory_model()
    members, initial = _parallel_members(base, n_runs=2, n_bins=400)
    global_params = [name for name in base.param_names if not name.startswith("A")]

    def _closure_model(t, **kwargs):  # not picklable (local function)
        return base.function(t, **kwargs)

    result = fit_grouped_series(
        "batch",
        members,
        _closure_model,
        global_params=global_params,
        local_params=list(GROUP_NUISANCE_PARAMS),
        initial_params=initial,
        seeding="as_provided",
        cost="poisson",
        max_workers=4,
    )
    assert set(result.member_results) == {-1600001, -1600002, -1601001, -1601002}


def _relaxing_cosine_model(t, frequency, Lambda, phase=0.0):  # noqa: N803
    """Module-level damped cosine (picklable for the parallel block-solver test)."""
    t = np.asarray(t, dtype=float)
    return np.cos(2.0 * np.pi * frequency * t + phase) * np.exp(-Lambda * t)


def _mixed_global_local_scenario(true_lambda):
    """Build a mixed grouped series: shared frequency, per-run Lambda, 2 groups/run.

    Returns ``(members, initial, common_kwargs)`` ready for ``fit_grouped_series``.
    """
    time = np.linspace(0.0, 8.0, 1601)
    frequency = 3.0
    n0, background, amplitude = 5.0e4, 50.0, 0.22

    def _counts(lam, phase):
        pol = _relaxing_cosine_model(time, frequency, lam, phase)
        return n0 * (1.0 + amplitude * pol) + background * np.exp(time / float(MUON_LIFETIME_US))

    members: dict[int, list[GroupedTimeDomainGroup]] = {}
    for run, lam in true_lambda.items():
        groups = []
        for gid, phase in ((1, 0.2), (2, np.deg2rad(170.0))):
            counts = _counts(lam, phase)
            groups.append(
                GroupedTimeDomainGroup(
                    group_id=gid,
                    group_name=f"g{gid}",
                    time=time.copy(),
                    counts=counts,
                    error=np.sqrt(np.clip(counts, 1.0, None)),
                    metadata={},
                )
            )
        members[run] = groups

    def _initial() -> ParameterSet:
        return ParameterSet(
            [
                Parameter("N0", 4.0e4),
                Parameter("background", 40.0),
                Parameter("amplitude", 0.2),
                Parameter("relative_phase", 0.0, min=-2.0 * np.pi, max=2.0 * np.pi),
                Parameter("frequency", 3.1),
                Parameter("Lambda", 0.6, min=0.0, max=10.0),
                Parameter("phase", 0.0, fixed=True),
            ]
        )

    initial = {run: {g.group_id: _initial() for g in groups} for run, groups in members.items()}
    common = dict(
        global_params=["frequency", "Lambda", "phase"],
        local_params=["N0", "background", "amplitude", "relative_phase"],
        initial_params=initial,
        cost="gaussian",
        cross_run_local_params=["Lambda"],
    )
    return members, frequency, common


def test_grouped_global_is_large_counts_free_params() -> None:
    rep = ParameterSet(
        [
            Parameter("frequency", 3.0),  # shared, free
            Parameter("phase", 0.0, fixed=True),  # shared, fixed
            Parameter("Lambda", 0.5),  # per-run, free
            Parameter("N0", 1.0),
            Parameter("background", 1.0),
            Parameter("amplitude", 0.1),
            Parameter("relative_phase", 0.0),  # 4 nuisances, free
        ]
    )
    # 2 runs x 2 groups: 1 shared + 1 per-run*2 + 4 nuis*4 = 19 (< 64 default → monolithic).
    assert not _grouped_global_is_large(
        rep,
        truly_global=["frequency", "phase"],
        per_run_physics=["Lambda"],
        nuisances=["N0", "background", "amplitude", "relative_phase"],
        n_runs=2,
        n_members=4,
    )
    # 8 runs x 2 groups: 1 + 1*8 + 4*16 = 73 (>= 64 → block-separable).
    assert _grouped_global_is_large(
        rep,
        truly_global=["frequency", "phase"],
        per_run_physics=["Lambda"],
        nuisances=["N0", "background", "amplitude", "relative_phase"],
        n_runs=8,
        n_members=16,
    )


def test_grouped_global_blockwise_matches_monolithic(monkeypatch) -> None:
    """The block-separable solver recovers the same physics as the monolithic fit."""
    true_lambda = {10: 0.3, 11: 1.0, 12: 0.6, 13: 1.4}
    members, frequency, common = _mixed_global_local_scenario(true_lambda)

    monolithic = fit_grouped_series("global", members, _relaxing_cosine_model, **common)

    # Force the block-separable route regardless of size, then re-fit the same problem.
    monkeypatch.setattr(
        "asymmetry.core.fitting.grouped_time_domain._grouped_global_is_large",
        lambda *a, **k: True,
    )
    block = fit_grouped_series(
        "global",
        members,
        _relaxing_cosine_model,
        block_separable=True,
        max_workers=1,
        **common,
    )

    assert monolithic.success and block.success
    assert "block-separable solver" in block.message
    shared_block = {p.name: p.value for p in block.shared_parameters}
    assert shared_block["frequency"] == pytest.approx(frequency, abs=1e-3)
    for run, lam in true_lambda.items():
        mono_key = min(k for k, src in monolithic.member_source_run.items() if src == run)
        block_key = min(k for k, src in block.member_source_run.items() if src == run)
        mono_lambda = monolithic.member_results[mono_key].parameters["Lambda"].value
        block_lambda = block.member_results[block_key].parameters["Lambda"].value
        assert block_lambda == pytest.approx(mono_lambda, abs=2e-3)
        assert block_lambda == pytest.approx(lam, abs=2e-2)
        # Each member carries the shared frequency + a (conditional) uncertainty.
        member = block.member_results[block_key]
        assert member.parameters["frequency"].value == pytest.approx(frequency, abs=1e-3)
        assert member.uncertainties.get("frequency") is not None


def test_grouped_global_blockwise_default_is_monolithic_below_threshold() -> None:
    """Below the free-parameter threshold a global fit stays monolithic even when allowed."""
    members, _frequency, common = _mixed_global_local_scenario({10: 0.3, 11: 1.0})
    result = fit_grouped_series(
        "global", members, _relaxing_cosine_model, block_separable=True, **common
    )
    assert result.success
    assert "block-separable solver" not in result.message


def test_grouped_global_blockwise_parallel_matches_sequential(monkeypatch) -> None:
    """Block-solver results are independent of inner-fit worker count."""
    members, _frequency, common = _mixed_global_local_scenario({10: 0.3, 11: 1.0, 12: 0.6})
    monkeypatch.setattr(
        "asymmetry.core.fitting.grouped_time_domain._grouped_global_is_large",
        lambda *a, **k: True,
    )
    serial = fit_grouped_series(
        "global", members, _relaxing_cosine_model, block_separable=True, max_workers=1, **common
    )
    parallel = fit_grouped_series(
        "global", members, _relaxing_cosine_model, block_separable=True, max_workers=3, **common
    )
    assert serial.success and parallel.success
    assert {p.name: p.value for p in serial.shared_parameters} == pytest.approx(
        {p.name: p.value for p in parallel.shared_parameters}, rel=0, abs=0
    )
    assert set(serial.member_results) == set(parallel.member_results)
    for key, s in serial.member_results.items():
        p = parallel.member_results[key]
        for name in s.parameters.names:
            assert float(s.parameters[name].value) == pytest.approx(
                float(p.parameters[name].value), rel=0, abs=0
            )


def test_grouped_global_blockwise_profiled_errors_widen_and_recover(monkeypatch) -> None:
    """Profiling the shared params yields marginal (>= conditional) errors, same values."""
    members, frequency, common = _mixed_global_local_scenario({10: 0.3, 11: 1.0, 12: 0.6})
    monkeypatch.setattr(
        "asymmetry.core.fitting.grouped_time_domain._grouped_global_is_large",
        lambda *a, **k: True,
    )

    def _freq_unc(result):
        key = min(result.member_results)
        return result.member_results[key].uncertainties.get("frequency")

    conditional = fit_grouped_series(
        "global",
        members,
        _relaxing_cosine_model,
        block_separable=True,
        max_workers=1,
        profile_shared_errors=False,
        **common,
    )
    profiled = fit_grouped_series(
        "global",
        members,
        _relaxing_cosine_model,
        block_separable=True,
        max_workers=1,
        profile_shared_errors=True,
        **common,
    )

    assert "errors are conditional" in conditional.message
    assert "errors profiled over the locals" in profiled.message
    # Same physics; the profiled error marginalises over the locals so it cannot shrink.
    cond_freq = {p.name: p.value for p in conditional.shared_parameters}["frequency"]
    prof_freq = {p.name: p.value for p in profiled.shared_parameters}["frequency"]
    assert prof_freq == pytest.approx(cond_freq, abs=1e-4)
    assert prof_freq == pytest.approx(frequency, abs=1e-2)
    cond_unc, prof_unc = _freq_unc(conditional), _freq_unc(profiled)
    assert cond_unc is not None and prof_unc is not None
    assert prof_unc >= cond_unc * (1.0 - 1e-6)
