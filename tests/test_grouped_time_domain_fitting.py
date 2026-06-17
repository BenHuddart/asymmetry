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
from asymmetry.core.fitting.engine import FitResult
from asymmetry.core.fitting.grouped_time_domain import (
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
