"""The count-array MaxEnt entry point, and its equivalence with ``maxent(run, …)``.

Every run here is synthetic and built from invented round numbers; nothing in
this file depends on measured data.
"""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.maxent import (
    MaxEntConfig,
    build_maxent_input,
    build_maxent_input_from_counts,
    maxent,
    maxent_from_counts,
)

# A synthetic four-detector transverse-field run: a 2.5 MHz line at four
# phases, on a 320 G nominal field, with 40 ns bins.
_FREQUENCY_MHZ = 2.5
_FIELD_GAUSS = 320.0
_BIN_WIDTH_US = 0.04
_N_BINS = 256
_T0_BIN = 0


def _synthetic_run(*, t0_bin: int = _T0_BIN) -> Run:
    rng = np.random.default_rng(90210)
    time = (np.arange(_N_BINS, dtype=float) - t0_bin) * _BIN_WIDTH_US
    histograms: list[Histogram] = []
    for phase in (0.0, 90.0, 180.0, 270.0):
        modulation = 1.0 + 0.2 * np.cos(2.0 * np.pi * _FREQUENCY_MHZ * time + np.deg2rad(phase))
        expected = 3000.0 * np.exp(-np.clip(time, 0.0, None) / 2.1969811) * modulation
        counts = rng.poisson(np.clip(expected, 1.0, None)).astype(float)
        histograms.append(Histogram(counts=counts, bin_width=_BIN_WIDTH_US, t0_bin=t0_bin))
    return Run(
        run_number=77,
        histograms=histograms,
        metadata={"field": _FIELD_GAUSS, "temperature": 4.0},
        grouping={
            "groups": {1: [1], 2: [2], 3: [3], 4: [4]},
            "group_names": {1: "G1", 2: "G2", 3: "G3", 4: "G4"},
            "first_good_bin": 0,
            "last_good_bin": _N_BINS - 1,
            "deadtime_correction": False,
            "alpha": 1.0,
            "forward_group": 1,
            "backward_group": 2,
        },
    )


def _counts_from(run: Run) -> dict[int, np.ndarray]:
    """The same per-group counts the Run path derives from its histograms."""
    return {
        index + 1: np.asarray(hist.counts, dtype=float) for index, hist in enumerate(run.histograms)
    }


def _config(**overrides) -> MaxEntConfig:
    base = {
        "n_spectrum_points": 96,
        "f_min_mhz": 0.5,
        "f_max_mhz": 5.0,
        "auto_window": False,
        "outer_cycles": 3,
    }
    base.update(overrides)
    return MaxEntConfig(**base)


def _from_counts(run: Run, config: MaxEntConfig, **overrides):
    kwargs = {
        "bin_width_us": _BIN_WIDTH_US,
        "t0_bin": _T0_BIN,
        "group_names": run.grouping["group_names"],
        "field_gauss": _FIELD_GAUSS,
        "alpha": run.grouping["alpha"],
        "run_number": run.run_number,
    }
    kwargs.update(overrides)
    return _counts_from(run), config, kwargs


def test_input_from_counts_matches_the_run_path_exactly() -> None:
    run = _synthetic_run()
    config = _config()

    from_run = build_maxent_input(run, config)
    counts, cfg, kwargs = _from_counts(run, config)
    from_counts = build_maxent_input_from_counts(counts, cfg, **kwargs)

    assert from_counts.run_number == from_run.run_number
    assert from_counts.n_spectrum_points == from_run.n_spectrum_points
    assert from_counts.f_min_mhz == from_run.f_min_mhz
    assert from_counts.f_max_mhz == from_run.f_max_mhz
    assert from_counts.default_level == from_run.default_level
    assert from_counts.mode == from_run.mode
    assert len(from_counts.groups) == len(from_run.groups)
    for expected, actual in zip(from_run.groups, from_counts.groups, strict=True):
        assert actual.group_id == expected.group_id
        assert actual.group_name == expected.group_name
        assert np.array_equal(actual.time_us, expected.time_us)
        assert np.array_equal(actual.signal, expected.signal)
        assert np.array_equal(actual.sigma, expected.sigma)
        assert np.array_equal(actual.mask, expected.mask)
        assert actual.phase_degrees == expected.phase_degrees
        assert actual.amplitude == expected.amplitude
        assert actual.background == expected.background


def test_maxent_from_counts_matches_maxent_exactly() -> None:
    run = _synthetic_run()
    config = _config()

    from_run = maxent(run, config, cycles=3, early_stop=False)
    counts, cfg, kwargs = _from_counts(run, config)
    from_counts = maxent_from_counts(counts, cfg, cycles=3, early_stop=False, **kwargs)

    assert np.array_equal(from_counts.frequencies_mhz, from_run.frequencies_mhz)
    assert np.array_equal(from_counts.spectrum, from_run.spectrum)
    assert from_counts.stop_reason == from_run.stop_reason
    assert from_counts.diagnostics.chi2 == from_run.diagnostics.chi2


@pytest.mark.parametrize(
    "overrides",
    [
        {"t_min_us": 0.2, "t_max_us": 6.0},
        {"time_binning_factor": 2},
        {"exclude_t_min_us": 1.0, "exclude_t_max_us": 2.0},
        {"auto_window": True, "f_min_mhz": None, "f_max_mhz": None},
        {"selected_group_ids": [1, 3]},
        {"auto_phase_seed": True},
    ],
    ids=lambda o: "-".join(sorted(o)),
)
def test_equivalence_holds_across_configurations(overrides: dict) -> None:
    run = _synthetic_run()
    config = _config(**overrides)

    from_run = build_maxent_input(run, config)
    counts, cfg, kwargs = _from_counts(run, config)
    from_counts = build_maxent_input_from_counts(counts, cfg, **kwargs)

    assert from_counts.f_min_mhz == from_run.f_min_mhz
    assert from_counts.f_max_mhz == from_run.f_max_mhz
    assert from_counts.n_spectrum_points == from_run.n_spectrum_points
    assert [g.group_id for g in from_counts.groups] == [g.group_id for g in from_run.groups]
    for expected, actual in zip(from_run.groups, from_counts.groups, strict=True):
        assert np.array_equal(actual.time_us, expected.time_us)
        assert np.array_equal(actual.signal, expected.signal)
        assert np.array_equal(actual.sigma, expected.sigma)
        assert actual.phase_degrees == pytest.approx(expected.phase_degrees)


def test_a_non_zero_t0_bin_reproduces_the_run_time_axis() -> None:
    run = _synthetic_run(t0_bin=20)
    config = _config()

    from_run = build_maxent_input(run, config)
    counts, cfg, kwargs = _from_counts(run, config, t0_bin=20)
    from_counts = build_maxent_input_from_counts(counts, cfg, **kwargs)

    assert np.array_equal(from_counts.groups[0].time_us, from_run.groups[0].time_us)
    assert np.array_equal(from_counts.groups[0].signal, from_run.groups[0].signal)


def test_zf_lf_mode_ties_the_pair_in_the_order_given() -> None:
    run = _synthetic_run()
    config = _config(mode="zf_lf", selected_group_ids=[1, 2])

    from_run = build_maxent_input(run, config)
    counts, cfg, kwargs = _from_counts(run, config)
    from_counts = build_maxent_input_from_counts(counts, cfg, **kwargs)

    assert from_counts.mode == "zf_lf" == from_run.mode
    assert from_counts.zf_lf_alpha == from_run.zf_lf_alpha
    assert [g.phase_degrees for g in from_counts.groups] == [
        g.phase_degrees for g in from_run.groups
    ]


# --- the count-domain boundary ----------------------------------------------


def test_a_sequence_of_arrays_numbers_the_groups_from_one() -> None:
    run = _synthetic_run()
    config = _config()

    mapped = build_maxent_input_from_counts(
        _counts_from(run), config, bin_width_us=_BIN_WIDTH_US, field_gauss=_FIELD_GAUSS
    )
    sequenced = build_maxent_input_from_counts(
        [_counts_from(run)[gid] for gid in (1, 2, 3, 4)],
        config,
        bin_width_us=_BIN_WIDTH_US,
        field_gauss=_FIELD_GAUSS,
    )

    assert [g.group_id for g in sequenced.groups] == [g.group_id for g in mapped.groups]
    for expected, actual in zip(mapped.groups, sequenced.groups, strict=True):
        assert np.array_equal(actual.signal, expected.signal)


def test_an_asymmetry_curve_is_refused_by_the_negative_count_guard() -> None:
    """The boundary this entry point exists to make explicit."""
    time = np.linspace(0.0, 8.0, 256)
    asymmetry = 20.0 * np.cos(2.0 * np.pi * 2.5 * time)

    with pytest.raises(ValueError, match="not an asymmetry curve"):
        maxent_from_counts({1: asymmetry}, bin_width_us=_BIN_WIDTH_US)


def test_ragged_groups_are_refused() -> None:
    with pytest.raises(ValueError, match="same number of count bins"):
        build_maxent_input_from_counts({1: np.ones(64), 2: np.ones(32)}, bin_width_us=_BIN_WIDTH_US)


def test_empty_counts_are_refused() -> None:
    with pytest.raises(ValueError, match="at least one group"):
        build_maxent_input_from_counts({}, bin_width_us=_BIN_WIDTH_US)


def test_a_non_positive_bin_width_is_refused() -> None:
    with pytest.raises(ValueError, match="positive number of"):
        build_maxent_input_from_counts({1: np.ones(64)}, bin_width_us=0.0)


def test_two_dimensional_counts_are_refused() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        build_maxent_input_from_counts({1: np.ones((2, 32))}, bin_width_us=_BIN_WIDTH_US)


def test_non_finite_counts_are_refused() -> None:
    counts = np.ones(64)
    counts[3] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        build_maxent_input_from_counts({1: counts}, bin_width_us=_BIN_WIDTH_US)


def test_a_dataset_is_not_a_counts_argument() -> None:
    dataset = MuonDataset(
        time=np.linspace(0.0, 1.0, 16),
        asymmetry=np.ones(16),
        error=np.ones(16),
        metadata={},
    )
    with pytest.raises(TypeError, match="mapping .* or a sequence"):
        maxent_from_counts(dataset, bin_width_us=_BIN_WIDTH_US)


def test_the_docstring_states_there_is_no_asymmetry_domain_maxent() -> None:
    """The plan's docs requirement, pinned so it cannot be edited away."""
    import asymmetry.core.fourier.maxent as fourier_maxent

    for text in (
        maxent_from_counts.__doc__,
        maxent.__doc__,
        fourier_maxent.__doc__,
        fourier_maxent.maxent.__doc__,
    ):
        assert text is not None
        assert "no asymmetry-domain MaxEnt" in text
    assert "fft_arrays" in (maxent_from_counts.__doc__ or "")
    assert "fft_arrays" in (maxent.__doc__ or "")
