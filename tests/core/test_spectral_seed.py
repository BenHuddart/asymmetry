from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.spectral import (
    default_frequency_model,
    seed_peak_parameters_from_dataset,
)


def _dc_dominated_spectrum(
    *, peak_freq: float = 30.0, n: int = 2000, f_max: float = 100.0
) -> MuonDataset:
    """Synthetic (Power)^1/2 spectrum shaped like the recorded EuO 2960 fixture.

    A tall apodisation/DC spike near 0 MHz dwarfs the genuine ~30 MHz
    precession peak, as observed live (ratio ~3x).
    """
    freq = np.linspace(0.0, f_max, n)
    dc_spike = 30.0 * np.exp(-0.5 * (freq / 0.5) ** 2)
    physical_peak = 10.0 * np.exp(-0.5 * ((freq - peak_freq) / 1.5) ** 2)
    bg = 0.2
    values = bg + dc_spike + physical_peak
    return MuonDataset(
        time=freq,
        asymmetry=values,
        error=np.full_like(freq, 0.05),
        metadata={"run_number": 2960, "plot_domain": "frequency"},
    )


def test_seed_peak_excludes_dc_spike_and_finds_physical_peak() -> None:
    dataset = _dc_dominated_spectrum(peak_freq=30.0)
    model = default_frequency_model()

    # Sanity: unguarded argmax would land on the DC spike, not the physical peak.
    y = np.asarray(dataset.asymmetry)
    assert float(dataset.time[int(np.nanargmax(y))]) < 1.0

    seeds = seed_peak_parameters_from_dataset(dataset, model)

    assert seeds["nu0"] == pytest.approx(30.0, abs=1.0)


def test_seed_peak_falls_back_to_global_argmax_when_guard_empties_array() -> None:
    freq = np.linspace(0.0, 1.0, 50)
    values = 5.0 - freq  # monotonic decay entirely inside a wide guard band
    dataset = MuonDataset(
        time=freq,
        asymmetry=values,
        error=np.full_like(freq, 0.05),
        metadata={"run_number": 1, "plot_domain": "frequency"},
    )
    model = default_frequency_model()

    seeds = seed_peak_parameters_from_dataset(dataset, model, guard_freq_mhz=2.0)

    assert seeds["nu0"] == 0.0


def test_seed_peak_bg_uses_guarded_median_not_dc_spike() -> None:
    dataset = _dc_dominated_spectrum(peak_freq=30.0)
    model = default_frequency_model()

    seeds = seed_peak_parameters_from_dataset(dataset, model)

    assert seeds["bg"] < 1.0


def test_seed_peak_fwhm_excludes_dc_spike_from_half_max_crossing() -> None:
    """The half-max crossing search must stay inside the same guarded region
    as the peak search, or a DC spike that also exceeds half_height drags the
    span out to the DC spike's edge and inflates fwhm by an order of
    magnitude even though nu0 correctly skipped it.
    """
    dataset = _dc_dominated_spectrum(peak_freq=30.0)  # physical peak sigma=1.5 MHz
    model = default_frequency_model()

    seeds = seed_peak_parameters_from_dataset(dataset, model)

    # True FWHM = 2.3548 * sigma ~= 3.53 MHz; a DC-contaminated crossing would
    # instead span from near the DC spike's edge to the peak's far edge (~30 MHz).
    assert seeds["fwhm"] < 10.0


def _two_peak_spectrum(
    *, first: float = 2.0, second: float = 3.5, second_height: float = 3.0
) -> MuonDataset:
    """A frequency spectrum with two Gaussian lines plus a flat background."""
    freq = np.linspace(1.0, 5.0, 401)
    values = (
        0.4
        + 6.0 * np.exp(-4.0 * np.log(2.0) * ((freq - first) / 0.2) ** 2)
        + second_height * np.exp(-4.0 * np.log(2.0) * ((freq - second) / 0.3) ** 2)
    )
    return MuonDataset(
        time=freq,
        asymmetry=values,
        error=np.full_like(freq, 0.05),
        metadata={"run_number": 7, "plot_domain": "frequency"},
    )


def _two_peak_model() -> CompositeModel:
    return CompositeModel(
        ["GaussianPeak", "GaussianPeak", "ConstantBackground"], operators=["+", "+"]
    )


def test_seed_two_peaks_assigns_strongest_first_to_each_component() -> None:
    dataset = _two_peak_spectrum(first=2.0, second=3.5, second_height=3.0)
    seeds = seed_peak_parameters_from_dataset(dataset, _two_peak_model())

    # Both peak components are seeded onto real lines (not the nu0=1.0 default).
    assert seeds["nu0_1"] == pytest.approx(2.0, abs=0.05)
    assert seeds["nu0_2"] == pytest.approx(3.5, abs=0.05)
    # Strongest-first: the taller line seeds component 1.
    assert seeds["height_1"] > seeds["height_2"]


def test_seed_two_peaks_seeds_a_weak_second_line() -> None:
    """A weak-but-real second line the user declared is still seeded (not gated)."""
    dataset = _two_peak_spectrum(first=2.0, second=3.5, second_height=0.6)
    seeds = seed_peak_parameters_from_dataset(dataset, _two_peak_model())

    assert seeds["nu0_2"] == pytest.approx(3.5, abs=0.1)


def test_seed_two_peaks_under_detection_stays_on_screen() -> None:
    """One real peak but two components: the extra nu0 must stay in the window."""
    freq = np.linspace(1.0, 5.0, 401)
    values = 0.4 + 6.0 * np.exp(-4.0 * np.log(2.0) * ((freq - 2.0) / 0.2) ** 2)
    dataset = MuonDataset(
        time=freq,
        asymmetry=values,
        error=np.full_like(freq, 0.05),
        metadata={"run_number": 8, "plot_domain": "frequency"},
    )
    seeds = seed_peak_parameters_from_dataset(dataset, _two_peak_model())

    assert 1.0 <= seeds["nu0_1"] <= 5.0
    assert 1.0 <= seeds["nu0_2"] <= 5.0


def test_seed_two_peaks_with_linear_background_seeds_slope() -> None:
    model = CompositeModel(
        ["GaussianPeak", "GaussianPeak", "LinearBackground"], operators=["+", "+"]
    )
    seeds = seed_peak_parameters_from_dataset(_two_peak_spectrum(), model)

    assert "slope" in seeds
    assert "nu0_1" in seeds and "nu0_2" in seeds


# ---------------------------------------------------------------------------
# Golden output-identity tests for the bounded (wlen) prominence search.
#
# ``_detect_top_n_local_maxima`` bounds find_peaks' prominence window to
# ``max(2001, odd(n // 8))`` to avoid an O(n^2) worst case on quasi-monotonic
# spectra.  The bound must not change which peaks are selected: each test
# below compares the seeded parameter dict from the bounded implementation
# against an unbounded reference (the same seeder with the ``wlen`` argument
# stripped from find_peaks), exactly on peak positions and allclose on the
# derived values.
# ---------------------------------------------------------------------------


def _golden_dataset(freq: np.ndarray, values: np.ndarray) -> MuonDataset:
    return MuonDataset(
        time=freq,
        asymmetry=values,
        error=np.full_like(freq, 0.05),
        metadata={"run_number": 9, "plot_domain": "frequency"},
    )


def _golden_benign_spectrum(n: int) -> MuonDataset:
    """Flat background, DC spike, two well-separated narrow lines, noise."""
    rng = np.random.default_rng(0)
    freq = np.linspace(0.0, 100.0, n)
    values = (
        0.2
        + 30.0 * np.exp(-0.5 * (freq / 0.5) ** 2)
        + 10.0 * np.exp(-0.5 * ((freq - 25.0) / 1.5) ** 2)
        + 6.0 * np.exp(-0.5 * ((freq - 45.0) / 2.0) ** 2)
        + rng.normal(scale=0.05, size=n)
    )
    return _golden_dataset(freq, values)


def _golden_trending_spectrum(n: int) -> MuonDataset:
    """Two lines on a quasi-monotonic drifting background (decay skirt shape).

    This is the pathological regime for the unbounded prominence search: most
    noise maxima on the drift scan all the way to the array edge.  The drift
    magnitude is kept independent of ``n`` so the shape (not the sampling)
    defines the regime.
    """
    rng = np.random.default_rng(1)
    freq = np.linspace(0.0, 100.0, n)
    drift = 5.0 - 0.001 * np.arange(n) * (262144.0 / n)
    values = (
        drift
        + 10.0 * np.exp(-0.5 * ((freq - 25.0) / 1.5) ** 2)
        + 6.0 * np.exp(-0.5 * ((freq - 60.0) / 2.0) ** 2)
        + rng.normal(scale=0.02, size=n)
    )
    return _golden_dataset(freq, values)


def _golden_weak_broad_spectrum(n: int, *, fwhm_fraction: float) -> MuonDataset:
    """A tall narrow line plus a weak *broad* second line.

    The broad line's FWHM is ``fwhm_fraction`` of the spectrum span — the
    regime where a too-small ``wlen`` silently truncates the broad line's
    prominence and a noise maximum outranks it.
    """
    rng = np.random.default_rng(2)
    freq = np.linspace(0.0, 100.0, n)
    sigma = fwhm_fraction * 100.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    values = (
        0.4
        + 6.0 * np.exp(-0.5 * ((freq - 25.0) / 1.5) ** 2)
        + 0.6 * np.exp(-0.5 * ((freq - 60.0) / sigma) ** 2)
        + rng.normal(scale=0.02, size=n)
    )
    return _golden_dataset(freq, values)


def _bounded_and_unbounded_seeds(
    dataset: MuonDataset, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, float], dict[str, float]]:
    """Seed once as shipped and once with the wlen bound stripped."""
    import scipy.signal

    model = _two_peak_model()
    bounded = seed_peak_parameters_from_dataset(dataset, model)

    real_find_peaks = scipy.signal.find_peaks

    def unbounded_find_peaks(y, **kwargs):  # noqa: ANN001, ANN202
        kwargs.pop("wlen", None)
        return real_find_peaks(y, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(scipy.signal, "find_peaks", unbounded_find_peaks)
        reference = seed_peak_parameters_from_dataset(dataset, model)
    return bounded, reference


def _assert_seeds_identical(bounded: dict[str, float], reference: dict[str, float]) -> None:
    assert set(bounded) == set(reference)
    # Peak positions must match exactly: identical selected bins give
    # bit-identical parabolic-refined centres.
    assert bounded["nu0_1"] == reference["nu0_1"]
    assert bounded["nu0_2"] == reference["nu0_2"]
    for name in bounded:
        assert bounded[name] == pytest.approx(reference[name], rel=1e-12, abs=1e-15)


@pytest.mark.parametrize("n", [16384, 262144])
def test_bounded_prominence_matches_unbounded_on_benign_spectrum(
    n: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    bounded, reference = _bounded_and_unbounded_seeds(_golden_benign_spectrum(n), monkeypatch)
    _assert_seeds_identical(bounded, reference)


# The trending golden uses n=65536 (not 262144) because the *unbounded
# reference* is the O(n^2) case being fixed (~4 s at 262144); the bounded
# path itself is fast at any n.
@pytest.mark.parametrize("n", [16384, 65536])
def test_bounded_prominence_matches_unbounded_on_trending_spectrum(
    n: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    bounded, reference = _bounded_and_unbounded_seeds(_golden_trending_spectrum(n), monkeypatch)
    _assert_seeds_identical(bounded, reference)


@pytest.mark.parametrize("n", [16384, 262144])
@pytest.mark.parametrize("fwhm_fraction", [0.10, 0.20])
def test_bounded_prominence_matches_unbounded_on_weak_broad_line(
    n: int, fwhm_fraction: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A weak line as broad as 10-20% of the spectrum still ranks identically."""
    dataset = _golden_weak_broad_spectrum(n, fwhm_fraction=fwhm_fraction)
    bounded, reference = _bounded_and_unbounded_seeds(dataset, monkeypatch)
    _assert_seeds_identical(bounded, reference)
    # Sanity: both actually landed on the real broad line, not a noise maximum.
    assert reference["nu0_2"] == pytest.approx(60.0, abs=3.0)
