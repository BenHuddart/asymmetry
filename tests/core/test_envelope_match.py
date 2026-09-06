"""Tests for the time-domain envelope matched filter (R2).

The banks recognise damped-envelope signatures (F-mu-F, mu-F, Kubo-Toyabe) that
FFT peak detection misses. Datasets use the percent scale and the realistic
exploding, capped error bars of dying-muon statistics (mirroring the R1 evidence
dataset in ``test_peak_detection``), so the significance null is exercised on the
same noise model as real data.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting import envelope_match as em
from asymmetry.core.fitting.envelope_match import match_envelope_banks
from asymmetry.core.fitting.models import longitudinal_field_kubo_toyabe
from asymmetry.core.fitting.muon_fluorine.polarization import (
    linear_fmuf_polarization,
    mu_f_polarization,
)
from asymmetry.core.fitting.peak_detection import (
    deserialize_multiplet_match,
    serialize_multiplet_match,
)


def _exploding_dataset(signal_fn, *, seed: int, metadata: dict | None = None) -> MuonDataset:
    """Percent-scale record with realistic dying-muon statistics.

    σ(t) = 0.7·exp(t / (2·2.2)) capped at 100 %, Gaussian noise drawn per point;
    ~1/3 of the record is pure noise at the cap (same construction as the R1
    exploding-error evidence dataset).
    """
    t = np.linspace(0.15, 32.6, 2000)
    sigma = np.minimum(0.7 * np.exp(t / (2.0 * 2.2)), 100.0)
    rng = np.random.default_rng(seed)
    payload = {"run_number": 1}
    payload.update(metadata or {})
    return MuonDataset(
        time=t,
        asymmetry=signal_fn(t) + rng.normal(0.0, sigma),
        error=sigma,
        metadata=payload,
    )


def _best(matches, kind):
    hits = [m for m in matches if m.kind == kind]
    return max(hits, key=lambda m: m.quality) if hits else None


# --------------------------------------------------------------------------- #
# 1. Bank matches on the family it exists for, with recovered parameter
# --------------------------------------------------------------------------- #


def test_fmuf_bank_matches_and_recovers_r() -> None:
    # S4: linear F-mu-F at r = 1.17 A under an exp(-0.2 t) envelope, ZF metadata.
    dataset = _exploding_dataset(
        lambda t: 20.0 * np.exp(-0.2 * t) * linear_fmuf_polarization(t, 1.17) + 4.0,
        seed=101,
    )
    match = _best(match_envelope_banks(dataset), "fmuf_envelope")
    assert match is not None
    assert match.family_key == "fmuf"
    assert match.quality > 0.5
    r = match.derived("r_muF_angstrom")
    assert r is not None
    assert abs(r - 1.17) <= 0.10 * 1.17  # within 10 %


def test_muf_bank_matches_and_recovers_r() -> None:
    dataset = _exploding_dataset(
        lambda t: 20.0 * np.exp(-0.2 * t) * mu_f_polarization(t, 1.10) + 4.0,
        seed=107,
    )
    match = _best(match_envelope_banks(dataset), "muF_envelope")
    assert match is not None
    r = match.derived("r_muF_angstrom")
    assert r is not None
    assert abs(r - 1.10) <= 0.10 * 1.10


def test_kt_bank_matches_and_recovers_delta() -> None:
    dataset = _exploding_dataset(
        lambda t: 20.0 * longitudinal_field_kubo_toyabe(t, 1.0, 0.3, 0.0, 0.0) + 4.0,
        seed=102,
    )
    match = _best(match_envelope_banks(dataset), "kt_envelope")
    assert match is not None
    assert match.family_key == "kt"
    delta = match.derived("Delta")
    assert delta is not None
    assert abs(delta - 0.3) <= 0.15 * 0.3  # within ~15 %


# --------------------------------------------------------------------------- #
# 2. Significance: structureless / smooth-relaxation data must NOT match.
#    Pure noise and flat data are the R3 controls; plain/stretched exp is the
#    dangerous *smooth-residual* false positive the monotonic detrend defeats.
# --------------------------------------------------------------------------- #


def test_pure_noise_matches_no_bank() -> None:
    dataset = _exploding_dataset(lambda t: np.full_like(t, 4.0), seed=105)
    assert match_envelope_banks(dataset) == ()


def test_plain_exponential_decay_matches_no_bank() -> None:
    # A monotonic decay is annihilated by the monotonic detrend, so the KT bank
    # (whose dangerous FP mode is exactly a smooth decay) must not fire.
    dataset = _exploding_dataset(
        lambda t: 20.0 * np.exp(-0.2 * t) + 4.0,
        seed=103,
        metadata={"field_direction": "Zero field"},
    )
    assert match_envelope_banks(dataset) == ()


def test_stretched_exponential_matches_no_bank() -> None:
    dataset = _exploding_dataset(
        lambda t: 20.0 * np.exp(-((0.2 * t) ** 0.6)) + 4.0,
        seed=104,
        metadata={"field_direction": "Zero field"},
    )
    assert match_envelope_banks(dataset) == ()


def test_compressed_exponential_matches_no_bank() -> None:
    # beta > 1 is the sharpest KT-bank false-positive mode: static-KT early-time
    # decay is Gaussian (beta ~ 2), so the fixed-exponential detrend leaves a
    # compressed-exponential residual the bank matched at quality ~0.96 before
    # the monotonic veto existed. The veto must reject it: a lone monotonic
    # relaxation explains this signal at least as well as any enveloped
    # template.
    dataset = _exploding_dataset(
        lambda t: 20.0 * np.exp(-((0.35 * t) ** 1.6)) + 4.0,
        seed=106,
        metadata={"field_direction": "Zero field"},
    )
    assert match_envelope_banks(dataset) == ()


# --------------------------------------------------------------------------- #
# 3. Normalization invariance to amplitude scale and DC offset
# --------------------------------------------------------------------------- #


def test_match_invariant_to_amplitude_scale_and_offset() -> None:
    base = _exploding_dataset(
        lambda t: 20.0 * np.exp(-0.2 * t) * linear_fmuf_polarization(t, 1.17) + 4.0,
        seed=101,
    )
    scaled = MuonDataset(
        time=base.time,
        asymmetry=3.0 * base.asymmetry + 15.0,  # amplitude ×3, offset +15
        error=3.0 * base.error,  # weights track the amplitude scale
        metadata=dict(base.metadata),
    )
    r_base = _best(match_envelope_banks(base), "fmuf_envelope").derived("r_muF_angstrom")
    r_scaled = _best(match_envelope_banks(scaled), "fmuf_envelope").derived("r_muF_angstrom")
    # Same recovered distance regardless of amplitude/offset (up to grid quantum).
    assert abs(r_base - r_scaled) <= 0.02


# --------------------------------------------------------------------------- #
# 4. Scope gating skips out-of-scope banks
# --------------------------------------------------------------------------- #


def test_include_families_gates_banks() -> None:
    dataset = _exploding_dataset(
        lambda t: 20.0 * np.exp(-0.2 * t) * linear_fmuf_polarization(t, 1.17) + 4.0,
        seed=101,
    )
    kt_only = match_envelope_banks(dataset, include_families=frozenset({"kt"}))
    assert all(m.family_key == "kt" for m in kt_only)  # fmuf bank was skipped


# --------------------------------------------------------------------------- #
# 5. Reproducibility: the seeded surrogate null gives a stable boundary
# --------------------------------------------------------------------------- #


def test_matching_is_deterministic() -> None:
    dataset = _exploding_dataset(
        lambda t: 20.0 * longitudinal_field_kubo_toyabe(t, 1.0, 0.3, 0.0, 0.0) + 4.0,
        seed=102,
    )
    first = match_envelope_banks(dataset)
    second = match_envelope_banks(dataset)
    assert [(m.kind, m.quality, m.derived_values) for m in first] == [
        (m.kind, m.quality, m.derived_values) for m in second
    ]


def test_surrogate_seed_is_stable_across_hash_salting() -> None:
    # The surrogate seed must NOT depend on builtin hash() (salted per process by
    # PYTHONHASHSEED) or the threshold — and thus the match/no-match boundary —
    # would differ across CI processes. Derive it in two subprocesses with
    # different PYTHONHASHSEED values and require the same seed.
    import subprocess
    import sys

    script = (
        "import numpy as np;"
        "from asymmetry.core.fitting.envelope_match import _seed_from_signal;"
        "print(_seed_from_signal(np.linspace(-1.0, 1.0, 257)))"
    )
    seeds = []
    for hashseed in ("0", "1"):
        env = {"PYTHONHASHSEED": hashseed, "PYTHONPATH": "src", "PATH": os.environ["PATH"]}
        out = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        seeds.append(out.stdout.strip())
    assert seeds[0] == seeds[1] and seeds[0]


# --------------------------------------------------------------------------- #
# 6. MultipletMatch serialization is kind-agnostic (round-trips new kinds)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("kind", "family", "derived"),
    [
        ("fmuf_envelope", "fmuf", (("r_muF_angstrom", 1.17),)),
        ("muF_envelope", "fmuf", (("r_muF_angstrom", 1.10),)),
        ("kt_envelope", "kt", (("Delta", 0.31), ("B_L", 0.0))),
    ],
)
def test_envelope_match_serialization_round_trip(kind, family, derived) -> None:
    dataset = _exploding_dataset(
        lambda t: 20.0 * longitudinal_field_kubo_toyabe(t, 1.0, 0.3, 0.0, 0.0) + 4.0,
        seed=102,
    )
    # Build a real match of the requested kind if produced; else synthesise one
    # to prove the serializer is kind-agnostic for the new kinds regardless.
    from asymmetry.core.fitting.peak_detection import MultipletMatch

    match = MultipletMatch(
        kind=kind,
        family_key=family,
        peak_indices=(),
        quality=0.72,
        derived_values=derived,
        note="synthetic",
    )
    payload = serialize_multiplet_match(match)
    restored = deserialize_multiplet_match(payload)
    assert restored == match
    # Also confirm a genuinely produced envelope match round-trips.
    produced = _best(match_envelope_banks(dataset), "kt_envelope")
    assert produced is not None
    assert deserialize_multiplet_match(serialize_multiplet_match(produced)) == produced


# --------------------------------------------------------------------------- #
# Matching a rebinned copy of a record
# --------------------------------------------------------------------------- #


def test_rebinned_copy_matches_the_same_bank_and_parameter() -> None:
    """The banks need no bandwidth, so a value-rebinned copy carries the match.

    This is what lets the wizard match on its analysed (rebinned) record instead
    of the raw one: on a 90 000-point 0.1 ns record the banks cost 19.9 s on the
    full record and 4.2 s on the ×5 copy.
    """
    dataset = _exploding_dataset(
        lambda t: 20.0 * np.exp(-0.2 * t) * linear_fmuf_polarization(t, 1.17) + 4.0,
        seed=101,
    )
    window = 32.6 - 0.15

    full = _best(match_envelope_banks(dataset), "fmuf_envelope")
    rebinned = _best(
        match_envelope_banks(dataset.rebin(4), analysis_window_us=window), "fmuf_envelope"
    )

    assert full is not None and rebinned is not None
    assert rebinned.derived("r_muF_angstrom") == pytest.approx(
        full.derived("r_muF_angstrom"), rel=0.1
    )


def test_an_explicit_window_overrides_the_datasets_own() -> None:
    """The caller may pin the window the banks are built for.

    A rebinned copy's averaged σ carries less scatter, so its own
    ``effective_analysis_window`` verdict can run past the record's; the wizard
    passes the full record's window rather than letting a rebin move it.
    """
    dataset = _exploding_dataset(
        lambda t: 20.0 * np.exp(-0.2 * t) * linear_fmuf_polarization(t, 1.17) + 4.0,
        seed=101,
    )

    full = _best(match_envelope_banks(dataset), "fmuf_envelope")
    cropped = _best(match_envelope_banks(dataset, analysis_window_us=1.0), "fmuf_envelope")

    assert full is not None
    # One microsecond of a 32 µs record cannot support the same verdict — the
    # override really does bind.
    assert cropped is None or cropped.quality != pytest.approx(full.quality)


# --------------------------------------------------------------------------- #
# 7. The variable-projection detrend against the curve_fit path it replaced
# --------------------------------------------------------------------------- #


def _legacy_monotonic_detrend(
    t: np.ndarray, y: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, float, bool]:
    """The pre-variable-projection detrend, kept here as the reference.

    A verbatim copy of the three-parameter ``curve_fit`` implementation
    ``_monotonic_detrend_rows`` replaced: same model, same seed, same weighting.
    Returns the residual, the fitted ``lambda`` and whether the fit converged.
    """
    import warnings

    from scipy.optimize import OptimizeWarning, curve_fit

    def _model(tt: np.ndarray, amp: float, lam: float, base: float) -> np.ndarray:
        return amp * np.exp(np.clip(-lam * tt, -700.0, 700.0)) + base

    sigma = 1.0 / np.sqrt(np.clip(weights, 1e-12, None))
    p0 = [float(y[0] - y[-1]), 0.2, float(y[-1])]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, _ = curve_fit(_model, t, y, p0=p0, sigma=sigma, maxfev=2000)
        return y - _model(t, *popt), float(popt[1]), True
    except Exception:
        return y - float(np.average(y, weights=weights)), float("nan"), False


def _detrend_window() -> tuple[np.ndarray, np.ndarray]:
    """2 000 points over 0.15-16 µs with realistic exploding error bars."""
    t = np.linspace(0.15, 16.0, 2000)
    sigma = np.minimum(0.7 * np.exp(t / (2.0 * 2.2)), 100.0)
    return t, 1.0 / np.clip(sigma, 1e-12, None) ** 2


_BANK_SPECS = (
    ("fmuf_envelope", em._R_GRID),
    ("muF_envelope", em._R_GRID),
    ("kt_envelope", em._DELTA_GRID),
)


def _weighted_sse(residual: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(weights * residual**2))


@pytest.mark.parametrize(("kind", "grid"), _BANK_SPECS, ids=[spec[0] for spec in _BANK_SPECS])
def test_vector_detrend_never_fits_a_bank_template_worse_than_curve_fit(kind, grid) -> None:
    """Variable projection is a *global* solve over ``lambda``; curve_fit is local.

    The old three-parameter fit starts from a fixed seed and, on these bank
    templates, frequently stalls near ``lambda = 0`` (a handful of its optima are
    even slightly negative — a growing exponential, which is not a monotonic
    relaxation at all). The grid-plus-polish solve searches the whole resolvable
    range of positive rates, so it can only match or beat that minimum.
    """
    t, weights = _detrend_window()
    raw = np.asarray([em._template_values(kind, t, float(p)) for p in grid], dtype=float)

    new = em._monotonic_detrend_rows(t, raw, weights)

    for row, template in zip(new, raw, strict=True):
        legacy, _lam, converged = _legacy_monotonic_detrend(t, template, weights)
        assert converged
        legacy_sse = _weighted_sse(legacy, weights)
        # The 1e-9 slack is the polish's own convergence tolerance, not headroom.
        assert _weighted_sse(row, weights) <= legacy_sse * (1.0 + 1e-9)


@pytest.mark.parametrize(("kind", "grid"), _BANK_SPECS, ids=[spec[0] for spec in _BANK_SPECS])
def test_vector_detrend_matches_curve_fit_where_both_found_the_same_minimum(kind, grid) -> None:
    """Same minimum, same residual — to 1e-4 of the residual norm.

    Rows where curve_fit reached a *different* (strictly worse) minimum are
    excluded by the SSE agreement test rather than by tolerance: on this window
    the old fit stalls on most bank templates, and reproducing a stall is not the
    goal. Some row must actually be compared, or the test would be vacuous.
    """
    t, weights = _detrend_window()
    raw = np.asarray([em._template_values(kind, t, float(p)) for p in grid], dtype=float)

    new = em._monotonic_detrend_rows(t, raw, weights)

    compared = 0
    for row, template in zip(new, raw, strict=True):
        legacy, _lam, converged = _legacy_monotonic_detrend(t, template, weights)
        assert converged
        legacy_sse = _weighted_sse(legacy, weights)
        if abs(_weighted_sse(row, weights) - legacy_sse) > 1e-9 * legacy_sse:
            continue  # curve_fit landed elsewhere; the SSE test covers this row
        compared += 1
        assert np.linalg.norm(row - legacy) <= 1e-4 * np.linalg.norm(legacy)
    assert compared > 0


@pytest.mark.parametrize(("kind", "grid"), _BANK_SPECS, ids=[spec[0] for spec in _BANK_SPECS])
def test_vector_detrend_keeps_the_same_templates_past_the_variance_floor(kind, grid) -> None:
    """The bank's surviving rows are the set the curve_fit detrend produced.

    Which templates clear ``_TEMPLATE_VARIANCE_FLOOR`` decides what the matcher
    can score at all, so this is the property that keeps match/no-match verdicts
    identical — not the per-row residual.
    """
    t, weights = _detrend_window()
    raw = np.asarray([em._template_values(kind, t, float(p)) for p in grid], dtype=float)

    new_var = np.var(em._monotonic_detrend_rows(t, raw, weights), axis=1)
    legacy_var = np.array(
        [np.var(_legacy_monotonic_detrend(t, template, weights)[0]) for template in raw]
    )

    new_kept = new_var >= em._TEMPLATE_VARIANCE_FLOOR * new_var.max()
    legacy_kept = legacy_var >= em._TEMPLATE_VARIANCE_FLOOR * legacy_var.max()
    assert np.array_equal(new_kept, legacy_kept)
    assert new_kept.any()


@pytest.mark.parametrize(
    ("name", "signal_fn"),
    [
        ("plain_relaxation", lambda t: 20.0 * np.exp(-0.35 * t) + 1.0),
        ("kt_dip", lambda t: 18.0 * longitudinal_field_kubo_toyabe(t, 1.0, 0.45, 0.0, 0.0)),
        ("fmuf_beat", lambda t: 15.0 * linear_fmuf_polarization(t, 1.17) * np.exp(-0.2 * t)),
    ],
)
def test_vector_detrend_reproduces_curve_fit_on_signals(name, signal_fn) -> None:
    """On the signal side (m = 1) both solves reach the same minimum.

    Unlike the bank templates, a realistic signal seeds curve_fit well enough to
    converge globally, so here the residuals must agree outright.
    """
    t, weights = _detrend_window()
    rng = np.random.default_rng(2026)
    signal = signal_fn(t) + rng.normal(0.0, 0.05, t.size)

    new = em._monotonic_detrend_rows(t, signal[None, :], weights)[0]
    legacy, lam, converged = _legacy_monotonic_detrend(t, signal, weights)

    assert converged
    assert lam > 0.0  # curve_fit agrees the detrend is a decay, not a growth
    assert _weighted_sse(new, weights) <= _weighted_sse(legacy, weights) * (1.0 + 1e-9)
    assert np.linalg.norm(new - legacy) <= 1e-4 * np.linalg.norm(legacy)


def test_vector_detrend_lambda_stays_positive_and_spans_the_window() -> None:
    """The grid is strictly positive and brackets every rate the window resolves."""
    t, _weights = _detrend_window()
    grid = em._detrend_lambda_grid(t - t[0])

    assert grid.size == em._DETREND_LAMBDA_STEPS
    assert grid[0] > 0.0
    assert np.all(np.diff(grid) > 0.0)
    span = float(t[-1] - t[0])
    # Slow end: indistinguishable from the model's straight-line lambda -> 0 limit.
    assert grid[0] * span < 1e-4
    # Fast end: a 1/e time of a few sample intervals.
    assert grid[-1] > 1.0 / (10.0 * (t[1] - t[0]))


@pytest.mark.skipif(
    os.environ.get("PYTEST_XDIST_WORKER") is not None,
    reason="wall-clock ratio is not measurable while other workers share the box",
)
def test_vector_detrend_bank_build_is_much_faster_than_per_row_curve_fit() -> None:
    """The whole point: one vectorised pass instead of 146 nonlinear fits.

    A deliberately loose bound — the measured ratio on an idle box is ~9x on this
    2 000-point window and ~20x on a realistic ~18 000-point record, where the
    per-row fit's cost grows with the record and the shared grid sweep's does not.
    """
    t, weights = _detrend_window()
    raw = np.asarray(
        [em._template_values("kt_envelope", t, float(p)) for p in em._DELTA_GRID], dtype=float
    )

    start = time.perf_counter()
    em._monotonic_detrend_rows(t, raw, weights)
    vectorised = time.perf_counter() - start

    start = time.perf_counter()
    for template in raw:
        _legacy_monotonic_detrend(t, template, weights)
    per_row = time.perf_counter() - start

    assert per_row > 5.0 * vectorised
