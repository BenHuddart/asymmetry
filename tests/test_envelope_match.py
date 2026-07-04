"""Tests for the time-domain envelope matched filter (R2).

The banks recognise damped-envelope signatures (F-mu-F, mu-F, Kubo-Toyabe) that
FFT peak detection misses. Datasets use the percent scale and the realistic
exploding, capped error bars of dying-muon statistics (mirroring the R1 evidence
dataset in ``test_peak_detection``), so the significance null is exercised on the
same noise model as real data.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
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
