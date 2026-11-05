"""Core-side ``reference_run`` resolution.

The resolver previously lived only in ``MainWindow``, so scripted/core
reductions and the grouped Fourier path silently skipped ``reference_run``
backgrounds. These tests pin the moved-into-core behaviour: registry match,
frame-scale arithmetic, caching, failure messages, and the Fourier path
actually subtracting a resolved reference.
"""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.io import (
    BackgroundReference,
    resolve_background_reference,
)


def _run(run_number: int, *, good_frames: float, level: float) -> Run:
    histograms = [
        Histogram(counts=np.full(64, float(level)), bin_width=0.016, t0_bin=8) for _ in range(4)
    ]
    return Run(
        run_number=run_number,
        histograms=histograms,
        metadata={"run_number": run_number},
        grouping={"good_frames": good_frames},
    )


def _dataset(run: Run) -> MuonDataset:
    return MuonDataset(
        time=np.array([]),
        asymmetry=np.array([]),
        error=np.array([]),
        metadata={},
        run=run,
    )


def test_resolve_matches_loaded_dataset_by_run_number():
    reference = _run(8103, good_frames=1000.0, level=5.0)
    registry = [_dataset(_run(1, good_frames=1.0, level=1.0)), _dataset(reference)]
    result = resolve_background_reference(
        {"run_number": 8103, "source_file": ""},
        sample_good_frames=2000.0,
        datasets=registry,
    )
    assert isinstance(result, BackgroundReference)
    assert result.run_number == 8103
    # Scale is the good-frame ratio sample/reference (the WiMDA exposure scale).
    assert result.scale == pytest.approx(2.0)
    assert result.histograms is reference.histograms or len(result.histograms) == 4


def test_resolve_falls_back_to_payload_frames_then_scale():
    reference = _run(42, good_frames=0.0, level=3.0)  # no good_frames on the run
    registry = [_dataset(reference)]
    # sample frames unknown -> payload good_frames_* used.
    result = resolve_background_reference(
        {"run_number": 42, "good_frames_sample": 500.0, "good_frames_reference": 250.0},
        sample_good_frames=None,
        datasets=registry,
    )
    assert result.scale == pytest.approx(2.0)
    # When no frames at all are resolvable, the payload's scale snapshot wins.
    result2 = resolve_background_reference(
        {"run_number": 42, "scale": 1.7},
        sample_good_frames=None,
        datasets=registry,
    )
    assert result2.scale == pytest.approx(1.7)


def test_resolve_cache_reused_across_calls(monkeypatch):
    """Loader fallback loads once per source path when a cache is supplied."""
    reference = _run(7, good_frames=100.0, level=2.0)
    calls = {"n": 0}

    def _fake_load(payload):
        calls["n"] += 1
        return _dataset(reference)

    monkeypatch.setattr("asymmetry.core.io.load_background_run", _fake_load)
    cache: dict[str, object] = {}
    payload = {"run_number": 7, "source_file": "/some/ref.nxs"}
    first = resolve_background_reference(payload, sample_good_frames=200.0, cache=cache)
    second = resolve_background_reference(payload, sample_good_frames=200.0, cache=cache)
    assert calls["n"] == 1  # second call served from cache
    assert first.scale == pytest.approx(2.0)
    assert second.scale == pytest.approx(2.0)


@pytest.mark.parametrize(
    "payload, match",
    [
        (None, "no reference is recorded"),
        ({"run_number": 999}, "not loaded and no source file"),
    ],
)
def test_resolve_raises_human_readable(payload, match):
    with pytest.raises(ValueError, match=match):
        resolve_background_reference(payload, datasets=[])


def test_resolve_does_not_mix_live_reference_with_stale_sample_snapshot():
    """When the sample's live good_frames is unknown, the scale must come from
    the payload snapshot ratio (same epoch), not a live reference over a stale
    sample snapshot."""
    reference = _run(42, good_frames=1000.0, level=3.0)  # live reference frames
    registry = [_dataset(reference)]
    result = resolve_background_reference(
        {
            "run_number": 42,
            "good_frames_sample": 800.0,
            "good_frames_reference": 400.0,  # snapshot ratio 2.0
        },
        sample_good_frames=None,  # sample live frames unknown
        datasets=registry,
    )
    # Snapshot ratio 800/400 = 2.0 — NOT 800/1000 (mixed) or any live/stale mix.
    assert result.scale == pytest.approx(2.0)
    assert result.good_frames_sample == 800.0
    assert result.good_frames_reference == 400.0
    assert result.scale_source == "saved_frames"


def test_resolve_cache_short_circuits_registry_scan(monkeypatch):
    """A cached reference (per source path) is reused without scanning the
    registry or re-loading, even when a run_number is present."""
    reference = _run(7, good_frames=100.0, level=2.0)
    cache: dict[str, object] = {"/some/ref.nxs": reference}

    def _fail_load(payload):
        raise AssertionError("loader must not run on a cache hit")

    monkeypatch.setattr("asymmetry.core.io.load_background_run", _fail_load)

    class _ExplodingRegistry:
        def __iter__(self):
            raise AssertionError("registry must not be scanned on a cache hit")

    result = resolve_background_reference(
        {"run_number": 7, "source_file": "/some/ref.nxs"},
        sample_good_frames=200.0,
        datasets=_ExplodingRegistry(),
        cache=cache,
    )
    assert result.scale == pytest.approx(2.0)
    assert result.source_file == "ref.nxs"


@pytest.mark.parametrize("reference_length", [40, 80])
def test_fourier_reference_must_cover_the_sample(monkeypatch, reference_length):
    """A longer reference is cropped; a missing reference tail is rejected."""
    from asymmetry.core.fourier.grouped import build_group_signal_dataset

    sample = _run(100, good_frames=1000.0, level=10.0)
    sample.histograms = [
        Histogram(counts=np.full(64, 10.0), bin_width=0.016, t0_bin=8) for _ in range(4)
    ]
    short_reference = [
        Histogram(counts=np.full(reference_length, 4.0), bin_width=0.016, t0_bin=8)
        for _ in range(4)
    ]
    grouping = {
        "groups": {1: [1, 2], 2: [3, 4]},
        "forward_group": 1,
        "backward_group": 2,
        "alpha": 1.0,
        "t0_bin": 8,
        "first_good_bin": 8,
        "background_correction": True,
        "background_mode": "reference_run",
        "background_run": {"run_number": 200, "source_file": "/ref.nxs"},
        "good_frames": 1000.0,
    }
    sample.grouping = grouping
    monkeypatch.setattr(
        "asymmetry.core.io.resolve_background_reference",
        lambda *a, **k: BackgroundReference(histograms=short_reference, scale=1.0),
    )

    no_bkg = dict(grouping)
    no_bkg["background_correction"] = False
    sample.grouping = no_bkg
    baseline = build_group_signal_dataset(
        sample, 1, center_signal=False, apply_lifetime_correction=False
    )
    sample.grouping = grouping
    if reference_length < 64:
        with pytest.raises(
            ValueError, match="reference run has 40 aligned bins; sample requires 64"
        ):
            build_group_signal_dataset(
                sample, 1, center_signal=False, apply_lifetime_correction=False
            )
    else:
        corrected = build_group_signal_dataset(
            sample, 1, center_signal=False, apply_lifetime_correction=False
        )
        assert corrected.asymmetry.size == baseline.asymmetry.size
        np.testing.assert_allclose(corrected.asymmetry, 12.0)


def test_fourier_grouped_path_satisfies_reference_run(monkeypatch):
    """The grouped Fourier input subtracts a resolved reference (previously a
    silent no-op outside the GUI)."""
    from asymmetry.core.fourier.grouped import build_group_signal_dataset

    sample = _run(100, good_frames=1000.0, level=10.0)
    reference = _run(200, good_frames=1000.0, level=4.0)
    grouping = {
        "groups": {1: [1, 2], 2: [3, 4]},
        "forward_group": 1,
        "backward_group": 2,
        "alpha": 1.0,
        "t0_bin": 8,
        "first_good_bin": 8,
        "background_correction": True,
        "background_mode": "reference_run",
        "background_run": {"run_number": 200, "source_file": "/ref.nxs"},
        "good_frames": 1000.0,
    }
    sample.grouping = grouping
    sample.histograms = [
        Histogram(counts=np.full(64, 10.0), bin_width=0.016, t0_bin=8) for _ in range(4)
    ]

    monkeypatch.setattr(
        "asymmetry.core.io.resolve_background_reference",
        lambda *a, **k: BackgroundReference(histograms=reference.histograms, scale=1.0),
    )

    # With lifetime correction and centring off, the raw grouped signal is the
    # summed counts; subtracting scale·reference shifts each group's level.
    no_bkg_grouping = dict(grouping)
    no_bkg_grouping["background_correction"] = False
    sample.grouping = no_bkg_grouping
    baseline = build_group_signal_dataset(
        sample, 1, center_signal=False, apply_lifetime_correction=False
    )
    sample.grouping = grouping
    corrected = build_group_signal_dataset(
        sample, 1, center_signal=False, apply_lifetime_correction=False
    )
    # group 1 = two detectors at 10 each = 20; reference two detectors at 4 = 8.
    n = min(baseline.asymmetry.size, corrected.asymmetry.size)
    np.testing.assert_allclose(
        baseline.asymmetry[:n] - corrected.asymmetry[:n],
        np.full(n, 8.0),
        atol=1e-9,
    )


@pytest.mark.parametrize("scale", [0, -1, float("nan"), "invalid"])
def test_invalid_reference_scale_does_not_become_unity(scale):
    reference = _run(42, good_frames=1000.0, level=3.0)
    with pytest.raises(ValueError, match="scale must be finite and positive"):
        resolve_background_reference(
            {"run_number": 42, "scale": scale},
            sample_good_frames=None,
            datasets=[_dataset(reference)],
        )


def test_fourier_reference_deadtime_uses_reference_frames(monkeypatch):
    from asymmetry.core.fourier.grouped import _resolve_group_reference_counts
    from asymmetry.core.transform.grouping import resolve_alignment

    reference = BackgroundReference(
        [Histogram(np.array([20 / 1.4]), 1.0)],
        2.0,
        good_frames_sample=10.0,
        good_frames_reference=5.0,
    )
    monkeypatch.setattr("asymmetry.core.io.resolve_background_reference", lambda *a, **k: reference)
    counts, scale = _resolve_group_reference_counts(
        {
            "background_correction": True,
            "background_mode": "reference_run",
            "deadtime_mode": "manual",
            "dead_time_us": [0.1],
            "good_frames": 10.0,
        },
        group_indices=[0],
        common_t0=0,
        apply_deadtime=True,
        detector_t0_bins=[0],
        sample_alignment=resolve_alignment(reference.histograms, [0]),
    )
    np.testing.assert_allclose(counts.counts, [20.0], rtol=1e-12)
    np.testing.assert_allclose(counts.variance, [20 / 1.4], rtol=1e-12)
    assert scale == 2.0


def test_reference_file_coefficients_are_acquired_from_the_reference():
    from asymmetry.core.data.dataset import MuonDataset, Run
    from asymmetry.core.io import resolve_background_reference
    from asymmetry.core.transform.reduce import corrected_grouped_counts

    reference_histograms = [Histogram(np.array([20.0]), 1.0) for _ in range(2)]
    reference_run = Run(
        run_number=42,
        histograms=reference_histograms,
        grouping={
            "deadtime_file_us": [0.2, 0.2],
            "dead_time_us": [0.9, 0.9],
            "deadtime_method": "manual",
            "good_frames": 100.0,
        },
    )
    dataset = MuonDataset(np.array([]), np.array([]), np.array([]), run=reference_run)
    reference = resolve_background_reference(
        {"run_number": 42, "scale": 1.0}, sample_good_frames=100.0, datasets=[dataset]
    )
    assert reference.file_dead_time_us == (0.2, 0.2)
    actual = corrected_grouped_counts(
        histograms=[Histogram(np.array([50.0]), 1.0) for _ in range(2)],
        grouping={
            "dead_time_us": [0.1, 0.1],
            "good_frames": 100.0,
            "background_mode": "reference_run",
        },
        forward_idx=[0],
        backward_idx=[1],
        use_deadtime=True,
        deadtime_mode="file",
        use_background=True,
        reference_resolver=lambda _: reference,
    )
    np.testing.assert_allclose(actual.forward.counts, [50 / (1 - 0.05) - 20 / (1 - 0.04)])
    assert actual.corrections.deadtime.status == "applied"


@pytest.mark.parametrize("mode", ["off", "manual", "file"])
@pytest.mark.parametrize("payload", [None, {}, {"source_file": "invalid\x00path", "scale": -1}])
def test_fourier_digest_does_not_validate_broken_reference_requests(mode, payload):
    from asymmetry.core.fourier.spectrum import fourier_grouping_digest

    run = _run(1, good_frames=1000.0, level=20.0)
    run.grouping.update(
        deadtime_correction=mode != "off",
        deadtime_mode=mode,
        dead_time_us=[0.001] * 4,
        background_correction=True,
        background_mode="reference_run",
        background_run=payload,
    )
    assert fourier_grouping_digest(run) == fourier_grouping_digest(run)


@pytest.mark.parametrize("mode", ["off", "manual", "estimate"])
def test_fourier_digest_skips_external_file_metadata_when_file_mode_is_inert(mode, monkeypatch):
    from asymmetry.core.fourier.spectrum import fourier_grouping_digest

    monkeypatch.setattr(
        "asymmetry.core.io.reference_file_identity", lambda _: pytest.fail("inert reference lookup")
    )
    run = _run(1, good_frames=1000.0, level=20.0)
    run.grouping.update(
        deadtime_correction=mode != "off",
        deadtime_mode=mode,
        dead_time_us=[0.001] * 4,
        background_correction=True,
        background_mode="reference_run",
        background_run={"source_file": "unused.nxs"},
    )
    assert fourier_grouping_digest(run)
