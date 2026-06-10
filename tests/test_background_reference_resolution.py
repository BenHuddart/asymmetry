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
