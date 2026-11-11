"""Detector-first correction of a declared raw-observation union."""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix

from asymmetry.core.data.dataset import Histogram
from asymmetry.core.transform.corrections import (
    CorrectionOutcome,
    CorrectionOutcomes,
    CorrectionStatus,
)
from asymmetry.core.transform.counting_noise import CountingNoise, Influence
from asymmetry.core.transform.deadtime import (
    DeadtimeDomain,
    apply_deadtime_correction,
    resolve_deadtime,
)
from asymmetry.core.transform.fb_counts import AlignmentCrop, BinWindow, GroupCounts


def observation_domain(
    parts: list[tuple[AlignmentCrop, list[int], list[BinWindow]]],
) -> DeadtimeDomain:
    """Map local k to raw k-offset and merge repeated observations exactly."""
    by_detector: dict[int, list[BinWindow]] = {}
    for alignment, indices, windows in parts:
        offsets = dict(alignment.detector_offsets)
        for index in indices:
            for window in windows:
                by_detector.setdefault(index, []).append(
                    BinWindow(window.first - offsets[index], window.last - offsets[index])
                )
    observations = []
    for index, windows in sorted(by_detector.items()):
        merged: list[BinWindow] = []
        for window in sorted(windows, key=lambda window: window.first):
            if merged and window.first <= merged[-1].last + 1:
                merged[-1] = BinWindow(merged[-1].first, max(merged[-1].last, window.last))
            else:
                merged.append(window)
        observations.append((index, tuple(merged)))
    return DeadtimeDomain(tuple(observations))


def require_independent_noise(histograms, domain, *, raw_sources=False):
    """Reject unknown noise only where a declared calculation consumes it."""
    for index, windows in domain.observations:
        hist = histograms[index]
        for window in windows:
            variance = hist.count_variance
            values = hist.counts[window.slice]
            unknown = (
                (
                    raw_sources
                    and not np.all(
                        np.isfinite(values) & (values >= 0) & (values == np.floor(values))
                    )
                )
                if variance is None
                else np.any(np.isnan(variance[window.slice]))
            )
            if unknown:
                raise ValueError(
                    f"Detector {index + 1} requires supplied variance for consumed bins with unknown independent count noise"
                )


@dataclass(frozen=True)
class ScopedCounts:
    """Raw source plus corrected samples, never a prepared acquisition.

    Values outside the domain remain raw context for existing estimators/plots.
    The caller declares every estimator dependency before constructing this value.
    """

    histograms: list[Histogram]
    domain: DeadtimeDomain
    outcome: CorrectionOutcome
    corrected: dict[int, tuple[np.ndarray, np.ndarray]]
    source_digest: str

    @classmethod
    def prepare(
        cls, histograms, grouping, domain, *, use_deadtime, mode=None, allow_incomplete=False
    ):
        """Resolve once, then invert each required detector/bin once."""
        require_independent_noise(histograms, domain)
        outcome = resolve_deadtime(grouping, len(histograms), use_deadtime, mode, domain=domain)
        if not allow_incomplete:
            CorrectionOutcomes(deadtime=outcome).require_complete()
        if outcome.warning is not None:
            warnings.warn(outcome.warning, UserWarning, stacklevel=2)
        corrected = {}
        if outcome.status is CorrectionStatus.APPLIED:
            for index, windows in domain.observations:
                coefficient = outcome.effective["tau_us"][index]
                if coefficient == 0:
                    continue
                histogram = histograms[index]
                bins = np.concatenate(
                    [np.arange(w.first, w.last + 1, dtype=np.intp) for w in windows]
                )
                values = apply_deadtime_correction(
                    histogram.counts[bins],
                    coefficient,
                    histogram.bin_width,
                    num_good_frames=outcome.effective["good_frames"],
                    detector=index,
                    raw_bin_indices=bins,
                )
                bins.flags.writeable = values.flags.writeable = False
                corrected[index] = bins, values
        return cls(histograms, domain, outcome, corrected, scoped_source_digest(histograms, domain))

    def require_source(self):
        """Reject reuse after required acquired values or supplied variance change."""
        if self.source_digest != scoped_source_digest(self.histograms, self.domain):
            raise ValueError(
                "Prepared grouped counts do not match the requested source observations"
            )

    def aligned(
        self, indices: list[int], alignment: AlignmentCrop, *, retain_variance: bool = False
    ) -> GroupCounts:
        """Project required samples and noise; preserve excluded values as raw context.

        Outside-domain variance is an excluded statistical contribution, never a
        measured source variance or an eligible observation. Original histograms
        retain their full counts and variance; only declared consumers use this value.
        """
        total = np.zeros(alignment.length, dtype=np.float64)
        offsets = dict(alignment.detector_offsets)
        for index in indices:
            start = -offsets[index]
            values = np.asarray(
                self.histograms[index].counts[start : start + alignment.length], dtype=np.float64
            ).copy()
            if index in self.corrected:
                bins, corrected = self.corrected[index]
                local = bins + offsets[index]
                inside = (local >= 0) & (local < alignment.length)
                values[local[inside]] = corrected[inside]
            total += values
        required = dict(self.domain.observations)
        complete = all(
            any(
                window.first <= -offsets[i] and window.last >= -offsets[i] + alignment.length - 1
                for window in required.get(i, ())
            )
            for i in indices
        )
        variance = None
        if (
            (retain_variance and not complete)
            or self.corrected
            or any(
                self.histograms[i].count_variance is not None for i, _ in alignment.detector_offsets
            )
        ):
            variance = np.zeros(alignment.length, dtype=np.float64)
            for index in indices:
                hist = self.histograms[index]
                values = hist.counts if hist.count_variance is None else hist.count_variance
                start = -offsets[index]
                for window in required.get(index, ()):
                    first = max(start, window.first)
                    stop = min(start + alignment.length, window.last + 1)
                    if first < stop:
                        variance[first - start : stop - start] += values[first:stop]
        terms = []
        for index in indices:
            hist = self.histograms[index]
            if hist.counting_observation is not None:
                start = -offsets[index]
                local = np.arange(alignment.length)
                raw = local + start
                consumed = np.zeros(alignment.length, dtype=bool)
                for window in required.get(index, ()):
                    consumed |= (raw >= window.first) & (raw <= window.last)
                direct = csr_matrix(
                    (np.ones(np.sum(consumed)), (local[consumed], raw[consumed])),
                    shape=(alignment.length, hist.n_bins),
                )
                terms.append(Influence(hist.counting_observation, direct))
        noise = CountingNoise(tuple(terms)) if terms else None
        if noise is not None and variance is None:
            variance = total.copy()
        return GroupCounts(total, variance, noise=noise)


def scoped_source_digest(histograms, domain):
    """Fingerprint only consumed raw values and their supplied variance."""
    digest = hashlib.sha256()
    for index, windows in domain.observations:
        histogram = histograms[index]
        for window in windows:
            for values in (histogram.counts, histogram.count_variance):
                if values is not None:
                    digest.update(np.asarray(values[window.slice], dtype="<f8").tobytes())
    return digest.hexdigest()
