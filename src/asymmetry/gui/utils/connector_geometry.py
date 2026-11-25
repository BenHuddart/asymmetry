"""Source-index M4 geometry for data guides (Jugel et al., PVLDB 7, 2014).

https://www.vldb.org/pvldb/vol7/p797-jugel.pdf
Dashed, antialiased strokes do not share the paper's binary-raster exactness.
"""

from __future__ import annotations

import numpy as np


class ConnectorGeometry:
    """Eligible samples and uninterrupted source spans of one owned display trace."""

    def __init__(self, x: np.ndarray, y: np.ndarray, error: np.ndarray, mask: np.ndarray):
        self.x, self.y = x, y
        self.indices = np.flatnonzero(mask & np.isfinite(x) & np.isfinite(y) & np.isfinite(error))
        steps = np.diff(x)
        positive = steps[np.isfinite(steps) & (steps > 0)]
        cadence = float(np.median(positive)) if positive.size else np.inf
        gaps = np.diff(self.indices) != 1
        self.coordinates = x[self.indices]
        self.coordinates.flags.writeable = False
        dx = np.diff(self.coordinates)
        gaps |= (dx <= 0) | (dx > 1.5 * cadence)
        self.spans = np.r_[0, np.cumsum(gaps)] if self.indices.size else np.empty(0, dtype=int)
        self.ordered = bool(np.all(dx > 0))

    def select(self, window: tuple[float, float], columns: int | None) -> np.ndarray:
        """Original indices in source order, with one clipping neighbour per edge.

        Each (span, horizontal bucket) retains first/last/min/max, without
        interpolation. ``None`` columns requests full visible source geometry.
        """
        lo, hi = sorted(window)
        if not self.indices.size or hi <= lo:
            return np.empty(0, dtype=int)
        x = self.coordinates
        if self.ordered:
            first = max(0, int(np.searchsorted(x, lo)) - 1)
            last = min(x.size, int(np.searchsorted(x, hi, side="right")) + 1)
            positions = np.arange(first, last)
        else:
            inside = (x >= lo) & (x <= hi)
            adjacent = self.spans[1:] == self.spans[:-1]
            crossing = adjacent & (x[:-1] <= hi) & (x[1:] >= lo)
            inside[:-1] |= crossing
            inside[1:] |= crossing
            positions = np.flatnonzero(inside)
        if not positions.size:
            return np.empty(0, dtype=int)
        indices = self.indices[positions]
        if columns is None or indices.size <= 4 * columns:
            return indices
        spans = self.spans[positions]
        buckets = np.floor((self.x[indices] - lo) * columns / (hi - lo))
        starts = np.r_[0, np.flatnonzero((np.diff(buckets) != 0) | (np.diff(spans) != 0)) + 1]
        stops = np.r_[starts[1:], indices.size]
        values = self.y[indices]
        sizes = stops - starts
        minima = np.minimum.reduceat(values, starts)
        maxima = np.maximum.reduceat(values, starts)
        order = np.arange(indices.size)
        low = np.minimum.reduceat(
            np.where(values == np.repeat(minima, sizes), order, indices.size), starts
        )
        high = np.minimum.reduceat(
            np.where(values == np.repeat(maxima, sizes), order, indices.size), starts
        )
        return indices[np.unique(np.concatenate([starts, stops - 1, low, high]))]

    def vertices(self, window: tuple[float, float], columns: int | None) -> np.ndarray:
        """NaN-separated selected geometry; excluded source bins never get joined."""
        selected = self.select(window, columns)
        if not selected.size:
            return np.empty((0, 2))
        positions = np.searchsorted(self.indices, selected)
        gaps = np.r_[False, np.diff(self.spans[positions]) != 0]
        destination = np.arange(selected.size) + np.cumsum(gaps)
        vertices = np.full((selected.size + int(gaps.sum()), 2), np.nan)
        vertices[destination, 0] = self.x[selected]
        vertices[destination, 1] = self.y[selected]
        return vertices
