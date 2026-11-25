"""Resolve file-backed and array-backed GLE extents in displayed coordinates."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class GleBounds:
    """Collect each axis independently before rendering unsupported y2 fills."""

    log_x: bool = False
    log_y: frozenset[str] = frozenset()
    extents: dict[str, list[float]] = field(default_factory=dict, init=False)
    fills: list[tuple] = field(default_factory=list, init=False)

    def include(self, x, y, error=None, *, yaxis="y"):
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        valid = np.isfinite(x) & np.isfinite(y)
        if self.log_x:
            valid &= x > 0
        if yaxis in self.log_y:
            valid &= y > 0
        values = {"x": x[valid], yaxis: y[valid]}
        if error is not None:
            error = np.broadcast_to(np.asarray(error, dtype=float), y.shape)
            usable = valid & np.isfinite(error) & (error >= 0)
            values[yaxis] = np.concatenate(
                (y[valid], y[usable] - error[usable], y[usable] + error[usable])
            )
        for name, arr in values.items():
            if name in self.log_y or (name == "x" and self.log_x):
                arr = arr[arr > 0]
            if arr.size:
                extent = self.extents.setdefault(name, [float("inf"), float("-inf")])
                extent[0] = min(extent[0], float(np.min(arr)))
                extent[1] = max(extent[1], float(np.max(arr)))
        return bool(np.any(valid))

    def fill_between(self, x, lower, upper, *, yaxis="y", **style):
        lower_visible = self.include(x, lower, yaxis=yaxis)
        upper_visible = self.include(x, upper, yaxis=yaxis)
        if lower_visible or upper_visible:
            self.fills.append((x, lower, upper, yaxis, style))

    def apply(self, ax):
        """Retain each explicit endpoint; pad only automatically resolved bounds."""
        resolved = {}
        if "y" not in self.extents and any(axis == "y2" for _, _, _, axis, _ in self.fills):
            # With no finite primary values, its unused linear coordinate is
            # normalized; secondary fills still align with their own y2 edges.
            lo, hi = ax.get_ylim()
            if lo is None:
                lo = 0.0 if hi is None else hi - 1.0
            if hi is None:
                hi = lo + 1.0
            resolved["y"] = (lo, hi)
            ax.set_ylim(lo, hi)
        for name, (lo, hi) in self.extents.items():
            logarithmic = name in self.log_y or (name == "x" and self.log_x)
            if logarithmic:
                lo, hi = np.log10([lo, hi])
            singleton_pad = 0.05 if logarithmic else max(abs(lo) * 0.05, 0.05)
            pad = (hi - lo) * 0.05 if hi != lo else singleton_pad
            limits = (lo - pad, hi + pad)
            if logarithmic:
                limits = tuple(10 ** np.asarray(limits))
            explicit = ax.get_xlim() if name == "x" else ax.get_ylim(axis=name)
            resolved[name] = tuple(
                value if value is not None else auto
                for value, auto in zip(explicit, limits, strict=True)
            )
            if name == "x":
                ax.set_xlim(*resolved[name])
            else:
                ax.set_ylim(*resolved[name], axis=name)
        for x, lower, upper, yaxis, style in self.fills:
            if yaxis == "y2":
                # GLE fills use primary units; this affine map preserves the
                # secondary stack's position between its resolved linear limits.
                primary_lo, primary_hi = resolved["y"]
                secondary_lo, secondary_hi = resolved["y2"]
                scale = (primary_hi - primary_lo) / (secondary_hi - secondary_lo)
                lower = primary_lo + (np.asarray(lower) - secondary_lo) * scale
                upper = primary_lo + (np.asarray(upper) - secondary_lo) * scale
            ax.fill_between(x, lower, upper, **style)
