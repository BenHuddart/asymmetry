"""Core data containers for μSR measurements.

The data model mirrors the structure of a typical μSR experiment:

* **Histogram** — a single detector time histogram (counts vs time bin).
* **Run** — one measurement run containing multiple histograms plus metadata.
* **MuonDataset** — a processed view with time, asymmetry, and error arrays
  ready for plotting or fitting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class Histogram:
    r"""Raw detector histogram.

    Parameters
    ----------
    counts : NDArray
        Positron counts per time bin.
    bin_width : float
        Bin width in microseconds.
    t0_bin : int
        Bin index of *t* = 0 (muon implantation).
    good_bin_start : int
        First usable bin (offset from *t*\ :sub:`0`).
    good_bin_end : int
        Last usable bin.
    """

    counts: NDArray[np.float64]
    bin_width: float
    t0_bin: int = 0
    good_bin_start: int = 0
    good_bin_end: int = -1

    @property
    def n_bins(self) -> int:
        return len(self.counts)

    @property
    def time_axis(self) -> NDArray[np.float64]:
        r"""Time axis in microseconds, centred on *t*\ :sub:`0`."""
        bins = np.arange(self.n_bins)
        return (bins - self.t0_bin) * self.bin_width


@dataclass
class Run:
    """A single μSR measurement run.

    Holds raw histograms, grouping definitions, and run metadata.
    """

    run_number: int = 0
    histograms: list[Histogram] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    grouping: dict[str, Any] = field(default_factory=dict)
    source_file: str = ""

    # --- convenience accessors ------------------------------------------

    @property
    def title(self) -> str:
        return self.metadata.get("title", "")

    @property
    def temperature(self) -> float:
        """Sample temperature *setpoint* (``sample/temperature`` in NeXus)."""
        return float(self.metadata.get("temperature", 0.0))

    @property
    def sample_temperature_logged(self) -> float | None:
        """Representative *logged* sample temperature, if recorded.

        Sourced from the ``Temp_Sample`` NXlog (the actual measured sample
        temperature), as distinct from :attr:`temperature` (the setpoint).
        ``None`` when no logged series is present.
        """
        value = self.metadata.get("sample_temperature_logged")
        return None if value is None else float(value)

    @property
    def field(self) -> float:
        return float(self.metadata.get("field", 0.0))

    def summary(self) -> str:
        lines = [
            f"Run #{self.run_number}  —  {self.title}",
            f"  File        : {self.source_file}",
            f"  Temperature : {self.temperature} K",
            f"  Field       : {self.field} G",
            f"  Histograms  : {len(self.histograms)}",
        ]
        if self.histograms:
            h = self.histograms[0]
            lines.append(f"  Bins        : {h.n_bins}  (Δt = {h.bin_width:.6f} μs)")
        for key in ("comment", "started", "stopped"):
            if key in self.metadata:
                lines.append(f"  {key.title():12s}: {self.metadata[key]}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"Run(number={self.run_number}, title={self.title!r}, "
            f"T={self.temperature} K, B={self.field} G)"
        )


@dataclass
class MuonDataset:
    """Processed asymmetry dataset ready for plotting / fitting.

    This is the primary object that users interact with after loading and
    reducing a run.

    Scale convention
    ----------------
    ``asymmetry`` and ``error`` are stored on the **percent** scale (the
    WiMDA-style convention the loaders and the time-domain fit models share —
    a 16 % asymmetry is stored as ``16.0``, not ``0.16``, and the built-in
    models default ``A0`` to ``25``). The low-level
    :func:`asymmetry.core.transform.compute_asymmetry` primitive instead
    returns the dimensionless **fraction** :math:`A \\in [-1, 1]`; the loaders
    multiply by 100 to populate these arrays. When a specific scale is needed,
    prefer the explicit :attr:`asymmetry_percent` / :attr:`asymmetry_fraction`
    (and :attr:`error_percent` / :attr:`error_fraction`) accessors so the two
    can never be silently confused — seeding a fit with fraction-scale
    amplitudes against this percent-scale data converges to the wrong minimum.
    """

    time: NDArray[np.float64]
    asymmetry: NDArray[np.float64]
    error: NDArray[np.float64]
    metadata: dict[str, Any] = field(default_factory=dict)
    run: Run | None = None

    # Convenience -------------------------------------------------------

    @property
    def n_points(self) -> int:
        return len(self.time)

    # Scale accessors ---------------------------------------------------
    #
    # Explicit percent/fraction views over the stored ``asymmetry``/``error``
    # arrays (which are percent — see the class docstring). These exist so
    # callers state the scale they want rather than guessing, removing the
    # percent-vs-fraction trap that otherwise corrupts fit seeds and trends.

    @property
    def asymmetry_percent(self) -> NDArray[np.float64]:
        """Asymmetry on the percent scale (the stored convention)."""
        return np.asarray(self.asymmetry, dtype=np.float64)

    @property
    def asymmetry_fraction(self) -> NDArray[np.float64]:
        r"""Asymmetry as the dimensionless fraction :math:`A \in [-1, 1]` (percent / 100)."""
        return np.asarray(self.asymmetry, dtype=np.float64) / 100.0

    @property
    def error_percent(self) -> NDArray[np.float64]:
        """Asymmetry error on the percent scale (the stored convention)."""
        return np.asarray(self.error, dtype=np.float64)

    @property
    def error_fraction(self) -> NDArray[np.float64]:
        """Asymmetry error on the fractional scale (percent / 100)."""
        return np.asarray(self.error, dtype=np.float64) / 100.0

    @property
    def run_number(self) -> int:
        return self.run.run_number if self.run else self.metadata.get("run_number", 0)

    @property
    def temperature(self) -> float | None:
        """Sample temperature setpoint (``metadata["temperature"]``), or ``None``.

        Unlike :attr:`Run.temperature` (which floors a missing value to ``0.0``),
        this returns ``None`` when no temperature was recorded so a trend point
        with genuinely-missing metadata is marked off-axis rather than planted
        at 0 K.
        """
        value = self.metadata.get("temperature")
        return None if value is None else float(value)

    @property
    def field(self) -> float | None:
        """Applied field in gauss (``metadata["field"]``), or ``None`` when absent."""
        value = self.metadata.get("field")
        return None if value is None else float(value)

    @property
    def sample_temperature_logged(self) -> float | None:
        """Representative *logged* sample temperature, if recorded.

        Sourced from the ``Temp_Sample`` NXlog (the actual measured sample
        temperature), as distinct from the ``metadata['temperature']``
        setpoint. ``None`` when no logged series is present.
        """
        value = self.metadata.get("sample_temperature_logged")
        return None if value is None else float(value)

    @property
    def run_label(self) -> str:
        """Return a user-facing run label.

        For co-added datasets this prefers ``metadata['run_label']`` (for
        example ``"3039 + 3040"``) so UI text does not display internal
        negative combined IDs.
        """
        label = self.metadata.get("run_label")
        if label is not None:
            text = str(label).strip()
            if text:
                return text
        return str(self.run_number)

    def rebin(self, factor: int) -> MuonDataset:
        r"""Return a copy with every ``factor`` consecutive bins merged.

        Thin convenience wrapper over
        :func:`asymmetry.core.transform.rebin.rebin`: the time/asymmetry/error
        arrays are rebinned by that primitive and a *new* dataset is returned
        with the same metadata and run (this dataset is left unchanged).

        Combining ``factor`` adjacent bins trades time resolution for
        statistics — the per-point error shrinks as :math:`1/\sqrt{factor}` on
        flat data. ``factor = 1`` is a no-op copy; a length that is not a
        multiple of ``factor`` drops the trailing remainder bins; ``factor < 1``
        raises ``ValueError``.

        Useful on high-rate continuous-source data (e.g. PSI GPS 1.25 ns bins)
        where the raw sampling is far finer than the physics requires.
        """
        from asymmetry.core.transform.rebin import rebin

        time, asymmetry, error = rebin(self.time, self.asymmetry, self.error, factor)
        return MuonDataset(
            time=time,
            asymmetry=asymmetry,
            error=error,
            metadata=dict(self.metadata),
            run=self.run,
        )

    def time_range(self, t_min: float | None = None, t_max: float | None = None) -> MuonDataset:
        """Return a copy restricted to [t_min, t_max]."""
        mask = np.ones(self.n_points, dtype=bool)
        if t_min is not None:
            mask &= self.time >= t_min
        if t_max is not None:
            mask &= self.time <= t_max
        return MuonDataset(
            time=self.time[mask].copy(),
            asymmetry=self.asymmetry[mask].copy(),
            error=self.error[mask].copy(),
            metadata=dict(self.metadata),
            run=self.run,
        )

    def summary(self) -> str:
        lines = [f"MuonDataset  ({self.n_points} points)"]
        if self.run:
            lines.append(self.run.summary())
        if self.n_points:
            lines.append(f"  Time range  : {self.time[0]:.4f} – {self.time[-1]:.4f} μs")
            lines.append(f"  Asymmetry   : {self.asymmetry.min():.4f} – {self.asymmetry.max():.4f}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"MuonDataset(n={self.n_points}, run={self.run_number})"
