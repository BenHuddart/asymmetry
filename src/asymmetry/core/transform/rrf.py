"""Rotating-reference-frame (RRF) demodulation of the FB asymmetry.

A transverse-field precession signal A(t) = a(t)·cos(2πνt + φ_d) collapses to
its slow envelope when multiplied by the complex carrier 2·e^{−i(2πν₀t+φ)}:
the product contains the rotating-frame signal a(t)·e^{i(2π(ν−ν₀)t+φ_d−φ)}
plus an image at ν + ν₀, which a low-pass filter removes.  The real part is
the in-phase component, the imaginary part the quadrature, and the magnitude
the phase-free envelope (Blundell, De Renzi, Lancaster & Pratt, *Muon
Spectroscopy*, OUP 2022, the rotating-reference-frame section of the
time-domain-analysis chapter; T. M. Riseman and J. H. Brewer, Hyperfine
Interact. 65, 1107 (1991)).

Two methods are provided:

``"fir"``
    Complex demodulation followed by a zero-phase windowed-sinc FIR low-pass
    (Blackman window).  The image is suppressed by the filter's stopband
    (≤ −74 dB) rather than by hoping a smoothing null lands on it.  This is
    the display method.

``"wimda"``
    WiMDA's scheme, reproduced bin-for-bin for comparison: multiply by
    2·cos(2πν₀t + φ) and box-average over a time window (default one image
    period), zeroing the output within half a window of either edge and
    *linearly* averaging the scaled errors — including WiMDA's statistically
    loose error treatment (``$WIMDA_SRC/src/Plot.pas``, ``plotdata``).

The demodulated curve is a **visualization**, not fit input: the low-pass
correlates neighbouring bins over the filter support (and the textbook warns
the filtering distorts lineshapes), so χ² against these points with their
per-point errors would be wrong.  Quantitative work fits the raw data — see
:mod:`asymmetry.core.fitting.rrf_offset` for fitting in the rotating frame
with exact statistics.

This module is Qt-free and operates on plain arrays; datasets are never
mutated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "RRFCurve",
    "default_bandwidth_mhz",
    "folded_image_frequency_mhz",
    "rrf_demodulate",
    "rrf_demodulate_values",
]

#: Minimum fraction of the filter-kernel mass that must fall on finite input
#: bins for an output bin to be flagged valid.
_MIN_KERNEL_COVERAGE = 0.5

_COMPONENTS = ("real", "imag", "magnitude")


@dataclass(frozen=True)
class RRFCurve:
    """A demodulated rotating-frame curve with per-point errors.

    ``real``/``imag`` are the in-phase and quadrature components with
    per-point standard deviations propagated exactly through the demodulation
    and the filter.  Neighbouring bins are **correlated** over roughly
    ``filter_taps`` bins; ``effective_independent_fraction`` (Σh²ₖ of the
    DC-normalised kernel) is the approximate fraction of output points that
    carry independent information.  Real/imaginary covariance from the shared
    input error is not reported.
    """

    time: NDArray[np.float64]
    real: NDArray[np.float64]
    real_error: NDArray[np.float64]
    imag: NDArray[np.float64]
    imag_error: NDArray[np.float64]
    #: False inside the filter edge region or where the kernel saw mostly
    #: non-finite input.
    valid: NDArray[np.bool_]
    frequency_mhz: float
    phase_deg: float
    method: str
    #: FIR cutoff (``"fir"``) in MHz; ``None`` for the WiMDA mode, whose
    #: width parameter is ``box width`` in µs (stored in ``filter_taps``).
    bandwidth_mhz: float | None
    filter_taps: int
    effective_independent_fraction: float

    @property
    def magnitude(self) -> NDArray[np.float64]:
        """The phase-free envelope |z|; Rician-biased where |z| ≲ σ."""
        return np.hypot(self.real, self.imag)

    @property
    def magnitude_error(self) -> NDArray[np.float64]:
        """First-order error on |z|; unreliable where |z| ≲ σ (Rician bias)."""
        mag = self.magnitude
        floor = np.finfo(float).tiny
        safe = np.where(mag > floor, mag, 1.0)
        err = (
            np.sqrt(
                np.square(self.real * self.real_error) + np.square(self.imag * self.imag_error)
            )
            / safe
        )
        # Where the magnitude vanishes the first-order expansion collapses;
        # fall back to the quadrature mean so the bar is not spuriously zero.
        fallback = np.sqrt(0.5 * (np.square(self.real_error) + np.square(self.imag_error)))
        return np.where(mag > floor, err, fallback)

    def component(self, name: str) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return ``(values, errors)`` for ``"real"``, ``"imag"`` or ``"magnitude"``."""
        key = str(name).strip().lower()
        if key == "real":
            return self.real, self.real_error
        if key == "imag":
            return self.imag, self.imag_error
        if key == "magnitude":
            return self.magnitude, self.magnitude_error
        raise ValueError(f"Unknown RRF component {name!r}; expected one of {_COMPONENTS}.")

    def frame_label(self, component: str = "real") -> str:
        """A short self-describing frame annotation for plots and exports."""
        parts = [f"frame: ν₀ = {self.frequency_mhz:g} MHz"]
        if self.phase_deg:
            parts.append(f"φ = {self.phase_deg:g}°")
        key = str(component).strip().lower()
        if key != "real":
            parts.append({"imag": "quadrature", "magnitude": "magnitude"}.get(key, key))
        return ", ".join(parts)


def default_bandwidth_mhz(
    frequency_mhz: float,
    sample_rate_mhz: float | None = None,
) -> float:
    """Default single-sided FIR cutoff between the envelope and the image.

    The image sits at ν + ν₀ ≈ 2ν₀ for ν₀ tuned near the line, so ν₀/2 leaves
    generous room for the envelope and any deliberate detuning beat.  On a
    sampled signal the image *folds*: for 2ν₀ beyond Nyquist it aliases to
    \\|2ν₀ − k·fs\\| and can land back inside the baseband — common for
    high-TF data binned close to Nyquist.  Given the sample rate, the default
    therefore stays below 70% of the folded image frequency as well.  When
    the image folds onto DC (2ν₀ ≈ k·fs) no filter can separate it from the
    envelope; the default falls back to ν₀/2 and the curve will show the
    contamination — rebin differently or move ν₀.
    """
    freq = float(frequency_mhz)
    if not np.isfinite(freq) or freq <= 0.0:
        raise ValueError(f"frequency_mhz must be positive and finite, got {frequency_mhz!r}.")
    bandwidth = 0.5 * freq
    if sample_rate_mhz is not None:
        folded = folded_image_frequency_mhz(freq, sample_rate_mhz)
        if folded > 0.0:
            bandwidth = min(bandwidth, 0.7 * folded)
    return bandwidth


def folded_image_frequency_mhz(frequency_mhz: float, sample_rate_mhz: float) -> float:
    """Where the 2ν₀ demodulation image lands after folding about Nyquist."""
    fs = float(sample_rate_mhz)
    if not np.isfinite(fs) or fs <= 0.0:
        raise ValueError(f"sample_rate_mhz must be positive and finite, got {sample_rate_mhz!r}.")
    image = 2.0 * float(frequency_mhz)
    return float(abs(image - fs * np.round(image / fs)))


def _uniform_bin_width_us(time: NDArray[np.float64]) -> float:
    """Return the (assumed uniform) bin width, failing fast on a bad grid."""
    if time.size < 3:
        raise ValueError("RRF demodulation needs at least 3 time bins.")
    steps = np.diff(time)
    dt = float(np.median(steps))
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("Time axis must be increasing with a positive bin width.")
    return dt


def _fir_kernel(dt_us: float, bandwidth_mhz: float, n_bins: int) -> NDArray[np.float64]:
    """Design the DC-normalised Blackman windowed-sinc low-pass kernel.

    Blackman buys a −74 dB stopband (amplitude 2×10⁻⁴ — invisible against
    any real envelope) at the cost of a wider transition; the tap count
    follows the Blackman transition-width relation Δf ≈ 5.5·fs/N with the
    transition width set to half the cutoff, clamped to the data length
    (a clamped kernel keeps DC gain 1 but a degraded stopband — the valid
    range shrinks accordingly either way).  A cutoff at or above Nyquist
    degenerates to the identity kernel.
    """
    fs = 1.0 / dt_us  # MHz
    nyquist = 0.5 * fs
    if bandwidth_mhz >= nyquist:
        return np.ones(1)
    transition = 0.5 * bandwidth_mhz
    taps = int(np.ceil(5.5 * fs / transition))
    taps = min(taps, max(3, n_bins - (1 - n_bins % 2)))
    if taps % 2 == 0:
        taps += 1
    if taps < 3:
        return np.ones(1)
    from scipy.signal import firwin

    kernel = firwin(taps, bandwidth_mhz, window="blackman", fs=fs)
    return np.asarray(kernel, dtype=np.float64)


def _normalised_convolution(
    values: NDArray,
    variances: NDArray[np.float64],
    finite: NDArray[np.bool_],
    kernel: NDArray[np.float64],
) -> tuple[NDArray, NDArray[np.float64], NDArray[np.float64]]:
    """Convolve with NaN-aware renormalisation.

    Returns the filtered values, filtered variances, and the kernel coverage
    (fraction of kernel mass that landed on finite bins; 1 in the interior of
    a fully finite signal because the kernel is DC-normalised).
    """
    if kernel.size == 1:
        coverage = finite.astype(np.float64)
        return (
            np.where(finite, values, 0.0),
            np.where(finite, variances, 0.0),
            coverage,
        )
    from scipy.signal import fftconvolve

    weights = finite.astype(np.float64)
    coverage = fftconvolve(weights, kernel, mode="same")
    coverage = np.clip(coverage, 0.0, None)
    safe = np.where(coverage > 1e-12, coverage, 1.0)
    if np.iscomplexobj(values):
        num = fftconvolve(np.where(finite, values.real, 0.0), kernel, mode="same") + (
            1j * fftconvolve(np.where(finite, values.imag, 0.0), kernel, mode="same")
        )
    else:
        num = fftconvolve(np.where(finite, values, 0.0), kernel, mode="same")
    var = fftconvolve(np.where(finite, variances, 0.0), np.square(kernel), mode="same")
    var = np.clip(var, 0.0, None)
    return num / safe, var / np.square(safe), coverage


def rrf_demodulate(
    time: ArrayLike,
    asymmetry: ArrayLike,
    error: ArrayLike,
    *,
    frequency_mhz: float,
    phase_deg: float = 0.0,
    bandwidth_mhz: float | None = None,
    method: str = "fir",
    wimda_box_width_us: float | None = None,
) -> RRFCurve:
    """Demodulate an asymmetry curve into the rotating frame at ν₀.

    Parameters
    ----------
    time, asymmetry, error : array-like
        The curve in µs with per-bin standard deviations.  Non-finite bins
        are excluded from the filter without poisoning their neighbourhood.
        Bins within one filter support of a non-finite hole keep a bounded
        bias (up to ~missing-bins/taps × twice the local amplitude): the
        kernel renormalisation restores the baseband mean but the hole
        impairs the image cancellation locally.
    frequency_mhz : float
        Frame frequency ν₀ (positive).  Gauss-entered values are converted by
        the caller through :mod:`asymmetry.core.fourier.units` — this module
        speaks MHz only.
    phase_deg : float
        Frame phase φ in degrees; the carrier is e^{−i(2πν₀t + φ)}.
    bandwidth_mhz : float, optional
        Single-sided FIR cutoff; defaults to :func:`default_bandwidth_mhz`.
        Ignored by the WiMDA mode.
    method : str
        ``"fir"`` (complex demodulation, the display method) or ``"wimda"``
        (2·cos + box average, comparison only; see the module docstring).
    wimda_box_width_us : float, optional
        WiMDA-mode box width in µs; defaults to one image period 1/(2ν₀),
        WiMDA's own default when ν₀ sits at the applied field.

    Returns
    -------
    RRFCurve
        Complex curve with exact per-point error propagation and a validity
        mask covering filter edges and non-finite holes.  See the class note
        on inter-bin correlation.
    """
    t = np.asarray(time, dtype=np.float64)
    a = np.asarray(asymmetry, dtype=np.float64)
    e = np.asarray(error, dtype=np.float64)
    if not (t.shape == a.shape == e.shape):
        raise ValueError("time, asymmetry and error must have matching shapes.")
    freq = float(frequency_mhz)
    if not np.isfinite(freq) or freq <= 0.0:
        raise ValueError(f"frequency_mhz must be positive and finite, got {frequency_mhz!r}.")
    phase = float(phase_deg)
    dt = _uniform_bin_width_us(t)

    method_key = str(method).strip().lower()
    if method_key == "fir":
        return _demodulate_fir(t, a, e, dt, freq, phase, bandwidth_mhz)
    if method_key == "wimda":
        return _demodulate_wimda(t, a, e, dt, freq, phase, wimda_box_width_us)
    raise ValueError(f"Unknown RRF method {method!r}; expected 'fir' or 'wimda'.")


def rrf_demodulate_values(
    time: ArrayLike,
    values: ArrayLike,
    *,
    frequency_mhz: float,
    phase_deg: float = 0.0,
    bandwidth_mhz: float | None = None,
    method: str = "fir",
    wimda_box_width_us: float | None = None,
) -> RRFCurve:
    """Demodulate an error-free curve (e.g. a stored fit-curve overlay).

    The model curve drawn over RRF-displayed data must pass through the same
    pipeline as the data so the two stay in step — the structural fix for
    WiMDA's display-only frame shift (see the study's comparison ledger).
    """
    values_arr = np.asarray(values, dtype=np.float64)
    return rrf_demodulate(
        time,
        values_arr,
        np.zeros_like(values_arr),
        frequency_mhz=frequency_mhz,
        phase_deg=phase_deg,
        bandwidth_mhz=bandwidth_mhz,
        method=method,
        wimda_box_width_us=wimda_box_width_us,
    )


def _demodulate_fir(
    t: NDArray[np.float64],
    a: NDArray[np.float64],
    e: NDArray[np.float64],
    dt: float,
    freq: float,
    phase_deg: float,
    bandwidth_mhz: float | None,
) -> RRFCurve:
    bandwidth = (
        default_bandwidth_mhz(freq, sample_rate_mhz=1.0 / dt)
        if bandwidth_mhz is None
        else float(bandwidth_mhz)
    )
    if not np.isfinite(bandwidth) or bandwidth <= 0.0:
        raise ValueError(f"bandwidth_mhz must be positive and finite, got {bandwidth_mhz!r}.")

    theta = 2.0 * np.pi * freq * t + np.deg2rad(phase_deg)
    finite = np.isfinite(a) & np.isfinite(e) & np.isfinite(t)
    carrier = np.exp(-1j * theta)
    z = 2.0 * a * carrier
    # Per-quadrature input variances: Re scales with 2cosθ, Im with 2sinθ.
    var_re = np.square(2.0 * e * np.cos(theta))
    var_im = np.square(2.0 * e * np.sin(theta))

    kernel = _fir_kernel(dt, bandwidth, t.size)
    z_f, var_re_f, coverage = _normalised_convolution(z, var_re, finite, kernel)
    _, var_im_f, _ = _normalised_convolution(z.imag, var_im, finite, kernel)

    half = kernel.size // 2
    valid = coverage >= _MIN_KERNEL_COVERAGE
    if half:
        valid[:half] = False
        valid[t.size - half :] = False
    eif = float(np.sum(np.square(kernel)))
    return RRFCurve(
        time=t,
        real=np.asarray(z_f.real, dtype=np.float64),
        real_error=np.sqrt(var_re_f),
        imag=np.asarray(z_f.imag, dtype=np.float64),
        imag_error=np.sqrt(var_im_f),
        valid=valid,
        frequency_mhz=freq,
        phase_deg=phase_deg,
        method="fir",
        bandwidth_mhz=bandwidth,
        filter_taps=int(kernel.size),
        effective_independent_fraction=min(eif, 1.0),
    )


def _demodulate_wimda(
    t: NDArray[np.float64],
    a: NDArray[np.float64],
    e: NDArray[np.float64],
    dt: float,
    freq: float,
    phase_deg: float,
    box_width_us: float | None,
) -> RRFCurve:
    """WiMDA's 2·cos demodulation + box smooth, bin-for-bin.

    Mirrors ``$WIMDA_SRC/src/Plot.pas`` (``plotdata``): values scaled by
    2·cos(2πν₀t + φ), errors by |2·cos|, box half-width ``(box/dt) div 2``,
    box values *and errors* divided by the in-range count (the linear error
    average of comparison-ledger item 1), and output zeroed within half a box
    of either edge while edge errors keep their averages (ledger item 3).
    """
    box = 1.0 / (2.0 * freq) if box_width_us is None else float(box_width_us)
    if not np.isfinite(box) or box <= 0.0:
        raise ValueError(f"wimda_box_width_us must be positive and finite, got {box_width_us!r}.")

    theta = 2.0 * np.pi * freq * t + np.deg2rad(phase_deg)
    factor = 2.0 * np.cos(theta)
    finite = np.isfinite(a) & np.isfinite(e)
    values = np.where(finite, a * factor, 0.0)
    errors = np.where(finite, e * np.abs(factor), 0.0)

    n = t.size
    half = int(box / dt) // 2
    if half <= 0:
        smoothed = values.copy()
        smoothed_err = errors.copy()
        taps = 1
        valid = finite.copy()
    else:
        taps = 2 * half + 1
        # Running in-bounds windowed sums via cumulative sums; the count is
        # the number of in-bounds indices, exactly as the Pascal loop counts.
        cum_v = np.concatenate(([0.0], np.cumsum(values)))
        cum_e = np.concatenate(([0.0], np.cumsum(errors)))
        idx = np.arange(n)
        lo = np.clip(idx - half, 0, n)
        hi = np.clip(idx + half + 1, 0, n)
        count = (hi - lo).astype(np.float64)
        smoothed = (cum_v[hi] - cum_v[lo]) / count
        smoothed_err = (cum_e[hi] - cum_e[lo]) / count
        interior = (idx >= half) & (idx <= n - half - 1)
        smoothed = np.where(interior, smoothed, 0.0)
        valid = interior & finite

    return RRFCurve(
        time=t,
        real=smoothed,
        real_error=smoothed_err,
        imag=np.zeros_like(smoothed),
        imag_error=np.zeros_like(smoothed),
        valid=valid,
        frequency_mhz=freq,
        phase_deg=phase_deg,
        method="wimda",
        bandwidth_mhz=None,
        filter_taps=taps,
        effective_independent_fraction=1.0 / float(taps),
    )
