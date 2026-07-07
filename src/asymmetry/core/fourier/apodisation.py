"""Data-driven matched-apodisation suggestion.

Advisory only — nothing in the core (or the GUI) ever applies a suggested
filter automatically. A matched filter maximises a line's peak S/N at the cost
of roughly doubling its apparent width, so applying one is a decision the user
must make knowingly; this module only computes what the matched values *would
be* from the unapodised spectrum.

Widths are measured on the POWER spectrum ``|F|^2``, not the magnitude: the
one-sided transform of a damped cosine carries a dispersion part alongside the
absorption line, and at half-maximum the *magnitude* of a Lorentzian is a
factor ``sqrt(3)`` wider than the absorption shape (the first implementation
measured magnitude and recovered relaxation rates ~1.7x too large). On the
power spectrum both shapes have closed-form half-widths.

The window parameterisation matches :func:`asymmetry.core.fourier.window.
apply_fft_filter` (``start_time_us=0``); with the power-spectrum FWHM
``Gamma`` in MHz and time constants in µs:

* Lorentzian weight ``exp(-t/tau)``. The one-sided transform of
  ``exp(-lambda t) cos(w0 t)`` has power ``1/4 / (lambda^2 + dw^2)`` exactly —
  a true Lorentzian of FWHM ``Gamma = lambda / pi`` (MHz) — so
  ``tau = 1 / (pi Gamma)``.
* Gaussian weight ``exp(-(t/tau)^2)``. For a Gaussian envelope
  ``exp(-sigma^2 t^2 / 2)`` the one-sided power line is
  ``(pi/2) exp(-u^2) + 2 Dawson(u / sqrt 2)^2`` (in ``u = dw / sigma`` units);
  its half-maximum falls at ``u* = 1.42294`` (numerical root — the Dawson
  dispersion tail broadens it well past the pure-Gaussian ``sqrt(ln 2)``), so
  ``sigma = pi Gamma / u*`` and the matched
  ``tau = sqrt(2) / sigma = sqrt(2) u* / (pi Gamma)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: A candidate line's POWER must rise this far above the search window's
#: baseline power to be worth matching — 4x in amplitude, squared because the
#: search runs on the power spectrum (the plot framing's amplitude convention
#: is 4.0). Below it, "matching" would chase a noise spike.
_LINE_PROMINENCE_POWER = 16.0

#: Half-maximum of the one-sided Gaussian-envelope power line in units of
#: ``dw / sigma`` — the numerical root of
#: ``(pi/2) exp(-u^2) + 2 Dawson(u / sqrt 2)^2 = pi / 4``.
_GAUSSIAN_POWER_HALF_WIDTH = 1.42294

#: Fraction of the frequency span treated as the DC region (zero-frequency
#: peak plus filter rolloff) and excluded from the line search, matching the
#: plot-framing convention.
_DC_CUT_FRACTION = 0.02

#: A measured FWHM narrower than this many frequency bins is resolution-limited
#: — the width is the transform's, not the sample's — so there is no physical
#: relaxation to match a filter to.
_MIN_FWHM_BINS = 2.0


@dataclass(frozen=True)
class ApodisationSuggestion:
    """A matched-filter suggestion derived from one spectral line."""

    #: ``"lorentzian"`` or ``"gaussian"`` — the window kind that was matched.
    window: str
    #: Matched filter time constant in µs (``apply_fft_filter`` convention).
    time_constant_us: float
    #: Frequency (MHz) of the line the suggestion was matched to.
    line_frequency_mhz: float
    #: Measured POWER-spectrum FWHM (MHz) of that line, unapodised.
    line_fwhm_mhz: float


def _half_maximum_crossing(
    freqs: np.ndarray,
    values: np.ndarray,
    peak_index: int,
    half_level: float,
    step: int,
) -> float | None:
    """Interpolated frequency where *values* first crosses *half_level*.

    Walks from the peak in *step* direction (±1); ``None`` when the edge of the
    search window is reached first (the width cannot be measured).
    """
    index = peak_index
    while 0 <= index + step < freqs.size:
        nxt = index + step
        if values[nxt] <= half_level:
            v0, v1 = values[index], values[nxt]
            if v1 == v0:
                return float(freqs[nxt])
            fraction = (v0 - half_level) / (v0 - v1)
            return float(freqs[index] + fraction * (freqs[nxt] - freqs[index]))
        index = nxt
    return None


def suggest_matched_apodisation(
    freqs: np.ndarray,
    magnitude: np.ndarray,
    *,
    window: str = "lorentzian",
    min_frequency_mhz: float | None = None,
    max_frequency_mhz: float | None = None,
) -> ApodisationSuggestion | None:
    """Suggest the matched apodisation for the dominant line of a spectrum.

    *freqs*/*magnitude* are the UNAPODISED magnitude spectrum (MHz axis); an
    already-filtered spectrum would match the filter, not the sample. The
    optional frequency window restricts the line search (callers narrow it
    around the field-expected region, as phase estimation does). Returns
    ``None`` — meaning "leave apodisation off" — when no line clears the
    prominence threshold, when the dominant line is resolution-limited, or
    when its width cannot be measured inside the window.
    """
    window_key = str(window).strip().lower()
    if window_key not in {"lorentzian", "gaussian"}:
        raise ValueError(f"Unknown apodisation window {window!r}.")

    f = np.asarray(freqs, dtype=float)
    # Measure on the power spectrum — see the module docstring for why the
    # magnitude's half-width is the wrong observable.
    v = np.square(np.abs(np.asarray(magnitude, dtype=float)))
    finite = np.isfinite(f) & np.isfinite(v)
    f = f[finite]
    v = v[finite]
    if f.size < 8:
        return None
    order = np.argsort(f)
    f = f[order]
    v = v[order]

    f_max = float(np.max(f))
    if f_max <= 0.0:
        return None
    lower = max(float(min_frequency_mhz or 0.0), f_max * _DC_CUT_FRACTION)
    upper = float(max_frequency_mhz) if max_frequency_mhz is not None else f_max
    in_window = (f > lower) & (f <= upper)
    if np.count_nonzero(in_window) < 8:
        return None
    f_win = f[in_window]
    v_win = v[in_window]

    baseline = float(np.median(v_win))
    if baseline <= 0.0:
        baseline = float(np.mean(v_win))
    peak_index = int(np.argmax(v_win))
    peak = float(v_win[peak_index])
    if baseline <= 0.0 or peak <= baseline * _LINE_PROMINENCE_POWER:
        return None

    half_level = baseline + 0.5 * (peak - baseline)
    left = _half_maximum_crossing(f_win, v_win, peak_index, half_level, -1)
    right = _half_maximum_crossing(f_win, v_win, peak_index, half_level, +1)
    if left is None or right is None:
        return None
    fwhm = float(right - left)
    bin_width = float(np.median(np.diff(f_win)))
    if not np.isfinite(fwhm) or fwhm < _MIN_FWHM_BINS * bin_width:
        return None

    if window_key == "lorentzian":
        time_constant_us = 1.0 / (math.pi * fwhm)
    else:
        time_constant_us = math.sqrt(2.0) * _GAUSSIAN_POWER_HALF_WIDTH / (math.pi * fwhm)
    return ApodisationSuggestion(
        window=window_key,
        time_constant_us=float(time_constant_us),
        line_frequency_mhz=float(f_win[peak_index]),
        line_fwhm_mhz=fwhm,
    )
