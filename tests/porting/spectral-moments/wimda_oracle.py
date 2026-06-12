"""Behavioural oracle: WiMDA ``Moments.pas`` arithmetic, transcribed.

A faithful, independent re-derivation of the moment arithmetic in
``$WIMDA_SRC/src/Moments.pas`` (``TMoment.FormShow`` + ``parabpkextrap``), used to
pin :func:`asymmetry.core.fourier.moments.spectrum_moments` on a shared spectrum.

GPL note: ``Moments.pas`` is GPL. This is a *behavioural* oracle — the arithmetic
is re-implemented from the algorithm, with the explicit normal-equation
accumulators kept (so the parabolic vertex is cross-checked against the core's
``np.polyfit`` route by a different code path), and with WiMDA's index off-by-one
(divergence D1) deliberately *not* reproduced: indices run a consistent
``0 … n-1``. It is test-only and never imported by the application.
"""

from __future__ import annotations

import math


def _parabolic_peak(b: list[float], amp: list[float], ipk: int, nexpts: int = 5) -> float:
    """WiMDA ``parabpkextrap`` — 5-point quadratic vertex via normal equations.

    Returns the refined peak position, or the discrete ``b[ipk]`` when the edge
    guard fires (peak within ``nexpts//2`` points of either end).
    """
    n = len(b)
    half = nexpts // 2
    if not (half <= ipk <= n - half - 1):
        return b[ipk]
    x0 = b[ipk]
    xi = b[ipk + 1] - b[ipk]
    s1 = sx = sy = syx = sx2 = syx2 = sx3 = sx4 = 0.0
    for i in range(ipk - half, ipk + half + 1):
        xx = (b[i] - x0) / xi
        yy = amp[i]
        xxx = xx * xx
        s1 += 1.0
        sx += xx
        sy += yy
        syx += xx * yy
        sx2 += xxx
        syx2 += yy * xxx
        sx3 += xxx * xx
        sx4 += xxx * xxx
    m3 = sx3 - sx2 * sx / s1
    m4 = sx2 - sx * sx / s1
    m1 = sx4 - sx2 * sx2 / s1
    m2 = sx3 - sx2 * sx / s1
    m6 = syx - sx * sy / s1
    m5 = syx2 - sx2 * sy / s1
    anum = m5 * m4 - m2 * m6
    bnum = m1 * m6 - m3 * m5
    if anum == 0.0:
        return b[ipk]
    xvertex = -bnum / 2.0 / anum
    return xvertex * xi + x0


def wimda_moments(
    b: list[float],
    amp: list[float],
    *,
    cutoff_fraction: float,
    x_range: tuple[float, float] | None = None,
) -> dict[str, float]:
    """Return WiMDA's moment set for a (field, amplitude) spectrum.

    *b* must be ascending. ``cutoff_fraction`` is the fraction-of-peak cutoff and
    *x_range* the ``[xnmin, xnmax]`` window (full axis when ``None``). The discrete
    peak is taken over *x_range* to match the core (the windowed-peak divergence).
    """
    if x_range is not None:
        lo, hi = sorted(x_range)
        idx = [i for i in range(len(b)) if lo <= b[i] <= hi]
        b = [b[i] for i in idx]
        amp = [amp[i] for i in idx]
    n = len(b)

    # Discrete peak (consistent 0 … n-1; D1 bug excluded).
    ppk = -math.inf
    ipk = 0
    for i in range(n):
        if amp[i] > ppk:
            ppk, ipk = amp[i], i
    b_pk = _parabolic_peak(b, amp, ipk)

    thr = cutoff_fraction * ppk
    sel = [i for i in range(n) if amp[i] > thr]

    m0 = sum(amp[i] for i in sel)
    if m0 <= 0.0:
        return {"n_sample": 0}
    b_ave = sum(amp[i] * b[i] for i in sel) / m0

    m2 = m2pk = m3 = 0.0
    for i in sel:
        d = b[i] - b_ave
        dpk = b[i] - b_pk
        p = amp[i]
        m2 += p * d * d
        m2pk += p * dpk * dpk
        m3 += p * d * d * d
    m2 /= m0
    m2pk /= m0
    m3 /= m0
    rootm2 = math.sqrt(m2)
    rootm2pk = math.sqrt(m2pk)
    alpha = (abs(m3) ** (1.0 / 3.0)) / rootm2
    if m3 < 0:
        alpha = -alpha
    beta = (b_ave - b_pk) / rootm2pk
    return {
        "b_pk": b_pk,
        "b_ave": b_ave,
        "b_diff": b_ave - b_pk,
        "b_rms_mean": rootm2,
        "b_rms_peak": rootm2pk,
        "skewness": alpha,
        "skewness_g1": m3 / m2**1.5,
        "beta": beta,
        "n_sample": len(sel),
    }
