Rotating-reference-frame display
================================

The rotating-reference-frame (RRF) view shows the **slow envelope of a fast
transverse-field precession**. A run at 2 T precesses at ~271 MHz — about
2000 cycles across an 8 μs window — so the raw FB-asymmetry plot is a solid
band that tells you nothing about how the signal relaxes. Demodulating into
a frame rotating at a frequency :math:`\nu_0` near the line leaves only what
is *slow*: the relaxation envelope, and a beat at the offset
:math:`\delta\nu = \nu - \nu_0` if :math:`\nu_0` is deliberately detuned.
RRF display is standard practice for vortex-lattice and Knight-shift work at
high field.

This page describes the *filtered* display of a single forward–backward
asymmetry, available under **Options → Advanced → Rotating reference frame**.
When a grouping measures both transverse projections — the :math:`P_x` and
:math:`P_y` of vector-polarisation mode — the frame is reached exactly instead,
bin by bin and without a filter, and the result can be fitted: see
:doc:`rotating_frame_projection`. The filtered controls are not offered for
such groupings.

The controls live in a row above the time-domain plot and appear only on the
**FB Asymmetry** representation: an enable box, the frame frequency
:math:`\nu_0` (entered in MHz or Gauss — the two are equivalent through
:math:`\nu = (\gamma_\mu/2\pi)B`), a frame phase, the low-pass bandwidth,
and the displayed component. When you first enable it, :math:`\nu_0` is
seeded from the run's field metadata; type over it freely. The plot carries
a **frame badge** ("frame: ν₀ = … MHz") so exported figures remain
self-describing, and the frame travels with exported data too. With several
runs overlaid, the one frame applies to all of them — comparing envelopes
across a temperature series at fixed field is the natural workflow.

What the transform does
-----------------------

The measured asymmetry :math:`A(t) = a(t)\cos(2\pi\nu t + \phi_d)` is
multiplied by the complex carrier :math:`2e^{-i(2\pi\nu_0 t + \phi)}`. The
product contains the rotating-frame signal at :math:`\delta\nu` and an
*image* at :math:`\nu + \nu_0`, which a low-pass filter (a Blackman
windowed-sinc FIR, stopband below −74 dB) removes. What remains is complex:

- **In-phase (Re)** — the envelope, when the frame phase matches the data.
  The default view.
- **Quadrature (Im)** — zero when the phase matches; signal leaking here
  diagnoses a phase error (or a frequency offset, which rotates the signal
  between the two components at :math:`\delta\nu`).
- **Magnitude** — the phase-free envelope :math:`|a(t)|`. Convenient, but
  noise-biased upward where the signal is comparable to its error bars
  (a non-Gaussian magnitude distribution), so treat near-zero magnitude qualitatively.

.. figure:: /_generated/screenshots/scalar_rrf_magnitude.png
   :width: 100%
   :alt: Synthetic transverse-field magnitude display with conditional error bars.

   **Magnitude** of a synthetic EuO signal at 30 K, with a detuned
   frame and shared real/imaginary noise included in each error bar.

The **bandwidth** is the single-sided filter cutoff. *Auto* picks
:math:`\nu_0/2`, reduced automatically when coarse binning folds the image
back into the displayed band (an aliasing trap that affects data binned
close to the Nyquist rate of the precession). Narrower bandwidth = smoother
envelope but slower response; anything genuinely oscillating faster than
the cutoff — including a large deliberate detuning beat — is filtered away
with the image, so keep the bandwidth comfortably above :math:`|\delta\nu|`.

.. _scalar-rrf-magnitude-errors:

Magnitude errors
----------------

Both quadratures use the same scalar observations. For supplied independent
input errors :math:`e_j`, carrier phase :math:`\theta_j=2\pi\nu_0t_j+\phi`,
and actual normalised filter weights :math:`w_{ij}`, the same-bin covariance is

.. math::

   C_{RI,i}=\sum_j w_{ij}^2[-4\cos\theta_j\sin\theta_j]e_j^2.

The weights include the finite-input mask and coverage normalisation used for
the values and marginal variances. The covariance is signed. For
:math:`r=\sqrt{R^2+I^2}>0`, first-order propagation [3]_ gives

.. math::

   \operatorname{var}(r)=\frac{R^2\operatorname{var}(R)
   +I^2\operatorname{var}(I)+2RI C_{RI}}{r^2}.

An identity filter therefore takes a scalar input :math:`10\pm1` to magnitude
:math:`20\pm2`, independently of carrier phase. Percent inputs retain percent
values and standard errors; covariance has squared percent units. Fractional
inputs retain the corresponding fractional units.

These errors are conditional on the supplied scalar input-noise model.
Correlations already present through estimated backgrounds or other reduction
steps cannot be reconstructed from marginal input errors alone. Filtering also
correlates output bins; same-bin covariance does not define a joint likelihood.
At low signal-to-noise, magnitude remains biased and first-order errors are
unreliable. At vanishing magnitude the display uses
:math:`\sqrt{[\operatorname{var}(R)+\operatorname{var}(I)]/2}` as a finite
error-bar fallback, not a confidence interval. Asymmetry also checks for a
materially negative propagated magnitude variance and raises an invariant error;
only cancellation within floating-point accumulation roundoff is clipped to zero.
Zero-noise model overlays retain exactly zero errors.

*When to use it — and when not to.* Use the RRF view to **look**: to see whether a high-field signal relaxes
exponentially or Gaussian-like, to spot beats from closely-spaced lines
(two frequencies :math:`\delta` apart show up as an envelope oscillating at
:math:`\delta`), and to make figures where the physics is the envelope, not
the carrier.

Do **not** fit the demodulated curve. The low-pass filter correlates
neighbouring points over the filter support (the drawn error bars are
conditional per-point errors, but the points are far from independent) and distorts
lineshapes at the bandwidth scale; a χ² fit of the displayed curve gives
wrong parameters *and* wrong uncertainties. The textbook is explicit that
the RRF transform is a visualisation tool rather than a fit preprocessor
[1]_. Quantitative results come from fitting the **raw** data — which the
fit panel always does, whatever the display shows. For working in
rotating-frame numbers there, the core provides a frequency-offset fit
using the raw data with the model's precession frequencies offset by
:math:`\nu_0`, so the fitted frequency reads as the small offset :math:`\delta\nu` while the statistics
remain those of the laboratory observations. The fitted curve drawn over an
RRF display is demodulated through the same pipeline as the data, so the overlay
always remains comparable.

A high-field caveat: at fields of several tesla, the geometric phases of
the individual detectors span the full circle, and an asymmetry built by
summing many detectors into one forward and one backward group can wash the
precession line out almost completely. If the RRF view of a high-field run
shows nothing, inspect a single near-opposite detector pair first — the
proper multi-detector treatment (mapping all detectors onto one quadrature
pair before the frame rotation [2]_) is not yet implemented.

Compared with other programs
----------------------------

WiMDA 1.326 uses real-cosine demodulation and running box smoothing, with
linearly averaged scaled standard errors. The core comparison API
(``method="wimda"``) retains this distinct convention. The FIR display instead
retains both quadratures and their shared-noise covariance. Independent identity
and measured linear-response checks establish its uncertainty propagation;
they do not establish numerical agreement with another program's estimator.

musrfit 1.12.1 offers scalar asymmetry RRF preparation and packing through its
`RRF fit types <https://lmu.web.psi.ch/musrfit/user/html/user-manual.html>`_.
In Mantid, `RRFMuon v1
<https://docs.mantidproject.org/nightly/algorithms/RRFMuon-v1.html>`_
(documentation associated with the 6.16 development snapshot) rotates two supplied
components rather than constructing FIR quadratures from a scalar observation.
Neither operation supplies a verified numerical covariance oracle for this FIR.
No new native-program error agreement is asserted. WiMDA's published description
is given by F. L. Pratt, Physica B **289–290**, 710 (2000),
`doi:10.1016/S0921-4526(00)00328-8 <https://doi.org/10.1016/S0921-4526(00)00328-8>`_.

References
----------

.. [1] S. J. Blundell, R. De Renzi, T. Lancaster, and F. L. Pratt,
   *Muon Spectroscopy: An Introduction* (Oxford University Press, Oxford, 2022) — the
   rotating-reference-frame section of the time-domain-analysis chapter;
   T. M. Riseman and J. H. Brewer, Hyperfine Interact. **65**, 1107 (1991).

.. [2] B. D. Rainford, in *Muon Science*, edited by S. L. Lee, S. H. Kilcoyne,
   and R. Cywinski (Institute of Physics Publishing, Bristol, 1999), p. 463.

.. [3] JCGM 100:2008, *Evaluation of measurement data — Guide to the expression
   of uncertainty in measurement*, Sec. 5.2 (correlated input quantities),
   `BIPM <https://www.bipm.org/documents/20126/2071204/JCGM_100_2008_E.pdf>`_.
