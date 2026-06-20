Rotating-reference-frame display
================================

The rotating-reference-frame (RRF) view shows the **slow envelope of a fast
transverse-field precession**. A run at 2 T precesses at ~271 MHz — about
2000 cycles across an 8 µs window — so the raw FB-asymmetry plot is a solid
band that tells you nothing about how the signal relaxes. Demodulating into
a frame rotating at a frequency :math:`\nu_0` near the line leaves only what
is *slow*: the relaxation envelope, and a beat at the offset
:math:`\delta\nu = \nu - \nu_0` if :math:`\nu_0` is deliberately detuned.
RRF display is standard practice for vortex-lattice and Knight-shift work at
high field.

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
  (a Rician distribution), so treat near-zero magnitude qualitatively.

The **bandwidth** is the single-sided filter cutoff. *Auto* picks
:math:`\nu_0/2`, reduced automatically when coarse binning folds the image
back into the displayed band (an aliasing trap that affects data binned
close to the Nyquist rate of the precession). Narrower bandwidth = smoother
envelope but slower response; anything genuinely oscillating faster than
the cutoff — including a large deliberate detuning beat — is filtered away
with the image, so keep the bandwidth comfortably above :math:`|\delta\nu|`.

When to use it — and when not to
--------------------------------

Use the RRF view to **look**: to see whether a high-field signal relaxes
exponentially or Gaussian-like, to spot beats from closely-spaced lines
(two frequencies :math:`\delta` apart show up as an envelope oscillating at
:math:`\delta`), and to make figures where the physics is the envelope, not
the carrier.

Do **not** fit the demodulated curve. The low-pass filter correlates
neighbouring points over the filter support (the drawn error bars are
per-point honest, but the points are far from independent) and distorts
lineshapes at the bandwidth scale; a χ² fit of the displayed curve gives
wrong parameters *and* wrong uncertainties. The textbook is explicit that
the RRF transform is a visualisation tool rather than a fit preprocessor
[1]_. Quantitative results come from fitting the **raw** data — which the
fit panel always does, whatever the display shows. For working in
rotating-frame numbers there, the core provides a frequency-offset fit
(:mod:`asymmetry.core.fitting.rrf_offset`): the raw data are fitted with
the model's precession frequencies offset by :math:`\nu_0`, so the fitted
frequency reads as the small offset :math:`\delta\nu` while the statistics
stay exact. The fitted curve drawn over an RRF display is demodulated
through the same pipeline as the data, so the overlay always remains
comparable.

A high-field caveat: at fields of several tesla, the geometric phases of
the individual detectors span the full circle, and an asymmetry built by
summing many detectors into one forward and one backward group can wash the
precession line out almost completely. If the RRF view of a high-field run
shows nothing, inspect a single near-opposite detector pair first — the
proper multi-detector treatment (mapping all detectors onto one quadrature
pair before the frame rotation [2]_) is not yet implemented.

History
-------

WiMDA's RRF display demodulates with :math:`2\cos(\omega t + \phi)` and
smooths with a running box average — the real part of the same operation,
with a filter whose image rejection depends delicately on the box width
landing on the image period. Complex demodulation with a designed filter
suppresses the image unconditionally and keeps the quadrature and magnitude
views. The WiMDA-equivalent mode survives in the core API
(``method="wimda"``) for comparison.

References
----------

.. [1] S. J. Blundell, R. De Renzi, T. Lancaster, and F. L. Pratt,
   *Muon Spectroscopy: An Introduction* (Oxford University Press, Oxford, 2022) — the
   rotating-reference-frame section of the time-domain-analysis chapter;
   T. M. Riseman and J. H. Brewer, Hyperfine Interact. **65**, 1107 (1991).

.. [2] B. D. Rainford, in *Muon Science*, eds. S. L. Lee, S. H. Kilcoyne,
   and R. Cywinski (CRC Press, 1999), p. 463.
