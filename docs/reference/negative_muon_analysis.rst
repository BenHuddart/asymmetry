Negative-Muon Capture-Lifetime Analysis (Experimental)
=======================================================

.. warning::

   **Experimental — negative-muon analysis is a work in progress.**

   This page documents a scriptable, **API-only** μ⁻ capture-lifetime analysis
   that is **unvalidated against real μ⁻ data**. It has been exercised only on
   synthetic histograms. The element lifetimes are literature-anchored
   (Suzuki, Measday & Roalsvig 1987), but the fitting and capture-ratio
   machinery have not been checked against an established μ⁻ tool on measured
   data, and the API may change. There is **no GUI** for this feature. Verify
   any physical interpretation against the primary literature and an established
   tool (WiMDA, Mantid) before relying on it.

What this measures
------------------

A negative muon implanted in matter forms a muonic atom at the lattice site,
cascading to the 1s muonic orbit on a picosecond timescale. From there it either
decays freely — just as a positive muon does, with lifetime τ\ :sub:`μ` =
2.197 μs — or is captured by the nucleus at a rate Λ\ :sub:`cap`\ (Z) that grows
steeply with atomic number. The observable disappearance lifetime

.. math::

   \tau(Z) = \frac{1}{\Lambda_\mathrm{cap}(Z) + \Lambda_\mathrm{decay}}

is therefore element-characteristic: τ is 2.195 μs for hydrogen (capture
negligible) and falls to ≈ 72–96 ns for the heavy elements from Ru to U.

In a sample containing several elements the decay-electron histogram summed over
all detectors is a sum of exponentials with those element-characteristic
lifetimes, and the amplitude of each term is proportional to the fraction of
muon captures on that element. Fitting the histogram and forming amplitude ratios
gives the relative capture probabilities, which are directly related to the
stoichiometric composition weighted by each element's capture rate. This is
nuclear-charge-selective elemental analysis from the muon decay electrons.

The model
---------

The raw detector count histogram is modelled as

.. math::

   N(t) = \sum_i N_i\, e^{-t/\tau_i}
        + N_\mathrm{bg}\, e^{-t/\tau_\mu}
        + b,

where the sum runs over the declared elemental components (each with a
literature lifetime τ\ :sub:`i` and a free amplitude N\ :sub:`i`\ ), the
second term is a free-μ⁻ decay background with the fixed muon lifetime
τ\ :sub:`μ` and a free amplitude N\ :sub:`bg`\ , and *b* is a flat background
per bin. The model is fit directly to raw counts — not to the reduced asymmetry
— using a Poisson (Cash) or Gaussian cost. The free amplitudes carry the
composition information; the lifetimes are held fixed at their table values
by default.

The lifetime table
------------------

Lifetimes are transcribed from Table C.1 of Blundell, De Renzi, Lancaster &
Pratt, *Muon Spectroscopy: An Introduction* (OUP, 2022), Appendix C, which
combines measurements from Suzuki, Measday & Roalsvig (1987). Values for 79
elements from H (Z = 1) to Np (Z = 93) are included. A small number of entries
marked ``source="WiMDA-provisional"`` carry no published uncertainty and are
drawn from WiMDA's older table where the primary compilation has no entry.

Access lifetimes programmatically via
:func:`~asymmetry.core.negmu.lifetimes.tau_us` and
:func:`~asymmetry.core.negmu.lifetimes.lifetime`:

.. code-block:: python

   from asymmetry.core.negmu.lifetimes import tau_us, lifetime

   tau_us("Fe")             # → 0.206 μs
   lifetime("Fe").sigma_us  # → 0.001 μs

Lifetimes are **fixed** at their table values by default. Free a specific
element's lifetime by including its symbol in
:class:`~asymmetry.core.negmu.fit.CaptureModelSpec`\ ``.free_tau``:

.. code-block:: python

   from asymmetry.core.negmu.fit import CaptureModelSpec

   spec = CaptureModelSpec(
       elements=("Ca", "Fe"),
       include_decay_background=True,
       free_tau=frozenset({"Fe"}),   # fit the Fe lifetime as a free parameter
   )

WiMDA's older element table has several transcription errors — a Ti/Tl symbol
swap and divergent values for Ne, Zn, Sr, and Ba — that were identified during
the porting study (``docs/porting/negative-muon-analysis/``). The Table C.1
values from the primary Suzuki/Measday/Roalsvig compilation are adopted
throughout.

When to use this
----------------

Reach for the negative-muon capture-lifetime API when:

* you have a raw μ⁻ histogram from a mixed-element sample and want relative
  capture probabilities — for example, the stoichiometric composition of a
  compound from a single implantation measurement;
* you can assert which elements are present and use their fixed table lifetimes
  to constrain the fit.

Do **not** use this for:

* μ\ :sup:`+`\ SR spin-relaxation analysis — this module fits raw counts with
  element-characteristic lifetimes, not the reduced asymmetry with a
  depolarisation function; use
  :class:`~asymmetry.core.fitting.engine.FitEngine` with an asymmetry model
  for that;
* muonic X-ray (μ-XRF) elemental analysis, which identifies elements from the
  characteristic X-ray spectrum rather than the decay-electron histogram;
* samples where the elements of interest all have lifetimes shorter than the
  time resolution (the sub-100 ns regime for Z ≳ 45) — the binned histogram
  must resolve the exponential decays.

Expect to declare the candidate elements explicitly rather than free-scanning
the whole table; the fit has one amplitude per element and becomes
underdetermined when many lifetimes are simultaneously free.

Worked example
--------------

The snippet below synthesises a two-element capture run (Ca + Fe at amplitude
ratio ≈ 5:3), fits the single-group histogram, and recovers the ratio. In a
real measurement, replace ``template`` and ``run`` with a loaded
:class:`~asymmetry.core.data.dataset.Run` — everything downstream is identical.

.. code-block:: python

   import numpy as np
   from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
   from asymmetry.core.negmu.fit import CaptureModelSpec, fit_capture_group
   from asymmetry.core.negmu.ratio import capture_ratio_report
   from asymmetry.core.simulate import simulate_capture_run

   # --- geometry template --------------------------------------------------
   # In practice supply a loaded μ⁻ run; it provides the bin structure, t0,
   # and grouping.  This minimal template is sufficient for a synthetic test.
   n_bins, bw = 1000, 0.016          # bins; 16 ns bin width (μs)
   template = Run(
       run_number=1,
       histograms=[
           Histogram(
               counts=np.zeros(n_bins),
               bin_width=bw,
               t0_bin=0,
               good_bin_start=0,
               good_bin_end=n_bins - 1,
           )
           for _ in range(2)
       ],
       metadata={},
       grouping={
           "groups": {1: [1, 2]},
           "first_good_bin": 0,
           "last_good_bin": n_bins - 1,
           "bunching_factor": 1,
           "deadtime_correction": False,
       },
       source_file="",
   )

   # --- model spec and simulated run ----------------------------------------
   spec = CaptureModelSpec(elements=("Ca", "Fe"), include_decay_background=True)

   # Weights allocate the event budget per component.  The fitted amplitude is
   # proportional to weight × (1−exp(−bw/τ)), which simplifies to weight/τ when
   # bw ≪ τ.  For Ca and Fe with bw = 0.016 μs the linear approximation holds to
   # within ~4%, so w_Ca/w_Fe ≈ (5/3) × (τ_Ca/τ_Fe) targets amplitude ratio ≈ 5/3.
   # For heavier elements (τ ≲ 5×bw) use the exact formula directly.
   run = simulate_capture_run(
       template,
       components=spec.components(),
       weights={"Ca": 1.68, "Fe": 0.618},   # → Ca/Fe amplitude ratio ≈ 5/3
       total_events=150_000,
       seed=5,
   )

   # --- fit ----------------------------------------------------------------
   # fit_capture_group wraps build_count_group, which expects a MuonDataset
   # whose .run carries the histograms; time/asymmetry arrays are not used.
   dataset = MuonDataset(
       time=np.array([]), asymmetry=np.array([]),
       error=np.array([]), metadata={}, run=run,
   )
   result = fit_capture_group(dataset, group_id=1, spec=spec)

   # --- capture-ratio report -----------------------------------------------
   report = capture_ratio_report(result, spec, reference="Fe")
   for r in report.ratios:
       print(f"{r.numerator}/{r.denominator} = {r.ratio:.2f}({round(r.sigma * 100):d})")
   # Ca/Fe = 1.67(5)

The ratio Ca/Fe = 1.67(5) recovers the input weight ratio 5/3.

Forward/backward fitting and α
-------------------------------

For a μ⁻ run where forward and backward detector banks are read out separately,
:func:`~asymmetry.core.negmu.fit.fit_capture_fb_alpha` fits both banks
simultaneously with a shared set of elemental amplitudes and a free
detector-balance parameter α:

.. math::

   N_F(t) &= \sqrt{\alpha}\; \sum_i N_i\, e^{-t/\tau_i} + b_F \\
   N_B(t) &= \frac{1}{\sqrt{\alpha}}\; \sum_i N_i\, e^{-t/\tau_i} + b_B

The amplitudes N\ :sub:`i` are **shared** between the two banks (isotropic
capture populations), so the per-side capture ratios are identical by
construction. This diverges from WiMDA, which fits independent per-side
amplitudes. Use per-group calls to
:func:`~asymmetry.core.negmu.fit.fit_capture_group` on each bank when a
genuine forward/backward amplitude difference is needed.

:func:`~asymmetry.core.negmu.ratio.fb_capture_ratio_report` returns a
dictionary of per-side :class:`~asymmetry.core.negmu.ratio.CaptureRatioReport`
objects (``{"forward": ..., "backward": ...}``); for the shared-amplitude fit
the two sides report identical ratios and differ only in the side label.

References
----------

* T. Suzuki, D. F. Measday, and J. P. Roalsvig, Phys. Rev. C **35**, 2212
  (1987) — primary lifetime compilation used for the element table.
* D. F. Measday, Phys. Rep. **354**, 243 (2001) — review of the nuclear
  physics of μ⁻ capture; Λ\ :sub:`cap`\ (Z) systematics.
* S. J. Blundell, R. De Renzi, T. Lancaster, and F. L. Pratt,
  *Muon Spectroscopy: An Introduction* (Oxford University Press, Oxford, 2022),
  Ch. 22 and Appendix C — the lifetime table (Table C.1) and negative-muon
  physics overview.
