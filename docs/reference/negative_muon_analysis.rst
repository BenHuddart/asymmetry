Negative-muon capture-lifetime analysis (experimental)
=======================================================

.. warning::

   **Experimental — negative-muon analysis is a work in progress.**

   This page documents a scriptable, **API-only** μ⁻ capture-lifetime analysis
   that is **unvalidated against real μ⁻ data**. It has been exercised only on
   synthetic histograms. Lifetime source associations include unresolved
   discrepancies, and the fitting and capture-ratio machinery have not been checked against an established μ⁻ tool on measured
   data, and the API may change. There is **no GUI** for this feature. Verify
   any physical interpretation against the primary literature and an established
   tool (WiMDA, Mantid) before relying on it.

What this measures
------------------

A negative muon implanted in matter forms a muonic atom at the lattice site,
cascading to the 1s muonic orbit on a picosecond timescale. From there it either
decays while bound or is captured by the nucleus. The bound-decay rate is
:math:`Q/\tau_\mu`, where :math:`Q` is the Huff correction and
:math:`\tau_\mu` is the free-muon lifetime. Suzuki et al. Eq. (4) distinguishes
this rate from the nuclear capture rate. The observable disappearance lifetime

.. math::

   \tau(Z) = \frac{1}{\Lambda_\mathrm{cap}(Z) + Q/\tau_\mu}

depends on the element, isotope and bound-state conditions. A capture rate
cannot be inverted to obtain this lifetime without the bound-decay contribution.

In a sample containing several elements the decay-electron histogram summed over
all detectors is a sum of exponentials with those element-characteristic
lifetimes. The fitted coefficient of each exponential is a count amplitude,
not an integrated yield or an elemental abundance. Capture and transfer
probabilities, bound-decay branching, detection efficiency and the observation
window are needed to infer composition. The API reports amplitude ratios;
it does not perform that material-dependent calibration.

.. _derived-negative-muon-definitions:

Amplitude and uncertainty conventions
-------------------------------------

For a single exponential, its integrated yield on :math:`[t_a,t_b]` is
:math:`N_i\tau_i(e^{-t_a/\tau_i}-e^{-t_b/\tau_i})` in the continuous-time
convention. A ratio of fitted :math:`N_i` therefore differs from a yield ratio
when lifetimes differ. The simulator weights integrated populations and converts
them to bin amplitudes; those weights are not the reported amplitude ratios.

The ratio report uses the first-order covariance propagation for
:math:`R=N_i/N_r`, including the fitted cross covariance when available.
Without it the cross term is zero by assumption. This conditional fit uncertainty
does not include lifetime-table uncertainty, transfer/efficiency calibration or
model selection. Background-component subtraction uses unpolarised exponential
terms; it does not propagate fitted background nuisance covariance into
residual-count errors.

The optional ``lorgau`` mode defines a damped cosine and ``diamagnetic`` an
undamped cosine, with cyclic frequency in MHz, time in microseconds and phase
in radians. They multiply the exponential count sum by :math:`1+P(t)` and
leave the flat background untouched. These are count-modulation definitions,
not a complete bound-muon atomic depolarisation calculation.

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
— using a Poisson (Cash) or Gaussian cost. The lifetimes are held fixed at their table values by default. The exponential
sum is a defined count model; a successful synthetic fit does not validate a
material's capture physics.

.. _derived-negative-muon-lifetimes:

The lifetime table
------------------

The authority contains 79 element-labelled defaults, with mean life and quoted
uncertainty stored in microseconds. Mean life is the **total disappearance
lifetime**, including capture and bound decay; it is not the inverse capture
rate. Each :class:`~asymmetry.core.negmu.lifetimes.ElementLifetime` records
``source`` and a separate ``status``:

* ``selected_measurement`` identifies a directly checked published mean-life
  row, with ``source_location`` and ``selection_note``;
* ``unverified`` retains a source-labelled transcription without certifying
  which measurement or combination it represents;
* ``unconfirmed`` identifies an unresolved source association or selection;
* ``provisional`` retains one of seven provisional entries with no quoted
  uncertainty.

The selected measurements for Rh, Pd, Ag, Ho and Er are printed in Suzuki,
Measday and Roalsvig (1987), Table IV, pages 2219–2220. Rh and Pd use ref. 49
(Eckhause *et al.*, 1966); Ho uses ref. 54 (Filippas *et al.*, 1963).
Their target isotope composition and detection channel are unspecified in
this authority; the original experiments' full texts are not independently
verified here. Ag and Er use Suzuki's own measurements, labelled ref. a.
Section II describes elemental self-supporting targets and decay-electron
timing for these measurements, without resolving isotope composition.
Ag is selected for that directly documented material and timing context;
the other Table IV Ag measurements remain alternatives. No smallest-error
rule or weighted average is applied, and isotope/channel measurements are
not pooled into an elemental recommendation.

The remaining unconfirmed and provisional values stay available with their
explicit status. Ru and Tm have no mean-life row in Suzuki Table IV;
this does not establish that no measurement exists elsewhere. Actinide
isotope and detection-channel alternatives remain unresolved. A ``source``
identifier alone is not evidence that every stored value was verified.

Access the single authority programmatically:

.. code-block:: python

   from asymmetry.core.negmu.lifetimes import tau_us, lifetime

   tau_us("Fe")             # 0.206 μs
   lifetime("Fe").sigma_us  # 0.001 μs, quoted uncertainty
   selected = lifetime("Ag")
   selected.status          # "selected_measurement"
   selected.source_location
   selected.selection_note

Lifetimes are **fixed** at their defaults unless their symbols occur in
:class:`~asymmetry.core.negmu.fit.CaptureModelSpec`\ ``.free_tau``. The source
uncertainty is retained as metadata; it is not automatically propagated into
a fit with fixed lifetimes.

.. code-block:: python

   from asymmetry.core.negmu.fit import CaptureModelSpec

   spec = CaptureModelSpec(
       elements=("Ca", "Fe"),
       include_decay_background=True,
       free_tau=frozenset({"Fe"}),
   )

A model specification resolves the current authority when creating components
or default parameters. Constructed components and explicit fit parameters
retain their numerical lifetimes. Saved fit parameters retain both the
``tau_<element>`` value and its fixed flag; evaluation and refitting give
those explicit values precedence over component defaults. To repeat a fit
with stored lifetime choices, pass its parameter set explicitly:

.. code-block:: python

   from asymmetry.core.negmu.fit import fit_capture_histogram

   result = fit_capture_histogram(time, counts, spec,
                                  parameters=stored_result.parameters)

*When to use this.* Reach for the negative-muon capture-lifetime API when:

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

The amplitude ratio Ca/Fe = 1.67(5) recovers the targeted exponential-coefficient
ratio, approximately 5/3. The simulator's integrated population weights are
1.68/0.618; unequal lifetimes make this a different ratio.

Forward/backward fitting and α
-------------------------------

For a μ⁻ run where forward and backward detector banks are read out separately,
:func:`~asymmetry.core.negmu.fit.fit_capture_fb_alpha` fits both banks
simultaneously with a shared set of elemental amplitudes and a free
calibration constant α:

.. math::

   N_F(t) &= \sqrt{\alpha}\; \sum_i N_i\, e^{-t/\tau_i} + b_F \\
   N_B(t) &= \frac{1}{\sqrt{\alpha}}\; \sum_i N_i\, e^{-t/\tau_i} + b_B

The amplitudes N\ :sub:`i` are **shared** between the two banks (isotropic
capture populations), so the per-side capture ratios are identical by
construction. This is an imposed shared-population model, not an independent
check of forward/backward equality. Use per-group calls to
:func:`~asymmetry.core.negmu.fit.fit_capture_group` on each bank when a
genuine forward/backward amplitude difference is needed.

:func:`~asymmetry.core.negmu.ratio.fb_capture_ratio_report` returns a
dictionary of per-side :class:`~asymmetry.core.negmu.ratio.CaptureRatioReport`
objects (``{"forward": ..., "backward": ...}``); for the shared-amplitude fit
the two sides report identical ratios and differ only in the side label.

References
----------

* T. Suzuki, D. F. Measday, and J. P. Roalsvig,
  `Phys. Rev. C **35**, 2212 (1987)
  <https://doi.org/10.1103/PhysRevC.35.2212>`_, Section II and Table IV,
  pages 2219–2220 — directly checked mean-life selections and context.
* D. F. Measday, Phys. Rep. **354**, 243 (2001) — review of the nuclear
  physics of μ⁻ capture; Λ\ :sub:`cap`\ (Z) systematics.
* S. J. Blundell, R. De Renzi, T. Lancaster, and F. L. Pratt,
  *Muon Spectroscopy: An Introduction* (Oxford University Press, Oxford, 2022),
  Ch. 22 and Appendix C — the lifetime table (Table C.1) and negative-muon
  physics overview.
