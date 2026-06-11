.. _exclusions-glossary:

Exclusions: A Glossary of Five Mechanisms
=========================================

Five different controls in Asymmetry are described as "excluding" or "skipping"
part of the data, and the word hides real differences. They operate on
different domains (time bins, time points, frequency bins, detectors), with
deliberately different **semantics** — one *drops* data, another *de-weights*
it, others *hard-zero* it — and they persist under different schema keys. They
are *not* interchangeable, and the most dangerous pair (the two time-window
controls) share a label while meaning opposite things. This page is the
reference; for the mechanics of each, follow the cross-link.

The five at a glance
--------------------

.. list-table::
   :header-rows: 1
   :widths: 22 14 26 22 16

   * - Mechanism (GUI label)
     - Domain
     - Semantics
     - Parameterisation
     - Schema key
   * - **Count-fit exclude window**
       (:doc:`count_domain_fitting` — "Skip window (μs)")
     - time bins
     - **hard drop** from the fit — the bins inside are removed (endpoints
       inclusive) and contribute nothing to the cost
     - :math:`(t_1, t_2)` µs
     - *(persisted from reconciliation Phase 2; see note)*
   * - **MaxEnt exclude window**
       (:doc:`fourier_analysis` — "Exclude from / to (μs)")
     - time points
     - **σ-inflate ×10⁸** — the points are de-weighted (error bars blown up)
       but kept, so the time grid and any derived frequency resolution are
       unchanged
     - :math:`(t_1, t_2)` µs
     - ``exclude_t_min_us`` /
       ``exclude_t_max_us``
   * - **Fourier exclusion ranges**
       (:doc:`frequency_finishers` — "Exclusions" table)
     - frequency bins
     - **hard zero** of the displayed spectrum inside each window
     - centre ± half-width
     - ``exclusion_ranges`` (``exclude_enabled``)
   * - **Diamagnetic band**
       (:doc:`frequency_finishers` — "Diamagnetic line")
     - frequency bins
     - **hard zero** of the displayed spectrum (a derived special case of the
       exclusion ranges, centred on the run's Larmor frequency)
     - derived centre ± half-width
     - ``diamag_exclusion`` /
       ``diamag_half_width_mhz``
   * - **Detector exclusion**
       (:doc:`data_reduction/detector_exclusion` — "Exclude Detectors")
     - detectors
     - **drop** the detector from *every* group sum at reduction time; raw
       histograms untouched
     - 1-based id list (e.g. ``1,5,10-15``)
     - ``excluded_detectors``
       (grouping)

Three semantics, not one
------------------------

The five controls realise only three underlying operations, and confusing them
changes the result:

- **Drop** (count-fit window, detector exclusion) removes data from the
  calculation entirely. A dropped time bin does not enter the fit cost; a
  dropped detector does not enter any group sum.
- **De-weight** (MaxEnt window) keeps the data on the grid but inflates its
  error by a factor of :math:`10^8` so it carries essentially no weight. The
  grid spacing — and therefore the FFT/MaxEnt frequency resolution derived from
  it — is preserved. This is the crucial difference from a drop.
- **Hard-zero** (Fourier exclusion ranges, diamagnetic band) sets the displayed
  spectrum to zero inside a frequency window. It is a *display* operation on a
  computed spectrum, not a change to the transform or the fit input.

.. warning::

   **The two time-window controls share a label but mean opposite things.**
   Both the count-fit and MaxEnt windows are labelled in µs with a
   :math:`(t_1, t_2)` start/end, but the count fit **drops** its bins while
   MaxEnt **de-weights** its points and keeps the grid. Setting "the same"
   window in both panels does not do the same thing: the count fit loses those
   bins from the statistics; MaxEnt keeps its frequency resolution intact and
   merely stops trusting the excluded stretch. Read the panel each control
   lives in, not just the number.

When to reach for each
----------------------

- **Count-fit / MaxEnt time windows** — to reject a corrupted interior stretch
  (a laser flash, an RF burst, a detector glitch) from a *single run's* analysis
  without splitting the run. Use the count-fit window when you want those bins
  gone from a count fit; use the MaxEnt window when you want the reconstruction
  to ignore them but keep its frequency resolution.
- **Fourier exclusion ranges / diamagnetic band** — to blank a known *spectral*
  artefact (RF pickup, a mains harmonic, a saturating diamagnetic line) so it
  does not dominate the plot or bias a baseline or peak search. Never blank a
  feature you have not identified, and keep the half-widths tight. The
  diamagnetic band is the dedicated, field-derived case (see
  :doc:`frequency_finishers` for the leave / fit-and-subtract / exclude-band
  choice).
- **Detector exclusion** — to remove a dead or hot *detector* from the whole
  analysis at reduction time; re-estimate α afterwards, because the group
  balance changes with group membership.

.. note::

   The count-fit window is labelled "Skip window (µs)" (drop) as of
   reconciliation Phase 2, which also gave it a persisted schema key. The MaxEnt
   window is still labelled "Exclude from / to (μs)"; Phase 3 relabels it
   "De-weight window (µs)" to encode that it de-weights rather than drops. This
   glossary's MaxEnt label column is updated when that phase lands.

See also
--------

* :doc:`count_domain_fitting` — the count-fit window and the count model.
* :doc:`fourier_analysis` — the MaxEnt exclude window.
* :doc:`frequency_finishers` — the Fourier exclusion table and the diamagnetic
  band.
* :doc:`data_reduction/detector_exclusion` — excluding a detector at reduction
  time.
