ALC Mode (Integral-Asymmetry Field Scans)
=========================================

**ALC mode** turns a set of runs taken at different fields (or temperatures)
into a single *field scan*: one integral-asymmetry value per run, plotted
against the swept variable. It is the workflow for **avoided level crossing
(ALC)**, **repolarisation / decoupling**, and **quadrupolar level-crossing
resonance (QLCR)** measurements, where a resonance appears as a dip or step in
the asymmetry as the field is stepped through the crossing.

Unlike the time-domain fitting workflows, ALC mode does **not** fit a model to
each spectrum. It reduces every run to one number — the asymmetry integrated
over a time window — and then lets you fit a baseline and resonance peaks to the
resulting scan. The method follows WiMDA's count-integral ALC and Mantid's
``PlotAsymmetryByLogValue``; with :math:`\alpha = 1` it reproduces the WiMDA
count integral exactly.

The integral-asymmetry observable
---------------------------------

For each run, the forward and backward counts are summed over the integration
window :math:`[t_\mathrm{min}, t_\mathrm{max}]` and combined into a single
asymmetry value:

.. math::

   A = \frac{\sum F - \alpha \sum B}{\sum F + \alpha \sum B},

where the sums run over the good bins inside the window, :math:`F` and :math:`B`
are the forward and backward grouped counts, and :math:`\alpha` is the current
balance factor from your detector grouping. Summing the counts *before* forming
the ratio (the WiMDA "count integral") gives Poisson-weighted statistics across
the whole window rather than averaging per-bin asymmetries.

Each value is plotted against the run's swept variable to form the scan. A
resonance shifts the time-averaged asymmetry, so it shows up directly as a
feature in :math:`A` versus field. The **dA/dB** view (see below) plots the
field-derivative of the scan — WiMDA's *differential ALC* — which sharpens a
resonance into a peak–trough signature.

Entering ALC mode
-----------------

ALC mode is a toggle on the **main toolbar**, labelled **ALC mode**. It is
**only available in the Forward–Backward asymmetry view** — the toggle is
disabled in the frequency, groups, and MaxEnt views, and auto-exits if you
leave the F–B asymmetry view.

When you enable it:

* the toggle highlights **red** while ALC mode is active, so the bespoke mode is
  always obvious;
* the **Fit** dock is replaced by the **Integral scan (ALC)** build panel;
* the **Parameters** dock is replaced by the **ALC scan view** (the scan plot
  plus its baseline and peak controls).

Toggling ALC mode off restores the normal **Fit** and **Parameters** docks; your
scan and its analysis are kept and reappear when you re-enter ALC mode.

The ALC workflow
----------------

Step 1 — Build the scan
~~~~~~~~~~~~~~~~~~~~~~~~~

Load the runs of the scan and select them in the **Data Browser** (the scan is
built from the current selection, exactly like a batch fit). Then:

* set the **integration window**. The window *is* the time-spectrum fit range:
  drag the shaded range directly on the time plot, or type precise
  :math:`t_\mathrm{min}` and :math:`t_\mathrm{max}` values (in μs) into the
  spinboxes in the **Integration window** group of the build panel. The two stay
  in sync;
* press **Build Scan**.

Each selected run is integrated over the window to one point, and the scan is
plotted in the **Parameters** dock, which is raised automatically. Runs that are
missing the chosen x-axis log (for example, a run with no recorded field on a
field scan) are dropped and listed in the log. Re-pressing **Build Scan** after
changing the window rebuilds the scan in place — it does not accumulate copies.

Step 2 — Choose the x-axis and (optionally) the differential view
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Above the scan plot:

* the **x-axis selector** chooses what the scan is plotted against — **B (G)**
  (field, the usual ALC axis), **T (K)** (temperature), or **Run**;
* the **dA/dB** checkbox switches the plot to the field-derivative of the scan
  (relabelled **dA/dT** or **dA/d(run)** to match the chosen axis). This is the
  *differential ALC* view and has one fewer point than the raw scan (each point
  is the midpoint slope between adjacent runs);
* the **Data table…** button opens the per-point values (run, x, asymmetry,
  error) in a separate window, keeping the plot itself uncluttered.

Changing the x-axis clears any baseline and peaks, because regions and peak
positions are expressed in the units of the current axis.

Step 3 — Fit a baseline
~~~~~~~~~~~~~~~~~~~~~~~~~

The **Baseline** section subtracts the non-resonant background so that only the
resonance remains. The workflow mirrors Mantid's two-step ALC analysis:

* choose a baseline **Model** — **Linear** or **Constant**;
* mark the **non-resonant regions** (the parts of the scan *away* from the
  resonance). Press **+ region** to add a row, then either type the
  ``start``/``end`` values or **drag the shaded region edges** directly on the
  plot. Add as many regions as you need; **− region** removes the selected one;
* press **Fit baseline**.

The fitted baseline is drawn over the scan (red line) and stored as the
baseline-corrected curve used for peak fitting. Editing a region — by typing or
by dragging an edge — marks the baseline stale and removes the overlay, so you
always know when a re-fit is needed.

Step 4 — Fit the resonance peaks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The **Peaks** section fits one or more centred peaks to the baseline-corrected
scan:

* press **+ Gaussian** or **+ Lorentzian** to add a peak. Each peak row carries
  an initial **B0** (centre), **Width**, and **Amp** (amplitude) guess; edit them
  in the table, or **drag the peak-centre marker** (the dashed line) on the plot
  to position :math:`B_0`. **− peak** removes the selected peak;
* press **Fit peaks**.

The two shapes share the same parameters,

.. math::

   \mathrm{Gaussian:}\; f\,e^{-(B-B_0)^2 / 2B_\mathrm{wid}^2}
   \qquad
   \mathrm{Lorentzian:}\; \frac{f}{1 + \left((B-B_0)/B_\mathrm{wid}\right)^2},

so peaks of either shape can be mixed in one fit. The total fit (baseline plus
all peaks) is overlaid on the scan, and the peaks table updates to the fitted
values.

Reading off the results
~~~~~~~~~~~~~~~~~~~~~~~~~

Below the peaks table, a summary line reports, for each peak, the resonance
field :math:`B_0` with its uncertainty, the full width at half maximum, and the
amplitude — for example::

   Peak 1 (Gaussian): B₀ = 3104 ± 1.7 G, FWHM = 553.3 G, amp = -3.32 %

The FWHM is derived from the fitted width per shape
(:math:`\mathrm{FWHM} = 2\sqrt{2\ln 2}\,B_\mathrm{wid} \approx 2.355\,B_\mathrm{wid}`
for a Gaussian; :math:`2\,B_\mathrm{wid}` for a Lorentzian).

Saving and reopening
--------------------

The scan and its full analysis are saved inside the project (``.asymp``) file.
Reopening the project rebuilds the scan, restores the x-axis choice, the baseline
regions, and the peaks, re-runs whichever fits were active so the overlays and
read-out reappear, and re-enters ALC mode if it was active when you saved — so
you resume exactly where you left off. See :doc:`project_files`.

Common pitfalls
---------------

* **ALC mode is greyed out.** It is only available in the Forward–Backward
  asymmetry view. Switch the central plot to that view first.
* **A run is missing from the scan.** Field-axis scans drop runs with no recorded
  field (and likewise for temperature); check the log for the list of excluded
  runs, or switch the x-axis to **Run**.
* **"Add at least one non-resonant region".** The baseline needs at least one
  region, and a **Linear** baseline needs at least two usable points across the
  marked regions. Widen the regions or use a **Constant** baseline.
* **Regions or peaks disappeared.** Changing the x-axis clears them, because they
  are expressed in the units of the axis they were drawn on.
* **The baseline or fit overlay vanished after an edit.** Editing a region or
  dragging a handle marks the fit stale on purpose; press **Fit baseline** (and
  then **Fit peaks**) again.

Further reading
---------------

* F. L. Pratt, *WiMDA: a muon data analysis program for the Windows PC*,
  Physica B **289–290**, 710 (2000) — the count-integral ALC this mode follows.
* The Mantid ``PlotAsymmetryByLogValue`` algorithm — the alpha-aware
  integral/differential field-scan reference.

See also
--------

* :doc:`fitting` — time-domain model fitting (the per-spectrum alternative).
* :doc:`parameter_trending` — trending fitted parameters across a run series.
* :doc:`detector_grouping` — the F/B grouping and :math:`\alpha` used by the
  integral.
* :doc:`project_files` — what is saved in a project and how it is restored.
