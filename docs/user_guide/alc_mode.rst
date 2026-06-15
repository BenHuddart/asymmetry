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
* the **Fit** dock is replaced by the **Integral scan (ALC)** build panel —
  this is where the **Integration window** and **Build Scan** controls live;
* the **Parameters** dock is replaced by the **ALC scan view** (the scan plot
  plus its baseline and peak controls).

Both docks are raised on entry, but the **Parameters** dock lands *on top*,
showing the placeholder *"Build a scan to see the ALC curve."* The **Build
Scan** button is in the **Fit** dock *behind* it — so until you raise the Fit
dock and build a scan, it looks like nothing happened. Click the **Fit**
toolbar button to bring the build panel forward (Step 1 below).

Toggling ALC mode off restores the normal **Fit** and **Parameters** docks; your
scan and its analysis are kept and reappear when you re-enter ALC mode.

The ALC workflow
----------------

Step 1 — Build the scan
~~~~~~~~~~~~~~~~~~~~~~~~~

The **Build Scan** control is in the **Fit** dock, which ALC mode leaves
*behind* the **Parameters** dock and its *"Build a scan to see the ALC curve"*
placeholder. So the build is a four-step sequence — and raising the Fit dock is
the step most people miss:

#. **Load and multi-select the runs.** Load the field-scan runs, then select
   them all in the **Data Browser**: click the first row and **Shift-click** the
   last (the scan is built from the current selection, exactly like a batch
   fit). The Data Browser resolves the **B (G)** column from each run's field;
   that column becomes the scan's field axis.
#. **Raise the Fit dock.** Click the **Fit** toolbar button to bring the
   **Integral scan (ALC)** build panel to the front, in front of the Parameters
   dock. This is the step that is easy to miss — the placeholder in the
   Parameters dock makes it look like nothing happened, when the **Build Scan**
   button was simply hidden behind it.
#. **Set the integration window.** The window *is* the time-spectrum fit range:
   drag the shaded range directly on the time plot, or type precise
   :math:`t_\mathrm{min}` and :math:`t_\mathrm{max}` values (in μs) into the
   spinboxes in the **Integration window** group of the build panel. The two
   stay in sync.
#. **Press Build Scan.**

Each selected run is integrated over the window to one point, and the scan is
plotted in the **Parameters** dock, which is raised automatically. Runs that are
missing the chosen x-axis log (for example, a run with no recorded field on a
field scan) are dropped and listed in the log. Re-pressing **Build Scan** after
changing the window rebuilds the scan in place — it does not accumulate copies.

As a worked example, the TCNQ ALC scan (31 runs at 350 K, stepped over
2000–5000 G) with an integration window of roughly 0.2–8 μs resolves a clean D1
dip at :math:`B_0 \approx 3100` G. See :doc:`workflows/alc_scan_tcnq`.

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

Fitting a scan from the API: single *and* multiple resonances
-------------------------------------------------------------

The GUI peak fitter above fits centred Gaussian/Lorentzian peaks on a
baseline-corrected scan. The same scan can also be fit programmatically with
:func:`~asymmetry.core.fitting.fit_scan_model`, which fits a model *and* its
background to the raw scan in a single call. The ``model`` argument is flexible::

   fit_scan_model(scan, model, *, parameters=None, initial=None,
                  x_min=None, x_max=None, method="migrad")

* ``model`` accepts a ``str``, a ``list[str]``, or a
  :class:`~asymmetry.core.fitting.ParameterCompositeModel` — so it is **not**
  limited to one resonance. Several resonances plus a polynomial background can
  be fit *simultaneously*; fitting overlapping peaks one at a time is neither
  necessary nor reliable.
* ``initial`` is an override dict (``{name: value}``) layered on the model's
  defaults; ``parameters`` is a full :class:`~asymmetry.core.fitting.ParameterSet`.
  The two are mutually exclusive — pass one or the other, not both.

Single resonance:

.. code-block:: python

   from asymmetry.core.fitting import fit_scan_model

   # scan is a FieldScan (e.g. built by the ALC mode "Build Scan" step,
   # or constructed directly from per-run integral-asymmetry values).
   result = fit_scan_model(scan, "LorentzianLCR")

Multiple resonances share a single fit. The component parameters are numbered
across the expression (``f_1``/``B0_1``/``Bwid_1`` for the first peak,
``f_2``/``B0_2``/``Bwid_2`` for the second, …) and the ``Polynomial``
background contributes ``c0`` … ``c5``:

.. code-block:: python

   from asymmetry.core.fitting import ParameterCompositeModel, fit_scan_model

   # Two Lorentzians on a polynomial background, fit together:
   pcm = ParameterCompositeModel.from_expression(
       "LorentzianLCR + LorentzianLCR + Polynomial"
   )
   print(pcm.param_names)
   # ['f_1', 'B0_1', 'Bwid_1', 'f_2', 'B0_2', 'Bwid_2',
   #  'c0', 'c1', 'c2', 'c3', 'c4', 'c5']

   result = fit_scan_model(
       scan,
       "LorentzianLCR + LorentzianLCR + Polynomial",
       initial={"B0_1": 3000.0, "B0_2": 3300.0},
   )

The motivating case is a multi-resonance ALC spectrum such as the
corannulene-style four-resonance scan, which fits cleanly as one mixed
Gaussian/Lorentzian model on a polynomial background::

   "GaussianLCR + LorentzianLCR + LorentzianLCR + LorentzianLCR + Polynomial"

Because every resonance and the background share one minimisation, overlapping
peaks are deconvolved correctly — which sequential single-peak fits cannot do.
The component parameter names follow the composite-model scheme documented in
:doc:`composite_models`; call ``.param_names`` (as above) before building an
``initial`` override or a ``ParameterSet`` so the names always match.

Repolarisation: a complementary route through parameter trending
----------------------------------------------------------------

For longitudinal-field **decoupling / repolarisation** measurements there are
two complementary ways to reach the same physics, and they sit on opposite
sides of the model boundary:

* **The integral-asymmetry ALC scan** described here reduces each run to one
  integrated number and fits a *phenomenological* baseline plus Gaussian or
  Lorentzian peaks — the right tool for a **resonance** (an avoided level
  crossing or QLCR dip) where you want the resonance field and width without
  committing to a microscopic model.
* **A time-domain parameter trend** fits a model to *each run's* spectrum,
  trends a fitted quantity across the field, and fits a *physical* curve to that
  trend. For the smooth repolarisation of isotropic muonium the
  :ref:`MuRepolarisation <muonium-repolarisation>` model
  (:doc:`parameter_trending`) extracts the hyperfine constant
  :math:`A_\mathrm{hf}` directly from the field-decoupling shape — fit it to an
  initial-asymmetry trend, or to a scan built with this mode's
  integral-asymmetry observable.

In short: use the ALC scan for sharp resonances read off model-free, and the
``MuRepolarisation`` parameter trend for the broad decoupling curve when the
hyperfine constant is the quantity you want. The integral-asymmetry observable
built here feeds both.

Building a scan from one period or run type
-------------------------------------------

A field scan must not mix distinct measurement periods or run types recorded at
the same field, or the resonance positions shift and the fitted hyperfine
constants are biased. Two cases are common:

* **Period mode** (RF-on / RF-off, light-on / light-off, ALC steps): a single
  file holds several periods, and the swept observable is one period — or the
  difference of two.
* **Interleaved calibration runs**: a repolarisation or ALC scan often has
  reference runs (for example a 100 G α-calibration) repeated through the sweep;
  they sit at a different asymmetry level and pollute the scan.

:func:`~asymmetry.core.transform.build_field_scan` takes a ``filter`` predicate
``run -> bool`` for exactly this. It is called with each resolved run before
reduction; runs for which it returns false are dropped and listed in
``scan.excluded`` (reason ``"excluded by filter"``) so nothing is silently lost.
The default ``filter=None`` keeps every run — the historical behaviour.

To split a multi-period *file* into its periods first, select each period
upstream with :func:`asymmetry.core.io.periods.select_period`
(period extraction lives in ``io``; the scan transform stays free of it).

RF-µSR resonance (Green − Red): GUI and API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RF-µSR resonance is the **(Green − Red)** integral-asymmetry observable of a
muoniated radical (Green = RF-off, Red = RF-on, recorded as two periods of each
run): a W-shaped double dip whose two resonance fields give the muon
(:math:`A_\mu`) and proton (:math:`A_p`) hyperfine couplings from one field scan.
It has first-class support on the integral-scan path:

* In ALC mode, tick **RF resonance (Green − Red)** in the build panel before
  **Build Scan**. Each two-period run is reduced to its Green − Red period
  difference and integrated over the window — no hand-assembly of separate
  red/green scans.
* In the scan view, open the **RF resonance (A_µ, A_p)** section, set
  **ν_RF** (the RF frequency, held fixed) and the **A_µ₀ / A_p₀** starting
  guesses, and press **Fit RF resonance**. The fit uses the exact muon + electron
  + proton spin Hamiltonian (``RFResonanceMuP``) and reports :math:`A_\mu`
  (mean dip position) and :math:`A_p` (dip splitting) with uncertainties.

The same two steps from a script use
:func:`~asymmetry.core.io.periods.build_rf_difference_scan` (the Green − Red
scan builder) and :func:`~asymmetry.core.fitting.fit_rf_resonance` (which seeds
the amplitudes/widths/background from the data and holds ν_RF fixed) — worked
example: the benzene scan, DEVA runs 56426–56462:

.. code-block:: python

   from asymmetry.core.io import load
   from asymmetry.core.io.periods import build_rf_difference_scan
   from asymmetry.core.fitting import fit_rf_resonance

   combined = [load(f) for f in rf_files]  # 2-period (red/green) files
   # Green − Red integral scan vs field; non-two-period runs land in `excluded`.
   scan = build_rf_difference_scan(combined, order_key="field", t_min=0.0, t_max=1.0)
   result = fit_rf_resonance(scan, nu_rf=218.5)  # A_µ₀/A_p₀ default to 515/124
   # A_mu ~ 516 MHz, A_p ~ 124 MHz (ground truth 514.8 / 124.6); resonances ~775/865 G.

The two RF resonances sit at ~775 G and ~865 G (splitting ~90 G), recovering the
literature ``A_µ`` and ``A_p``. Because the difference is taken **within** each
run, mixing the RF-on and RF-off periods into one ordinary scan instead would
shift the apparent resonance positions and badly bias the fitted ``A_p``.

The lower-level route — reducing each period separately with
:func:`~asymmetry.core.io.periods.select_period` and subtracting two
``build_field_scan`` results — is still available when you need the individual
red and green scans; ``build_rf_difference_scan`` is the one-call equivalent for
the difference observable.

When the periods appear as *separate runs or datasets* in the series (rather
than periods of one file), ``filter`` selects one directly — for example
``filter=lambda run: run.metadata.get("period_label") == "red"``. And to drop
interleaved calibration runs from a repolarisation scan::

   scan = build_field_scan(
       runs, order_key="field",
       filter=lambda run: run.run_number not in calibration_run_numbers,
   )

Saving and reopening
--------------------

The scan and its full analysis are saved inside the project (``.asymp``) file.
Reopening the project rebuilds the scan, restores the x-axis choice, the baseline
regions, and the peaks, re-runs whichever fits were active so the overlays and
read-out reappear, and re-enters ALC mode if it was active when you saved — so
you resume exactly where you left off. See :doc:`project_files`.

Common pitfalls
---------------

* **"Build a scan to see the ALC curve" — nothing seems to happen.** That
  placeholder is in the **Parameters** dock, which sits *on top* when you enter
  ALC mode. The **Build Scan** button (and the **Integration window**) are in
  the **Fit** dock *behind* it — click the **Fit** toolbar button to raise it,
  set the window, and press **Build Scan**.
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
