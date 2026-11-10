.. _period-mapping:

Period mapping
==============

Pulsed-source runs can interleave acquisition conditions frame by frame —
laser on/off at half the muon pulse rate in photo-μSR, RF on/off
alternating every few seconds, stepped delays in pump–probe sequences. Each
condition accumulates into its own *period* histogram set within one file,
retaining its own beam exposure bookkeeping. Two-period files load as a
single dataset with the Red/Green selector (Red, Green, G−R, G+R) in the
grouping dialog; this page covers runs with **more than two** periods.

The **Map periods…** button (shown beside the Red/Green selector when the
reference run has three or more periods) opens a matrix: one row per
period, with its good-frame count, and a three-way choice

=========  =====================================================
Red        summed into the red set
Green      summed into the green set
Ignore     left out entirely
=========  =====================================================

On Apply, a combined dataset is created whose red and green sets are the
count-level sums of the chosen periods (independent supplied variances added,
raw Poisson counts added when no variance is supplied, good frames added per set), and the full Red/Green machinery applies to it:
plot either set, or the G−R difference that isolates the effect of the
stimulus. The mapping is recorded with the grouping for provenance.
Defaults follow WiMDA's period-mapping form: period 1 to red, period 2 to
green, the rest ignored; with no green periods the result behaves as a
single-set run and the G±R modes do not apply.

*When to use this.* Multi-condition sequences where several periods share a
condition — e.g. a laser-delay scan where periods 1 and 3 are light-off and
2 and 4 are light-on at different delays: map {1, 3} → red and {2} → green
to compare one delay against the off reference, then remap for the next
delay without reloading anything. Each period retains its own measured
good-frame count. G±R combines independently reduced asymmetries; it does not
assume equal exposure or subtract raw counts without normalisation.
Summing periods with *different* conditions into one set is statistically
fine but physically meaningless — the sums are labelled by your mapping,
not by what the hardware did, so record the period legend (the acquisition
log's period table) with the analysis.

The rows identify stored data periods, numbered from one, rather than every
slot in the acquisition's DAE sequence. Declared NeXus ``period_type`` and
``period_labels`` remain inspectable in file metadata. When the declared data
slots correspond to the stored arrays, each selected period records its DAE
slot separately; absent or inconsistent declarations record that limitation.
The dialog does not add dwell rows or fix them to **Ignore**. Zero counts do
not establish a dwell interval, and ROOT detector banks are not inferred to be
acquisition periods.

Original acquisitions, including ignored ones, retain their counts, supplied
variances, timing, good-bin limits, exposure, and file coefficients. Mapping
leaves the individually loaded observations available for another selection or
mapping. A red-only mapping reduces and can be selected as period 1; a green
selection or G±R operation requires an actual green set.

Mapped sets pool raw-index counts before the selected set's correction. Timing
follows the first contributor; this is not alignment of different period clocks.
Point errors retain first-order propagation at fixed α/β, with existing shared
background terms. This does not provide full cross-period/reference covariance.
Asymmetry also checks that detector lists correspond and that signed input has
supplied variance before independent pooling. An unsupported sum raises an
operation error; the original observations remain available.

Reference-background selection
------------------------------

A **Background run** chosen from a loaded period retains that acquired period
when loaded from its source file or reopened in a project. Period exposure is
selected with its histograms, and need not equal another period's exposure.
Distinct acquired periods in one file may serve as independent inputs;
selecting the identical observations as an independent reference is refused.
See :doc:`backgrounds` for the selection and uncertainty contract.

Compared with other programs
----------------------------

WiMDA 1.326 maps acquired periods into red/green count sets and exposes declared
dwell slots. Mantid's documented period workspaces retain separate observations;
musrfit 1.12.1 addresses period histogram families explicitly. These are selection
and representation comparisons, not equivalence of exposure or error models.
Asymmetry's current G+R adds independently reduced asymmetries; WiMDA's G+R forms
one pooled-count ratio. See :doc:`/explanation/comparison`.

**References**

- S. J. Blundell, R. De Renzi, T. Lancaster, and F. L. Pratt, *Muon
  Spectroscopy: An Introduction* (Oxford University Press, Oxford, 2022). —
  photoexcitation and RF alternating-frame acquisition.
- F. L. Pratt, Physica B **289–290**, 710 (2000).

- `WiMDA overview <https://shadow.nd.rl.ac.uk/wimda/>`_.
- Mantid, `MuonPreProcess v1 <https://docs.mantidproject.org/nightly/algorithms/MuonPreProcess-v1.html>`_
  and `LoadAndApplyMuonDetectorGrouping v1 <https://docs.mantidproject.org/nightly/algorithms/LoadAndApplyMuonDetectorGrouping-v1.html>`_
  (6.16 development documentation).
- musrfit, `User manual <https://lmu.pages.psi.ch/musrfit-docu/user-manual.html>`_
  (1.12.1).
