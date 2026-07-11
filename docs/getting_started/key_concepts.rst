Key concepts
============

Asymmetry organises an analysis around a small number of objects. They map onto
the physics of a muon experiment rather than onto files or windows, and the same
objects appear whether you drive the program from the graphical interface or
script against the :mod:`asymmetry.core` API. This page introduces them once, so
that the rest of the documentation can refer to them without ceremony. For the
underlying physics, see :doc:`/explanation/musr_primer`.

Runs and datasets
-----------------

A **run** is the raw output of one measurement: a set of per-detector time
histograms together with the metadata that describes how they were collected —
temperature, applied field, detector geometry, time-zero, the good-bin range, and
so on. Loading a file with :func:`~asymmetry.core.io.load` returns a **dataset**
(a :class:`~asymmetry.core.data.dataset.MuonDataset`), which wraps the run and
carries the reduction choices applied to it; the raw histograms themselves live
on ``ds.run``. A multi-period measurement loads as a list of datasets, one per
period. Provenance travels with the dataset — every correction is recorded rather
than baked into the histograms, so a saved analysis reproduces the same numbers
when it is reopened.

Detector grouping
-----------------

The individual detectors are rarely analysed one at a time. **Grouping** combines
them into a few channels — most often a *forward* and a *backward* group on
opposite sides of the initial muon spin — by summing the histograms of the
detectors assigned to each group. The grouping is a property of the analysis, not
of the data: detectors can be regrouped, excluded, or arranged into the named
groups a particular instrument expects, and the raw histograms are never altered.
See :doc:`/reference/detector_grouping`.

The asymmetry
-------------

The quantity of interest is the **asymmetry**, the normalised difference between
two opposed groups. For a forward count :math:`F(t)` and a backward count
:math:`B(t)`,

.. math::

   A(t) = \frac{F(t) - \alpha B(t)}{F(t) + \alpha B(t)},

where :math:`\alpha` is a calibration constant that corrects for the unequal
counting efficiency and solid angle of the two groups. Forming the asymmetry divides out
the exponential muon decay and leaves the time evolution of the muon spin
polarisation — the signal that every downstream fit and transform works on.
:math:`\alpha` is usually calibrated once, from a transverse-field run in which
the asymmetry is known to oscillate symmetrically about zero; see
:doc:`/reference/grouping_calibration`.

The reduction order
-------------------

The corrections that take raw histograms to an asymmetry are applied in a fixed
order — **deadtime → background → grouping → asymmetry** — because each step
assumes the previous one has been done. Deadtime correction restores the true
count rate where the instantaneous rate is high (PSI and ROOT histograms need it;
ISIS NeXus data arrive already corrected); background subtraction removes the
time-independent floor; only then are the detectors grouped and the asymmetry
formed. The reduction layer is documented in
:doc:`/reference/data_reduction/index`.

Representations
---------------

Most analyses look at the same run in more than one way: as a forward–backward
asymmetry, as a set of individual detector groups, or as a frequency spectrum.
Asymmetry calls these **representations**, and they are first-class — a fit, its
parameters, and any resulting trend all belong to the representation they were
made in, and switching representation switches the analysis that goes with it.
The parameter-trend panel, for example, shows the series for whichever
representation is active.

Fits, models, and components
----------------------------

A **fit** describes a representation with a **model** and a set of fitted
parameters. Models are assembled from **components** — normalised building blocks
such as ``Exponential``, ``Gaussian``, or an oscillation — combined with
arithmetic and fraction groups in an expression like
``Gaussian * Exponential + Constant``. Component names are the ones used inside
expressions; the standalone single-component models in the ``MODELS`` registry
carry longer names, a distinction that matters mainly when scripting and is
spelled out in the :doc:`/reference/cookbook`. Fitting is covered in
:doc:`/reference/fitting`.

Trends and series
-----------------

Fitting the same model across a series of runs — a temperature scan, a field
sweep, an angular rotation — produces a **trend**: a fitted parameter followed as
a function of an external variable and recorded as a
:class:`~asymmetry.core.fitting.FitSeries`. A trend is itself fittable, so the
order parameter of a magnetic transition or the penetration depth of a
superconductor is extracted by fitting a model to the trend rather than to any
single spectrum. See :doc:`/reference/parameter_trending`.

A run-membered series belongs to a **data group** — the named run collection
you organise in the Data Browser — rather than owning a frozen list of run
numbers of its own: the group is the batch vehicle, and the series' effective
membership tracks the group's live membership (minus any run the series has
individually excluded). A run may sit in more than one group at once, and a
group may carry more than one series (the same scan fit two different ways).
See :doc:`/reference/gui_usage` for the Data Browser's grouping controls.

Projects
--------

A complete session — the loaded datasets, their grouping and reduction choices,
the fits, the trends, and the state of the browser and plot — saves to a single
``.asymp`` **project** file and reopens exactly as it was left. See
:doc:`/reference/project_files`.
