Core data structures
====================

.. py:module:: asymmetry.core

.. py:module:: asymmetry.core.workflow


.. currentmodule:: asymmetry.core.data

The core data layer carries μSR measurements through the fit and
analysis pipeline. :class:`~asymmetry.core.data.dataset.MuonDataset` is
the working representation — a time axis, an asymmetry, and an
uncertainty, plus a metadata dictionary and a reference back to the
:class:`~asymmetry.core.data.dataset.Run` it was computed from. The
:class:`~asymmetry.core.data.dataset.Run` carries the raw per-detector
:class:`~asymmetry.core.data.dataset.Histogram` records and the
grouping payload (see :doc:`/reference/detector_grouping`) and is
what :func:`asymmetry.core.io.load` returns for raw-histogram files.
The :class:`~asymmetry.core.data.logbook.Logbook` collects multiple
runs into a searchable, tag-filterable run table (see
:doc:`/reference/logbook`), and the instrument-geometry classes below
define the detector banks and preset groupings for the supported
spectrometers.

Dataset
-------

.. autoclass:: asymmetry.core.data.dataset.MuonDataset
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: asymmetry.core.data.dataset.Run
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: asymmetry.core.data.dataset.Histogram
   :members:
   :undoc-members:
   :show-inheritance:

Effective reduction recipe
--------------------------

``MuonDataset.effective_recipe`` is independent of its transient
``histogram_observation``. Historical or unsupported curves have no captured raw
provenance. JSON conversion does not grant replay or covariance reconstruction.
Source locators are retained in the view and excluded from numerical identity;
``publishable()`` returns a detached numerical disclosure without nested local
locators or diagnostic prose and preserves numerical identity. Workdir and scan
codecs retain the local view. Project result metadata and fitted-input summaries
retain captured history separately from recomputed curves. A fit-input digest
records sampled values, errors, and support; recipe identity alone cannot verify
those observations.

.. autoclass:: asymmetry.core.data.reduction_recipe.EffectiveReductionRecipe
   :members: canonical_json, identity, to_dict, from_dict, publishable, scalar_units

Logbook
-------

.. autoclass:: asymmetry.core.data.logbook.LogbookEntry
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: asymmetry.core.data.logbook.Logbook
   :members:
   :undoc-members:
   :show-inheritance:


Scripted grouping preparation
-----------------------------

See :ref:`preset-scripting-example` for synthetic loading, preset application,
calibration, and reduction, and :doc:`/reference/detector_grouping` for policy.

.. autofunction:: asymmetry.core.workflow.apply_grouping_preset

Scripted calibration packing
----------------------------

Use ``CalibrationOptions.count_packing`` to request fixed TF count packing
independently of science reduction. See :ref:`calibration-count-packing` for
the public-call example, inherited-source behaviour, and quick-method limits.

The request and operation are documented in :doc:`fitting/calibration`.

Scripted temperature selection
------------------------------

See :ref:`logged-temperature-selection` for channel, statistic, unit, and failure
semantics and :ref:`temperature-selection-example` for a synthetic example.

.. autofunction:: asymmetry.core.workflow.select_logged_temperature

Instrument geometry
-------------------

.. autoclass:: asymmetry.core.instrument.DetectorSegment
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: asymmetry.core.instrument.BankLayout
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: asymmetry.core.instrument.GroupDefinition
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: asymmetry.core.instrument.PresetGrouping
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: asymmetry.core.instrument.ReferenceArrow
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: asymmetry.core.instrument.InstrumentLayout
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: asymmetry.core.instruments.geometry.BeamDirection
   :members:

.. autoclass:: asymmetry.core.instruments.geometry.ViewOrientation
   :members:

.. autoclass:: asymmetry.core.instruments.geometry.BeamProjection
   :members:

.. autofunction:: asymmetry.core.instruments.geometry.project_beam

.. autofunction:: asymmetry.core.instrument.get_instrument_layout

.. autofunction:: asymmetry.core.instrument.detect_instrument

.. autofunction:: asymmetry.core.instruments.resolution.resolve_instrument

.. autoclass:: asymmetry.core.instruments.resolution.RecordedInstrumentChoice
   :members: to_dict, from_dict

.. autofunction:: asymmetry.core.instruments.resolution.choice_for_layout

.. autofunction:: asymmetry.core.project.profiles.choose_instrument_for_run

Simulation
----------

Synthetic-run generation and statistics degradation (see
:doc:`/reference/simulation`): Poisson draws of expected per-detector
counts from an instrument template — a loaded run or a built-in idealised
instrument — exact binomial thinning of measured runs, per-group amplitude
and phase simulation, and the promoted screenshot-archetype builders.

.. automodule:: asymmetry.core.simulate
   :members:
   :undoc-members:

Archetype presets
~~~~~~~~~~~~~~~~~~

One-click textbook archetypes (Ag Kubo–Toyabe, EuO T-scan, F-μ-F, YBCO)
that generate badged synthetic runs through the simulate pipeline.

.. automodule:: asymmetry.core.simulate_presets
   :members:
   :undoc-members:

Pull-distribution diagnostic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Simulate and refit under an explicit ideal-count experiment, retaining seeds,
outcomes, source/synthetic recipes, and descriptive usable-subset pulls.

.. automodule:: asymmetry.core.pull_diagnostic
   :members:
   :undoc-members:

.. automodule:: asymmetry.core.pull_request
   :members: PullRequest, make_pull_request

.. automodule:: asymmetry.core.pull_refit
   :members: EngineRefit, PullFitInput

.. _physical-constants-api:

Physical constants
------------------

.. automodule:: asymmetry.core.utils.constants
   :members:
   :member-order: bysource

Run arithmetic
--------------

``asymmetry.core.data.combine.combine_runs`` accepts finite signed ``scales``.
The first coefficient equals ``scales[0]``; later coefficients equal
``sign * scales[i]``. Counts combine linearly and independent variances use
squared coefficients, retained in ``Histogram.count_variance``. NaN denotes
explicitly unknown per-bin noise; only a consumed unknown bin blocks preparation.
Zero terms are
recorded but excluded from operation domains; nonfinite/all-zero requests and
known direct reuse raise ``CombineError``. Corresponding period payloads use
the same coefficients; mixed/mismatched payloads require explicit selection.
``reduce_combined_run`` prepares each retained acquisition using its own correction
facts and resolved clock, then weights and pools on common observed good support.
The returned dataset records an immutable counts-first replay snapshot; the raw
inspection sum alone does not authorise reconstruction. Shared group background
is applied once. See :doc:`/reference/run_arithmetic` for exposure and likelihood
eligibility. Ordinary GUI co-add remains unit weighted.
Core weighted Runs are transient in-memory results; saved reduced results and recipes
do not supply histogram reconstruction. A project/browser definition that cannot
reproduce its saved coefficients or selected periods retains its observations or
unavailable state and history, including after a later source rebuild.

Calibration
-----------

.. autodata:: asymmetry.core.data.calibration.WEAK_TF_FIELD_RANGE_GAUSS
   :no-value:

.. autofunction:: asymmetry.core.data.calibration.classify_tf_calibration_run

Run combination
---------------

.. autoclass:: asymmetry.core.data.combine.CombineError
   :no-members:
   :no-special-members:

.. autofunction:: asymmetry.core.data.combine.combine_runs

Ratio validity
--------------

.. autoclass:: asymmetry.core.data.ratio_validity.RatioValidity
   :no-members:
   :no-special-members:

Instrument
----------

.. autodata:: asymmetry.core.instrument.INSTRUMENT_NAMES
   :no-value:

.. autofunction:: asymmetry.core.instrument.instrument_choices_for

Global fit studies
------------------

.. autoclass:: asymmetry.core.representation.global_fit_study.GlobalFitStudy
   :no-members:
   :no-special-members:

.. autofunction:: asymmetry.core.representation.global_fit_study.compute_group_input_digest

Joint fit records
-----------------

.. autoclass:: asymmetry.core.representation.joint_fit.JointFit
   :no-members:
   :no-special-members:

Fit series
----------

.. autoclass:: asymmetry.core.representation.series.FitSeries
   :no-members:
   :no-special-members:

.. py:attribute:: FitSeries.member_run_numbers
   :module: asymmetry.core.representation.series

   Ordered integer member keys: run numbers or synthetic detector-group keys.

.. py:attribute:: FitSeries.member_source_run
   :module: asymmetry.core.representation.series

   Source acquisition for each member key in a detector-group series.

Run surveys
-----------

.. autodata:: asymmetry.core.workflow.survey.LARMOR_FREQUENCY_TOLERANCE
   :no-value:

.. autodata:: asymmetry.core.workflow.survey.PRECESSION_SNR_FLOOR
   :no-value:

.. autoclass:: asymmetry.core.workflow.survey.ScanGroup
   :no-members:
   :no-special-members:

Work directories
----------------

.. automethod:: asymmetry.core.workflow.workdir.WorkDir.bind

.. autoclass:: asymmetry.core.workflow.workdir.WorkDirMismatchError
   :no-members:
   :no-special-members:
