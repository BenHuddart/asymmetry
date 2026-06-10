Core Data Structures
====================

.. currentmodule:: asymmetry.core.data

The core data layer carries μSR measurements through the fit and
analysis pipeline. :class:`~asymmetry.core.data.dataset.MuonDataset` is
the working representation — a time axis, an asymmetry, and an
uncertainty, plus a metadata dictionary and a reference back to the
:class:`~asymmetry.core.data.dataset.Run` it was computed from. The
:class:`~asymmetry.core.data.dataset.Run` carries the raw per-detector
:class:`~asymmetry.core.data.dataset.Histogram` records and the
grouping payload (see :doc:`/user_guide/detector_grouping`) and is
what :func:`asymmetry.core.io.load` returns for raw-histogram files.
The :class:`~asymmetry.core.data.logbook.Logbook` collects multiple
runs into a searchable, tag-filterable run table (see
:doc:`/user_guide/logbook`), and the instrument-geometry classes below
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

Instrument Geometry
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

.. autofunction:: asymmetry.core.instrument.get_instrument_layout

.. autofunction:: asymmetry.core.instrument.detect_instrument

Simulation
----------

Synthetic-run generation and statistics degradation (see
:doc:`/user_guide/simulation`): Poisson draws of expected per-detector
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

Re-simulate-and-refit a completed fit over many seeds to test that the
analysis chain's error bars are calibrated (pulls ~ :math:`N(0, 1)`).

.. automodule:: asymmetry.core.pull_diagnostic
   :members:
   :undoc-members:
