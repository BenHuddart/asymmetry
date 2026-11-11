Input/output
=============

.. currentmodule:: asymmetry.core.io

The I/O subsystem is a small registry of format-specific loaders
behind a single entry point: :func:`asymmetry.core.io.load` infers the
format from the file extension and dispatches to the appropriate
:class:`~asymmetry.core.io.base.BaseLoader` subclass. The current
loaders are :class:`~asymmetry.core.io.nexus.NexusLoader` (ISIS muon
NeXus V1 and V2, including multi-period files, which return a list of
:class:`~asymmetry.core.data.dataset.MuonDataset`),
:class:`~asymmetry.core.io.psi.PsiLoader` (PSI BIN and MDU raw
histograms, with discovery of PSI ``.mon`` temperature sidecars), and
:class:`~asymmetry.core.io.root.RootLoader` (MusrRoot and LEM ROOT
files, including slow-control histograms). The format-specific quirks
each loader handles — bin-index conventions, per-detector :math:`t_0`,
deadtime and background metadata — are documented in
:doc:`/reference/loading_data`. Custom formats can be added by
subclassing :class:`~asymmetry.core.io.base.BaseLoader` and
registering it with the
:class:`~asymmetry.core.io.base.LoaderRegistry`.

Main interface
--------------

.. autofunction:: asymmetry.core.io.load

Period selection
----------------

Period-mode (multi-period) runs — photo-μSR light-OFF/ON, RF on/off, ALC —
expose a scriptable selection API in :mod:`asymmetry.core.io.periods`. The same
helpers back the GUI red/green ("RG") selector, so scripts and the desktop app
share one implementation. See :ref:`selecting-periods` for a worked example.

.. autofunction:: asymmetry.core.io.periods.select_period

.. autofunction:: asymmetry.core.io.periods.period_count

.. autofunction:: asymmetry.core.io.periods.period_labels

.. autofunction:: asymmetry.core.io.periods.resolve_period_index

.. autofunction:: asymmetry.core.io.periods.combine_mapped_periods

.. autofunction:: asymmetry.core.io.periods.sum_period_histograms

.. autofunction:: asymmetry.core.io.periods.combine_period_asymmetry

The arrays-only helper supports G−R. Arrays of separate ratios cannot determine
pooled G+R; use the histogram preparation API with verified acquired counts.
Historical curve-only records remain readable, but a fresh RF pooled scan excludes
them with an explicit reload reason. Count-backed RF scans use current settings;
scan values/errors are fractions, whereas time curves are percentages.

Mapped and arithmetically combined runs retain original acquired ``Run`` records in
``run.grouping["period_inputs"]``. Mappings retain stored input order; run arithmetic
retains source order, then each source's acquired periods. Input ordinal and
original acquired-period identity remain distinct.
``period_mapping`` records authored membership; held histograms are pooled
inspection views. Normal reduction corrects each selected acquisition with its
own coefficient/exposure and aligns its own t0 on common observed support before
pooling. Extracted sets retain this context; selected-set reduction and histogram
replay use the same preparation. ``sum_period_histograms`` remains an explicit
raw-index addition helper. Pass the mapped dataset to ``combine_mapped_periods`` to remap these
retained acquisitions, including ignored periods. The inputs are transient: projects
reconstruct them from the source file and saved mapping. Fitting curve records
remove these replay arrays. NeXus selections distinguish ``data_period_number``
from an evidenced ``dae_period_number`` and retain ``period_declaration``
metadata, including unresolved correspondence.

Registry
--------

.. autoclass:: asymmetry.core.io.base.LoaderRegistry
   :members:
   :undoc-members:
   :show-inheritance:

Loaders
-------

Base loader
~~~~~~~~~~~

.. autoclass:: asymmetry.core.io.base.BaseLoader
   :members:
   :undoc-members:
   :show-inheritance:

NeXus loader
~~~~~~~~~~~~

.. autoclass:: asymmetry.core.io.nexus.NexusLoader
   :members:
   :undoc-members:
   :show-inheritance:

PSI loader
~~~~~~~~~~

Loads PSI BIN/MDU raw histogram files using the musrfit-compatible binary
metadata interpretation. PSI-BIN ``.mon`` temperature sidecars are discovered
and parsed using Mantid ``LoadPSIMuonBin``-compatible rules, then exposed as
``metadata["nexus_time_series"]`` and in the Get Info metadata tables.

.. autoclass:: asymmetry.core.io.psi.PsiLoader
   :members:
   :undoc-members:
   :show-inheritance:

ROOT loader
~~~~~~~~~~~

Loads MusrRoot/LEM ROOT histograms and header metadata. MusrRoot slow-control
histograms under ``histos/SCAnaModule`` are exposed through
``metadata["nexus_time_series"]`` and Get Info, including sample-temperature
logs such as ``hSampleTemperature``. Serialized LEM headers support numeric
v4/v5 calibration; ``metadata["root_format"]`` retains the actual version,
``declared_units`` records its format units, and ``root_histo_numbers`` maps
stored arrays to detector ordinals. See :doc:`/reference/loading_data` for
supported v4 banks and unavailable timing.

.. autoclass:: asymmetry.core.io.root.RootLoader
   :members:
   :undoc-members:
   :show-inheritance:

NeXus writer
------------

Writes synthetic or derived runs (see :doc:`/reference/simulation`) as
loadable ISIS muon NeXus V1 files. Built from the
:class:`~asymmetry.core.data.dataset.Run` alone — any loaded run, PSI and
ROOT formats included, can act as the instrument template — with the
simulation provenance stored in a ``/run/simulation`` group.

.. automodule:: asymmetry.core.io.nexus_writer
   :members:
   :undoc-members:

Run arithmetic metadata
------------------------

``combine_runs`` retains the first caller-ordered input's finite field and
temperature (or ``None``), title, and instrument/facility/beamline identity.
``combination.constituents`` records ordered leaf conditions, source identities,
effective signed coefficients, and the master relationship. ``input_ordinal``
binds a live record to its retained acquisition; inactive authored inputs absent
from the numerical payload use null and remain provenance only. Rebuilding
reads current acquired conditions rather than cached composite summaries. The finite active
ranges are ``temperature_spread`` and ``field_spread``; inactive terms stay
provenance. Composite sources do not inherit a single file identity or cached
single-acquisition diagnostics. ``implicit_period_inputs`` preserves implicit counterpart intent separately
from deliberate period selection/mapping. ``coadd_compatibility_error`` shares the core
correspondence/reuse checks without reducing or scanning counts.

The float-valued ``Run.field`` and ``Run.temperature`` convenience accessors
represent an explicit unknown (``None``) as NaN, so summaries and replay
identities can carry it without a fabricated zero.
