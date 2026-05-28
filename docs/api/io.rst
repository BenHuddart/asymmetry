Input/Output
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
:doc:`/user_guide/loading_data`. Custom formats can be added by
subclassing :class:`~asymmetry.core.io.base.BaseLoader` and
registering it with the
:class:`~asymmetry.core.io.base.LoaderRegistry`.

Main Interface
--------------

.. autofunction:: asymmetry.core.io.load

Registry
--------

.. autoclass:: asymmetry.core.io.base.LoaderRegistry
   :members:
   :undoc-members:
   :show-inheritance:

Loaders
-------

Base Loader
~~~~~~~~~~~

.. autoclass:: asymmetry.core.io.base.BaseLoader
   :members:
   :undoc-members:
   :show-inheritance:

NeXus Loader
~~~~~~~~~~~~

.. autoclass:: asymmetry.core.io.nexus.NexusLoader
   :members:
   :undoc-members:
   :show-inheritance:

PSI Loader
~~~~~~~~~~

Loads PSI BIN/MDU raw histogram files using the musrfit-compatible binary
metadata interpretation. PSI-BIN ``.mon`` temperature sidecars are discovered
and parsed using Mantid ``LoadPSIMuonBin``-compatible rules, then exposed as
``metadata["nexus_time_series"]`` and in the Get Info metadata tables.

.. autoclass:: asymmetry.core.io.psi.PsiLoader
   :members:
   :undoc-members:
   :show-inheritance:

ROOT Loader
~~~~~~~~~~~

Loads MusrRoot/LEM ROOT histograms and header metadata. MusrRoot slow-control
histograms under ``histos/SCAnaModule`` are exposed through
``metadata["nexus_time_series"]`` and Get Info, including sample-temperature
logs such as ``hSampleTemperature``.

.. autoclass:: asymmetry.core.io.root.RootLoader
   :members:
   :undoc-members:
   :show-inheritance:
