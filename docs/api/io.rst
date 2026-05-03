Input/Output
=============

.. currentmodule:: asymmetry.core.io

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
