Loading Data
============

Asymmetry supports WiMDA and ISIS muon NeXus files through a common API.

Supported Formats
-----------------

WiMDA Format (.wim)
~~~~~~~~~~~~~~~~~~~

WiMDA files are loaded with ``WimLoader`` or the convenience ``load`` function.

ISIS Muon NeXus (.nxs, .nexus)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Asymmetry supports ISIS muon NeXus files (legacy V1 and modern V2), including
multi-period runs. Multi-period files return a list of ``MuonDataset`` values.

Basic Usage
-----------

Loading a Single File
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from asymmetry.core.io import load
   
   dataset = load("path/to/data.wim")
   print(dataset.summary())

The same ``load`` API can be used with NeXus files:

.. code-block:: python

      dataset_or_periods = load("path/to/HIFI00206453.nxs")
      if isinstance(dataset_or_periods, list):
         print(f"Loaded {len(dataset_or_periods)} periods")

The returned ``MuonDataset`` contains:

* ``time``: Time axis in microseconds
* ``asymmetry``: Asymmetry values
* ``error``: Error bars
* ``metadata``: Run metadata (temperature, field, etc.)
* ``run``: Reference to the original Run object

Accessing Metadata
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   dataset = load("data.wim")
   
   print(f"Run number: {dataset.run_number}")
   print(f"Temperature: {dataset.metadata.get('temperature')} K")
   print(f"Field: {dataset.metadata.get('field')} G")
   print(f"Title: {dataset.metadata.get('title')}")

Loading Multiple Files
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pathlib import Path
   
   data_dir = Path("data")
   datasets = []
   
   for wim_file in data_dir.glob("*.wim"):
       dataset = load(str(wim_file))
       datasets.append(dataset)
   
   print(f"Loaded {len(datasets)} datasets")

Direct File Format Access
--------------------------

For advanced users, you can access the low-level file loaders:

.. code-block:: python

   from asymmetry.core.io.wim import WimLoader
   
   loader = WimLoader()
   dataset = loader.load("data.wim")
   print(dataset.run_number)

For NeXus files:

.. code-block:: python

   from asymmetry.core.io.nexus import NexusLoader

   loader = NexusLoader()
   dataset_or_periods = loader.load("HIFI00206453.nxs")

Loader Registry and Custom Formats
----------------------------------

``LoaderRegistry`` maps file extensions to loader classes.

.. code-block:: python

   from asymmetry.core.io.base import LoaderRegistry

   print(LoaderRegistry.supported_extensions())
   print(LoaderRegistry.file_dialog_filter())

You can register custom loader classes at runtime for additional formats.

Runnable Examples
-----------------

See the executable scripts:

* ``examples/basic_dataset_loading.py``
* ``examples/custom_loader.py``

