Loading Data
============

Asymmetry supports loading μSR data from various file formats.

Supported Formats
-----------------

WiMDA Format (.wim)
~~~~~~~~~~~~~~~~~~~

The primary format currently supported is WiMDA (.wim) files.

ISIS Muon NeXus (.nxs, .nexus)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Asymmetry also supports ISIS muon NeXus files (legacy V1 and modern V2),
including multi-period runs. For multi-period data, one dataset row is created
per period.

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

Using the Logbook
-----------------

For managing multiple runs, use the ``Logbook`` class:

.. code-block:: python

   from asymmetry.core.data.logbook import Logbook
   from asymmetry.core.io import load
   
   logbook = Logbook()
   
   # Add datasets
   for file in ["run1.wim", "run2.wim", "run3.wim"]:
       dataset = load(file)
       logbook.add(dataset, tags=["low_field"])
   
   # Filter by criteria
   low_temp_runs = logbook.filter(temperature=10.0)
   
   # Search by text
   results = logbook.search("copper")
   
   # Iterate over all entries
   for entry in logbook:
       print(f"Run {entry.run_number}: {entry.title}")

Direct File Format Access
--------------------------

For advanced users, you can access the low-level file loaders:

.. code-block:: python

   from asymmetry.core.io.wim import WimLoader
   
   loader = WimLoader()
   run = loader.load("data.wim")
   
   # Access raw histograms
   for i, hist in enumerate(run.histograms):
       print(f"Histogram {i}: {hist.n_bins} bins")

   For NeXus files:

   .. code-block:: python

      from asymmetry.core.io.nexus import NexusLoader

      loader = NexusLoader()
      dataset_or_periods = loader.load("HIFI00206453.nxs")
