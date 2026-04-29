Asymmetry Documentation
========================

**Asymmetry** is a Python library for muon-spin spectroscopy (μSR) data analysis, 
providing tools for loading, processing, analyzing, and visualizing μSR data.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   user_guide/index
   api/index
   contributing

Features
--------

* **Data Loading**: Support for NeXus, PSI, and ROOT formats with extensible I/O system
* **Data Processing**: Rebinning, grouping, and asymmetry calculation
* **Fourier Analysis**: FFT and Maximum Entropy for frequency-domain analysis
* **Fitting**: Integration with lmfit for flexible model fitting
* **Interactive GUI**: Qt-based interface with:
  
  - Multi-file loading and management
  - Sortable and filterable data browser
  - Co-adding of datasets with proper error propagation
  - Interactive plotting with zoom and axis controls
  - Data bunching/rebinning
  
Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/BenHuddart/asymmetry.git
   cd asymmetry
   python -m pip install -c constraints.txt ".[gui,hdf5,root,gle]"

Loading Data
~~~~~~~~~~~~

.. code-block:: python

   from asymmetry.core.io import load
   
   # Load a NeXus file
   dataset = load("mydata.nxs")
   print(dataset.summary())

Using the GUI
~~~~~~~~~~~~~

.. code-block:: bash

   asymmetry-gui

Or from Python:

.. code-block:: python

   from asymmetry.gui.app import main
   main()

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
