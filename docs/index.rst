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

* **Data Loading**: Support for WiMDA (.wim) format with extensible I/O system
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

   pip install asymmetry[gui]

Loading Data
~~~~~~~~~~~~

.. code-block:: python

   from asymmetry.core.io import load
   
   # Load a WiMDA file
   dataset = load("mydata.wim")
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
