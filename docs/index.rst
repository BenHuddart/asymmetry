Asymmetry Documentation
========================

**Asymmetry** is a Python library for muon-spin spectroscopy (μSR) data analysis,
providing tools for loading, processing, analyzing, and visualizing μSR data.

.. warning::

   **Verify the physics in this documentation against trusted sources.**

   Asymmetry is in an early, **alpha phase** of development, and this
   documentation is written alongside it. These pages describe µSR data,
   fitting-function forms, and analysis workflows to help you use the
   software — but they are **not a substitute for the primary literature or
   established analysis tools**. Statements about µSR data, the meaning and
   parameterisation of fitting models, and the correctness of a given
   workflow may be incomplete, simplified, or in error.

   Before relying on any physical interpretation, model definition, or
   procedure described here, **confirm it against peer-reviewed references
   and an established µSR tool** (for example WiMDA, Musrfit, or Mantid).
   Please report anything you find to be inaccurate.

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
