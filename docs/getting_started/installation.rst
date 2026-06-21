Installation
============

Asymmetry is not currently published on PyPI. Most users on **Windows** or
**Apple Silicon Macs** should install the pre-built desktop application from
the `GitHub Releases <https://github.com/BenHuddart/asymmetry/releases/latest>`__
page (Windows ``.exe`` installer; macOS ``.dmg``). **Intel Mac** and **Linux**
users should install from the Git repository as described below.

Requirements
------------

Asymmetry requires Python 3.10 or later, and is fully compatible with Python 3.13.

Core Dependencies
~~~~~~~~~~~~~~~~~

* numpy >= 1.24
* iminuit >= 2.0 (for fitting without scipy dependency)

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

For GUI support:

* PySide6 >= 6.5
* matplotlib >= 3.7

For HDF5 support:

* h5py >= 3.8

For HDF4 support (reading legacy ISIS muon NeXus V1 ``.nxs`` files):

* pyhdf >= 0.10

  On **Linux** the ``pyhdf`` wheel bundles the HDF4 C library. On **macOS
  (Apple Silicon)** the PyPI wheel does too. On **Intel Macs** there is no
  PyPI wheel — use conda-forge ``pyhdf`` or build from source. On
  **Windows** the wheel does not bundle the HDF4 runtime: ``pyhdf`` also
  needs ``hdf.dll`` / ``mfhdf.dll`` (e.g. from the conda-forge ``hdf4``
  package or ``packaging/windows/fetch_hdf4_dlls.py``), with
  ``ASYMMETRY_HDF4_DLL_DIR`` pointed at the directory holding them. The
  pre-built **Windows and Apple Silicon macOS** desktop releases bundle HDF4;
  no extra setup is needed there. See :doc:`/reference/loading_data`.

For ROOT file support:

* uproot >= 5.0

For GLE export support:

* gleplot installed from GitHub:

   .. code-block:: bash

       pip install "gleplot @ git+https://github.com/BenHuddart/gleplot.git"

  Recent Asymmetry releases use gleplot's foldered export support by default,
  so install a current git version rather than an older cached wheel.

* The GLE compiler itself (``gle``) needs to be on ``PATH`` for ``.gle``
  scripts to be compiled to PDF or EPS at export time. Without it, the
  scripts and sidecar data files are still written and can be compiled
  later. In a frozen macOS app the inherited ``PATH`` is minimal, so
  Asymmetry also looks for ``gle`` in the standard Homebrew, MacPorts,
  and QGLE locations. Override the search from **Setup → GLE Setup…**
  in the main menu; the chosen path is persisted across sessions.

Installation Methods
--------------------

From a local repository checkout (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/BenHuddart/asymmetry.git
   cd asymmetry

   # Full end-user feature set (GUI + optional I/O/export support)
   python -m pip install -c constraints.txt ".[gui,hdf5,hdf4,root,gle]"

Other options from the same checkout:

.. code-block:: bash

   # Core library only
   python -m pip install -c constraints.txt .

   # GUI support only
   python -m pip install -c constraints.txt ".[gui]"

   # All optional dependencies, including development tools
   python -m pip install -c constraints.txt ".[all]"

Install directly from GitHub with pip
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Core library
   python -m pip install "git+https://github.com/BenHuddart/asymmetry.git"

   # Full end-user feature set (no development extras)
   python -m pip install "asymmetry[gui,hdf5,hdf4,root,gle] @ git+https://github.com/BenHuddart/asymmetry.git"

Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~

For development work:

.. code-block:: bash

   git clone https://github.com/BenHuddart/asymmetry.git
   cd asymmetry
   python -m pip install -c constraints.txt -e ".[all]"

   # Run tests
   python -m pytest

Verifying Installation
----------------------

.. code-block:: python

   import asymmetry
   print(asymmetry.__version__)
   
   # Test GUI availability
   from asymmetry.gui.app import main
   # main()  # Uncomment to launch GUI
