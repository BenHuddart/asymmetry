Installation
============

Asymmetry is not currently published on PyPI. Install from the Git repository.

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

For ROOT file support:

* uproot >= 5.0

For GLE export support:

* gleplot installed from GitHub:

   .. code-block:: bash

       pip install "gleplot @ git+https://github.com/BenHuddart/gleplot.git"

Installation Methods
--------------------

From a local repository checkout (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/BenHuddart/asymmetry.git
   cd asymmetry

   # Full end-user feature set (GUI + optional I/O/export support)
   python -m pip install -c constraints.txt ".[gui,hdf5,root,gle]"

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
   python -m pip install "asymmetry[gui,hdf5,root,gle] @ git+https://github.com/BenHuddart/asymmetry.git"

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
