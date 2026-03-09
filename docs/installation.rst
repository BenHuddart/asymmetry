Installation
============

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

Installation Methods
--------------------

From PyPI (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Minimal installation
   pip install asymmetry

   # With GUI support
   pip install asymmetry[gui]

   # With all optional dependencies
   pip install asymmetry[all]

From Source
~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/your-org/asymmetry.git
   cd asymmetry
   pip install -e ".[gui,dev]"

Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~

For development work:

.. code-block:: bash

   pip install -e ".[dev]"
   
   # Run tests
   pytest

Verifying Installation
----------------------

.. code-block:: python

   import asymmetry
   print(asymmetry.__version__)
   
   # Test GUI availability
   from asymmetry.gui.app import main
   # main()  # Uncomment to launch GUI
