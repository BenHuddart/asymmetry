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

For GLE export support:

* gleplot installed from GitHub:

   .. code-block:: bash

       pip install "gleplot @ git+https://github.com/BenHuddart/gleplot.git"

Installation Methods
--------------------

From PyPI (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Minimal installation
   pip install asymmetry

   # With GUI support
   pip install asymmetry[gui]

   # With GUI + GLE export support
   pip install asymmetry[gui,gle]

   # With all optional dependencies
   pip install asymmetry[all]

From Source
~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/BenHuddart/asymmetry.git
   cd asymmetry
   pip install -c constraints.txt -e ".[gui,gle,dev]"

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
