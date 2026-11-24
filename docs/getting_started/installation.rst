Installation
============

Asymmetry is not currently published on PyPI. Most users on **Windows** or
**Apple Silicon Macs** should install the pre-built desktop application from
the `GitHub Releases <https://github.com/BenHuddart/asymmetry/releases/latest>`__
page (Windows ``.exe`` installer; macOS ``.dmg``). **Intel Mac** and **Linux**
users should install from the Git repository as described below.

After installing the Windows application, launch **Asymmetry** from the Start
menu. On macOS, open the disk image, drag **Asymmetry** to **Applications**,
and launch it there. These packaged applications include their own Python
runtime and GUI dependencies; the terminal commands below apply to Python
installations.

Desktop build dependencies
--------------------------

The Windows x64 and Apple Silicon macOS build pipelines use reviewed,
platform-specific dependency wheels and build-tool versions. Release rehearsals
and previews replay these inputs in two fresh environments and check the observed
versions before packaging. The Python library retains the flexible dependency
ranges described below; its minimum Python version is independent of the selected
desktop build interpreter.

Desktop download assets include a target-specific build/component record, a final
artifact record, and ``<target>-SHA256SUMS``. These bind the installer bytes to the packaged
version, source commit, reviewed inputs, and actual collected components and
third-party notices. The Windows build verifies its external HDF4 runtime archives
before extraction. The final installer payload is checked against the recorded
application components. Downloaded installer and record bytes are checked before
publication. SHA256 establishes byte identity and traceability; it is not a digital
signature.

These checks reproduce dependency versions and the selected GLE source commit.
Hosted operating-system tools and native packaging timestamps can change, so
identical installer bytes are not guaranteed. The existing packaged startup and
import smoke checks do not certify a complete scientific analysis workflow.

Desktop scientific diagnostic
-----------------------------

The application provides an explicit ``--scientific-acceptance`` diagnostic for
build verification. It reads tiny original ROOT and native HDF4 inputs, reduces
an explicitly selected correction-free count ratio to 20 percent asymmetry, fits
a free constant, checks the exported text values, and saves and replays a project
with its source identity, grouping, recipe, and fit state. The diagnostic uses a
fresh temporary state directory and bypasses user plugin discovery. Required
resource, reader, identity, numerical, export, or reload failures return nonzero.
A parent runner checks completion and a fresh report within a finite time budget.

This bounded scenario checks packaging and scientific wiring. It does not certify
instrument defaults, corrections, uncertainty coverage, or all model families.
Release verification runs this journey through the final installed Windows x64
launcher and read-only mounted Apple Silicon macOS app on fresh native hosts
without source or developer Python dependencies. Both must match the source
scenario numerically. Disposable payload copies must also fail when a required
logo, native HDF4 library, or reader backend is actually removed. Installation,
uninstallation, mounting, analysis, and cleanup use finite time budgets; timeouts
and missing or stale reports fail verification. Scientific reports remain separate
from artifact provenance records. Both native outcomes gate publication alongside
payload and download checks; the core-only Linux wheel gate remains separate.

Requirements
------------

Asymmetry requires Python 3.11 or later. CI checks core-only wheel installs on
Python 3.11 and 3.12, including public core imports, CLI help, and bundled skill
resources. These checks do not certify every optional extra or platform on newer
Python versions; optional packages also have their own interpreter requirements.

Core dependencies
~~~~~~~~~~~~~~~~~

* numpy >= 1.24, < 2.3
* scipy >= 1.10, < 1.18
* iminuit >= 2.0, < 3

The core library and CLI help need only these dependencies and their transitive
requirements. GUI, plotting, and format-specific libraries are optional. The
``constraints.txt`` file restricts versions of requested dependencies; it does
not install every package it lists.

Optional dependencies
~~~~~~~~~~~~~~~~~~~~~

Extra-install commands below run from the repository checkout described under
:ref:`installation-methods`. Asymmetry itself is not available from PyPI.

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
  package), with
  ``ASYMMETRY_HDF4_DLL_DIR`` pointed at the directory holding them. The
  pre-built **Windows and Apple Silicon macOS** desktop releases bundle HDF4;
  no extra setup is needed there. See :doc:`/reference/loading_data`.

For ROOT file support:

* uproot >= 5.0

For headless plots and NeXus loading in the command-line workflow and AI-agent
skill:

* matplotlib >= 3.7, h5py >= 3.8 — installed together as the ``agent`` extra
  (``python -m pip install ".[agent]"``). See :doc:`/reference/agent_workflow`.

For GLE export support:

* gleplot 6.0.0 from PyPI, installed by the ``gle`` extra:

   .. code-block:: bash

       python -m pip install ".[gle]"

  To install the script generator separately, use ``pip install gleplot==6.0.0``.

* The GLE compiler itself (``gle``) needs to be on ``PATH`` for ``.gle``
  scripts to be compiled to PDF or EPS at export time. Use a build with Cairo
  support for translucent uncertainty bands, component fills and phase shading.
  GLE's include library (``gleinc``, supplied by its release packages) supports
  native reference lines and spans. Without the compiler, the
  scripts and sidecar data files are still written and can be compiled
  later. In a frozen macOS app the inherited ``PATH`` is minimal, so
  Asymmetry also looks for ``gle`` in the standard Homebrew, MacPorts,
  and QGLE locations. Override the search from **Setup → GLE Setup…**
  in the main menu; the chosen path is persisted across sessions. On
  accept the dialog runs the chosen executable once to verify it is
  actually GLE — a path that cannot run (or is not GLE) is rejected with
  the reason shown in the dialog, rather than saved and left to fail at
  export time.

.. _installation-methods:

Installation methods
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

Development installation
~~~~~~~~~~~~~~~~~~~~~~~~

For development work:

.. code-block:: bash

   git clone https://github.com/BenHuddart/asymmetry.git
   cd asymmetry
   python -m pip install -c constraints.txt -e ".[all]"

   # Run tests
   python3 tools/harness.py test

Verifying installation
----------------------

.. code-block:: python

   import asymmetry
   print(asymmetry.__version__)
   
   # Exercise core imports, including the public workflow used by the CLI.
   from asymmetry.core import workflow
   from asymmetry.core.fitting.models import exponential_relaxation
   print(exponential_relaxation(0.0, A0=0.25, Lambda=0.7, baseline=0.05))
   # 0.3

The core install also provides CLI help:

.. code-block:: bash

   asymmetry --help
   python -m asymmetry fit --help

After installing the ``gui`` extra, ``asymmetry-gui`` launches the application.
