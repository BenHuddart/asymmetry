Installation
============

Choose the desktop download for a bundled application, or a Python installation
for the library, CLI, and optional GUI. The commands for PyPI below apply to
versions published there. If the version you need is unavailable, use
:ref:`source-installation`.

Desktop application
-------------------

Download the **Windows x64** installer or **macOS Apple Silicon** disk image
from `GitHub Releases <https://github.com/BenHuddart/asymmetry/releases/latest>`__.
These applications include Python, GUI dependencies, and their HDF4 runtime.
Open **Asymmetry** from the Windows Start menu; on macOS, drag it to
**Applications** and launch it there. Linux and Intel Mac users can use a
Python installation.

The desktop download does not install the ``asymmetry`` CLI on ``PATH``.
For scripting or an AI coding agent, use the Python or pipx route below.

.. _installation-methods:

Python installation
-------------------

Asymmetry requires Python 3.11 or later. Clean Linux wheel installations are
checked on Python 3.11, 3.12, and 3.13 with constrained and ordinary dependency
resolution. The checks exercise core, GUI with HDF5/ROOT, headless agent, and
full end-user profiles; they do not certify every platform, optional runtime,
or newer interpreter.

Create a virtual environment with one of these tested Python versions. If your
system calls Python 3 ``python3``, use that name for this command:

.. code-block:: bash

   python -m venv .venv

Activate it on Linux or macOS:

.. code-block:: bash

   source .venv/bin/activate

On Windows PowerShell:

.. code-block:: powershell

   .venv\Scripts\Activate.ps1

For a published release, install the full end-user Python feature set and launch:

.. code-block:: bash

   python -m pip install "asymmetry[all]"
   asymmetry-gui

The ``all`` extra includes the GUI, supported file readers, headless plotting,
and GLE script generation. It excludes development tools and CUDA. Native HDF4
libraries, the external GLE compiler, and GPU drivers have separate requirements
under :ref:`external-programs-and-runtimes`.

For the core library and CLI alone:

.. code-block:: bash

   python -m pip install asymmetry
   asymmetry --help

For the GUI with modern NeXus and ROOT readers:

.. code-block:: bash

   python -m pip install "asymmetry[gui,hdf5,root]"
   asymmetry-gui

Continue with :doc:`quickstart` to generate data and complete a first analysis.
That walkthrough needs the ``gui`` and ``hdf5`` extras, both included in ``all``.

Optional dependencies
~~~~~~~~~~~~~~~~~~~~~

Extras can be combined inside the quoted brackets. ``gui`` installs the interface
only; choose the readers that your acquisitions need.

.. list-table:: Installation extras
   :header-rows: 1
   :widths: 15 45 40

   * - Extra
     - Provides
     - Python dependencies
   * - ``all``
     - All ordinary end-user features
     - GUI, readers, headless plots, and GLE support
   * - ``gui``
     - Desktop interface
     - PySide6 and Matplotlib
   * - ``agent``
     - Headless plots and HDF5 loading
     - Matplotlib and h5py
   * - ``hdf5``
     - HDF5 NeXus reading and writing
     - h5py
   * - ``hdf4``
     - Legacy HDF4 NeXus reading
     - pyhdf
   * - ``root``
     - Supported μSR ROOT reading
     - uproot
   * - ``gle``
     - GLE script generation
     - gleplot
   * - ``gpu``
     - Optional CUDA MaxEnt acceleration
     - CUDA-13 CuPy wheel with CUDA components
   * - ``dev``
     - Tests, lint, and documentation tools
     - pytest, Ruff, Sphinx, and supporting tools

The core dependencies are NumPy >= 1.24, < 2.3; SciPy >= 1.10, < 1.18;
and iminuit >= 2.0, < 3. Minimum dependency versions are distinct from the
versions exercised by clean-install checks. Format support also depends on the
acquisition layout described in :doc:`/reference/loading_data`.

.. _external-programs-and-runtimes:

External programs and runtimes
------------------------------

HDF4
~~~~

For a published release, add the legacy NeXus reader with:

.. code-block:: bash

   python -m pip install "asymmetry[hdf4]"

Binary-wheel availability depends on the Python version, operating system, and
architecture. If pip needs to build pyhdf from source, its HDF4 libraries and
build dependencies must be installed; see the
`pyhdf installation guide <https://fhs.github.io/pyhdf/install.html>`__.

On Windows, the pyhdf wheel also needs the HDF4 runtime (``hdf.dll`` and
``mfhdf.dll``). For example, install conda-forge's ``hdf4`` package and point
``ASYMMETRY_HDF4_DLL_DIR`` at the directory holding those DLLs. The desktop
installers bundle this runtime. See :doc:`/reference/loading_data` for supported
containers and the missing-reader diagnostic. If you only need HDF5 and ROOT,
``asymmetry[gui,hdf5,root]`` avoids installing the legacy HDF4 reader.

GLE
~~~

The ``gle`` extra installs the Python script generator selected by Asymmetry:

.. code-block:: bash

   python -m pip install "asymmetry[gle]"

Compiling scripts to PDF or EPS also needs the external
`GLE program <https://glx.sourceforge.io/>`__ and its supporting runtime.
Cairo is needed for overlapping translucent fills and SVG output. Without the
compiler, scripts and sidecar data can still be written and compiled later.
The ordinary Matplotlib plot export does not require GLE.

The GLE executable must be on ``PATH`` or selected through
**Setup → GLE Setup…**. In a frozen macOS application, Asymmetry also checks the
standard Homebrew, MacPorts, and QGLE locations. The setup dialog runs the
chosen executable to verify it before saving the path.

CUDA
~~~~

GPU acceleration is opt-in and is not part of ``all``:

.. code-block:: bash

   python -m pip install "asymmetry[gpu]"

This selects ``cupy-cuda13x[ctk]``. For a CUDA-12 installation, install Asymmetry
without ``gpu`` and select the alternative CuPy wheel:

.. code-block:: bash

   python -m pip install asymmetry "cupy-cuda12x[ctk]>=13.0"

Choose one CuPy CUDA variant, rather than installing both. A compatible NVIDIA
GPU and driver are required; see the
`CuPy installation guide <https://docs.cupy.dev/en/stable/install.html>`__
and :ref:`maxent-gpu-acceleration` for the supported backend settings.

CLI and AI coding agents
------------------------

The ``agent`` extra adds headless plots and HDF5 loading to the core CLI.
For a published release, pipx can keep its environment separate:

.. code-block:: bash

   pipx install "asymmetry[agent]"
   asymmetry --help
   asymmetry skill install --agent claude   # or --agent codex
   asymmetry skill check

Use ``agent,hdf4`` or ``agent,root`` when those readers are needed. If Asymmetry
is already installed in your activated virtual environment, use its
``asymmetry`` command directly; a second pipx installation is unnecessary.
See :doc:`/reference/agent_workflow` for skill upgrades, restart requirements,
and the command reference.

.. _source-installation:

Source installation
-------------------

Use this route for development versions or when the requested release is not
available on PyPI. Install Git, then, in an activated Python environment:

.. code-block:: bash

   git clone https://github.com/BenHuddart/asymmetry.git
   cd asymmetry
   python -m pip install -c constraints.txt ".[all]"
   asymmetry-gui

The checkout's ``constraints.txt`` restricts versions of requested dependencies;
it does not install every package it lists. PyPI commands use the distribution's
own dependency declarations without requiring a checkout or that file.
Replace ``.[all]`` with ``.`` for the core, or ``.[agent]`` for headless plotting
and HDF5 loading; other extras use the same bracket syntax.

An alternative source installation uses pip directly with Git:

.. code-block:: bash

   python -m pip install "asymmetry[all] @ git+https://github.com/BenHuddart/asymmetry.git"

For a contributor installation from the checkout:

.. code-block:: bash

   python -m pip install -c constraints.txt -e ".[all,dev]"

See :doc:`/contributing` for the harness, source documentation builds, and
contribution workflow.

Verifying installation
----------------------

.. code-block:: python

   import asymmetry
   print(asymmetry.__version__)

   from asymmetry.core import workflow
   from asymmetry.core.fitting.models import exponential_relaxation
   print(exponential_relaxation(0.0, A0=0.25, Lambda=0.7, baseline=0.05))
   # 0.3

The core install also provides CLI help:

.. code-block:: bash

   asymmetry --help
   python -m asymmetry fit --help

After installing the ``gui`` or ``all`` extra, ``asymmetry-gui`` launches the
application.

Build verification for maintainers
----------------------------------

The following checks describe release verification; they are not steps needed
to start the application or follow the quickstart.

Desktop build dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~

The Windows x64 and Apple Silicon macOS build pipelines use reviewed,
platform-specific dependency wheels and build-tool versions. Release rehearsals
and previews replay these inputs in two fresh environments and check the observed
versions before packaging. The Python library retains the flexible dependency
ranges described above; its minimum Python version is independent of the selected
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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
