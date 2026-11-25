Asymmetry
=========

**Asymmetry** is a Python library, command-line tool, and desktop application
for muon-spin spectroscopy (μSR). It provides tools for reducing, analysing,
and visualising data: detector grouping and calibration, time- and
frequency-domain fitting, Fourier and maximum-entropy analysis, parameter
trends, and saving analysis sessions for reuse.

.. note::

   **Asymmetry is beta software under active development.** Its code and
   documentation have been written by AI coding agents (Claude Code and
   Codex). Throughout development, numerical behaviour has been checked
   through automated tests, independent mathematical references, and
   comparisons with established μSR software. See the
   :doc:`software comparisons <explanation/comparison>` for tested cases,
   agreement, and known limitations.

   Check scientific claims and model definitions against primary literature
   and other trusted sources, and validate results for your analysis before
   relying on them for publication. APIs may change between releases.

Projects saved by released versions remain loadable. The
:doc:`numerical accuracy guide <explanation/numerical_accuracy>` explains
evaluation targets and their scientific scope.

Where to start
--------------

- :doc:`getting_started/index` — install Asymmetry and make your first
  asymmetry plot and fit.
- :doc:`workflows/index` — worked analyses with graphical instructions and selected
  Python examples. For scripting recipes, see :doc:`reference/cookbook`.
- :doc:`reference/index` — feature-by-feature documentation and the
  fit-function library.
- :doc:`explanation/index` — a μSR primer, the conventions Asymmetry follows,
  how it compares to other μSR tools, and a :doc:`glossary
  <explanation/glossary>` of terms.
- :doc:`api/index` — the Python API reference for scripting.
- :doc:`contributing` — how to report issues and contribute.

Use the :ref:`documentation search <search>` to find tasks such as
``deadtime``, ``grouping``, or ``alpha calibration``. Results show page titles
and section headings. To find a Python object, enter its full name, for example
``asymmetry.core.transform.deadtime``.

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Contents

   getting_started/index
   workflows/index
   reference/index
   explanation/index
   api/index
   contributing

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
