Logbook Workflows
=================

.. image:: /_generated/screenshots/logbook_view.png
   :alt: Data-browser logbook view of an EuO temperature scan sorted by T(K)
   :width: 100%

*Logbook view of the EuO temperature scan, sorted by sample temperature.*
*The data browser doubles as the run logbook in the Asymmetry GUI: the*
*Run, Title, T(K) and B(G) columns expose the run-level metadata directly,*
*and additional columns can be surfaced through* **Options → Add column…**

The Data Browser doubles as the run logbook in the Asymmetry GUI: a
searchable, filterable, sortable run table that drives every downstream
analysis. The :class:`~asymmetry.core.data.logbook.Logbook` Python class
exposes the same machinery for scripted use — tagging runs with
user-defined labels (``zf-cooled`` vs ``fc``, ``sample-A`` vs
``reference``), grouping them into named collections, free-text
searching across titles, comments and tags, and exporting to TSV for
further processing or RTF to paste directly into a manuscript. Every
case study under :doc:`workflows/index` begins with a logbook step to
set up the run series.

.. _logbook-columns:

Columns: metadata and custom
----------------------------

Beyond the fixed **Run**, **Title**, **T (K)** and **B (G)** columns, the Data
Browser shows two kinds of *extra* column. Both are managed the same way — a
right-click on any extra-column header offers **Rename…** and a remove action,
and both are saved with the project.

**Metadata columns** surface a value read from each run's metadata (a NeXus
field or a derived run-quality quantity). They are read-only. Add one in either
of two ways:

* **Get Info → "Include in Data Browser".** Right-click a run and choose *Get
  Info* (or use the toolbar), then tick a field's *Include in Data Browser* box —
  in the summary table or under *Advanced* for the full NeXus tree. The field
  becomes a column for every run.
* **Right-click a header → Add column…** for the built-in run-quality fields
  (points, histograms, counts, events/frame, …).

**Custom columns** are yours to fill in. Click the **＋ Column** button at the
bottom-right of the browser, give the column a name, and an empty, editable
column appears. Double-click any cell to type a value for that run — a label
like ``annealed`` / ``as-grown``, a sample batch, an off-axis angle, anything.
Values are free-form text: empty by default, entered per run, and saved with the
project.

Renaming and the underlying field
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Right-click any extra-column header and choose **Rename…** to change the
gui-facing name. For a metadata column the rename only changes the *display*
label — the underlying NeXus/metadata field is retained, so the column keeps
showing the same data and still round-trips correctly. Hover the header to see a
tooltip naming the source field (e.g. *From metadata field:
nexus_fields.sample.shape*). This is the recommended way to give a cryptic
advanced NeXus parameter a readable, manuscript-friendly name.

Using columns as the plot label and the trend x-axis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A custom column can drive two downstream surfaces:

* **Plot legend label.** Pick the column from the **Label:** selector above the
  main plot to label each curve by your custom value (falling back to the run
  number for any run you left blank).
* **Parameter-trend x-axis.** In the parameter trending panel, choose the custom
  column from the **X:** selector to plot fitted parameters against it.

Because custom values are free text, the trend x-axis coerces each one to a
number on demand. Runs whose value is **empty or non-numeric are dropped** from
the trend (they cannot be placed on a numeric axis), and a small note beside the
selector reports how many were skipped — e.g. *⚠ 2/8 skipped (empty/non-numeric)*.
Fill those runs in (or pick a different axis) to include them.

Basic Usage
-----------

.. code-block:: python

   from asymmetry.core.data.logbook import Logbook
   from asymmetry.core.io import load

   logbook = Logbook()
   dataset = load("run_3101.nxs")
   logbook.add(dataset, tags=["zf", "sample-a"])

   print(logbook.run_numbers)
   print(logbook.get_entry(dataset.run_number))

Filtering and Search
--------------------

.. code-block:: python

   # Exact metadata filtering
   entries = logbook.filter(temperature=5.0, field=30.0)

   # Free-text search across title, comment, and tags
   matches = logbook.search("baseline")

   # Access the dataset for plotting/fitting
   if matches:
       ds = logbook.get_dataset(matches[0].run_number)
       print(ds.n_points)

Collections
-----------

Collections are named groups of run numbers that are useful for repeated
analysis sets.

.. code-block:: python

   logbook.create_collection("LF series", [3102, 3103, 3104])
   print(logbook.collections)
   print(logbook.get_collection("LF series"))

Persistence
-----------

The logbook stores metadata and collection definitions in JSON.

.. code-block:: python

   logbook.save("my_logbook.json")

   restored = Logbook()
   restored.load_metadata("my_logbook.json")
   print(restored.run_numbers)

.. note::

   ``load_metadata`` restores logbook entries and collections, not raw datasets.
   Reload source data files and re-add datasets when needed.

GUI Export
----------

From the GUI Data Browser, use **Export logbook** on the main toolbar to export
the browser logbook.

* **TSV** is the default and recommended format for speed and column alignment.
* **RTF** is also available for rich-text sharing.
* Export uses active Data Browser columns and preserves data-group sections.
* Hidden datasets (filtered/collapsed) are still included in the export file.

Runnable Example
----------------

See ``examples/logbook_usage.py`` for a complete executable script.
