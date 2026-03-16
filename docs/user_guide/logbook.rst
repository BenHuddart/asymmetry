Logbook Workflows
=================

The ``Logbook`` provides a searchable run table for multi-run analysis.

Basic Usage
-----------

.. code-block:: python

   from asymmetry.core.data.logbook import Logbook
   from asymmetry.core.io import load

   logbook = Logbook()
   dataset = load("run_3101.wim")
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

Runnable Example
----------------

See ``examples/logbook_usage.py`` for a complete executable script.
