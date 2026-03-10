Data Processing
===============

Asymmetry provides several tools for processing and transforming μSR data.

Rebinning
---------

Reduce noise by combining adjacent time bins:

.. code-block:: python

   from asymmetry.core.transform.rebin import rebin
   import numpy as np
   
   # Original data
   time = np.linspace(0, 10, 1000)
   asymmetry = np.random.randn(1000) * 0.01
   error = np.ones(1000) * 0.01
   
   # Rebin by factor of 10
   t_rebinned, a_rebinned, e_rebinned = rebin(time, asymmetry, error, factor=10)
   
   print(f"Original: {len(time)} points")
   print(f"Rebinned: {len(t_rebinned)} points")
   # Errors are propagated: σ_new = √(Σσ²) / N

Time Range Selection
--------------------

Extract a subset of the data:

.. code-block:: python

   dataset = load("data.wim")
   
   # Select data between 0.5 and 5.0 μs
   subset = dataset.time_range(t_min=0.5, t_max=5.0)
   
   print(f"Original: {dataset.n_points} points")
   print(f"Subset: {subset.n_points} points")

Grouping Histograms
-------------------

Combine detector histograms into logical groups:

.. code-block:: python

   from asymmetry.core.transform.grouping import group_histograms
   
   run = load_run("data.wim")
   
   # Define grouping: forward and backward detectors
   grouping = {
       "forward": [0, 1, 2, 3],
       "backward": [4, 5, 6, 7]
   }
   
   # Group and calculate asymmetry
   from asymmetry.core.transform.asymmetry import calculate_asymmetry
   
   forward_counts = group_histograms(run.histograms, grouping["forward"])
   backward_counts = group_histograms(run.histograms, grouping["backward"])
   
   asymmetry = calculate_asymmetry(forward_counts, backward_counts)

Co-adding Datasets
------------------

Average multiple datasets with proper error propagation:

.. code-block:: python

   import numpy as np
   from asymmetry.core.io import load
   
   # Load multiple datasets
   datasets = [load(f"run_{i}.wim") for i in range(1, 4)]
   
   # Ensure same time grid (or interpolate if needed)
   time_grid = datasets[0].time
   
   asymmetries = [ds.asymmetry for ds in datasets]
   errors = [ds.error for ds in datasets]
   
   # Average
   avg_asymmetry = np.mean(asymmetries, axis=0)
   
   # Error propagation: σ_avg = √(Σσ²) / N
   avg_error = np.sqrt(np.sum([e**2 for e in errors], axis=0)) / len(datasets)

The GUI provides automatic co-adding via the context menu: select 2+ datasets,
right-click, and choose "Co-add Selected" to combine them with proper error propagation.

Windowing for Fourier Analysis
-------------------------------

Apply window functions before FFT:

.. code-block:: python

   from asymmetry.core.fourier.window import apply_window
   
   dataset = load("data.wim")
   
   # Apply Hann window
   windowed_data = apply_window(dataset.asymmetry, window_type="hann")
   
   # Available windows: 'hann', 'hamming', 'blackman', 'bartlett'
