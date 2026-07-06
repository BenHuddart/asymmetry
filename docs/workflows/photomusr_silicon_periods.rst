Photo-μSR in silicon (red/green periods)
========================================

This worked example shows how to analyse a **period-mode** run, where a
single file holds more than one detector histogram set recorded under
different conditions. The archetype is **photo-μSR**: the sample is
measured with the pump laser **ON** and **OFF** in alternating periods,
and the difference isolates the light-induced muonium dynamics.

The run
-------

``HIFI00103277`` (corpus *Semiconductors → Photo-μSR in silicon*,
``Data_hdf5/``) is a two-period HIFI run at :math:`T = 291\;\mathrm{K}`.
The two periods are labelled **red** (period 1) and **green** (period 2)
— the same convention as the GUI's **RG box** — and each carries its own
provenance, including its own per-period good-frame count
(:math:`\approx 28\,108` frames each here).

The GUI: the RG box
-------------------

Load the run and select it. Because the file is period-mode, the plot
toolbar shows the **RG (red/green) selector**. Switching between **Red**
and **Green** re-reduces the asymmetry from that period's histograms —
with its own good-frame normaliser, so dead-time correction and counting
statistics stay correct per period. Group, calibrate :math:`\alpha`, and
fit each period exactly as for an ordinary single-period run
(:doc:`calibration_grouping_emu`).

The scriptable API
------------------

The RG box calls the same core period-selection API your scripts use, so
the desktop app and a batch script agree on the per-period spectra. Pull
out a single period as an ordinary :class:`~asymmetry.core.data.dataset.MuonDataset`:

.. code-block:: python

   from asymmetry.core.io import load, select_period, period_count, period_labels

   run = load(".../Photo-muSR in silicon/Data_hdf5/HIFI00103277.nxs")
   print(period_count(run))                  # 2
   print(period_labels(run))                 # ['red', 'green']

   light_on = select_period(run, "red")      # period 1
   light_off = select_period(run, "green")   # period 2

   # ...or select a single period at load time:
   light_off = load(
       ".../Data_hdf5/HIFI00103277.nxs", period="green",
   )

Each returned dataset keeps its parent's :math:`t_0`, good-bin window,
grouping, field and temperature, plus its **own** per-period
``good_frames`` and ``dead_time_us``. For files with three or more
periods, pass a 1-based integer period number instead of a label.

.. note::

   In a photo-μSR experiment the usual convention is **light-ON = Red**
   (period 1) and **light-OFF = Green** (period 2). Confirm this against
   the relaxation for *your* instrument and run before interpreting the
   difference — the period ordering is a property of the data-acquisition
   setup, not a fixed rule.

The light-ON − light-OFF analysis
---------------------------------

With the two periods in hand, the photo-μSR observable is the difference
of their fitted (or raw) asymmetries: the light-OFF period is the dark
baseline, and the light-ON period adds the photo-excited muonium signal.
Fit each period with the model appropriate to silicon muonium (e.g. an
oscillatory or relaxing component; see
:doc:`/reference/fit_functions/oscillation`) and compare the fitted parameters, or
subtract the two asymmetry traces directly for a model-free view of the
light-induced change.

See also
--------

- :ref:`selecting-periods` — period-selection reference, including
  ``period_count`` and ``period_labels``.
- :doc:`/reference/loading_data` — supported formats and period-mode
  files.
- :doc:`calibration_grouping_emu` — grouping and :math:`\alpha` setup,
  applied per period.
