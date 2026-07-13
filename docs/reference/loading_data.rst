Loading Data
============

Asymmetry reads the major muSR raw-data formats through a common
:func:`~asymmetry.core.io.load` API and a uniform Grouping dialog: ISIS
muon NeXus (legacy V1 and V2), PSI BIN/MDU, and MusrRoot/LEM ROOT. The
practical consequence is that data from different facilities can be
compared in one session without manual format-by-format pre-processing.
Each loader is implemented against the same authoritative reference used
by the facility's own software — musrfit's ``PRunDataHandler`` for PSI
BIN/MDU and MusrRoot/LEM ROOT, with Mantid's ``LoadPSIMuonBin`` used as a
PSI-BIN cross-check — so format-specific subtleties (bin-index
conventions, per-detector :math:`t_0` offsets, multi-period ISIS files,
PSI temperature sidecars) are handled consistently with what those
packages do.

If you are loading a new dataset for the first time, the subsections
below document the metadata each loader extracts and the common failure
modes. Period-mode files (e.g. photo-μSR light-OFF/ON, RF on/off,
avoided-level-crossing) can
be navigated with the scriptable period-selection API — see
:ref:`selecting-periods` below.

For an end-to-end walk-through that starts with loading, see
:doc:`/workflows/temperature_scan_magnetism`.

Supported formats
-----------------

ISIS Muon NeXus (.nxs, .nexus)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Asymmetry supports ISIS muon NeXus files (legacy V1 and modern V2), including
multi-period runs. Multi-period files return a list of ``MuonDataset`` values.
Scalar sample temperature is read from the NeXus sample header. NXlog-style
temperature groups with ``time`` plus ``value``/``values`` datasets are also
summarised in Get Info; the Data Browser keeps the scalar header temperature
unless **Options → Use temperature from log** or the **Temperature (K)**
include checkbox is ticked, in which case the log mean is shown while that
selection remains active. The Options menu choice is global and resets all
loaded runs to the same behaviour. The Get Info checkbox affects only the
target run, making it useful for per-dataset overrides. Log-derived temperature
values are shown with red text in the Data Browser.

For NeXus good-data windows, Asymmetry treats integer bin metadata as
canonical (``first_good_bin``, ``last_good_bin``, and ``t0_bin``). When
``first_good_time`` / ``last_good_time`` are present, they are used only as a
fallback if the corresponding integer bin attributes are missing.

Asymmetry's internal ``Histogram`` and grouping bin indices are always
zero-based array indices. Real ISIS files (both V1 and V2) often encode
integer bin metadata using one-based centre-bin numbering. The loader compares
explicit ``t0_bin`` values with the time axis; when all available detector
``t0`` values point one sample past ``t = 0``, it subtracts one from
``t0_bin``, ``first_good_bin``, and ``last_good_bin`` and records
``bin_index_base = 1`` in grouping metadata. Otherwise it leaves the integers
unchanged with ``bin_index_base = 0``. For V1, this good-data window and
``t0_bin`` are read from the attributes of the ``counts`` dataset (where ISIS
stores them), matching the V2 behaviour.

HDF4 container (legacy ``.nxs``)
""""""""""""""""""""""""""""""""

The V1 ``/run`` ``muonTD`` schema is the format WiMDA reads natively, and ISIS
historically wrote it inside an **HDF4** container (pre-~2015 runs) rather than
HDF5. Asymmetry detects the container from the file magic and reads HDF4 V1
files directly — no manual pre-conversion to HDF5 is required. The same schema
reader is used for both containers, so an HDF4 ``.nxs`` and its HDF5-converted
twin reduce to identical asymmetry, counts, grouping, and metadata.

HDF4 reading needs the optional ``pyhdf`` dependency, installed with the
``hdf4`` extra::

    pip install asymmetry[hdf4]

On **Linux** the ``pyhdf`` wheel bundles the HDF4 C library. On **macOS
(Apple Silicon)** the PyPI wheel does too (Intel Macs: use conda-forge
``pyhdf`` or build from source). On **Windows**, ``pyhdf``'s wheel does *not*
bundle the HDF4 runtime: it also needs ``hdf.dll`` / ``mfhdf.dll`` (for
example from the conda-forge ``hdf4`` package, as Mantid uses, or via
``packaging/windows/fetch_hdf4_dlls.py``). Point the ``ASYMMETRY_HDF4_DLL_DIR``
environment variable at the directory holding those DLLs. The pre-built Windows
and Apple Silicon macOS desktop releases bundle HDF4. When ``pyhdf`` (or the
Windows runtime) is absent, opening an HDF4 ``.nxs`` raises a clear error
naming the ``hdf4`` extra; HDF5 ``.nxs`` loading is unaffected.

PSI BIN/MDU (.bin, .mdu)
~~~~~~~~~~~~~~~~~~~~~~~~

Asymmetry supports PSI raw histogram files in the classic PSI-BIN and PSI-MDU
formats. These files are loaded into normal raw histogram runs, so they use the
same full Grouping dialog as ISIS NeXus data.

This loader is intentionally based on established PSI readers rather than a
new file interpretation. The BIN/MDU binary layout and metadata offsets follow
the musrfit PSI reader, especially ``PRunDataHandler::ReadPsiBinFile`` and the
``MuSR_td_PSI_bin``/MDU structures it uses. Mantid's ``LoadPSIMuonBin`` was
also checked for PSI-BIN behaviour. Asymmetry keeps the implementation in pure
Python and maps the result into the existing ``Run``/``Histogram`` data model.

PSI files can store different ``t0`` values per detector. Asymmetry preserves
those detector-specific offsets and aligns histograms by their own ``t0`` when
regrouping.

For PSI-BIN, musrfit's ``MuSR_td_PSI_bin`` reader uses the stored ``t0`` and
``first good`` values directly as C++ vector indices, so Asymmetry treats them
as zero-based. The ``last good`` value has an end-bin convention: some PSI-BIN
headers store the histogram length itself, for example ``8192`` for an
8192-bin histogram. musrfit uses this effectively as an exclusive stop when it
extracts good-bin ranges. Asymmetry stores the corresponding valid inclusive
array index, so that example becomes ``8191`` internally.

For PSI-MDU, the tag metadata are converted to the same zero-based internal
coordinate system used by musrfit's public reader methods. The converted
``t0``, ``first good``, and ``last good`` values from the musrfit reader match
the values stored by Asymmetry.

PSI label metadata is used to build initial grouping suggestions. BIN labels
such as ``Forw``/``Back`` and MDU positron labels such as ``F1``/``B1`` become
group names and forward/backward defaults where possible. Users can then edit
the grouping in the same full Grouping dialog used for raw NeXus runs.
Repeated labels are not collapsed: each histogram remains visible as its own
group, with suffixes such as ``Forw 2`` added where needed.

PSI labels define forward/backward along the beam direction, whereas
Asymmetry's pair-asymmetry controls use forward/backward relative to the
initial muon spin. For PSI data, the analysis forward/backward selections are
therefore initialised with the beam-forward and beam-backward detector groups
swapped. The detector layout editor still displays the PSI detector convention.

If the explicit field entry is missing or zero but the PSI comment/title text
contains a recognisable field such as ``LF 32G`` or ``Bz=150 G``, the GUI
offers to apply that comment-derived value as the run field. This prompt is
available for PSI BIN/MDU and MusrRoot/LEM ROOT files.

PSI-BIN temperature metadata has two sources. The scalar run temperature is
read from the BIN header using the musrfit-compatible offsets. Optional
temperature-log sidecars use Mantid's ``LoadPSIMuonBin`` convention: Asymmetry
searches from the BIN file directory, to three directory levels below it, for
all ``.mon`` files whose filename contains the run number. The ``.mon`` header
date/title, equipment name, and backslash-delimited rows are parsed into
plottable ``nexus_time_series`` entries named like
``psi_temperature/Temp_<channel>`` for classic logs or
``psi_temperature/<equipment>/<channel>`` for FLAME ``tlog`` files. When FLAME
logs are present, ``flamesam0/SAM_ts`` is marked as the primary sample
temperature to match the MusrRoot slow-control convention; ``flamedil0/DIL_T_mix``
and ``variox0/Variox`` are retained as secondary sample-temperature traces.
The Get Info window shows these logs in the summary/advanced tables and records
the sidecar paths plus ``Mantid LoadPSIMuonBin-compatible`` provenance in
``psi_temperature_log`` metadata.

MusrRoot / LEM ROOT (.root)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Asymmetry supports ROOT files that follow the PSI/MusrRoot layout used by
musrfit. The loader reads the ROOT objects with ``uproot`` and maps the result
into the same raw ``Run``/``Histogram`` model as ISIS NeXus and PSI BIN/MDU
files, so ROOT runs use the full Grouping dialog.

The implementation follows musrfit's ``PRunDataHandler::ReadRootFile`` rather
than inventing a new interpretation. In particular:

* MusrRoot files are detected from ``RunHeader`` and ``hDecay%03d``
  histograms. Both the documented ``RunHeader`` ``TFolder`` layout and the
  newer ``TDirectory`` layout are supported. musrfit added the
  ``TDirectory``-based layout in a 2025 commit series, and the current
  MusrRoot specification defines it as canonical (the ``TFolder`` streaming is
  deprecated). PSI's 2026 FLAME DAQ writes this layout; other PSI bulk-µSR
  instruments are expected to follow, since the layout is instrument-agnostic.
* Histogram search follows musrfit's paths: ``histos/hDecay%03d`` for folder
  files and ``histos/DecayAnaModule/hDecay%03d`` for directory files.
* ``RunInfo`` values such as run number, title, laboratory, instrument,
  sample, temperature, field, time resolution, number of histograms, and
  red/green offsets are read from the header when present. Instrument names
  are matched case-insensitively (FLAME writes the lowercase ``flame``).
* ``DetectorInfo`` entries provide detector labels, histogram numbers,
  per-detector ``t0`` bins, and detector-specific good-bin ranges when the
  file supplies them.
* In the ``TDirectory`` layout, ``RunHeader`` and its subfolders
  (``RunInfo``, ``DetectorInfo/DetectorNNN``, ``SampleEnvironmentInfo``,
  ``MagneticFieldEnvironmentInfo``, ``BeamlineInfo``, ``RunSummary``) are
  themselves ``TDirectory``\ s, and every leaf is a ``TObjString`` encoding
  both its key name and its payload as ``"NNN - Label: Value -@type"`` (a
  physical-quantity value carries an optional error, unit, set-point, and
  free-text description, for example ``Time Resolution: 0.09765625 ns;
  SiPM``). The ``NNN`` prefix is a single counter shared across every
  subfolder, so numbering within one subfolder is not contiguous; Asymmetry
  parses these entries by label, never by number. The free-text
  ``RunSummary`` block — which musrfit itself does not read — is attached
  verbatim to the loaded run as
  ``metadata["musrroot_run_summary"]``, so it is preserved as provenance
  rather than discarded.
* MusrRoot slow-control histograms in ``histos/SCAnaModule`` are imported as
  plottable ``nexus_time_series`` logs. In particular, ``hSampleTemperature``
  is summarised as an average temperature in Get Info while the scalar
  ``RunInfo/Sample Temperature`` remains the default Data Browser value until
  **Options → Use temperature from log** or the **Temperature (K)** include
  checkbox is ticked. The Options menu choice applies to every loaded run,
  while the Get Info checkbox overrides only that run. Log-derived temperature
  values are shown with red text. FLAME ROOT files may use EPICS-style sensor
  names such as ``SAM_ts_value`` instead of a literal
  ``hSampleTemperature`` histogram; Asymmetry follows the ``Sens=...`` pointer
  in ``RunInfo/Sample Temperature`` so the Get Info temperature row still opens
  the corresponding log plot.

Legacy LEM ROOT files without a MusrRoot ``RunHeader`` are supported only when
their ``RunInfo`` and histogram objects are readable through ``uproot``. The
full PyROOT/TLemRunHeader object model used by very old files is not
reimplemented.

The optional ``uproot`` dependency is required for ROOT loading. Install with
the ``root`` extra if it is not already present.

Deadtime correction remains controlled by the Grouping dialog. ``File`` mode
uses per-detector deadtime values already supplied by a run file, while
``Manual``, ``Cal``, and ``Estimate`` resolve their deadtime payload in the
Grouping dialog and then apply that same payload to all selected runs. The
resolved payload is also reused for future datasets loaded into the same
project, matching the existing grouping-template workflow.

PSI BIN/MDU and MusrRoot/LEM ROOT files do not normally contain NeXus-style
file deadtime constants, so ``File`` mode is often unavailable there. Those
formats can still use manual per-detector values, the WiMDA-style ``Cal``
per-detector calibration routine, or the reference-run deadtime estimate.

Background correction is a separate Grouping dialog toggle for PSI-style data
(``.bin``, ``.mdu``, and PSI/LEM ``.root``). It subtracts a count background
from grouped raw forward/backward histograms before asymmetry calculation,
following musrfit ``PRunAsymmetry``. The toggle is off by default and disabled
for ISIS/NeXus data, keeping the correction split explicit: ISIS uses
file-based deadtime correction when available, while PSI uses grouped
background correction.

Reference provenance
~~~~~~~~~~~~~~~~~~~~

The format and correction behaviour is split by source so that the application
does not invent format rules:

* PSI BIN/MDU parsing follows musrfit's PSI raw-data reader, with Mantid's
  ``LoadPSIMuonBin`` used as a cross-check for BIN files.
* MusrRoot/LEM ROOT parsing follows musrfit
  ``PRunDataHandler::ReadRootFile`` and the MusrRoot documentation:
  ``RunHeader`` metadata plus ``hDecay`` histograms are mapped into
  Asymmetry's raw run model.
* Per-detector ``t0`` handling follows the PSI file metadata exposed by those
  readers; Asymmetry stores these as ``detector_t0_bins`` and aligns detector
  histograms before grouping.
* Background correction follows musrfit ``PRunAsymmetry``: group
  forward/backward histograms, subtract fixed values or mean background from
  inclusive bin ranges, then calculate asymmetry. If no ranges are supplied,
  the fallback range is ``0.1 * t0`` to ``0.6 * t0``. The corrected-count
  uncertainty follows musrfit-style propagation, while Asymmetry retains its
  standard alpha convention with ``alpha`` applied to the backward group.
* File-based deadtime correction uses the non-paralyzable formula implemented
  by musrfit ``PRunBase::DeadTimeCorrection`` and Mantid
  ``ApplyDeadTimeCorr``:
  ``N_corr = N / (1 - N * dead_time / (time_bin * good_frames))``. The
  ``good_frames`` normalisation is read from the file's top-level
  ``good_frames`` / ``goodfrm`` when present; legacy ISIS HDF4 / NeXus-V1 files
  (e.g. HiFi runs) omit it, so Asymmetry falls back to ``instrument/beam``
  (``frames_period`` per period, then the ``frames_good`` / ``frames`` run
  total). Without that fallback the count defaults to 1 and the correction
  over-normalises by orders of magnitude.
* The Grouping dialog also exposes WiMDA-style manual, calibrated, and
  estimated deadtime workflows alongside file-provided deadtimes. ``Estimate``
  is calculated from the selected reference run only and then broadcast to all
  selected detectors and runs. ``Cal`` fits each detector separately on that
  reference run and persists the resulting explicit per-detector table through
  the manual detector-value editor.
* Mantid's PSI-BIN loader can emit a deadtime table, but it fills the PSI-BIN
  table with zeros. Asymmetry therefore treats PSI deadtime as absent and uses
  ``File`` mode only when non-zero file values are actually present.
* PSI-BIN ``.mon`` temperature sidecar loading follows Mantid
  ``LoadPSIMuonBin`` rather than musrfit, because musrfit was found to read
  embedded PSI temperature fields but not these external log files.

Grouping behaviour by format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the GUI, supported raw data runs use the full Grouping dialog:

* NeXus runs use the full Grouping dialog.
* PSI BIN/MDU runs use the full Grouping dialog, with PSI label-derived
   forward/backward defaults when labels such as ``Forw``/``Back`` or
   ``F1``/``B1`` are present.
* MusrRoot/LEM ROOT runs use the full Grouping dialog. Detector names from
   ``DetectorInfo`` or ROOT histogram titles become the initial group names,
   and ROOT per-detector ``t0`` values are handled like PSI BIN/MDU ``t0``
   metadata.

Basic usage
-----------

Loading a single file
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from asymmetry.core.io import load
   
   dataset = load("path/to/data.nxs")
   print(dataset.summary())

The same ``load`` API can be used with NeXus files:

.. code-block:: python

      dataset_or_periods = load("path/to/HIFI00206453.nxs")
      if isinstance(dataset_or_periods, list):
         print(f"Loaded {len(dataset_or_periods)} periods")

And with PSI raw files:

.. code-block:: python

      dataset = load("deltat_pta_gps_3110.bin")
      dataset = load("tdc_hifi_2014_00153.mdu")

And with MusrRoot/LEM ROOT files:

.. code-block:: python

      dataset = load("lem15_his_2994.root")

The returned ``MuonDataset`` contains:

* ``time``: Time axis in microseconds
* ``asymmetry``: Asymmetry values
* ``error``: Error bars
* ``metadata``: Run metadata (temperature, field, etc.)
* ``run``: Reference to the original Run object

.. _selecting-periods:

Selecting periods (Red / Green)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Period-mode runs record several period histograms in one file — for example
**light-OFF** and **light-ON** in a photo-μSR experiment, RF on/off, or ALC
steps. Use the period-selection API to pull out a single period as an ordinary
``MuonDataset`` with its own provenance (``t0``, good-bin window, grouping,
temperature, field, per-period ``good_frames`` and deadtime):

.. code-block:: python

   from asymmetry.core.io import load, select_period, period_count, period_labels

   run = load("HIFI00103277.nxs")          # two-period photo-µSR run
   print(period_count(run))                 # 2
   print(period_labels(run))                # ['red', 'green']

   light_on = select_period(run, "red")     # period 1
   light_off = select_period(run, "green")  # period 2

   # ...or select at load time:
   light_off = load("HIFI00103277.nxs", period="green")

For the common two-period case the first period is labelled ``"red"`` and the
second ``"green"`` (the same convention as the GUI "RG box"). You can also pass
a 1-based integer period number, which is the way to address files with three
or more periods. In a photo-μSR experiment the convention is **light-OFF =
Green** (period 2) and **light-ON = Red** (period 1); confirm this against the
relaxation for your instrument. Out-of-range numbers and unknown labels raise a
clear error at the boundary. The GUI red/green selector calls this same core
API, so scripts and the desktop app agree on the per-period spectra.

Accessing metadata
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   dataset = load("data.nxs")

   print(f"Run number: {dataset.run_number}")
   print(f"Temperature: {dataset.metadata.get('temperature')} K")
   print(f"Field: {dataset.metadata.get('field')} G")
   print(f"Title: {dataset.metadata.get('title')}")

In the GUI the same preserved provenance is surfaced by right-clicking a run in
the Data Browser and choosing **Get Info**, which opens the run-information
dialog:

.. image:: /_generated/screenshots/run_info_provenance.png
   :alt: The Run Info dialog's Run Parameters table listing instrument, run
         number, title, comment, start/end times, temperature, field,
         detector-histogram count, bins, bin width, and total counts for a
         loaded transverse-field run.
   :width: 100%

*The* **Run Parameters** *table in the Get Info dialog for a loaded*
*transverse-field run, with each field selectable for inclusion in the Data*
*Browser via the* **Include in Data Browser** *column. The loader keeps the*
*experiment provenance explicit — instrument, title, temperature, field, and*
*the raw histogram geometry (four detector histograms, 2400 bins,*
*0.005 μs bin width, and the summed counts) — alongside the free-text*
*comment and run start/end timestamps.*

Loading multiple files
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pathlib import Path
   
   data_dir = Path("data")
   datasets = []
   
   for nexus_file in data_dir.glob("*.nxs"):
       dataset = load(str(nexus_file))
       datasets.append(dataset)
   
   print(f"Loaded {len(datasets)} datasets")

.. _loading-a-run-range:

Loading a run range
~~~~~~~~~~~~~~~~~~~

A contiguous run series — a field or temperature scan such as BiSCCO
1276–1289 — can be loaded in one step from a folder plus a first/last run
number, instead of selecting every file by hand.

**Why this exists.** The native Open dialog's *File name* field holds only a
bounded number of characters (~256), so a long quoted list of ~15 file names is
silently truncated and the load has to be split into batches. The run-range
path resolves the files itself and never passes them through that field, so a
whole scan loads at once.

**In the GUI.** Choose **File → Load Run Range…**. Pick the folder that holds
the run files; the dialog prefills the run *Prefix* (e.g. ``MUSR``) and the
first/last run numbers from the files it finds there. Adjust the range if you
want a subset, then click **OK**. The matching runs are loaded through the
ordinary multi-file path (duplicate prompts, auto-grouping, and the missing-run
gap warning all apply). The log records how many of the requested runs were
found, e.g. ``Loading run range 1276–1289: 14 of 14 runs found.``

The prefill scan is capped at 20,000 directory entries so a huge facility
folder (often on a slow network mount) cannot stall the dialog. When the cap
is hit, a warning appears under the folder field — *"This folder has too many
files to scan in full — showing the first N run files found. Adjust the range
by hand if runs are missing."* — and the prefilled first/last numbers may
cover only part of the folder; type the range you want and the load proceeds
normally.

**In scripts.** :func:`~asymmetry.core.io.resolve_run_range` is the pure,
GUI-free resolver behind the dialog. It scans a folder and returns the existing
files for an inclusive run range, sorted by run number:

.. code-block:: python

   from asymmetry.core.io import load, resolve_run_range

   files = resolve_run_range("data/BiSCCO", 1276, 1289, prefix="MUSR")
   datasets = [load(str(path)) for path in files]

Resolver semantics:

* **Inclusive range.** Both ``first`` and ``last`` are included. The result is
  sorted ascending by run number.
* **Padding-agnostic.** The run number is parsed from the trailing digits of
  each file name, so any zero-pad width matches — ``MUSR00001276.nxs`` resolves
  to run ``1276``.
* **Prefix.** ``prefix`` (the leading text before the run-number digits) is
  matched case-insensitively. When omitted it is auto-detected: if the files in
  range all share one prefix it is used, and if several prefixes are present a
  :class:`ValueError` lists them so you can pass an explicit ``prefix=``.
* **Extensions.** Only files with a loader-registered extension are considered
  (so sidecar logs such as ``.mon``/``.txt`` are ignored). Pass ``ext="nxs"``
  to restrict the scan to a single format.
* **Missing runs are skipped.** Gaps in the range are not an error — the
  resolver returns the runs that exist and omits the rest. The GUI separately
  warns when the loaded runs are non-contiguous.
* **Errors.** A missing or non-directory folder, or ``first > last``, raises
  :class:`ValueError`. A valid folder with no matching runs in range returns an
  empty list (the GUI reports this as "no run files found").

Direct file format access
--------------------------

For advanced users, you can access the low-level file loaders. For NeXus files:

.. code-block:: python

   from asymmetry.core.io.nexus import NexusLoader

   loader = NexusLoader()
   dataset_or_periods = loader.load("HIFI00206453.nxs")

For ROOT files:

.. code-block:: python

   from asymmetry.core.io.root import RootLoader

   loader = RootLoader()
   dataset = loader.load("lem15_his_2994.root")

Loader registry and custom formats
----------------------------------

``LoaderRegistry`` maps file extensions to loader classes.

.. code-block:: python

   from asymmetry.core.io.base import LoaderRegistry

   print(LoaderRegistry.supported_extensions())
   print(LoaderRegistry.file_dialog_filter())

You can register custom loader classes at runtime for additional formats.

Runnable examples
-----------------

See the executable scripts:

* ``examples/basic_dataset_loading.py``
* ``examples/custom_loader.py``
