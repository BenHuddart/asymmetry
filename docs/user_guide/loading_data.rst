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
modes. Multi-period ISIS data is currently limited to returning every
period as a list; explicit period arithmetic in the analysis path is on
the roadmap (:doc:`/user_guide/comparison`).

For an end-to-end walk-through that starts with loading, see
:doc:`workflows/temperature_scan_magnetism`.

Supported Formats
-----------------

ISIS Muon NeXus (.nxs, .nexus)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Asymmetry supports ISIS muon NeXus files (legacy V1 and modern V2), including
multi-period runs. Multi-period files return a list of ``MuonDataset`` values.
Scalar sample temperature is read from the NeXus sample header. NXlog-style
temperature groups with ``time`` plus ``value``/``values`` datasets are also
summarized in Get Info; the Data Browser keeps the scalar header temperature
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
zero-based array indices. For ISIS NeXus V2 files, some real files encode
integer bin metadata using one-based centre-bin numbering. The loader compares
explicit ``t0_bin`` values with the time axis; when all available detector
``t0`` values point one sample past ``t = 0``, it subtracts one from
``t0_bin``, ``first_good_bin``, and ``last_good_bin`` and records
``bin_index_base = 1`` in grouping metadata. Otherwise it leaves the integers
unchanged with ``bin_index_base = 0``. NeXus V1 is currently read as
zero-based.

PSI BIN/MDU (.bin, .mdu)
~~~~~~~~~~~~~~~~~~~~~~~~

Asymmetry supports PSI raw histogram files in the classic PSI-BIN and PSI-MDU
formats. These files are loaded into normal raw histogram runs, so they use the
same full Grouping dialog as ISIS NeXus data.

This loader is intentionally based on established PSI readers rather than a
new file interpretation. The BIN/MDU binary layout and metadata offsets follow
the musrfit PSI reader, especially ``PRunDataHandler::ReadPsiBinFile`` and the
``MuSR_td_PSI_bin``/MDU structures it uses. Mantid's ``LoadPSIMuonBin`` was
also checked for PSI-BIN behavior. Asymmetry keeps the implementation in pure
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
therefore initialized with the beam-forward and beam-backward detector groups
swapped. The detector layout editor still displays the PSI detector convention.

If the explicit field entry is missing or zero but the PSI comment/title text
contains a recognizable field such as ``LF 32G`` or ``Bz=150 G``, the GUI
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
  newer ``TDirectory`` layout are supported.
* Histogram search follows musrfit's paths: ``histos/hDecay%03d`` for folder
  files and ``histos/DecayAnaModule/hDecay%03d`` for directory files.
* ``RunInfo`` values such as run number, title, laboratory, instrument,
  sample, temperature, field, time resolution, number of histograms, and
  red/green offsets are read from the header when present.
* ``DetectorInfo`` entries provide detector labels, histogram numbers,
  per-detector ``t0`` bins, and detector-specific good-bin ranges when the
  file supplies them.
* MusrRoot slow-control histograms in ``histos/SCAnaModule`` are imported as
  plottable ``nexus_time_series`` logs. In particular, ``hSampleTemperature``
  is summarized as an average temperature in Get Info while the scalar
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

The format and correction behavior is split by source so that the application
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
  ``N_corr = N / (1 - N * dead_time / (time_bin * good_frames))``.
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

Grouping behavior by format
~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the GUI, supported raw data runs use the full Grouping dialog:

* NeXus runs use the full Grouping dialog.
* PSI BIN/MDU runs use the full Grouping dialog, with PSI label-derived
   forward/backward defaults when labels such as ``Forw``/``Back`` or
   ``F1``/``B1`` are present.
* MusrRoot/LEM ROOT runs use the full Grouping dialog. Detector names from
   ``DetectorInfo`` or ROOT histogram titles become the initial group names,
   and ROOT per-detector ``t0`` values are handled like PSI BIN/MDU ``t0``
   metadata.

Basic Usage
-----------

Loading a Single File
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

Accessing Metadata
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   dataset = load("data.nxs")
   
   print(f"Run number: {dataset.run_number}")
   print(f"Temperature: {dataset.metadata.get('temperature')} K")
   print(f"Field: {dataset.metadata.get('field')} G")
   print(f"Title: {dataset.metadata.get('title')}")

Loading Multiple Files
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pathlib import Path
   
   data_dir = Path("data")
   datasets = []
   
   for nexus_file in data_dir.glob("*.nxs"):
       dataset = load(str(nexus_file))
       datasets.append(dataset)
   
   print(f"Loaded {len(datasets)} datasets")

Direct File Format Access
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

Loader Registry and Custom Formats
----------------------------------

``LoaderRegistry`` maps file extensions to loader classes.

.. code-block:: python

   from asymmetry.core.io.base import LoaderRegistry

   print(LoaderRegistry.supported_extensions())
   print(LoaderRegistry.file_dialog_filter())

You can register custom loader classes at runtime for additional formats.

Runnable Examples
-----------------

See the executable scripts:

* ``examples/basic_dataset_loading.py``
* ``examples/custom_loader.py``
