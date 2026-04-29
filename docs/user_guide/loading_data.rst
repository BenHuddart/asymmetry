Loading Data
============

Asymmetry supports ISIS muon NeXus, PSI BIN/MDU, and MusrRoot/LEM ROOT files
through a common API.

Supported Formats
-----------------

ISIS Muon NeXus (.nxs, .nexus)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Asymmetry supports ISIS muon NeXus files (legacy V1 and modern V2), including
multi-period runs. Multi-period files return a list of ``MuonDataset`` values.

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

If the explicit field entry is missing or zero but the PSI comment/title text
contains a recognizable field such as ``LF 32G`` or ``Bz=150 G``, the GUI
offers to apply that comment-derived value as the run field.

PSI-BIN temperature metadata has two sources. The scalar run temperature is
read from the BIN header using the musrfit-compatible offsets. Optional
temperature-log sidecars use Mantid's ``LoadPSIMuonBin`` convention: Asymmetry
searches from the BIN file directory, to three directory levels below it, for
a ``.mon`` file whose filename contains the run number. The ``.mon`` header
date/title and backslash-delimited rows are parsed into plottable
``nexus_time_series`` entries named like ``psi_temperature/Temp_<channel>``.
The Get Info window shows these logs in the summary/advanced tables and records
the sidecar path plus ``Mantid LoadPSIMuonBin-compatible`` provenance in
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

Legacy LEM ROOT files without a MusrRoot ``RunHeader`` are supported only when
their ``RunInfo`` and histogram objects are readable through ``uproot``. The
full PyROOT/TLemRunHeader object model used by very old files is not
reimplemented.

The optional ``uproot`` dependency is required for ROOT loading. Install with
the ``root`` extra if it is not already present.

Deadtime correction remains controlled by the Grouping dialog, but it is only
available when a loaded file provides per-detector deadtime values and
good-frame metadata. This keeps deadtime correction on the ISIS/NeXus-style
path. PSI BIN/MDU and MusrRoot/LEM ROOT files do not normally contain those
NeXus-style deadtime constants, so Asymmetry does not estimate a fallback.

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
  the fallback range is ``0.1 * t0`` to ``0.6 * t0``.
* File-based deadtime correction uses the non-paralyzable formula implemented
  by musrfit ``PRunBase::DeadTimeCorrection`` and Mantid
  ``ApplyDeadTimeCorr``:
  ``N_corr = N / (1 - N * dead_time / (time_bin * good_frames))``.
* musrfit's ``estimate`` deadtime mode is not implemented there. Asymmetry does
  not add its own deadtime estimator; if file deadtime values are missing,
  deadtime correction is unavailable for that run.
* Mantid's PSI-BIN loader can emit a deadtime table, but it fills the PSI-BIN
  table with zeros. Asymmetry therefore treats PSI deadtime as absent and uses
  the background-correction path instead.
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
