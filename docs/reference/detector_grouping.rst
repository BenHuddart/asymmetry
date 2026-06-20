Detector Grouping and Layout
============================

Detector grouping defines how the raw per-detector count histograms are
combined into the forward and backward sums that enter the asymmetry
formula. The choice is dictated by the experiment: a coarse two-group
arrangement is right for ordinary asymmetry analysis on a longitudinal
spectrometer; keeping detectors as individual groups is what enables the
per-detector amplitudes and phases needed for a paramagnetic Knight-shift
measurement (:doc:`grouped_time_domain_fitting`); and a vector-polarisation
experiment needs distinct pairs assigned to :math:`P_x`, :math:`P_y`, and
:math:`P_z` (:doc:`vector_polarization`). For new data from any of the
supported instruments — ISIS HiFi, MuSR, EMU and the PSI FLAME, HAL-9500 and
GPS spectrometers — the matching preset in the Detector Layout editor seeds
sensible defaults that can then be refined graphically before being applied.

Grouping is configured from the Grouping dialog and edited graphically
with the Detector Layout editor.

The grouping payload stores:

* detector groups (1-based detector IDs)
* per-group include flags for grouped plotting and grouped fitting
* group names
* selected forward/backward groups
* alpha and bin-range settings
* instrument and preset metadata
* optional per-detector ``t0`` metadata for formats such as PSI BIN/MDU and
  MusrRoot/LEM ROOT
* optional deadtime metadata: ``deadtime_mode``, ``deadtime_method``, and any
   resolved ``dead_time_us`` values used for manual, calibrated, or estimated
   deadtime correction
* optional background metadata: ``background_correction``,
  ``background_ranges``, ``background_values``, and ``background_method``

These settings are persisted in project files and in ``.grp`` files.

PSI Grouping
------------

PSI BIN/MDU files use the full Grouping dialog, matching the interaction model
used for raw ISIS NeXus files. Initial group names and forward/backward
defaults are derived from PSI detector labels where the file provides them
(``Forw``/``Back`` in BIN files, and labels such as ``F1``/``B1`` in MDU
files). For PSI FLAME BIN files, filenames, detector labels, or metadata
containing ``FLAME`` select the FLAME detector layout automatically; this
includes PSI instrument strings such as ``LMU_BULKMUSR_FLAME``. PSI HAL-9500
runs (the high-field πE3 spectrometer) are recognised from their ``HIFI``
instrument string and ``tdc_hifi_*`` run names and open with the HAL-9500
octagonal layout; this is a distinct instrument from the ISIS HiFi
spectrometer despite the shared ``hifi`` token. PSI GPS BIN files
(``deltat_tdc_gps_*`` with the six ``Forw/Back/Up/Down/Righ/Left`` histograms)
select the GPS two-panel layout automatically (see `PSI GPS`_ below). This
behaviour
follows the detector metadata exposed by musrfit's PSI raw-data reader, with
Mantid's PSI-BIN loader used as a cross-check for BIN layout details.
When labels repeat, Asymmetry keeps one visible group per histogram and makes
the displayed names unique with numeric suffixes.

PSI data can carry a separate ``t0`` for each detector. Asymmetry stores these
values as ``detector_t0_bins`` and aligns each detector histogram to its own
``t0`` before summing groups. This avoids shifting all PSI spectra through a
single global time-zero before grouping.

PSI detector names use the PSI instrument convention: ``Forward`` and
``Backward`` are measured along the beam direction. Asymmetry's pair-asymmetry
formula uses forward/backward relative to the initial muon spin direction.
For PSI runs, the detector layout editor keeps the PSI detector convention,
but the main Grouping dialog swaps the analysis dropdown defaults. For a
longitudinal PSI/FLAME layout, **Forward Group** is set to ``Group 2:
Backward`` and **Backward Group** is set to ``Group 1: Forward``. ISIS runs are
not swapped.

ROOT Grouping
-------------

MusrRoot/LEM ROOT files also use the full Grouping dialog. The ROOT loader
follows musrfit's ``PRunDataHandler::ReadRootFile`` and reads detector labels
from ``DetectorInfo`` when available, falling back to the ROOT histogram title
or ``hDecay`` name. As with PSI BIN/MDU files, repeated detector labels are
kept as separate visible groups with numeric suffixes.

ROOT files with ``RunInfo/Instrument`` set to ``FLAME`` are opened with the
FLAME detector layout available by default. If that metadata field is absent,
Asymmetry also recognises ``flame`` in the source filename. GPS MusrRoot files
(instrument ``LMU_BULKMUSR_GPS``) expose eleven raw sub-detectors and open with
the eleven-detector GPS variant (see `PSI GPS`_ below).

ROOT ``DetectorInfo`` entries can provide detector-specific ``Time Zero Bin``,
``First Good Bin``, and ``Last Good Bin`` values. Asymmetry stores these in the
grouping payload and aligns detector histograms by their own ``t0`` before
constructing the initial asymmetry.

Deadtime Correction
-------------------

The Grouping dialog includes a deadtime correction toggle plus three deadtime
modes for raw histogram formats. ``File`` stays selected by default so the
deadtime workflow starts from the same file-first assumption WiMDA uses, even
when the current reference run does not provide file deadtime values.

* ``File`` uses per-detector deadtime values already present in a run file.
* ``Manual`` exposes a detector-value combo box. It shows one deadtime value
  per detector, allows direct editing, and is also where calibrated values are
  stored.
* ``Estimate`` fits the reference run's early-time average detector rate,
   following WiMDA's uniform deadtime estimate workflow, then applies that one
   estimated value to every detector.

The deadtime panel also includes a ``Cal`` button that ports WiMDA's
per-detector calibration routine. It fits each detector histogram in the
reference run separately, produces a resolved per-detector deadtime table, and
then populates the manual detector-value table with those calibrated values.

When the checkbox is off, deadtime correction is disabled. When ``Estimate`` is
selected, the estimate is calculated from the currently selected reference run
only, then applied to every checked run in the dialog. ``Cal`` also uses the
selected reference run only, but calibrates one deadtime value per detector
instead of a single shared value. The resolved deadtime payload is also
preserved in grouping metadata, so future datasets loaded into the same project
inherit the same manual, calibrated, or estimated deadtime values just
as they already inherit alpha and grouping settings.

All modes use the same non-paralysable correction form used by musrfit
``PRunBase::DeadTimeCorrection`` and Mantid ``ApplyDeadTimeCorr``:

.. math::

   N_\mathrm{corr} =
   \frac{N}{1 - N\,t_\mathrm{dead}/(\Delta t\,N_\mathrm{frames})}

For ``File`` mode, ``t_dead`` comes from the run file when available. For
``Manual``, ``Cal``, and ``Estimate`` workflows, Asymmetry resolves the
deadtime values in the Grouping dialog and stores them in grouping metadata
before the correction is applied. ``N_frames`` still comes from each dataset's
own good-frame metadata when it is available.

PSI BIN/MDU and MusrRoot/LEM ROOT data usually do not ship NeXus-style file
deadtime constants, so ``File`` mode is commonly unavailable there. Those runs
can still use ``Manual`` or ``Estimate`` deadtime correction, with ``Cal``
available to populate per-detector manual values from the selected reference
run. The
background correction path remains separate and optional.

Background Correction
---------------------

The full Grouping dialog also includes a background correction toggle for
PSI-style raw histogram formats, including PSI BIN/MDU and PSI/LEM ROOT data.
This is separate from fit-model background parameters such as ``A_bg``: it
subtracts a count background from grouped raw forward/backward histograms
before the asymmetry is calculated.

This follows musrfit's ``PRunAsymmetry`` ordering. Histograms are first grouped
into forward and backward sums, then background is subtracted, and then
asymmetry is calculated. If grouping metadata provide fixed forward/backward
background values, those values are subtracted. Otherwise Asymmetry estimates
the background as the mean count in an inclusive bin range. If no range is
provided, it uses musrfit's fallback range from ``0.1 * t0`` to ``0.6 * t0``.
For corrected histograms, Asymmetry propagates musrfit-style count
uncertainties through its standard pair formula, with ``alpha`` multiplying the
backward group.

Background subtraction can make late-time corrected forward/backward sums very
small or negative. Those bins may therefore produce asymmetries at or beyond
``+/-100%``. The plot keeps such low-confidence PSI points visible in grey,
matching the low-count visual treatment used for raw grouped data, and excludes
them from automatic Y-axis scaling.

The correction is off by default and disabled for ISIS/NeXus data, where
deadtime correction is the file-metadata correction path. When enabled for PSI
data, the applied method, estimated values, and ranges are stored in the
grouping payload.

Detector Layout Editor Workflow
-------------------------------

1. Open Grouping from the toolbar or menu.
2. Click Detector Layout...
3. Choose instrument and preset in the right-hand panel.
4. Click detector sectors in the schematic to refine groups.
5. Apply and return to the Grouping dialog.

A detector can belong to multiple groups. This is required for transverse and
vector-polarisation workflows.

The Grouping table includes an **Include** checkbox for each group. This does
not change the stored detector membership of the group. Instead, it controls
whether that group participates in the **Individual Groups** plot view and in
grouped time-domain fitting.

In-App Arrangement Schematics
-----------------------------

HiFi
~~~~

.. figure:: images/hifi-program-schematic.png
   :width: 90%
   :align: center
   :alt: HiFi detector schematic generated from the program layout model.

   HiFi schematic matching the in-app detector arrangement.

MuSR
~~~~

.. figure:: images/musr-program-schematic.png
   :width: 90%
   :align: center
   :alt: MuSR detector schematic generated from the program layout model.

   MuSR schematic matching the in-app detector arrangement.

EMU
~~~

.. figure:: images/emu-program-schematic.png
   :width: 90%
   :align: center
   :alt: EMU detector schematic generated from the program layout model.

   EMU schematic matching the in-app detector arrangement.

PSI FLAME
~~~~~~~~~

.. figure:: images/flame-program-schematic.png
   :width: 90%
   :align: center
   :alt: PSI FLAME detector schematic generated from the program layout model.

   PSI FLAME top-view detector layout. The beam and main magnetic field are
   drawn along +z, and the initial muon spin points toward the Backward
   detector. FLAME detectors are rectangular plates: 1 Forward, 2 Backward,
   3 Right, 4 Left, 5 R_F, 6 R_B, 7 L_F, and 8 L_B. The Left and Right banks
   use equal-height rectangles with the central detector drawn wider than the
   front/back side detectors.

PSI HAL-9500
~~~~~~~~~~~~

.. figure:: images/hal-program-schematic.png
   :width: 90%
   :align: center
   :alt: PSI HAL-9500 detector schematic generated from the program layout model.

   PSI HAL-9500 detector layout, viewed along the beam axis. The 16 positron
   detectors form two octagonal rings of eight — a forward ring (F1–F8) and a
   backward ring (B1–B8) — drawn as separate octagons. The central muon-veto
   detector (MV) is shown at the centre of the forward ring. The histograms are
   stored in the order ``MV, F1…F8, B1…B8``, so detector IDs run MV → 1,
   F1–F8 → 2–9, and B1–B8 → 10–17. Presets include **Longitudinal**
   (forward ring vs backward ring), **Transverse (opposed pairs)** (each
   forward detector as its own group, defaulting to the F1–F5 diametric pair),
   and **Per-octant** (each azimuthal sector combining its forward and
   backward wedge).

PSI GPS
~~~~~~~

.. figure:: images/gps-program-schematic.png
   :width: 95%
   :align: center
   :alt: PSI GPS detector schematic generated from the program layout model.

   PSI GPS layout, drawn as two plan panels. The beam runs along +z toward the
   Forward detector. GPS surrounds the sample with six positron detectors on
   three orthogonal axes — Forward/Backward (beam), Up/Down (vertical) and
   Left/Right (horizontal-transverse) — so a single flat view cannot place all
   six. The **Top view** shows the horizontal plane (Forward/Backward and
   Left/Right in place; Up/Down drawn end-on, ⊙ toward you and ⊗ away); the
   **Side view** shows the vertical plane (Up/Down in place; Forward/Backward
   read-only; Left/Right end-on). Each detector is editable in its home panel and
   shown read-only for context in the other.

GPS is recognised automatically from PSI data carrying a ``GPS`` instrument
string or a ``deltat_tdc_gps_*`` run name. Two histogram conventions are
supported and presented to the user as a single "GPS" layout, selected
automatically from the histogram count:

* the **PSI-BIN** export with six combined detectors (``Forw, Back, Up, Down,
  Righ, Left``); and
* the **MusrRoot** export with eleven raw sub-detectors (``Forw, Back, Up_B,
  Up_F, Down_B, Down_F, Right_B, Right_F, Left_B, Left_F, Mob-RL``), where each
  transverse plate is split into an upstream (``_B``) and downstream (``_F``)
  half and a Mobile detector is added.

Detector IDs match the histogram order in each format (detector *N* maps to
histogram *N − 1*). Presets:

* **Longitudinal** (default) — Forward vs Backward.
* **Transverse (Vector)** — the Up–Down and Left–Right pairs exposed as two
  asymmetry projections (musrfit's ``WED(L)`` transverse setup).
* **Spin-rotated (F+U/B+D)** — Forward+Up vs Backward+Down. When the spin
  rotator (a Wien filter on πM3.2; GPS User Guide, Section 13) is used in
  transverse geometry the muon spin is rotated up by about 50°, so the
  polarisation points along the Forward–Up diagonal; summing those detectors
  realigns one asymmetry axis with the rotated spin and recovers the full
  amplitude.
* **WEP (spin-rotated)** — the same rotated-spin mode following musrfit's
  convention. This preset **follows musrfit's GPS instrument
  definition** (``musredit_qt5/musrWiz/instrument_defs/instrument_def_psi.xml``,
  ``<tf name="WEP">``): rather than summing detectors it keeps Forward, Backward,
  Up and Down as four separate groups and exposes the **FB** and **UD**
  asymmetry pairs. The FB projection declares musrfit's default ``alpha = 0.75``
  and UD ``alpha = 1.0``, and the reduction applies each projection's own
  declared alpha — reducing or fitting the FB pair uses 0.75 and the UD pair uses
  1.0. The per-detector phase offsets musrfit uses to encode the rotation are a
  fitting detail and are not stored in the layout.

The Mobile sub-detector (``Mob-RL``) is left ungrouped by default: it is added
to either the Right or Left detector depending on the cryostat port in use,
which the data file does not record.

Related Topics
--------------

* :doc:`data_processing` for grouping and asymmetry APIs
* :doc:`gui_usage` for UI workflows
* :doc:`vector_polarization` for vector mode (P_x, P_y, P_z)
