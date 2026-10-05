Vector-polarisation mode
========================

.. image:: /_generated/screenshots/vector_polarization_emu.png
   :alt: Three EMU vector-polarisation projections P_x, P_y, P_z overlaid
   :width: 100%

*Synthetic EMU-style three-axis polarisation projections overlaid in the*
*central plot. The* P_z *trace carries the dominant slow exponential decay,*
P_x *carries a weak transverse oscillation, and* P_y *is centred near*
*zero with statistical noise — the standard signature of a sample whose*
*local field is aligned along the* z *axis of the spectrometer.*

Vector-polarisation mode treats the muon-spin polarisation as a
three-component vector, exposing the :math:`P_x`, :math:`P_y`, and
:math:`P_z` projections separately rather than collapsing the detector
counts onto a single forward/backward asymmetry. This is the right
analysis path for anisotropic single crystals — where the precession
axis is set by the crystallography rather than by the spectrometer
geometry — and for any measurement where the local field at the muon
site is canted away from :math:`\hat{z}`, since the off-axis precession
is then carried by the :math:`P_x` and :math:`P_y` components and is
lost in a one-dimensional asymmetry. Powder samples have an orientational
average that already collapses onto a single non-trivial component along
:math:`\hat{z}`, so the ordinary F-B asymmetry workflow is sufficient
there. EMU's octant geometry is the canonical example; vector mode is
activated automatically when grouping names contain the canonical vector
pairs:

* ``Pz Forward`` / ``Pz Backward`` from forward/backward detector groups
* ``Py Top`` / ``Py Bottom`` (or ``Py Up`` / ``Py Down``) from top/bottom
  detector groups
* ``Px Left`` / ``Px Right`` from left/right detector groups

so the same vector workflow can be applied to any instrument whose
detector layout supports the same six-group naming convention.

Setup
-----

1. Open Grouping and choose (or create) a profile for EMU.
2. Open Detector Layout...
3. Select instrument ``EMU``.
4. Apply the ``Vector Polarization`` preset.
5. Return to Grouping and calibrate each axis's alpha (see below).

Per-axis alpha
--------------

When vector mode is active, the Grouping window switches to a vector table
with separate alpha values for each axis:

* ``alpha_x``
* ``alpha_y``
* ``alpha_z``

Each axis is its own asymmetry projection (see :doc:`detector_grouping`,
"Per-projection alpha"), so each calibrates independently through the same
alpha calibration dialog used for an ordinary two-group run — open it per
axis with that row's **Calibrate…** button, or calibrate all three in one
action with **Estimate All alpha**. Because each axis's alpha and provenance
are stored on its own projection, switching between axes never mixes up
which value came from which calibration run.

Backwards compatibility:

* Existing scalar ``alpha`` is still supported.
* Older projects are migrated so vector groupings initialise
  ``alpha_x``, ``alpha_y``, and ``alpha_z`` from scalar ``alpha``.

Display in the main plot
------------------------

The plot toolbar shows one chip per projection — ``P_x``, ``P_y`` and
``P_z``, each in its own tint. Click a chip to show or hide that projection;
selecting more than one stacks them as subplots, and **all** selects every
projection at once. At least one chip always stays selected. A run whose
grouping has a single forward/backward pair has only one polarisation
component, so it gets one plain plot and no chips. Each grouping profile
remembers its own chip selection: browse from a vector run showing ``P_x``
and ``P_z`` to a single-pair run and back, and both subplots return, while a
run on another profile keeps the selection last made there. A run released
from its profile remembers its own, and an overlay of runs from several
profiles remembers that combination. Renaming a profile keeps its
selection, and the remembered selections are saved with the project.

The chips never make the plot wider. When the toolbar is too narrow for the
full names they switch to short ones (a detector pair such as ``Top-Bottom``
becomes ``T–B``; the full name stays in the tooltip). When even those do not
fit, the chips fold into a single button that summarises the selection — the
selected projection's short name (``P_x ▾``), ``2 of 3 ▾``, or ``All 3 ▾`` —
and opens a menu of the projections with a **Show all** entry.

Alpha display behaviour:

* Single-axis views show the alpha for the selected axis.
* Views with several projections hide alpha in the header.

Runs from different groupings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Overlaying runs whose groupings differ shows every projection any of them
declares, and each subplot holds only the runs that measure that projection.
A single-pair run joins the projection whose forward and backward groups hold
exactly its detectors on the same instrument: an EMU **Longitudinal** run sits
on the ``P_z`` subplot beside the vector runs and is absent from ``P_x`` and
``P_y``; a GPS **Longitudinal** run joins the **WEP (spin-rotated)** ``FB``
projection. A run keeps one trace colour on every subplot. When a single-pair
run matches no projection (a MuSR **Longitudinal** run beside the
**Transverse (Vector)** grouping, whose detector split differs), the overlay
falls back to one plain plot without chips.

A fit on a single-pair run always belongs to the run's own asymmetry, even
when it is selected from a shared projection subplot.

Persistence
-----------

Per-axis alpha values are persisted in:

* project files (schema v4+)
* dataset grouping state

This preserves axis-specific alpha values across save/load cycles and across
axis switching in vector mode.

Transverse-field dual grouping
------------------------------

The same projection workflow generalises beyond EMU's three-axis vector mode.
A forward/backward asymmetry *is* the muon polarisation projected onto the axis
joining that detector pair, so any preset that exposes more than one such pair is
a set of projections. MuSR and HiFi each ship a combined ``Transverse (Vector)``
preset that exposes **two** transverse projections of the same run:

* MuSR — ``Top-Bottom`` and ``Fwd-Back``
* HiFi — ``Left-Right`` and ``Top-Bottom``

Apply it from Detector Layout exactly as for EMU vector mode (select the
instrument, apply the ``Transverse (Vector)`` preset). The projection chip bar
then shows one chip per transverse projection; selecting two stacks them as
subplots, and clicking a subplot makes it the fit target with its own
per-projection single fit — identical behaviour to the EMU :math:`P_x`/
:math:`P_y`/:math:`P_z` projections, including the tinted ``Fitting: <label>``
echo and save/load persistence. Unlike EMU's octant model, the two transverse
pairs use four distinct detector groups so both coexist (the legacy split
presets reused the same group IDs and were mutually exclusive).

Detector group composition
--------------------------

EMU vector mode follows octant-style detector composition. The detector layout
reference in :doc:`detector_grouping` should be used for instrument-consistent
verification of assigned groups.

EMU has no facility-documented "vector-polarisation" grouping — the EMU User
Guide and the Mantid EMU instrument definition only describe the physical
detector numbering (Section 8.1), not a Px/Py/Pz preset. The ``Vector
Polarization`` preset is therefore an **Asymmetry construct**: it is verified
internally consistent (each octant selection matches the geometric
half-plane of the layout's own detector angles) but is not itself a
published EMU convention.

Assumptions and limitations
---------------------------

The vector treatment rests on a small set of geometry assumptions; read the
projections with them in mind:

- **Each projection is a forward/backward asymmetry along one detector-pair
  axis.** A forward/backward asymmetry measures the muon polarisation projected
  onto the axis joining that detector pair, so :math:`P_x`, :math:`P_y`, and
  :math:`P_z` are the three orthogonal projections only insofar as the three
  detector-pair axes are mutually orthogonal and aligned with the spectrometer
  frame. On a real instrument the octant groups approximate those axes; the
  reconstruction is exact only for an idealised orthogonal layout.
- **Per-axis α decouples the three projections.** Each axis carries its own
  calibration constant (``alpha_x`` / ``alpha_y`` / ``alpha_z``) and is reduced
  independently, so a miscalibrated α on one axis biases that projection's
  amplitude without contaminating the other two — but all three must be
  calibrated for the vector to be quantitatively balanced.
- **Powder samples do not need it.** An orientational average collapses the
  polarisation onto a single non-trivial component along :math:`\hat{z}`, so the
  ordinary F–B asymmetry workflow is sufficient; the vector projections add
  information only when the local field is canted away from :math:`\hat{z}`, as
  in an oriented single crystal.
- **The EMU Px/Py/Pz preset is an Asymmetry construct**, not a published EMU
  convention (see `Detector Group Composition`_): it is verified internally
  consistent against the layout's own detector angles, but should be
  cross-checked against the facility detector numbering before quantitative use.

References
----------

- S. J. Blundell, R. De Renzi, T. Lancaster, and F. L. Pratt,
  *Muon Spectroscopy: An Introduction* (Oxford University Press, Oxford, 2022) —
  detector geometry and the polarisation-projection observable.

Related topics
--------------

* :doc:`detector_grouping`
* :doc:`gui_usage`
* :doc:`data_processing`
