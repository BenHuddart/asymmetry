Vector Polarization Mode
========================

.. image:: /_generated/screenshots/vector_polarization_emu.png
   :alt: Three EMU vector-polarization projections P_x, P_y, P_z overlaid
   :width: 100%

*Synthetic EMU-style three-axis polarisation projections overlaid in the*
*central plot. The* P_z *trace carries the dominant slow exponential decay,*
P_x *carries a weak transverse oscillation, and* P_y *is centred near*
*zero with statistical noise — the standard signature of a sample whose*
*local field is aligned along the* z *axis of the spectrometer.*

Vector polarization mode treats the muon-spin polarisation as a
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

1. Open Grouping.
2. Open Detector Layout...
3. Select instrument ``EMU``.
4. Apply the ``Vector Polarization`` preset.
5. Return to Grouping and adjust alpha values.

Per-Axis Alpha
--------------

When vector mode is active, the Grouping dialog switches to a vector table with
separate alpha values for each axis:

* ``alpha_x``
* ``alpha_y``
* ``alpha_z``

You can estimate alpha per axis with row-level Estimate buttons, or use
Estimate All alpha to calculate all three in one action.

Backwards compatibility:

* Existing scalar ``alpha`` is still supported.
* Older projects are migrated so vector groupings initialise
  ``alpha_x``, ``alpha_y``, and ``alpha_z`` from scalar ``alpha``.

Display in the Main Plot
------------------------

The Polarization selector in the plot header provides:

* ``x`` (``P_x``)
* ``y`` (``P_y``)
* ``z`` (``P_z``)
* ``All``

Alpha display behaviour:

* Single-axis views show the alpha for the selected axis.
* ``All`` mode hides alpha in the header.

Persistence
-----------

Per-axis alpha values are persisted in:

* project files (schema v4+)
* dataset grouping state
* ``.grp`` files

This preserves axis-specific alpha values across save/load cycles and across
axis switching in vector mode.

Transverse-Field Dual Grouping
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

Detector Group Composition
--------------------------

EMU vector mode follows octant-style detector composition. The detector layout
reference in :doc:`detector_grouping` should be used for instrument-consistent
verification of assigned groups.

EMU has no facility-documented "vector polarization" grouping — the EMU User
Guide and the Mantid EMU instrument definition only describe the physical
detector numbering (Section 8.1), not a Px/Py/Pz preset. The ``Vector
Polarization`` preset is therefore an **Asymmetry construct**: it is verified
internally consistent (each octant selection matches the geometric
half-plane of the layout's own detector angles) but is not itself a
published EMU convention.

Related Topics
--------------

* :doc:`detector_grouping`
* :doc:`gui_usage`
* :doc:`data_processing`
