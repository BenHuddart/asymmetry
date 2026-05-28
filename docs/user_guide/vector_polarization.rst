Vector Polarization Mode
========================

.. image:: /_generated/screenshots/vector_polarization_emu.png
   :alt: Three EMU vector-polarization projections P_x, P_y, P_z overlaid
   :width: 100%

*Synthetic EMU-style three-axis polarisation projections overlaid in the*
*central plot. The* P_z *trace carries the dominant slow exponential decay,*
*P_x *carries a weak transverse oscillation, and* P_y *is centred near*
*zero with statistical noise — the standard signature of a sample whose*
*local field is aligned along the* z *axis of the spectrometer (textbook*
*Ch 6.3).*

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
* Older projects are migrated so vector groupings initialize
  ``alpha_x``, ``alpha_y``, and ``alpha_z`` from scalar ``alpha``.

Display in the Main Plot
------------------------

The Polarization selector in the plot header provides:

* ``x`` (``P_x``)
* ``y`` (``P_y``)
* ``z`` (``P_z``)
* ``All``

Alpha display behavior:

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

Detector Group Composition
--------------------------

EMU vector mode follows octant-style detector composition. The detector layout
reference in :doc:`detector_grouping` should be used for instrument-consistent
verification of assigned groups.

Related Topics
--------------

* :doc:`detector_grouping`
* :doc:`gui_usage`
* :doc:`data_processing`
