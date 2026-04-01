Vector Polarization Mode
========================

Overview
--------

Vector polarization mode provides three detector-pair projections of the muon
polarization:

* ``P_x`` from left/right detector groups
* ``P_y`` from top/bottom detector groups
* ``P_z`` from forward/backward detector groups

In Asymmetry, vector mode is activated when grouping names contain canonical
vector pairs:

* ``Pz Forward`` and ``Pz Backward``
* ``Py Top`` and ``Py Bottom`` (or ``Py Up`` and ``Py Down``)
* ``Px Left`` and ``Px Right``

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
