Detector Grouping and Layout
============================

Overview
--------

Asymmetry supports detector grouping for HiFi, MuSR, and EMU instruments.
Grouping is configured from the Grouping dialog and can be edited graphically
with the Detector Layout editor.

The grouping payload stores:

* detector groups (1-based detector IDs)
* group names
* selected forward/backward groups
* alpha and bin-range settings
* instrument and preset metadata

These settings are persisted in project files and in ``.grp`` files.

Detector Layout Editor Workflow
-------------------------------

1. Open Grouping from the toolbar or menu.
2. Click Detector Layout...
3. Choose instrument and preset in the right-hand panel.
4. Click detector sectors in the schematic to refine groups.
5. Apply and return to the Grouping dialog.

A detector can belong to multiple groups. This is required for transverse and
vector-polarization workflows.

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

Related Topics
--------------

* :doc:`data_processing` for grouping and asymmetry APIs
* :doc:`gui_usage` for UI workflows
* :doc:`vector_polarization` for vector mode (P_x, P_y, P_z)
