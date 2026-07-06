GUI components
==============

.. currentmodule:: asymmetry.gui

The GUI is a PySide6 application organised around a
:class:`~asymmetry.gui.mainwindow.MainWindow` that hosts the four
persistent panels (data browser, plot, fit, Fourier) plus a log dock
and a workspace selector for the time / individual-groups / frequency
views. Non-modal dialog windows (Grouping, Detector Layout, Fit Wizard,
Global Fit Wizard, Run Info, GLE setup) are launched from the main
window. The end-user reference for the GUI lives in
:doc:`/reference/gui_usage`; this API page lists the classes for
developers who need to embed or extend the components, not for
day-to-day use.

Main application
----------------

.. automodule:: asymmetry.gui.app
   :members:
   :undoc-members:

Main window
-----------

.. autoclass:: asymmetry.gui.mainwindow.MainWindow
   :members:
   :undoc-members:
   :show-inheritance:

Panels
------

Data Browser
~~~~~~~~~~~~

.. autoclass:: asymmetry.gui.panels.data_browser.DataBrowserPanel
   :members:
   :undoc-members:
   :show-inheritance:

Plot panel
~~~~~~~~~~

.. autoclass:: asymmetry.gui.panels.plot_panel.PlotPanel
   :members:
   :undoc-members:
   :show-inheritance:

Fit panel
~~~~~~~~~

.. autoclass:: asymmetry.gui.panels.fit_panel.FitPanel
   :members:
   :undoc-members:
   :show-inheritance:

Fourier panel
~~~~~~~~~~~~~

.. autoclass:: asymmetry.gui.panels.fourier_panel.FourierPanel
   :members:
   :undoc-members:
   :show-inheritance:

Log panel
~~~~~~~~~

.. autoclass:: asymmetry.gui.panels.log_panel.LogPanel
   :members:
   :undoc-members:
   :show-inheritance:

Dialogs
-------

Grouping dialog
~~~~~~~~~~~~~~~

.. autoclass:: asymmetry.gui.windows.grouping_dialog.GroupingDialog
   :members:
   :undoc-members:
   :show-inheritance:

Detector layout dialog
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: asymmetry.gui.windows.detector_layout_dialog.DetectorLayoutDialog
   :members:
   :undoc-members:
   :show-inheritance:

Widgets
-------

Detector schematic
~~~~~~~~~~~~~~~~~~

.. autoclass:: asymmetry.gui.widgets.detector_schematic.DetectorSchematicWidget
   :members:
   :undoc-members:
   :show-inheritance:
