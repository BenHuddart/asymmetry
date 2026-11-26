Fitting
=======

.. py:module:: asymmetry.core.fitting

.. currentmodule:: asymmetry.core.fitting

To fit time-domain asymmetry with an expression from the GUI model builder, use
:class:`~asymmetry.core.fitting.composite.CompositeModel`. The
:ref:`worked damped-precession fit <python-damped-precession-fit>` compiles
``Oscillatory * Exponential + Constant``, seeds the generated parameter names,
and uses :class:`~asymmetry.core.fitting.engine.FitEngine` to recover the curve
from synthetic data. The :doc:`/reference/cookbook` shows how to fit a loaded
dataset.

For a single built-in time-domain curve with its own amplitude and baseline,
pass the callable from :data:`~asymmetry.core.fitting.models.MODELS` to the
same engine. To fit values already extracted from several runs, such as
precession frequency versus temperature, use :doc:`parameter_models`.

The :doc:`/reference/fit_functions/index` guide describes the physics of the
builder components. The :data:`~asymmetry.core.fitting.parameters.PARAM_INFO_REGISTRY`
supplies parameter symbols and units. :doc:`spectral` covers Fourier-peak fitting,
and :doc:`/reference/sc_penetration_depth` explains the superconducting models
used for both time-domain and parameter-trend fits.

Public imports such as ``from asymmetry.core.fitting import FitEngine`` load
the module that defines the requested object.

.. toctree::
   :maxdepth: 1

   models
   composite
   engine
   calibration
   parameter_models
   experiment_design
   knight_shift
   spectral
   sc
   parameters
   wizards
   count_domain
   grouped_time_domain
   joint
