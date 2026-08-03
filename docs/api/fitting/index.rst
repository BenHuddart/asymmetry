Fitting
=======

.. currentmodule:: asymmetry.core.fitting

The fitting subsystem is split into four parts: the iminuit-backed
:class:`~asymmetry.core.fitting.engine.FitEngine` that minimises any
``f(t, **params) -> array`` callable; the
:data:`~asymmetry.core.fitting.models.MODELS` registry of standalone
time-domain models with explicit baselines (one-shot Python use); the
:class:`~asymmetry.core.fitting.composite.CompositeModel` builder that
parses arithmetic expressions over the
:data:`~asymmetry.core.fitting.composite.COMPONENTS` registry into a
compiled callable (the canonical path for any non-trivial muSR model,
mirroring the GUI **Edit Function...** dialog); and the
:mod:`~asymmetry.core.fitting.parameter_models` package that fits
*extracted* fit parameters as a function of field, temperature, or run
number with the same machinery. Superconducting models live under
:mod:`asymmetry.core.fitting.sc` and are surfaced through both the
composite registry and the parameter-trending registry — the physics
context is in :doc:`/reference/sc_penetration_depth`. Symbols, units,
and physical descriptions for every fit parameter are sourced from the
:data:`~asymmetry.core.fitting.parameters.PARAM_INFO_REGISTRY`, which
is canonical: GUI labels, GLE export labels, and the autodoc parameter
tables all derive from it.

Frequency-domain peak fitting reuses the same engine and composite-model path.
The spectral helper module supplies the default Fourier-spectrum peak model,
MHz/G conversion, and derived field parameters for trend tables.

.. toctree::
   :maxdepth: 1

   engine
   models
   composite
   parameter_models
   experiment_design
   knight_shift
   spectral
   sc
   parameters
   wizards
