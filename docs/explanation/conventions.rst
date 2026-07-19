Conventions and implementation notes
====================================

This page records the conventions Asymmetry follows and points to where each is
implemented, so that a number read off a fit, a plot, or an exported file can be
interpreted unambiguously.

Units
-----

Unless a page states otherwise, Asymmetry uses these units throughout the
interface, the fitted parameters, and the exported data:

.. list-table::
   :header-rows: 1
   :widths: 28 22 50

   * - Quantity
     - Unit
     - Used for
   * - Time
     - μs
     - histogram bins, fit ranges, lifetimes
   * - Frequency
     - MHz
     - precession and oscillation frequencies
   * - Magnetic field
     - Gauss (G)
     - applied and internal fields
   * - Distance
     - Å
     - muon–nucleus distances in dipolar models
   * - Relaxation rate
     - μs⁻¹
     - relaxation rates and field-distribution widths
   * - Phase
     - radians
     - oscillation phases

The muon gyromagnetic ratio is taken as
:math:`\gamma_\mu / 2\pi = 135.539` MHz/T.

The asymmetry convention
------------------------

The asymmetry of a forward group :math:`F(t)` and a backward group :math:`B(t)`
is

.. math::

   A(t) = \frac{F(t) - \alpha B(t)}{\beta F(t) + \alpha B(t)},

with the calibration constant :math:`\alpha` applied to the **backward** group
and the optional intrinsic-asymmetry balance
:math:`\beta = A_{0,b}/A_{0,f}` (default 1, giving the familiar
:math:`(F - \alpha B)/(F + \alpha B)`) applied to the forward group in the
denominator — the musrfit asymmetry-fit (fit type 2) correction pair, set in
the Grouping window's Corrections column.
This convention is used consistently across the interface, the loaders, the
grouping tools, and the fitting inputs. Some other programs place :math:`\alpha`
on the forward group instead, so take care when comparing α values between
tools (:math:`\beta` is numerically identical in both conventions).
The corrections that produce the asymmetry are applied in a fixed order —
deadtime, then background, then grouping, then asymmetry — as described in
:doc:`/getting_started/key_concepts` and
:doc:`/reference/data_reduction/index`.

Errors
------

Per-bin asymmetry uncertainties use exact Poisson error propagation through the
asymmetry expression, rather than treating the numerator and denominator as
independent, so the reported errors are well calibrated — a property the test
suite checks with a pull distribution on every build (:doc:`/reference/simulation`).

Fit functions
-------------

A fit component evaluates a normalised polarisation or relaxation shape scaled by
an amplitude, and a fittable model is assembled by combining components with
arithmetic and fraction groups. Component notation follows Blundell *et al.*
(2022), and each function cites the original literature for its form. The full
catalogue, with the naming and parameter conventions, is
:doc:`/reference/fit_functions/index`.

Where the behaviour lives
-------------------------

The analysis engine is pure Python and free of any interface code, so every
convention above is exercised identically from the GUI and from a script. The
relevant packages are :mod:`asymmetry.core.transform` (grouping, asymmetry,
deadtime, background), :mod:`asymmetry.core.fitting` (models and engines),
:mod:`asymmetry.core.fourier` (frequency-domain tools), and
:mod:`asymmetry.core.io` (loaders). The API reference is :doc:`/api/index`, and
the design rationale and study notes behind individual features are kept in the
repository under ``docs/porting/``.
