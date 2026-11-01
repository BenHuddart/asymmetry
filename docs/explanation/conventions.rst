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

.. _gaussian-width-conventions:

Gaussian widths
---------------

The width :math:`\Delta=\gamma_\mu B_c` is an angular-frequency width (μs⁻¹),
where :math:`B_c` is the standard deviation of one Cartesian component of an
isotropic, zero-mean Gaussian field. Its total vector second moment is
:math:`\langle|\mathbf B|^2\rangle=3B_c^2`. A transverse-field Gaussian
frequency distribution and the early-time zero-field polarisation use different
mappings to the ``Gaussian`` component:

.. list-table::
   :header-rows: 1
   :widths: 27 73

   * - Model
     - Width convention and relevant limit
   * - ``Gaussian``
     - :math:`e^{-\sigma^2t^2}`. Transverse-field envelope:
       :math:`\sigma=\gamma_\mu B_c/\sqrt{2}`; zero-field early-time match:
       :math:`\sigma=\gamma_\mu B_c`.
   * - ``StaticGKT_ZF``
     - :math:`\Delta=\gamma_\mu B_c`; starts as :math:`1-\Delta^2t^2`
       and includes the static dip and :math:`1/3` tail.
   * - ``LongitudinalFieldKT``
     - The same per-component :math:`\Delta`; its zero-field limit is
       ``StaticGKT_ZF``.
   * - ``DynamicGaussianKT``
     - The same per-component :math:`\Delta`; vanishing fluctuation rate
       recovers the static KT response.
   * - ``Abragam``
     - Static transverse-field limit :math:`e^{-\Delta^2t^2/2}`;
       ``Gaussian`` at :math:`\sigma=\Delta/\sqrt{2}`.
   * - ``Keren``
     - At zero applied field and vanishing fluctuation rate,
       :math:`e^{-\Delta^2t^2}`; ``Gaussian`` at :math:`\sigma=\Delta`.
       This limit matches the early-time KT curvature, not its full static tail.

The Gaussian and zero-field KT functions first differ at fourth order:
:math:`G_{\mathrm{KT}}(t)-e^{-\Delta^2t^2}=-\Delta^4t^4/12+O(t^6)`.
See :ref:`fit-gaussian` and :ref:`fit-static-gkt-zf` for applicability.

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
the Grouping window's Corrections column. :math:`\beta` is entered as a
fixed scalar or measured from a weak-TF calibration run's precession fit;
see :doc:`/reference/data_reduction/beta_calibration`.
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


.. rubric:: References

- R. S. Hayano, Y. J. Uemura, J. Imazato, N. Nishida, T. Yamazaki, and
  R. Kubo, Phys. Rev. B **20**, 850 (1979).
- A. Abragam, *The Principles of Nuclear Magnetism* (Clarendon Press,
  Oxford, 1961).
- A. Keren, Phys. Rev. B **50**, 10039 (1994).
