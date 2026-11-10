Conventions and implementation notes
====================================

This page records the conventions Asymmetry follows, so that a number read off
a fit, a plot, or an exported file can be interpreted unambiguously.

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

Physical constants
------------------

The :ref:`physical constants API <physical-constants-api>` gives
values, units, uncertainties, and sources. Exact defining values follow the
2019 SI; magnetic constants follow CODATA 2018, lifetimes follow PDG 2024,
the fluorine-19 ratio follows Stone’s recommended 2019 nuclear moment,
and the vacuum muonium hyperfine starting value follows Liu et al. (1999).
Derived SI constants use the defining values directly. This fixed reference set
keeps calculations independent of the SciPy version installed.

The fluorine ratio is calculated from the recommended nuclear moment and
CODATA 2018 nuclear magneton using :math:`\gamma_F/(2\pi)=2\mu_F/h` for spin
:math:`I=1/2`. Its standard uncertainty is given in the API alongside the other
measured constants. The pion lifetime is rounded to three decimal places in μs,
and the proton gyromagnetic ratio is truncated to the stated precision.

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

   * - Temperature-dependent ``SC_*``
     - Width parameters inherit the convention of the input trend;
       normalised :math:`\rho_s(T)` is unchanged by constant width rescaling.
   * - ``SC_Brandt_*`` and σ ↔ λ helpers
     - Second-moment width :math:`\sigma_2=\gamma_\mu B_\mathrm{rms}` for
       :math:`e^{-\sigma_2^2t^2/2}`. Helpers require ``width=`` to select
       :math:`\sigma_2` or :math:`\sigma_G=\sigma_2/\sqrt{2}`.
   * - ``VortexLattice`` and ``VortexLatticePowder``
     - The field distribution is calibrated to the Brandt second moment;
       the time signal is its characteristic function, without a Gaussian-width
       substitution. See :ref:`sc-width-convention`.

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
on the forward group instead. In musrfit type 2,
:math:`\alpha_{\mathrm{musrfit}}=1/\alpha`; :math:`\beta` is numerically
identical after this conversion. See :ref:`comparison-asymmetry-errors` for
the distinction between corrected curves and raw fitting coordinates.
Count-to-ratio primitives return fractions; datasets and time-domain models use
percent. Convert errors by the same factor; see :doc:`/reference/asymmetry_units`.
Signed corrected counts or general :math:`\beta` can give estimates outside
:math:`[-1,1]`; that range is not a universal validity condition.
The standard count-based F/B path applies enabled detector deadtime correction
before alignment and grouping, then enabled group background subtraction.
Counts are binned before asymmetry is formed. The operation order and
calibration dependencies are described in :doc:`reduction_pipeline`.

.. _asymmetry-counting-errors:

Errors
------

For fixed :math:`\alpha,\beta`, Asymmetry uses first-order propagation through
:math:`A=(F-\alpha B)/D`, where :math:`D=\beta F+\alpha B`. For independent
forward/backward groups with retained count variances :math:`V_F,V_B`,

.. math::

   \frac{\partial A}{\partial F}=\frac{\alpha(1+\beta)B}{D^2},\qquad
   \frac{\partial A}{\partial B}=-\frac{\alpha(1+\beta)F}{D^2},

   \operatorname{var}(A)\simeq
   \frac{\alpha^2(1+\beta)^2(B^2V_F+F^2V_B)}{D^4}.

This retains the covariance between numerator and denominator. For raw Poisson
counts, :math:`V_F=F,V_B=B`; at unity balances it reduces to
:math:`\operatorname{var}(A)\simeq(1-A^2)/(F+B)`. For example, 6000 forward
and 4000 backward counts give 20 % asymmetry with a standard error of
0.979795897 percentage points. This is a local approximation, not an exact
finite-count variance or confidence interval; sparse counts and cancellation
can make the Gaussian approximation poor. The uncertainty framework is the
first-order law in `JCGM 100:2008, sections 5.1–5.2
<https://doi.org/10.59161/JCGM100-2008E>`_.

Corrected count means are not new Poisson observations. Corrections retain
measured or supplied variances; count merging includes shared-background and
estimation-window overlap terms as described in
:ref:`background-count-uncertainty`. Ordinary Gaussian fits use marginal point
errors, not the full between-bin covariance of those corrections.

.. _conditional-calibration-errors:

Calibration conditioning
~~~~~~~~~~~~~~~~~~~~~~~~

Ordinary reduced errors condition on the chosen :math:`\alpha,\beta`.
Available calibration errors and covariance remain in the calibration record;
they are not added independently to each reduced point or propagated as a full
reduced-bin/run covariance into the ordinary fit. A common calibration shifts
many observations together. Reusing the same counts for calibration also
introduces count–calibration covariance. Fitting calibration coefficients within
a count model is a separate treatment; see
:doc:`/reference/data_reduction/alpha_calibration` and
:doc:`/reference/data_reduction/beta_calibration`.

.. _asymmetry-error-fallback:

Undefined and degenerate bins
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a zero denominator the current count-to-ratio operation returns a fallback
value of zero and error of one fraction (100 percentage points). A defined
ratio with zero first-order count contribution also receives that fallback
error; raw one-sided Poisson counts are an example. These software values are
not measured zero asymmetries or calibrated uncertainties. The Gaussian fit
boundary accepts their finite positive errors, giving them finite weight;
it does not identify or omit them automatically.

A corrected zero group count can retain positive variance and a usable error.
For raw :math:`F=5,B=10` with a fixed forward background of 10, the corrected
counts are :math:`-5,10` with variances :math:`5,10`, giving :math:`A=-3` and
:math:`\sigma_A=\sqrt{4.8}` in fractions. Finite small denominators are retained,
and values are not clipped to :math:`[-1,1]`. Neither :math:`A=0` nor an error
of 100 percentage points alone diagnoses an unusable observation. See
:doc:`/reference/count_domain_fitting` for supported count-fitting alternatives.

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
:mod:`asymmetry.core.io` (loaders). The API reference is :doc:`/api/index`.


.. rubric:: References

- R. S. Hayano, Y. J. Uemura, J. Imazato, N. Nishida, T. Yamazaki, and
  R. Kubo, Phys. Rev. B **20**, 850 (1979).
- A. Abragam, *The Principles of Nuclear Magnetism* (Clarendon Press,
  Oxford, 1961).
- A. Keren, Phys. Rev. B **50**, 10039 (1994).
