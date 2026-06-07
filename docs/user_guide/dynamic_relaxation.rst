.. _dynamic-relaxation:

Dynamic and Fluctuating-Field Relaxation Functions
==================================================

This page documents the relaxation functions for **fluctuating** (dynamic) local
fields: the dynamic Kubo–Toyabe functions (Gaussian and Lorentzian), the Keren
longitudinal-field function, and the Abragam function. It also states **how each
is evaluated numerically and how accurate it is**, so the fitted results can be
trusted.

The notation follows Blundell, De Renzi, Lancaster & Pratt, *Muon Spectroscopy:
An Introduction* (OUP, 2022), Chapter 5: the Gaussian width is
:math:`\Delta` (μs⁻¹), the Lorentzian half-width is :math:`a` (μs⁻¹), the field
fluctuation (hop) rate is :math:`\nu` (MHz ≡ μs⁻¹), and the applied longitudinal
field is :math:`B_L` (Gauss, the book's :math:`B_0`).

Overview
--------

A muon in a **static** random local field dephases according to a static
Kubo–Toyabe function :math:`G^{\mathrm{s}}(t)` (see :doc:`static_gkt_zf` and
:doc:`lf_kubo_toyabe`). When that field **reorients stochastically** at rate
:math:`\nu` — through muon hopping, ionic motion, or thermally fluctuating
electronic moments — the polarisation becomes the *dynamic* function
:math:`G^{\mathrm{d}}(t)`. Two universal limits bracket every dynamic function:

- :math:`\nu \to 0` recovers the static function (with its zero-field
  :math:`1/3` tail);
- :math:`\nu \gg \Delta` gives **motional narrowing** — exponential decay with
  rate :math:`2\Delta^2/\nu` (Gaussian, zero field), washing the tail away.

A longitudinal field adds a Larmor term :math:`\omega_0 = \gamma_\mu B_L` that
**decouples** the muon (:math:`G^{\mathrm{d}} \to 1` as :math:`B_L \to \infty`).

The components
--------------

**DynamicGaussianKT** — strong-collision dynamic Gaussian Kubo–Toyabe
(ZF and LF). Parameters :math:`A,\ \Delta,\ \nu,\ B_L`. The model for muon
hop-rate / fluctuation-rate studies in metals and ionic conductors.

**DynamicLorentzianKT** — strong-collision dynamic Lorentzian Kubo–Toyabe
(ZF and LF). Parameters :math:`A,\ a_L,\ \nu,\ B_L`. Use for dilute-moment /
spin-glass field distributions, where the field distribution is Lorentzian
rather than Gaussian.

**Keren** — Keren's analytic longitudinal-field dynamic Gaussian relaxation
(Keren, PRB 50, 10039 (1994)). Parameters :math:`A,\ \Delta,\ \nu,\ B_L`:

.. math::

   A(t) = A\,\exp[-\Gamma(t)],\quad
   \Gamma(t)=\frac{2\Delta^2}{(\omega_0^2+\nu^2)^2}
   \Big[(\omega_0^2+\nu^2)\,\nu t+(\omega_0^2-\nu^2)(1-e^{-\nu t}\cos\omega_0 t)
   -2\nu\omega_0 e^{-\nu t}\sin\omega_0 t\Big]

It reduces to the Abragam function at :math:`B_L = 0` and is an excellent
analytic approximation in the fast/intermediate-fluctuation regime.

**Abragam** — the Gaussian-to-exponential crossover function
(Abragam 1961; textbook eqn 5.52). Parameters :math:`A,\ \Delta,\ \nu`:

.. math::

   A(t) = A\,\exp\!\left[-\frac{\Delta^2}{\nu^2}\left(e^{-\nu t}-1+\nu t\right)\right]

with the limits :math:`\nu\to 0:\ \exp(-\Delta^2 t^2/2)` (Gaussian) and
:math:`\nu\gg\Delta:\ \exp(-(\Delta^2/\nu)\,t)` (exponential, rate
:math:`\lambda=\Delta^2/\nu`).

.. _dynamic-relaxation-accuracy:

Numerical evaluation and accuracy
---------------------------------

Different functions in this family are evaluated by different means. The table
summarises the method and the expected accuracy; details follow.

.. list-table::
   :header-rows: 1
   :widths: 26 40 34

   * - Function
     - Evaluation
     - Accuracy
   * - Abragam
     - Closed form (NumPy)
     - Exact (machine precision)
   * - Keren
     - Closed form (NumPy)
     - Exact (machine precision)
   * - DynamicGaussianKT
     - Strong-collision solver of the static Gaussian KT (ZF analytic; LF via the
       Hayano integral)
     - Grid-independent to < 0.5 % over 0–16 μs
   * - DynamicLorentzianKT (ZF)
     - Strong-collision solver of the analytic ZF Lorentzian KT (eqn 5.47)
     - Grid-independent to < 0.5 %
   * - DynamicLorentzianKT (LF)
     - Strong-collision solver of the **numerically field-averaged** static
       Lorentzian-LF line shape
     - ≈ 1 % over 0–16 μs

**Closed-form functions (Abragam, Keren).** These are evaluated directly from
their analytic expressions and are exact to machine precision. They are also the
cheapest to fit and are the recommended first choice for longitudinal-field
dynamic Gaussian analyses where their approximations apply.

**Dynamic Kubo–Toyabe — the strong-collision (dynamicisation) integral.** The
dynamic polarisation is obtained from the *static* Kubo–Toyabe function
:math:`G^{\mathrm{s}}(t)` by the strong-collision (Markovian) relation
(Blundell et al. eqn 5.30; equivalently Hayano et al. 1979):

.. math::

   G^{\mathrm{d}}(t) = G^{\mathrm{s}}(t)\,e^{-\nu t}
   + \nu\int_0^t G^{\mathrm{d}}(t-t')\,G^{\mathrm{s}}(t')\,e^{-\nu t'}\,dt' .

This Volterra integral equation is solved on a uniform time grid with the
trapezoidal rule. The grid step :math:`h` is chosen automatically from
:math:`\nu` so that :math:`\nu h` stays small (the explicit scheme is accurate
when :math:`\nu h \lesssim 0.02` and only becomes unstable for
:math:`\nu h \gtrsim 0.4`); the point count is capped to bound the cost. In the
physical regime (:math:`\nu \lesssim 10` MHz) the result is **grid-independent to
better than 0.5 %**, verified by halving the step. For very fast fluctuations the
function is already close to its analytic motional-narrowing limit, so the
capped-grid error there remains small and bounded. Solutions are cached per
:math:`(\Delta\,\text{or}\,a_L,\ \nu,\ B_L,\ t_{\max})`.

**Static Lorentzian-LF — numerical field average.** Unlike the Gaussian case
(which has the analytic Hayano longitudinal-field form), the Lorentzian
Kubo–Toyabe function *"becomes modified in applied field … [and] must be computed
numerically"* (Blundell et al., §5.3). Asymmetry therefore evaluates the static
Lorentzian-LF line shape directly from the stochastic field average
(textbook eqn 5.3),

.. math::

   G^{\mathrm{s,LF}}_{\mathrm{Lor}}(t)
   = \int \mathrm{d}^3 w\; p(\mathbf{w})
     \left[\cos^2\Theta + \sin^2\Theta\,\cos(|\mathbf{W}|t)\right],
   \quad \mathbf{W} = \omega_0\hat{\mathbf z} + \mathbf{w},

over an isotropic Lorentzian local-field distribution
:math:`p(\mathbf{w}) = (a_L/\pi^2)/(a_L^2 + w^2)^2` (in rate units
:math:`w=\gamma_\mu B_{\mathrm{local}}`), where :math:`\Theta` is the angle of the
total field from :math:`\hat{\mathbf z}`. The 2-D quadrature is compressed into a
binned frequency spectrum so evaluation is fast (~1 ms, cached) and is then
dynamicised with the same strong-collision solver. This field average is
**accurate to about 1 %** over 0–16 μs; it reduces exactly to the analytic
zero-field Lorentzian KT (eqn 5.47) as :math:`B_L \to 0` and to full decoupling
(:math:`G \to 1`) at large :math:`B_L`.

**How this is verified.** The test suite asserts, to tight tolerances, that:
the dynamic functions reduce to the correct static functions as
:math:`\nu \to 0`; the dynamic Gaussian KT result is grid-independent (matches a
much finer grid); the Keren zero-field limit equals the Abragam form; the Abragam
slow/fast limits equal the Gaussian and exponential envelopes; the
Lorentzian-LF average is normalised, decouples at large field, and recovers the
eqn 5.47 line shape at zero field. See
``tests/test_dynamic_relaxation.py``.

When the ~1 % Lorentzian-LF tolerance matters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For most fits the ~1 % numerical accuracy of the Lorentzian-LF line shape is well
below the statistical scatter of the data and does not affect the fitted
parameters. Where it does matter — high-statistics data, or a parameter that is
sensitive to the line-shape tail — prefer a **zero-field** Lorentzian measurement
(exact, eqn 5.47) or, if the distribution is Gaussian, use **DynamicGaussianKT**
(whose LF branch uses the analytic Hayano form).

Using these components in the Fit Builder
-----------------------------------------

All four appear in the **Edit Function…** dialog of the fit panel. Common
composites are ``DynamicGaussianKT + Constant`` (hop-rate fits),
``Keren + Constant`` (analytic LF dynamic Gaussian), and ``Abragam + Constant``
(transverse-field line-shape / hop-rate). When a dataset's metadata carries a
known applied field, ``B_L`` is initialised from it.

Parameters
----------

- :math:`A` — initial asymmetry (%); :math:`A \ge 0`.
- :math:`\Delta` — Gaussian field-distribution width (μs⁻¹); KT-family and
  Abragam.
- :math:`a_L` — Lorentzian field-distribution half-width (μs⁻¹).
- :math:`\nu` — field fluctuation / hop rate (MHz ≡ μs⁻¹); :math:`\nu = 0`
  recovers the static function.
- :math:`B_L` — applied longitudinal field (Gauss).

Physics references
------------------

- R. S. Hayano *et al.*, *Phys. Rev. B* **20**, 850 (1979) — Gaussian KT in
  longitudinal field and the strong-collision dynamic generalisation.
- Y. J. Uemura *et al.*, *Phys. Rev. B* **31**, 546 (1985) — Lorentzian (dilute)
  Kubo–Toyabe.
- A. Keren, *Phys. Rev. B* **50**, 10039 (1994) — analytic LF dynamic Gaussian
  relaxation.
- A. Abragam, *The Principles of Nuclear Magnetism* (Oxford, 1961), Ch. X.
- S. J. Blundell, R. De Renzi, T. Lancaster, F. L. Pratt, *Muon Spectroscopy: An
  Introduction* (OUP, 2022), Chapter 5 (notation, eqns 5.26, 5.30, 5.47, 5.52).
