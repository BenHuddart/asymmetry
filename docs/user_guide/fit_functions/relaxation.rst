.. _fit-relaxation:

Relaxation
==========

Relaxation components describe the decay of the muon-spin polarisation
without coherent oscillation. They are used standalone for zero- and
longitudinal-field relaxation, and multiplicatively as damping envelopes on
oscillating components (``Oscillatory * Exponential``,
``FmuF_Linear * Exponential``, …). This page covers the simple envelopes
(``Exponential``, ``Gaussian``, ``StretchedExponential``), the
dynamic-crossover functions (``Abragam``, ``Keren``), and the
1D-transport function ``RischKehr``. Relaxation from *static field
distributions* with the characteristic :math:`1/3` tail lives under
:doc:`kubo_toyabe`.

.. _fit-exponential:

Exponential
-----------

.. math::

   A(t) = A\,e^{-\lambda t}

The workhorse depolarisation function for any system in which the muon
experiences a rapidly fluctuating local field. It is the Redfield
motional-narrowing limit of slower-relaxation forms: when the fluctuation
rate :math:`\nu` of the local field is large compared with its static second
moment :math:`\Delta`, the depolarisation collapses from a Kubo–Toyabe shape
onto a pure exponential with rate :math:`\lambda \simeq 2\Delta^2/\nu`. In
practice this covers paramagnetic spin fluctuations, electronic relaxation in
metals and semiconductors where coherent dynamics are absent, and the
high-temperature regime of essentially every diffusive system once the
correlation time has dropped below the muon precession scale. The same shape
also arises from a *static but dilute* (Lorentzian) field distribution in
transverse field.

==========  ===============  =====  ===========================================
Name        Symbol           Unit   Description
==========  ===============  =====  ===========================================
``A``       :math:`A`        %      Component asymmetry amplitude.
``Lambda``  :math:`\lambda`  μs⁻¹   Exponential relaxation rate.
==========  ===============  =====  ===========================================

Both parameters are constrained non-negative by default. For a paramagnetic
salt at room temperature :math:`\lambda \sim 0.1{-}1\;\mu s^{-1}` is typical;
above roughly :math:`\lambda \sim 10\;\mu s^{-1}` the relaxation occurs
almost entirely inside the instrumental deadtime window and ``Lambda``
becomes ill-conditioned — in that regime the signal is better described as
missing initial asymmetry than as a fast exponential. When :math:`\lambda` is
recovered alongside a slow Gaussian or Kubo–Toyabe channel, expect
correlation between the two rates; LF-decoupling data on the same sample is
usually the cleanest way to break the degeneracy.

A standalone ``ExponentialRelaxation`` model (with explicit ``baseline``) is
available in the Python ``MODELS`` registry for scripted single-channel fits.

**References**

- A. Yaouanc and P. Dalmas de Réotier, *Muon Spin Rotation, Relaxation, and
  Resonance: Applications to Condensed Matter* (Oxford University Press,
  Oxford, 2011).

.. _fit-gaussian:

Gaussian
--------

.. math::

   A(t) = A\,e^{-(\sigma t)^2}

The natural relaxation envelope when the muon ensemble experiences a *static*
Gaussian distribution of local fields and the experimental time window is
short compared with :math:`1/\sigma`. Expanding the static Gaussian
Kubo–Toyabe function (:ref:`fit-static-gkt-zf`) for :math:`\Delta t \ll 1`
gives :math:`1 - \tfrac{1}{2}\Delta^2 t^2`, which matches
:math:`e^{-(\sigma t)^2}` to leading order with
:math:`\sigma = \Delta/\sqrt{2}` — note the convention, since rates quoted
for the :math:`e^{-\Delta^2 t^2/2}` form differ by :math:`\sqrt{2}`. The rate
is set by the second moment of the field distribution at the muon site,
:math:`\sigma = \gamma_\mu\sqrt{\langle B^2\rangle}\,/\sqrt{2}` in this
convention, and the same form describes the Gaussian damping envelope of a
TF precession line.

=========  ===============  =====  ===========================================
Name       Symbol           Unit   Description
=========  ===============  =====  ===========================================
``A``      :math:`A`        %      Component asymmetry amplitude.
``sigma``  :math:`\sigma`   μs⁻¹   Gaussian relaxation rate.
=========  ===============  =====  ===========================================

For polycrystalline samples with only nuclear moments, typical values lie in
the range :math:`\sigma \sim 0.1{-}0.5\;\mu s^{-1}`, up to about
:math:`1\;\mu s^{-1}` when light, high-moment nuclei such as :sup:`1`\ H or
:sup:`19`\ F are dense at the muon site. Values substantially larger than
this almost always signal that a coupled F–μ–F (or similar) entangled state
is being mis-fitted as a Gaussian envelope; use the dedicated components in
:doc:`nuclear_dipolar` instead. If the data reach
:math:`t \gtrsim 1/\sigma` without flattening onto a :math:`1/3` tail,
the relaxation is not purely static-Gaussian — try ``StretchedExponential``
or the full ``StaticGKT_ZF``.

A standalone ``GaussianRelaxation`` model is available in the ``MODELS``
registry.

**References**

- R. S. Hayano, Y. J. Uemura, J. Imazato, N. Nishida, T. Yamazaki, and
  R. Kubo, Phys. Rev. B **20**, 850 (1979).

.. _fit-stretched-exponential:

StretchedExponential
--------------------

.. math::

   A(t) = A\,e^{-(\lambda t)^{\beta}}

The Kohlrausch–Williams–Watts form, interpolating continuously between a
simple exponential (:math:`\beta = 1`) and a Gaussian (:math:`\beta = 2`). It
is the standard phenomenological relaxation function for systems in which the
muon ensemble samples a *distribution* of relaxation rates rather than a
single :math:`\lambda` — spin glasses near and below freezing, dilute and
concentrated magnetic alloys with broad RKKY-coupling distributions,
frustrated magnets with quenched disorder.

The stretching exponent :math:`\beta` is the physically informative
parameter. :math:`\beta` near 1 indicates a narrow distribution of dynamic
rates; :math:`\beta = 1/2` is the Walstedt–Walker form expected for dilute,
broadly distributed static moments in the fast-fluctuation limit; values
approaching 2 indicate a static near-Gaussian distribution better fitted with
a Kubo–Toyabe form. A fitted :math:`\beta` drifting from 1 toward
:math:`\approx 1/3` on cooling through a transition is one of the canonical
μSR signatures of glassy freezing.

==========  ===============  =====  ==========================================
Name        Symbol           Unit   Description
==========  ===============  =====  ==========================================
``A``       :math:`A`        %      Component asymmetry amplitude.
``Lambda``  :math:`\lambda`  μs⁻¹   Relaxation rate scale.
``beta``    :math:`\beta`    —      Stretching exponent.
==========  ===============  =====  ==========================================

Constrain :math:`\beta` away from 0 (``min=0.1`` or similar) and cap it at 2.
:math:`\lambda` and :math:`\beta` are strongly correlated: changing
:math:`\beta` rescales the effective rate, so the marginal uncertainty on
:math:`\lambda` from the Hessian usually understates the truth. Where this
matters, fix :math:`\beta` at a physically motivated value or quote the joint
covariance. A standalone ``StretchedExponential`` model is available in the
``MODELS`` registry.

**References**

- R. E. Walstedt and L. R. Walker, Phys. Rev. B **9**, 4857 (1974).
- Y. J. Uemura, T. Yamazaki, D. R. Harshman, M. Senba, and E. J. Ansaldo,
  Phys. Rev. B **31**, 546 (1985).
- I. A. Campbell *et al.*, Phys. Rev. Lett. **72**, 1291 (1994).

.. _fit-abragam:

Abragam
-------

.. math::

   A(t) = A\,\exp\!\left[-\frac{\Delta^2}{\nu^2}
   \left(e^{-\nu t} - 1 + \nu t\right)\right]

The Gaussian-to-exponential crossover function: a Gaussian static width
:math:`\Delta` fluctuating at rate :math:`\nu`, with the limits
:math:`\nu \to 0:\ \exp(-\Delta^2 t^2/2)` (static Gaussian) and
:math:`\nu \gg \Delta:\ \exp(-(\Delta^2/\nu)\,t)` (motionally narrowed
exponential). It is the classic model for extracting a hop or correlation
rate from a transverse-field line shape — the textbook example being the
Gaussian-to-Lorentzian change of the Cu line shape as muon diffusion sets in
on warming. Evaluated in closed form (machine precision).

=========  ===============  =====  ===========================================
Name       Symbol           Unit   Description
=========  ===============  =====  ===========================================
``A``      :math:`A`        %      Component asymmetry amplitude.
``Delta``  :math:`\Delta`   μs⁻¹   Static Gaussian field-distribution width.
``nu``     :math:`\nu`      MHz    Fluctuation (hop) rate.
=========  ===============  =====  ===========================================

**References**

- A. Abragam, *The Principles of Nuclear Magnetism* (Oxford University
  Press, Oxford, 1961), Ch. X.

.. _fit-keren:

Keren
-----

.. math::

   A(t) = A\,e^{-\Gamma(t)},\qquad
   \Gamma(t)=\frac{2\Delta^2}{(\omega_0^2+\nu^2)^2}
   \Big[(\omega_0^2+\nu^2)\,\nu t
   +(\omega_0^2-\nu^2)(1-e^{-\nu t}\cos\omega_0 t)
   -2\nu\omega_0 e^{-\nu t}\sin\omega_0 t\Big]

with :math:`\omega_0 = \gamma_\mu B_L`. Keren's analytic generalisation of
the Abragam function to a longitudinal field: an accurate strong-collision
result in the fast/intermediate fluctuation regime (:math:`\nu \gtrsim
\Delta`) that avoids the numerical convolution of the full dynamic
Kubo–Toyabe. It is the standard model for longitudinal-field decoupling
analyses (e.g. ionic diffusion) and reduces to the Abragam function at
:math:`B_L = 0`. Prefer the full :ref:`fit-dynamic-gaussian-kt` when
fluctuations are slow (:math:`\nu \lesssim \Delta`) or the static :math:`1/3`
tail matters. Evaluated in closed form (machine precision).

=========  ===============  =====  ===========================================
Name       Symbol           Unit   Description
=========  ===============  =====  ===========================================
``A``      :math:`A`        %      Component asymmetry amplitude.
``Delta``  :math:`\Delta`   μs⁻¹   Static Gaussian field-distribution width.
``nu``     :math:`\nu`      MHz    Fluctuation (hop) rate.
``B_L``    :math:`B_L`      G      Applied longitudinal field.
=========  ===============  =====  ===========================================

**References**

- A. Keren, Phys. Rev. B **50**, 10039 (1994).

.. _fit-risch-kehr:

RischKehr
---------

.. math::

   A(t) = A\, e^{\Gamma t}\,\mathrm{erfc}\!\left(\sqrt{\Gamma t}\right)

Relaxation of the muon (or muonium) polarisation by a spin carrier diffusing
in **one dimension** — a polaron moving along a conducting-polymer chain, or
an excitation confined to a structural channel. The 1D random walk keeps
returning the carrier to the muon, so instead of an exponential the
polarisation acquires a :math:`(\pi\Gamma t)^{-1/2}` long-time tail. A
stretched-exponential fit drifting toward :math:`\beta \approx 1/2` at early
times is the usual hint to try this form; prefer it over a stretched
exponential whenever 1D transport is physically motivated, since
:math:`\Gamma` then has a microscopic interpretation in terms of the
intrachain diffusion rate and hyperfine coupling.

=========  ===============  =====  ===========================================
Name       Symbol           Unit   Description
=========  ===============  =====  ===========================================
``A``      :math:`A`        %      Component asymmetry amplitude.
``Gamma``  :math:`\Gamma`   μs⁻¹   Risch–Kehr relaxation rate.
=========  ===============  =====  ===========================================

:math:`\Gamma` is constrained non-negative. The implementation evaluates the
scaled complementary error function (``erfcx``), which is numerically stable
for all :math:`\Gamma t` — there is no asymptotic-branch switch (WiMDA
changes form at :math:`\Gamma t = 20`), and WiMDA's mirrored branch for
negative rates is intentionally not reproduced.

**References**

- R. Risch and K. W. Kehr, Phys. Rev. B **46**, 5246 (1992).
