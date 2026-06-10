.. _fit-kubo-toyabe:

Kubo–Toyabe
===========

The Kubo–Toyabe family describes depolarisation by a random *static* (or
stochastically fluctuating) distribution of local fields, with the
characteristic zero-field dip and :math:`1/3` tail. The notation follows
Chapter 5 of Blundell, De Renzi, Lancaster and Pratt: the Gaussian width is
:math:`\Delta` (μs⁻¹), the Lorentzian half-width :math:`a_L` (μs⁻¹), the
fluctuation (hop) rate :math:`\nu` (MHz ≡ μs⁻¹), and the applied longitudinal
field :math:`B_L` (Gauss). Two universal limits bracket every dynamic member
of the family:

- :math:`\nu \to 0` recovers the static function (with its zero-field
  :math:`1/3` tail);
- :math:`\nu \gg \Delta` gives **motional narrowing** — exponential decay
  with rate :math:`2\Delta^2/\nu` (Gaussian, zero field), washing the tail
  away.

A longitudinal field adds a Larmor term :math:`\omega_0 = \gamma_\mu B_L`
that **decouples** the muon (:math:`G \to 1` as :math:`B_L \to \infty`);
recovery of the polarisation under a small decoupling field is the
unambiguous experimental signature that the local field is static. For an
end-to-end walk-through see :doc:`../workflows/lf_decoupling_dynamics`.

.. _fit-static-gkt-zf:

StaticGKT_ZF
------------

.. math::

   A(t) = A\left[\tfrac{1}{3}
   + \tfrac{2}{3}\left(1-\Delta^2 t^2\right)e^{-\Delta^2 t^2/2}\right]

The zero-field static Gaussian Kubo–Toyabe function: the foundational
response of a host in which a static, isotropic, Gaussian-distributed local
field (typically randomly oriented nuclear moments) dominates. The
:math:`1/3` tail has a geometric origin — one third of the ensemble has its
spin parallel to the local field and is not depolarised, while the remaining
two thirds precess and dephase into the dip at :math:`t = \sqrt{3}/\Delta`.
The width

.. math::

   \Delta = \gamma_\mu\sqrt{\langle B^2\rangle}

sets both the dip position and the early-time Gaussian behaviour,
:math:`1 - \Delta^2 t^2 + \dots`. For metals and diamagnetic hosts with only
nuclear moments :math:`\Delta \sim 0.1{-}0.5\;\mu s^{-1}`; appreciably larger
values imply an electronic contribution.

=========  ===============  =====  ===========================================
Name       Symbol           Unit   Description
=========  ===============  =====  ===========================================
``A``      :math:`A`        %      Component asymmetry amplitude.
``Delta``  :math:`\Delta`   μs⁻¹   Static Gaussian field-distribution width.
=========  ===============  =====  ===========================================

A good seed follows directly from the data: the dip sits at
:math:`t_{\min} = \sqrt{3}/\Delta`. If the data window does not extend past
the dip, the function is indistinguishable from a Gaussian with
:math:`\sigma = \Delta/\sqrt{2}` (:ref:`fit-gaussian`); if the data decay
monotonically *through* the plateau value, the field is partly dynamic — use
``DynamicGaussianKT``. Common composites are ``StaticGKT_ZF + Constant`` and
``StaticGKT_ZF * Exponential + Constant`` when an additional dynamic channel
modulates the static recovery. A standalone ``StaticGKT_ZF`` model is
available in the ``MODELS`` registry.

**References**

- R. Kubo and T. Toyabe, in *Magnetic Resonance and Relaxation*, edited by
  R. Blinc (North-Holland, Amsterdam, 1967), p. 810.
- R. S. Hayano, Y. J. Uemura, J. Imazato, N. Nishida, T. Yamazaki, and
  R. Kubo, Phys. Rev. B **20**, 850 (1979).

.. _fit-lf-kubo-toyabe:

LongitudinalFieldKT
-------------------

.. image:: /_generated/screenshots/lf_kt_series_plot.png
   :alt: Overlay of five Ag LF Kubo–Toyabe runs spanning the decoupling regime
   :width: 100%

*Synthetic Ag polycrystal LF series with Δ ≈ 0.39 μs⁻¹ and B_L = 0, 5, 10,
25, 50 G. The 0 G run shows the characteristic 1/3 tail; as B_L grows the
muon spins decouple from the nuclear dipolar field and the polarisation
recovers toward unity.*

.. math::

   A(t) = A\left\{1
   - \frac{2\Delta^2}{\omega_0^2}\left[1 - e^{-\Delta^2 t^2/2}
     \cos(\omega_0 t)\right]
   + \frac{2\Delta^4}{\omega_0^3}\int_0^t e^{-\Delta^2\tau^2/2}
     \sin(\omega_0\tau)\,d\tau\right\},
   \qquad \omega_0 = \gamma_\mu B_L

The static Gaussian Kubo–Toyabe function in a longitudinal field — the
workhorse for magnetically disordered hosts where the local field is static
on the muon time scale (frozen spin systems, dilute nuclear-dipole hosts) and
the experiment sweeps :math:`B_L` through the decoupling crossover
:math:`\gamma_\mu B_L \sim \Delta` to extract :math:`\Delta`. The
:math:`B_L \to 0` limit recovers ``StaticGKT_ZF`` exactly; at large
:math:`B_L` the polarisation decouples toward unity. If the polarisation does
not recover with field, the local field is dynamic — use
``DynamicGaussianKT`` or ``Keren``.

=========  ===============  =====  ===========================================
Name       Symbol           Unit   Description
=========  ===============  =====  ===========================================
``A``      :math:`A`        %      Component asymmetry amplitude.
``Delta``  :math:`\Delta`   μs⁻¹   Static Gaussian field-distribution width.
``B_L``    :math:`B_L`      G      Applied longitudinal field.
=========  ===============  =====  ===========================================

When a dataset's metadata carries a known applied field, the fit panel
initialises ``B_L`` from it; fix ``B_L`` whenever it is not the quantity of
interest. :math:`\Delta` is partially degenerate with the amplitude in a
single run — pin it with a decoupling field sweep, or a global fit sharing
:math:`\Delta` across runs (:doc:`../global_fit_wizard`). The oscillatory
integral is evaluated for all requested times at once by cumulative
trapezoidal integration on a shared fine grid (accurate to better than
10⁻⁶); for zero-field-only data use the cheaper ``StaticGKT_ZF`` directly.
A standalone ``LFKuboToyabe`` model is available in the ``MODELS`` registry.

**References**

- R. S. Hayano, Y. J. Uemura, J. Imazato, N. Nishida, T. Yamazaki, and
  R. Kubo, Phys. Rev. B **20**, 850 (1979).
- A. D. Hillier and R. Cywinski, Appl. Magn. Reson. **13**, 95 (1997).

.. _fit-dynamic-gaussian-kt:

DynamicGaussianKT
-----------------

The strong-collision (Markovian) dynamic generalisation of the Gaussian
Kubo–Toyabe function: the static field of width :math:`\Delta` reorients
stochastically at rate :math:`\nu` — muon hopping, ionic motion, or
thermally fluctuating moments. The dynamic polarisation is obtained from the
static function :math:`G^{\mathrm{s}}(t)` by the strong-collision relation

.. math::

   G^{\mathrm{d}}(t) = G^{\mathrm{s}}(t)\,e^{-\nu t}
   + \nu\int_0^t G^{\mathrm{d}}(t-t')\,G^{\mathrm{s}}(t')\,e^{-\nu t'}\,dt' ,

solved on a uniform grid (trapezoidal rule) with the step chosen so the
result is **grid-independent to better than 0.5 %**; solutions are cached per
:math:`(\Delta, \nu, B_L, t_{\max})`. This is the standard model for
extracting a hop/fluctuation rate and its activation energy in metals (Cu)
and ionic conductors. :math:`\nu \to 0` recovers the static (LF) function;
:math:`\nu \gg \Delta` approaches exponential decay at rate
:math:`2\Delta^2/\nu`.

=========  ===============  =====  ===========================================
Name       Symbol           Unit   Description
=========  ===============  =====  ===========================================
``A``      :math:`A`        %      Component asymmetry amplitude.
``Delta``  :math:`\Delta`   μs⁻¹   Static Gaussian field-distribution width.
``nu``     :math:`\nu`      MHz    Fluctuation (hop) rate.
``B_L``    :math:`B_L`      G      Applied longitudinal field.
=========  ===============  =====  ===========================================

In the fast/intermediate regime the analytic :ref:`fit-keren` function is an
excellent and cheaper alternative. A standalone ``DynamicGaussianKT`` model
is available in the ``MODELS`` registry.

**References**

- R. S. Hayano, Y. J. Uemura, J. Imazato, N. Nishida, T. Yamazaki, and
  R. Kubo, Phys. Rev. B **20**, 850 (1979).

.. _fit-dynamic-lorentzian-kt:

DynamicLorentzianKT
-------------------

The Lorentzian analogue of ``DynamicGaussianKT``, for **dilute** or randomly
diluted moments (spin glasses, dilute-spin systems) whose local-field
distribution is Lorentzian with half-width :math:`a_L` rather than Gaussian.
The zero-field static limit is the analytic Lorentzian Kubo–Toyabe,

.. math::

   G^{\mathrm{s}}(t) = \tfrac{1}{3} + \tfrac{2}{3}(1 - a_L t)\,e^{-a_L t},

and dynamics are added with the same strong-collision solver. The
longitudinal-field static line shape has no closed form; it is computed from
the stochastic field average over an isotropic Lorentzian distribution, with
the angular and precession integrals done analytically so only a single
smooth 1-D quadrature remains. The result is accurate to ≈ 0.2 % for
:math:`B_L \gtrsim 20` G (≈ 0.3–0.5 % near 5 G) — well below the statistical
scatter of typical data; the line shape is cached, but it remains the most
expensive member of the family, so fix :math:`B_L` from the known applied
field where possible.

=========  ===============  =====  ===========================================
Name       Symbol           Unit   Description
=========  ===============  =====  ===========================================
``A``      :math:`A`        %      Component asymmetry amplitude.
``a_L``    :math:`a_L`      μs⁻¹   Lorentzian field-distribution half-width.
``nu``     :math:`\nu`      MHz    Fluctuation (hop) rate.
``B_L``    :math:`B_L`      G      Applied longitudinal field.
=========  ===============  =====  ===========================================

A standalone ``DynamicLorentzianKT`` model is available in the ``MODELS``
registry.

**References**

- Y. J. Uemura, T. Yamazaki, D. R. Harshman, M. Senba, and E. J. Ansaldo,
  Phys. Rev. B **31**, 546 (1985).

.. _fit-gaussian-broadened-kt:

GaussianBroadenedKT
-------------------

.. math::

   A(t) = A\int d\Delta'\,p(\Delta')\,
   G^{\mathrm{LF}}_{\mathrm{KT}}(t;\Delta',B_L),
   \qquad p = \mathcal{N}\!\left(\Delta,\,(w_\Delta\Delta)^2\right)

The static (longitudinal-field) Gaussian Kubo–Toyabe averaged over a Gaussian
**distribution of widths** :math:`\Delta` — for disordered hosts where a
single-width KT fit is qualitatively right but the dip is too sharp and the
:math:`1/3`-tail recovery too pronounced: structurally disordered systems,
dilute magnetic alloys, or several inequivalent muon sites. The fractional
standard deviation :math:`w_\Delta` is the broadening parameter
(:math:`w_\Delta = 0` reduces exactly to ``LongitudinalFieldKT``); the
average is evaluated by Gauss–Hermite quadrature and cached. WiMDA's
``Gau broad KT`` "rel width" equals :math:`w_\Delta\sqrt{2}`.

=========  =================  =====  =========================================
Name       Symbol             Unit   Description
=========  =================  =====  =========================================
``A``      :math:`A`          %      Component asymmetry amplitude.
``Delta``  :math:`\Delta`     μs⁻¹   Mean Gaussian field-distribution width.
``B_L``    :math:`B_L`        G      Applied longitudinal field.
``w_rel``  :math:`w_\Delta`   —      Fractional standard deviation of Δ.
=========  =================  =====  =========================================

Beware a fundamental ambiguity: width broadening and dynamics *both* fill in
the dip and soften the tail, and a single spectrum rarely distinguishes them
— vary temperature or field before preferring this model over
``DynamicGaussianKT``.

**References**

- D. R. Noakes and G. M. Kalvius, Phys. Rev. B **56**, 2352 (1997).
