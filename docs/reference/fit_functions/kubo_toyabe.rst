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
recovery under a field satisfying :math:`\gamma_\mu B_L\gg\Delta` is consistent
with static fields, but slow fluctuations (:math:`
u\ll\Delta`) can also
decouple. Field recovery alone does not uniquely establish a static state. For an
end-to-end walk-through see :doc:`/workflows/lf_decoupling_dynamics`.

.. _fit-static-gkt-zf:

StaticGKT_ZF
------------

.. math::

   A(t) = A\left[\tfrac{1}{3}
   + \tfrac{2}{3}\left(1-\Delta^2 t^2\right)e^{-\Delta^2 t^2/2}\right]

The zero-field static Gaussian Kubo–Toyabe function: the foundational
response of a host in which a static, isotropic, Gaussian-distributed local
field (typically randomly oriented nuclear moments) dominates. The :math:`1/3` tail is the angular average of the squared spin
projection
along the local field; that component does not dephase. The complementary
transverse projection precesses and dephases into the dip at
:math:`t=\sqrt{3}/\Delta`.
The width

.. math::

   \Delta = \gamma_\mu\sqrt{\langle B_x^2\rangle} = \gamma_\mu B_c

uses the standard deviation of one Cartesian component of the zero-mean
isotropic field distribution; :math:`\langle|\mathbf B|^2\rangle=3\Delta^2/\gamma_\mu^2`.
It sets both the dip position and the early-time response
:math:`1-\Delta^2t^2+(5/12)\Delta^4t^4+O(t^6)`. For metals and diamagnetic
hosts with only
nuclear moments :math:`\Delta \sim 0.1{-}0.5\;\mu s^{-1}`; appreciably larger
values can also reflect short muon–nucleus distances, several sites, or an
electronic contribution.

=========  ===============  =====  ===========================================
Name       Symbol           Unit   Description
=========  ===============  =====  ===========================================
``A``      :math:`A`        %      Component asymmetry amplitude.
``Delta``  :math:`\Delta`   μs⁻¹   Static Gaussian field-distribution width.
=========  ===============  =====  ===========================================

A good seed follows directly from the data: the dip sits at
:math:`t_{\min} = \sqrt{3}/\Delta`. For :math:`\Delta t\ll1` it agrees to leading
order with ``Gaussian`` at
:math:`\sigma=\Delta` (:ref:`fit-gaussian`), with a fourth-order difference
:math:`-\Delta^4t^4/12`. A window ending before the dip can still resolve that
difference if its precision is sufficient. Relaxation of the tail can indicate
fluctuations or an additional relaxation channel; use ``DynamicGaussianKT``
when field dynamics are required by the data. Common composites are
``StaticGKT_ZF + Constant`` and
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
interest. The :ref:`fit wizard <fit-wizard-applied-field>` does that for you —
it pins ``B_L`` from the recorded geometry and setpoint (at 0 on a zero-field
run) and leaves it ticked **Fix** in the parameter table when you apply the
candidate. :math:`\Delta` is partially degenerate with the amplitude in a
single run — pin it with a decoupling field sweep, or a global fit sharing
:math:`\Delta` across runs (:doc:`../global_fit_wizard`). Evaluation uses
:math:`u=|\Delta t|` and :math:`r=|\omega_0/\Delta|`. Weak fields retain
smooth field dependence all the way to zero; they do not select a finite
zero-field plateau. The zero-width limit is constant polarisation. For
zero-field-only data use the cheaper ``StaticGKT_ZF`` directly.

The normalised static Gaussian kernel targets absolute numerical error
:math:`\leq10^{-5}` before amplitude and baseline
(:doc:`/explanation/numerical_accuracy`). Independent three-dimensional
Gaussian field averages cover :math:`0\leq u\leq5` at
:math:`r=0,0.001,0.01,0.049,0.05,0.051,0.06,0.1,1`, including scaled
widths 0.05, 0.4, 1, and 5 μs⁻¹ and both field signs. Adaptive integration
of the published expression checks representative times through
:math:`u=20` at :math:`r=0.05,0.06,1,10,100`, including the numerical
transition near :math:`r=1`. These are numerical checks of the selected
static model, not evidence that a sample is static.

.. dropdown:: Stable evaluation of the static Gaussian expression

   Differentiating the expression above in dimensionless time gives

   .. math::

      \frac{dG}{du}=2e^{-u^2/2}
      \left[u^3\frac{j_1(ru)}{ru}-u\operatorname{sinc}(ru)\right],

   where :math:`\operatorname{sinc}(z)=\sin(z)/z` and
   :math:`j_1(z)/z\to1/3`. For :math:`r<1`, termwise integration of
   the Bessel series gives

   .. math::

      G(u,r)=1-\sum_{k=0}^{\infty}
      \frac{2(-r^2)^k}{(2k+1)!(2k+3)}
      \left[M_k(u)+u^{2k+2}e^{-u^2/2}\right],
      \qquad M_k(u)=\int_0^u x^{2k+1}e^{-x^2/2}\,dx.

   The moments follow :math:`M_0=1-e^{-u^2/2}` and
   :math:`M_k=2kM_{k-1}-u^{2k}e^{-u^2/2}`. Evaluation retains
   :math:`k=0,\ldots,15`. Each omitted term is bounded in magnitude by
   :math:`2^{k+1}k!/(2k+1)!`, whose sum for :math:`k\geq16`
   is below :math:`4\times10^{-19}` for :math:`0\leq r\leq1`, before
   floating-point rounding. The Gaussian envelope bounds the omitted
   derivative beyond :math:`u=12` by less than :math:`10^{-28}`.
   This finite series avoids inverse-field cancellation and evaluates
   smooth field dependence through zero with storage proportional to the
   number of requested times. For :math:`r\geq1`, the Gaussian sine
   integral is evaluated analytically with the complex scaled complementary
   error function. Both branches evaluate the same physical field average
   directly at the requested times, without interpolation.

A standalone ``LFKuboToyabe`` model is available in the ``MODELS`` registry.

**References**

- R. S. Hayano, Y. J. Uemura, J. Imazato, N. Nishida, T. Yamazaki, and
  R. Kubo, Phys. Rev. B **20**, 850 (1979).
- A. D. Hillier and R. Cywinski, Appl. Magn. Reson. **13**, 95 (1997).

.. _fit-dynamic-gaussian-kt:

DynamicGaussianKT
-----------------

.. image:: /_generated/screenshots/dynamic_kt_rate_series.png
   :alt: Synthetic Gaussian strong-collision rate sweep at the Ag nuclear-field width
   :width: 100%

*An imposed rate sweep at Δ = 0.39 μs⁻¹: ν = 0, 0.1, 0.5, 2, and 20 MHz.
Intermediate fluctuations relax the tail; fast fluctuations narrow the decay.
These rates illustrate the model, rather than measured Ag diffusion.*


The strong-collision (Markovian) dynamic generalisation of the Gaussian
Kubo–Toyabe function: the static field of width :math:`\Delta` reorients
stochastically at rate :math:`\nu` — muon hopping, ionic motion, or
thermally fluctuating moments. The dynamic polarisation is obtained from the
static function :math:`G^{\mathrm{s}}(t)` by the strong-collision relation

.. math::

   G^{\mathrm{d}}(t) = G^{\mathrm{s}}(t)\,e^{-\nu t}
   + \nu\int_0^t G^{\mathrm{d}}(t-t')\,G^{\mathrm{s}}(t')\,e^{-\nu t'}\,dt' ,

solved by exponential-weighted quadratic product integration. The physical
fluctuation rate is retained at every evaluation: ``Keren`` is an explicitly
selected approximation, not an automatic high-rate branch. The zero-width
limit is constant polarisation. This is the standard model for
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

For an explicitly chosen approximation in the fast/intermediate regime,
consider :ref:`fit-keren`; it is a different function and does not inherit
the full renewal model's numerical target. A standalone ``DynamicGaussianKT``
model is available in the ``MODELS`` registry.

The normalised dynamic evaluation targets absolute numerical error
:math:`\leq10^{-5}` before amplitude and baseline
(:doc:`/explanation/numerical_accuracy`). Independent scalar renewal solves
with grid refinement and high-precision inverse-Laplace evaluations check
both dynamic families at :math:`\nu/\mathrm{width}=0.01,0.1,1,2.4,10,100`
and :math:`\omega_0/\mathrm{width}=0,0.06,1,10`. The Laplace matrix samples
:math:`\mathrm{width}\,t=0,0.05,0.2,1,2,5`; high-rate cases and selected
slow/high-field cases also sample 8 and 20. Widths 0.05, 0.4, 1, and 5 μs⁻¹
and both field signs reproduce the same scaled curves within this target.
These checks establish numerical evaluation on the tested domain, not fit
convergence or statistical coverage.

.. dropdown:: Renewal quadrature and smooth parameter perturbations

   The factor :math:`e^{-\nu t'}` is integrated in each interval rather than
   sampled by a direct trapezoidal rule. Quadratic interpolation of the
   unknown dynamic polarisation gives a causal Toeplitz system; its first
   two steps use a refined linear product integration. This preserves a
   constant static kernel and avoids spurious high-rate growth. Coarse and
   half-step solutions remove the leading stiff-limit error.

   The step varies continuously with field and width. Adding samples at the
   future end of the window keeps existing nodes in place. Padding and cubic
   interpolation keep the requested times clear of the remote interpolation
   boundary. The static spline preserves its known initial derivative:
   zero for Gaussian fields, and :math:`-4a_L/3` for Lorentzian fields.
   Nearby-parameter tests check numerical second derivatives as well as
   curve values, because small value errors can otherwise bias fit curvature.

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

and in zero field the strong-collision dynamics have an **exact closed
form**: the rational Laplace transform of the static function gives one
real pole and a conjugate pair. The real pole is bracketed in scaled
coordinates, and the pair is evaluated with cosine and sinc functions.
This retains the exact strong-collision solution at very large rate-to-width
ratios and near coincident poles, without a fast-fluctuation crossover. Unlike
the Gaussian case there is no motional narrowing — a Lorentzian distribution
has no second moment — and the closed form shows the relaxation rate
saturating at :math:`4a_L/3` for :math:`\nu \gg a_L`.

In a longitudinal field the static line shape is evaluated from a closed-form
**spectral density**: writing the stochastic field average over the isotropic
Lorentzian distribution as a cosine transform in the total-field magnitude
:math:`W` and doing the magnitude integral analytically gives

.. math::

   S(W) = \frac{a_L}{2\pi\,\omega_0^{3}\,W}\left[
   \left(a_L^{2} + W^{2} + \omega_0^{2}\right)
   \ln\frac{a_L^{2} + (W + \omega_0)^{2}}{a_L^{2} + (W - \omega_0)^{2}}
   - 4\,\omega_0 W\right],

with :math:`\omega_0 = \gamma_\mu B_L`. Two rational tail terms with
scale :math:`\kappa^2=a_L^2+\omega_0^2` are subtracted before the FFT and their
analytic cosine transforms added back. They retain the original inverse-square
and inverse-fourth-power tails while resolving high-field oscillations even
for a narrow distribution. The conjugate time step is the integration grid's.
At weak field the spectral
density uses the finite series for
:math:`[\operatorname{atanh}(q)-q]/q^3`, with
:math:`q=2\omega_0W/(a_L^2+W^2+\omega_0^2)`, to avoid inverse-field
cancellation. Only exactly zero field selects the analytic zero-field
function; finite fields retain smooth field dependence. Independent checks
also use Uemura's spherical-Bessel time expression.

The dynamic field solution uses the same exponential-weighted renewal method
and numerical target as ``DynamicGaussianKT``. The requested fluctuation rate
is retained, including its effect in fast fluctuating fields. The field path
uses a control difference: the numerical finite-field and zero-field solutions
share exactly the same grids, startup, and refinement. Their interpolated
difference is added to the exact zero-field value at each requested time.
This cancels the discretisation offset as the field tends to zero and retains
physical field curvature. The field path costs more than the closed form, so fix :math:`B_L` from the
known applied field where it is not the quantity of interest.

Both dynamic models resolve the distribution width and Larmor frequency on
causally padded grids. Evaluation stops before allocating a grid requiring
more than 524,288 fine samples and reports the required count and budget.
The fit and preview error readout asks users to shorten the selected time
window or explicitly choose a suitable analytic approximation; no curve is
silently substituted. For example, 5 kG over 20 μs uses about 273,000 fine
samples and fits within this budget. Each numerical kernel cache retains at
most 16 parameter combinations. These are work and memory bounds, not a
physical rate or width restriction. A wizard seed attempt that exceeds this
budget is recorded as failed with its diagnostic; the other seed attempts
continue. Failed candidates have no model or residual curve. A manual fit reports the failure through its normal error readout.
At the sample limit the dynamic spline
cache can retain about 320 MiB and the static Lorentzian grid cache about
128 MiB, in addition to temporary solver arrays. A successful single preview
replaces the error readout with “Previewing the current parameter values.”

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
(:math:`w_\Delta = 0` reduces exactly and continuously to
``LongitudinalFieldKT``). Gaussian tilted moments analytically average the
width distribution inside the static LF time integral. Shared Gauss–Legendre
time panels and independently integrated endpoint remainders evaluate the
polarisation directly at the requested times, without interpolation.
WiMDA's ``Gau broad KT`` "rel width" equals :math:`w_\Delta\sqrt{2}`.

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

.. figure:: /_images/screenshots/gaussian_broadened_kt_series.png
   :alt: Synthetic Gaussian-width distributions at zero and longitudinal field
   :width: 100%

   *Synthetic width distributions at the Ag nuclear-field scale. The imposed
   broadening illustrates a distribution of static widths, rather than a
   measurement of disorder in Ag.*

The numerical target is absolute error :math:`\leq10^{-5}` in normalised
polarisation. Sampled checks cover :math:`0\leq w_\Delta\leq1`,
:math:`0\leq u=|\Delta t|\leq20`, and zero, weak, and strong fields at
:math:`r=|\omega_0/\Delta|=0,10^{-6},10^{-3},0.06,1,10,100`, including
nearby parameter values, both field signs, and width scales 0.05, 0.4, 1,
and 5 μs⁻¹. These checks are finite-domain evidence, not a continuum error
proof or accuracy certification outside those ranges. The numerical work
limit does not define the physical applicability of the model.

Independent checks use the exact continuous zero-field width average,
a converged adaptive average in the original width coordinate, and direct
classical spin averages at representative times through :math:`u=2`.
Nested integration of the published static LF expression supplies separate
long-time anchors at :math:`u=20`, :math:`r=0.06,1,10,100`; its quadrature
refinement changes the reference by less than :math:`2\times10^{-8}`.
Nearby parameter and panel-boundary checks retain smooth fit derivatives.
Fits to independent width-average curves test numerical bias; they do not
establish experimental parameter identifiability.

Finite-field integration permits at most 2,097,152 quadrature nodes per
model evaluation: 16 times the sum of full time panels and requested times.
The panel rate is :math:`\max(\Delta,|\omega_0|/8)`. All work is checked
before allocating panel arrays, and intervals are evaluated in blocks of
1,024 to bound temporary storage. The budget and blocking keep numerical
integration arrays below approximately 16 MiB, apart from caller-owned
arrays and library overhead. This supports 65,536 requested times in
ordinary short windows. An excessive request raises an evaluation-work
diagnostic asking you to shorten the time window or reduce the requested
times; no coarser rule or other physical model is substituted. Exact zero
field uses the analytic width average and exact zero broadening uses the
static kernel, so these limits need no width-integration work budget.

.. dropdown:: Analytically averaging the Gaussian widths

   Let :math:`X\sim\mathcal{N}(1,w_\Delta^2)`. Reflecting to
   :math:`|X|` leaves the static Gaussian variance :math:`X^2` unchanged.
   Define :math:`M_k(s)=\langle X^k e^{-X^2s^2/2}\rangle`. Completing the
   Gaussian square gives

   .. math::

      D=1+w_\Delta^2s^2,\quad
      M_0=D^{-1/2}e^{-s^2/(2D)},\quad m=D^{-1},\quad v=w_\Delta^2/D,

      M_2=M_0(m^2+v),\qquad M_4=M_0(m^4+6m^2v+3v^2).

   Interchanging the width average and the Hayano time integral yields

   .. math::

      G(u,r)=1+2\int_0^u\left[
      s^3M_4(s)\frac{j_1(rs)}{rs}-sM_2(s)\operatorname{sinc}(rs)
      \right]ds.

   The factors :math:`j_1(x)/x\to1/3` and
   :math:`\operatorname{sinc}(x)\to1` make the zero-field limit finite.
   GL16 panels have width at most one in :math:`u` and eight in phase
   :math:`ru`. Shared full-panel integrals are accumulated once; each
   endpoint remainder is integrated independently. At exactly zero field,
   with :math:`D=1+w_\Delta^2u^2`, the result is

   .. math::

      G(u,0)=\frac13+\frac23D^{-3/2}
      \left(1-\frac{u^2}{D}\right)e^{-u^2/(2D)}.

Compared with other programs
----------------------------

The musrfit manual documents ``statGssKTLF`` with cyclic frequency
:math:`f=\omega_0/(2\pi)` in MHz and Gaussian width :math:`\sigma=\Delta`
in μs⁻¹. Asymmetry's ``B_L`` is instead in Gauss. The documented static
Gaussian formula is the same after this conversion; the independent
mathematical checks above provide the accuracy oracle. In musrfit 1.12.1
(revision ``708f777b``, ROOT 6.40.04), fixed-parameter type-8 evaluations on
41 independent integral values over :math:`0\leq\Delta t\leq5`, with
unit supplied errors and packing one, give summed squared residuals of
:math:`1.20\times10^{-5}`, :math:`1.25\times10^{-12}`, and
:math:`2.09\times10^{-10}` at :math:`r=0.06,1,10`, respectively.
The weak-field discrepancy is retained; it does not validate either
implementation. The independent Gaussian spin average supports Asymmetry's
weak-field result.
WiMDA's ``Gau broad KT`` uses the relative-width convention described above;
no native broadening comparison is asserted here.

For dynamic curves, musrfit 1.12.1 (``708f777b``, ROOT 6.40.04)
``dynGssKTLF`` and ``dynExpKTLF`` use cyclic field frequency in MHz,
width in μs⁻¹, and fluctuation rate in μs⁻¹. Fixed-parameter type-8
checks use unit supplied errors, packing one, and the independent Laplace
reference times above. The total squared residuals are:

- Gaussian :math:`(\nu/w,\omega_0/w)=(2.4,0)`: 9.86018×10⁻⁶;
- Gaussian :math:`(100,1)`: 9.97993×10⁻¹⁰;
- Lorentzian :math:`(2.4,1)`: 6.16668×10⁻⁶;
- Lorentzian :math:`(100,1)`: 3.42067×10⁻⁸.

Here :math:`w` denotes the corresponding local-field width. The moderate-rate
checks use six times through :math:`wt=5`; the high-rate checks include 8 and
20. These numerical discrepancies are retained; they do not establish
pointwise agreement at :math:`10^{-5}`.

Mantid 6.16.1 ``DynamicKuboToyabe`` was checked in fresh processes on a
uniform 0–5 μs grid with ``Asym=1``, ``Delta=1``, ``Nu=2.4``, and ``Field=0``.
On the six common reference times, reducing ``BinWidth`` from 0.02 to 0.005
to 0.001 μs reduces the largest absolute difference from the independent
reference from 0.00594 to 0.00148 to 0.000295. This demonstrates refinement,
but does not reach Asymmetry's :math:`10^{-5}` target at the documented
minimum step. No native WiMDA dynamic comparison is asserted.


**References**

- `musrfit user manual, THEORY block
  <https://lmu.pages.psi.ch/musrfit-docu/user-manual.html#the-theory-block>`_.
- `Mantid DynamicKuboToyabe documentation
  <https://docs.mantidproject.org/v6.10.0/fitting/fitfunctions/DynamicKuboToyabe.html>`_.
- D. R. Noakes and G. M. Kalvius, Phys. Rev. B **56**, 2352 (1997).
