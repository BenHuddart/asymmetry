Muonium Reaction Kinetics at a Pulsed Source (Scripting)
========================================================

Muonium (Mu = μ⁺e⁻) is a light isotope of the hydrogen atom; the rate at which it
reacts with a dissolved scavenger is measured by the relaxation of its
transverse-field precession signal [1]. In a weak transverse field the Mu signal
is a *relaxing oscillation*

.. math::

   A(t) = A_\mathrm{Mu}\, e^{-\lambda_\mathrm{Mu} t}\, \cos(2\pi f_\mathrm{Mu} t + \varphi)
          \;[\, +\ \text{slow diamagnetic}\,]

and the kinetics are pseudo-first-order, so the relaxation rate is linear in the
reactant concentration :math:`[x]`,

.. math::

   \lambda_\mathrm{Mu} = \lambda_0 + k_\mathrm{Mu}\,[x],

with slope the bimolecular rate constant :math:`k_\mathrm{Mu}` and intercept the
solvent background :math:`\lambda_0`. Repeating across temperature gives the
activation energy from the Arrhenius form

.. math::

   \log_{10} k_\mathrm{Mu} = \log_{10} A - \frac{E}{2.3\,R\,T}.

The pulsed-source problem
-------------------------

At a pulsed source (e.g. ISIS EMU) the muon pulse and the resulting dead window
put the **first good bin** at :math:`t_g \approx 0.2\,\mu\mathrm{s}`. For a
*fast*-reacting sample the Mu oscillation has largely decayed before :math:`t_g`,
so a free per-run fit cannot separate the initial amplitude from the rate — the
two trade off through the conserved surviving amplitude and the fit rails to the
amplitude bound. Integrating the asymmetry does not rescue it either, since a
transverse-field oscillation averages toward zero.

.. dropdown:: Mathematical detail: why the truncated fit is degenerate

   Re-centred on the first good bin :math:`t_g`,

   .. math::

      A(t) = \bigl[A_\mathrm{Mu}\, e^{-\lambda_\mathrm{Mu} t_g}\bigr]\,
             e^{-\lambda_\mathrm{Mu}(t-t_g)} \cos(\dots),

   so the data fix the **surviving amplitude** (the bracket) but not
   :math:`\lambda_\mathrm{Mu}` itself: the initial amplitude
   :math:`A_\mathrm{Mu}` and the rate :math:`\lambda_\mathrm{Mu}` are coupled
   through the conserved product :math:`A_\mathrm{Mu} e^{-\lambda_\mathrm{Mu}
   t_g}`. Any increase in the rate can be absorbed by a compensating increase in
   the initial amplitude, which is what makes the free per-run fit degenerate.

The fix: share the muonium amplitude across the series
------------------------------------------------------

The initial muonium amplitude :math:`A_\mathrm{Mu}` is the **muonium fraction** —
a property of muon thermalisation in the solvent, *the same for every sample*. The
scavenger changes only the rate. So fit the whole series **simultaneously with**
:math:`A_\mathrm{Mu}` **(and the phase) shared** and :math:`\lambda_\mathrm{Mu}`
varying per run: the slow, well-surviving members pin the shared amplitude, which
then **forces** :math:`\lambda_\mathrm{Mu}` for the truncated fast members. This is
the muonium-chemistry "fraction" method [1], realised over
:func:`~asymmetry.core.fitting.fit_global`.

.. important::

   The series must contain at least one **slow reference** whose Mu signal
   survives past the first good bin — typically the **deoxygenated-water** run at
   :math:`[x] = 0`. That run anchors the shared :math:`A_\mathrm{Mu}`. Without it,
   if every member has reacted quickly, there is nothing to pin the amplitude and
   the fit is under-determined.

Step 1 — per-run :math:`\lambda_\mathrm{Mu}` from the shared-amplitude fit
--------------------------------------------------------------------------

:func:`~asymmetry.core.fitting.fit_mu_relaxation_series` takes a list of 2 G Mu
datasets at one temperature, **ordered by concentration**, and returns the
per-run rate with the muonium frequency held fixed
(:math:`f_\mathrm{Mu} = \gamma_\mathrm{Mu} B \approx 2.78\,\mathrm{MHz}` at 2 G):

.. code-block:: python

   from asymmetry.core.io import load
   from asymmetry.core.fitting import (
       fit_mu_relaxation_series, fit_bimolecular_rate, fit_arrhenius,
   )

   # 2 G Mu runs at room temperature, ordered by relative concentration.
   concentrations = [0.0, 1.0, 2.0, 4.0]            # water, quarter, half, full
   datasets = [load(p) for p in room_temperature_2G_files]

   relax = fit_mu_relaxation_series(datasets, f_mu=2.78, share_amplitude=True)
   print(relax.shared_amplitude)                    # the common muonium fraction
   print(relax.lambda_mu, relax.lambda_mu_error)    # per-run rate, input order

Set ``share_amplitude=False`` to reproduce the **degenerate** per-run baseline:
the fast members' :math:`\lambda_\mathrm{Mu}` then carry a far larger (often
non-finite) uncertainty, which is exactly the failure the shared fit removes.

Step 2 — the bimolecular rate :math:`k_\mathrm{Mu}`
---------------------------------------------------

Fit the recovered rates against concentration:

.. code-block:: python

   rate = fit_bimolecular_rate(concentrations, relax.lambda_mu, relax.lambda_mu_error)
   print(rate.k_mu, rate.k_mu_error)        # slope: bimolecular rate constant
   print(rate.lambda0, rate.lambda0_error)  # intercept: solvent background

Because the supplied data give **relative** concentrations only, ``k_mu`` is in
units of μs⁻¹ per relative-concentration unit; converting to an absolute
:math:`\mathrm{M^{-1}s^{-1}}` value needs the stock molarity (an external input).

Step 3 — the activation energy
-------------------------------

Repeat Steps 1–2 at each temperature to get :math:`k_\mathrm{Mu}(T)`, then:

.. code-block:: python

   arr = fit_arrhenius(temperatures, k_values, k_errors)   # energy_unit="kJ/mol"
   print(arr.activation_energy, arr.activation_energy_error)   # E_a in kJ/mol

:func:`~asymmetry.core.fitting.fit_arrhenius` linearises the guide's exact form in
:math:`(1/T, \log_{10} k)`; the slope is :math:`-E/(\ln 10 \cdot R)`. Pass
``energy_unit="J/mol"`` for joules.

Single-run cross-check
----------------------

When the muonium fraction is already known, :func:`~asymmetry.core.fitting.mu_relaxation_from_amplitude`
recovers :math:`\lambda_\mathrm{Mu}` from a *single* truncated run by holding
:math:`A_\mathrm{Mu}` **fixed** to that reference — the analytic limit of the
shared fit, useful as an independent check or a manual fallback:

.. code-block:: python

   lam, sigma = mu_relaxation_from_amplitude(
       dataset, reference_amplitude=relax.shared_amplitude,
       f_mu=2.78, phase=relax.shared_phase,
   )

Worked example (EMU maleic acid)
--------------------------------

On the ISIS muon-school maleic-acid dataset (EMU runs 78251–78302, Mu addition to
the C=C bond of maleic acid in water; ``first_good_bin`` 21, :math:`t_g \approx
0.203\,\mu\mathrm{s}`) the shared-amplitude fit recovers a physical rate for
*every* concentration — including the fast ``half`` and ``full`` samples a free
fit cannot reach. The room-temperature concentration line gives
:math:`k_\mathrm{Mu} \approx 0.68\ \mu\mathrm{s^{-1}}` per relative-concentration
unit and a water background :math:`\lambda_0 \approx 0.6\ \mu\mathrm{s^{-1}}`, and
the Arrhenius plot over 278–338 K gives an activation energy of order
:math:`10\ \mathrm{kJ\,mol^{-1}}` — the right order for the diffusion-controlled
regime expected for this π-addition, and consistent with the literature value of
:math:`\approx 17.6\ \mathrm{kJ\,mol^{-1}}` quoted for the reaction (see the
porting study under ``docs/porting/pulsed-fast-mu-kinetics/`` for the corpus
ground truth this is checked against).

References
----------

[1] E. Roduner, *The Positive Muon as a Probe in Free Radical Chemistry*,
Lecture Notes in Chemistry Vol. 40 (Springer, Berlin, 1988).

.. seealso::

   :doc:`asymmetry_domain_global_fit` — the shared/local global fit this method
   builds on. :doc:`parameter_trending` — trending fitted parameters across a
   run series in the GUI.
