Muonium reaction kinetics at a pulsed source (scripting)
========================================================

Muonium (Mu = μ⁺e⁻) is a light isotope of the hydrogen atom; the rate at which it
reacts with a dissolved scavenger is measured by the relaxation of its
transverse-field precession signal [1]. In a weak transverse field the Mu signal
is a *relaxing oscillation*

.. math::

   A(t) = A_\mathrm{Mu}\, e^{-\lambda_\mathrm{Mu} t}\, \cos(2\pi f_\mathrm{Mu} t + \varphi)
          \;[\, +\ \text{slow diamagnetic}\,]

For pseudo-first-order kinetics, the relaxation rate is linear in the
reactant concentration :math:`[x]`,

.. math::

   \lambda_\mathrm{Mu} = \lambda_0 + k_\mathrm{Mu}\,[x],

with slope the bimolecular rate constant :math:`k_\mathrm{Mu}` and intercept the
solvent background :math:`\lambda_0`. Repeating across temperature gives the
activation energy from the Arrhenius form

.. math::

   \log_{10} k_\mathrm{Mu} = \log_{10} A - \frac{E}{\ln(10)\,R\,T}.

The pulsed-source problem
-------------------------

A pulsed source and the selected good-bin window can exclude the early part of
an oscillation. Write the first selected time as :math:`t_g`; its value depends
on the acquisition and reduction settings. For a rapidly relaxing sample, only
a weak surviving signal may remain. The initial amplitude and rate can then be
strongly correlated, even when the fit converges.

.. dropdown:: Mathematical detail: the surviving amplitude

   Re-centred on :math:`t_g`,

   .. math::

      A(t) = \bigl[A_\mathrm{Mu}\, e^{-\lambda_\mathrm{Mu} t_g}\bigr]\,
             e^{-\lambda_\mathrm{Mu}(t-t_g)} \cos(\dots).

   The bracket is the surviving amplitude. A resolved decay over the selected
   window can still identify the rate; truncation alone does not create an exact
   degeneracy. When the remaining decay is too weak or too rapid to resolve,
   amplitude and rate can trade off within the observational uncertainty.

Sharing the muonium amplitude across a series
---------------------------------------------

:func:`~asymmetry.core.fitting.mu_kinetics.fit_mu_relaxation_series` can share
:math:`A_\mathrm{Mu}` and phase while allowing the rate to vary per member. The
fitted amplitude is in **percent asymmetry**, rather than a dimensionless
formation fraction. Interpreting a common amplitude as a common muonium fraction
requires comparable detector calibration and formation conditions. A scavenger
or temperature change is not guaranteed to preserve those conditions.

Use a shared amplitude when that assumption is justified and at least one
member retains enough signal to constrain it. A slow solvent reference can
supply this information. Sharing alone cannot recover a rate when the selected
observations contain no measurable muonium signal. Inspect parameter errors,
bounds and residuals for each member; successful optimisation is not a test of
the common-amplitude assumption.

Step 1 — per-run :math:`\lambda_\mathrm{Mu}` from the shared-amplitude fit
--------------------------------------------------------------------------

:func:`~asymmetry.core.fitting.mu_kinetics.fit_mu_relaxation_series` takes a list of 2 G Mu
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
   print(relax.shared_amplitude)                    # shared amplitude, percent
   print(relax.lambda_mu, relax.lambda_mu_error)    # per-run rate, input order

Set ``share_amplitude=False`` to fit independent amplitudes. This provides a
comparison under a different parameter-sharing assumption; poorly resolved
members can have large or non-finite rate errors in either configuration.

Step 2 — the bimolecular rate :math:`k_\mathrm{Mu}`
---------------------------------------------------

Fit the recovered rates against concentration:

.. code-block:: python

   rate = fit_bimolecular_rate(concentrations, relax.lambda_mu, relax.lambda_mu_error)
   print(rate.k_mu, rate.k_mu_error)        # slope: bimolecular rate constant
   print(rate.lambda0, rate.lambda0_error)  # intercept: solvent background

The slope uses μs⁻¹ per supplied concentration unit. For the relative values in
the example, converting to an absolute :math:`\mathrm{M^{-1}s^{-1}}` value needs
the stock molarity. The line fit accepts individual rate errors, not the
between-member covariance induced by a shared-amplitude fit. Its reported
uncertainty therefore does not propagate that full covariance or concentration
uncertainty.

Step 3 — the activation energy
-------------------------------

Repeat Steps 1–2 at each temperature to get :math:`k_\mathrm{Mu}(T)`, then:

.. code-block:: python

   arr = fit_arrhenius(temperatures, k_values, k_errors)   # energy_unit="kJ/mol"
   print(arr.activation_energy, arr.activation_energy_error)   # E_a in kJ/mol

:func:`~asymmetry.core.fitting.mu_kinetics.fit_arrhenius` linearises the Arrhenius form in
:math:`(1/T, \log_{10} k)`; the slope is :math:`-E/(\ln 10 \cdot R)`. Pass
``energy_unit="J/mol"`` for joules. Supplied rate errors are propagated to
:math:`\log_{10} k` to first order and used as diagonal weights. Temperature
uncertainty and between-temperature covariance are not inputs to this fit.

Single-run cross-check
----------------------

When the initial percent-asymmetry amplitude is supplied, :func:`~asymmetry.core.fitting.mu_kinetics.mu_relaxation_from_amplitude`
recovers :math:`\lambda_\mathrm{Mu}` from a *single* truncated run by holding
:math:`A_\mathrm{Mu}` **fixed** to that reference. This is a numerical fit
through the same global-fit engine. Its uncertainty is conditional on the
supplied reference amplitude; uncertainty in that reference is not propagated:

.. code-block:: python

   lam, sigma = mu_relaxation_from_amplitude(
       dataset, reference_amplitude=relax.shared_amplitude,
       f_mu=2.78, phase=relax.shared_phase,
   )

References
----------

[1] E. Roduner, *The Positive Muon as a Probe in Free Radical Chemistry*,
Lecture Notes in Chemistry Vol. 40 (Springer, Berlin, 1988).

.. seealso::

   :doc:`asymmetry_domain_global_fit` — the shared/local global fit this method
   builds on. :doc:`parameter_trending` — trending fitted parameters across a
   run series in the GUI.
