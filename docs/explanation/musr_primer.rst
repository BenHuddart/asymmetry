Muon-spin spectroscopy: a primer
================================

.. note::

   This is a short, self-contained introduction, not a substitute for the
   literature — see the references at the foot of the page. For how Asymmetry
   relates to the established μSR programs, see :doc:`comparison`.

Muon-spin spectroscopy (μSR) has grown from a specialist method of particle
physics into a mainstream probe of condensed matter, where it is used to study
magnetism, superconductivity, and molecular and ionic dynamics. A spin-polarised
beam of positive muons is implanted into the sample, and the subsequent evolution
of the muon spin reports on the local magnetic field where each muon comes to
rest. This page summarises how that works and what the measured signal means; it
assumes undergraduate physics but no prior knowledge of the technique.

The muon as a local probe
-------------------------

The positive muon (μ⁺) is a spin-½ particle with charge :math:`+e`, a mass about
one-ninth that of the proton, and a mean lifetime of
:math:`\tau_\mu = 2.197` μs. Implanted into a solid it thermalises within
nanoseconds — far faster than its lifetime, and with no measurable loss of spin
polarisation — and comes to rest at an interstitial site, where it acts as a
sensitive magnetometer for the local field. Two properties make the measurement
possible: the muon beam is produced already spin-polarised, and the muon decay is
anisotropic, so the direction of the spin at the moment of decay can be read out.

Spin precession
---------------

A muon in a local magnetic field :math:`B` precesses about that field at the
Larmor frequency

.. math::

   \omega = \gamma_\mu B,

where :math:`\gamma_\mu / 2\pi \approx 135.5` MHz/T is the muon gyromagnetic
ratio. Because :math:`\gamma_\mu` is known precisely, a measured precession
frequency is a direct measurement of the field at the muon site: a spontaneous
frequency in zero applied field, for instance, measures the internal field of a
magnetically ordered state, and its temperature dependence follows the magnetic
order parameter.

Measuring the polarisation
--------------------------

The muon decays to a positron (and two neutrinos), and the parity-violating weak
decay emits the positron preferentially along the direction of the muon spin at
that instant. Recording positrons in detectors placed around the sample —
typically a forward and a backward group, ahead of and behind the sample
relative to the initial spin — and histogramming their arrival times recovers the
time evolution of the spin polarisation as the **asymmetry** between the groups
(:doc:`/getting_started/key_concepts`). The asymmetry is proportional to the muon
spin polarisation function :math:`P(t)`, the quantity that carries the physics.

Static and dynamic fields
-------------------------

The shape of :math:`P(t)` reflects the distribution of local fields and how it
evolves in time. A single well-defined internal field gives a coherent
oscillation; a spread of static fields dephases the ensemble and relaxes the
polarisation; fluctuating fields relax it differently again, and the distinction
between static and dynamic disorder can be settled by applying a longitudinal
field. Two limiting cases recur throughout this documentation:

* In a **transverse field**, applied perpendicular to the initial spin, the muon
  precesses at the total field; the precession amplitude and its damping measure,
  for example, the field distribution of the vortex lattice in a superconductor.
* In **zero field**, randomly oriented nuclear dipolar fields produce the
  Kubo–Toyabe relaxation function — the characteristic dip and recovery whose
  width measures the width of the static field distribution
  (:doc:`/reference/fit_functions/kubo_toyabe`).

Asymmetry provides polarisation functions for these and many other static and
dynamic field distributions; they are catalogued in
:doc:`/reference/fit_functions/index`.

Relation to other probes
------------------------

As a local probe of magnetism, μSR yields information similar to nuclear magnetic
resonance (NMR), electron spin resonance (ESR), or Mössbauer spectroscopy, but
with two distinctions: no resonant electromagnetic field is required, since the
precessing muon is followed directly in the time domain; and the muon is a
sensitive, essentially universal probe — it stops in any material, responds to
very small fields, and works across the full temperature range.

A note of caution
-----------------

The muon's chief limitation is the mirror image of its strength: because it is an
implanted, positively charged interstitial, its stopping site is not known a
priori, and it can perturb its local environment. Quantitative interpretation
therefore often rests on a calculation of the stopping site and the distortion it
induces. This is an active area, and results should be checked against the
primary literature and, where possible, an established analysis tool
(:doc:`comparison`).

References
----------

- S. J. Blundell, R. De Renzi, T. Lancaster, and F. L. Pratt,
  *Muon Spectroscopy: An Introduction* (Oxford University Press, Oxford, 2022).
- A. Yaouanc and P. Dalmas de Réotier, *Muon Spin Rotation, Relaxation, and
  Resonance: Applications to Condensed Matter* (Oxford University Press, Oxford,
  2011).
