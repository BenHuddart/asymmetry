Muoniated-radical correlation spectrum
======================================

The correlation spectrum reads the **muon hyperfine coupling** of a muoniated
radical straight off a transverse-field FFT. A muoniated radical's transverse-
field spectrum shows a *pair* of precession lines, not one; the correlation
spectrum collapses that pair onto a single peak at the coupling
:math:`A_\mu` that produced it. It is the frequency-domain tool for
**identifying a muoniated radical and pinning** :math:`A_\mu`.

It lives in the Fourier panel as the **Correlation (radical)** display mode,
alongside the other spectrum views — a specialist tool, off the common path.

What a muoniated radical is
---------------------------

When a positive muon stops in a molecule with an unsaturated bond, it does not
always end up diamagnetic. If it **adds to a C=C double bond, an aromatic ring,
or a C=O group**, it sits at a β-position next to a carbon that now carries an
unpaired electron — a **muoniated radical**. The muon spin then couples, through
that electron, to the molecule, and the strength of the coupling is the
**isotropic muon hyperfine coupling constant** :math:`A_\mu` (in frequency
units, MHz). For organic radicals :math:`A_\mu` runs from a few MHz up to about
700 MHz, and it is a fingerprint of where the muon sits and how the spin density
is distributed.

The muon acts as a polarised spin label reporting on a *single, prompt* radical,
so muoniated-radical |muSR| sees clean first-order kinetics that conventional EPR,
which detects a mixture of products, cannot.

Why a radical gives a line pair
-------------------------------

The muon and the radical's unpaired electron form a coupled two-spin system —
the same Breit–Rabi physics as muonium. In a high transverse field the muon
precession is not one line at the diamagnetic Larmor frequency
:math:`\nu_\mathrm{d} = (\gamma_\mu/2\pi)B` but a **pair** of lines,
:math:`\nu_{12}` and :math:`\nu_{34}`, straddling it. The two frequencies are
fixed by the field and the coupling, and their **sum is the coupling itself**:

.. math::

   A_\mu = \nu_{12} + \nu_{34}.

Equivalently, the splitting between the two lines *is* :math:`A_\mu` (one of the
pair is often a negative frequency, so the measured spacing equals the sum). At
the kilogauss fields where radicals are measured the system is deep in the
high-field (Paschen–Back) regime, where this simple relation is exact and the
two lines carry equal weight.

So a radical's transverse-field FFT shows the diamagnetic line plus a symmetric
pair about it. Read the two line positions, add them, and you have
:math:`A_\mu` — that is the whole method.

How the correlation spectrum works
----------------------------------

Reading two lines by eye is easy for one clean radical and hard for a noisy
spectrum or several overlapping radicals. The correlation spectrum automates it.
For every candidate coupling :math:`A` it computes the exact Breit–Rabi pair
:math:`(\nu_{12}, \nu_{34})` that *would* arise at the measurement field, looks
up the spectral amplitude at **both** frequencies, and multiplies them together
(with a ratio penalty that rewards pairs of comparable height). The product is
large only when there really is a line at each of the two frequencies — a
genuine pair — and small otherwise. Plotted against :math:`A`, the result peaks
at the true coupling of each radical present.

The horizontal axis is therefore a **hyperfine-coupling axis** in MHz, not a
precession-frequency or field axis: a peak at 514 MHz means :math:`A_\mu = 514`
MHz. Because the axis is a coupling and not :math:`\gamma_\mu B`, the MHz / Gauss
/ Tesla field-unit selector is disabled for this view — converting a coupling to
"Gauss" would be meaningless.

*When to use this.* Reach for the correlation spectrum when you have transverse-
field data on a muoniated radical and want :math:`A_\mu`: to identify which
radical formed, or to track its coupling versus temperature, solvent, or
structure. It shines in **liquids, at high field, with resolvable precession and
good radical yield** — the conditions of classic radical-|muSR|.

*Pitfalls.* It is a **high-transverse-field** construction: at low field the
observable pair and the coupling relation differ, and the method does not apply.
It needs a **continuous muon source** (PSI, TRIUMF) — the radical spectrum runs
to hundreds of MHz, beyond a pulsed source's time resolution — and a **promptly
formed** radical, since a slowly forming one dephases in the transverse field
before it can be labelled. The diamagnetic line is skipped, and a single strong
line with no partner is suppressed rather than reported.

Using it in Asymmetry
---------------------

Compute a transverse-field FFT as usual, then select **Correlation (radical)**
in the Fourier panel's display-mode list. Two controls appear:

- **Correlation field (G)** — the transverse field used to compute the
  Breit–Rabi pairs. Leave it blank to use the run's applied field from metadata;
  set it to nudge the matching field if the header value is missing or slightly
  off.
- **Correlation order** — how aggressively unequal-amplitude (spurious) pairs
  are penalised. The default of 2 follows WiMDA; raise it to sharpen against
  noise, set 0 for a plain product.

With several detector groups selected the correlation is built from the averaged
spectrum; select a single group to correlate that group alone. The peak position
is the coupling — read it straight off the axis.

Worked example: the cyclohexadienyl radical
-------------------------------------------

The textbook muoniated radical is **cyclohexadienyl**, formed when a muon adds
to benzene. Its transverse-field spectrum at a few kilogauss shows the
diamagnetic line and a pair straddling it; their sum gives a muon hyperfine
coupling of :math:`A_\mu = 514.4(1)` MHz. Feed such a spectrum to the
correlation mode and a single peak appears at 514 MHz — the radical's signature.
A radical with two inequivalent muon environments would show two peaks, one per
coupling.

.. _radical-correlation-vs-alc:

TF correlation spectrum vs ALC — complementary routes to radical hyperfine couplings
------------------------------------------------------------------------------------

The correlation spectrum is one of **two** ways to measure a radical's hyperfine
couplings, and most radical studies use both. They probe the coupling network
from orthogonal directions.

**Transverse-field (TF) correlation** — the method on this page — applies a high
field *across* the initial muon spin and Fourier-transforms the precession. It
delivers the **isotropic muon coupling** :math:`A_\mu` from the line pair. It is
at its best in **liquids, at high field, where the precession is sharp and the
radical yield is good**, and it needs a continuous source and a promptly formed
radical.

**Avoided level crossing (ALC)** instead applies the field *along* the muon spin
and sweeps it, recording the time-integral asymmetry; a resonance appears as a
**dip** where two spin states cross and mix. ALC reads couplings that TF cannot:

- A :math:`\Delta_1` resonance (muon spin flip) sits near
  :math:`A_\mu / 2\gamma_\mu` and gives the **same** :math:`A_\mu` as the TF
  correlation — a useful cross-check.
- A :math:`\Delta_0` resonance (muon–nucleus flip-flop) appears once for **each
  coupled nucleus**, at a field set by both :math:`A_\mu` and that nucleus's
  hyperfine coupling — so ALC maps the **other (nuclear) couplings** the TF pair
  is blind to. In solids and oriented media the :math:`\Delta_0` resonance also
  carries the **dipolar (anisotropic)** part of the coupling, making it the route
  to molecular **orientation and dynamics**.

ALC therefore shines exactly where TF struggles: in **solids, liquid crystals,
polymers and oriented or complex media**, and wherever the precession is too
broad to resolve.

The practical workflow is to use **TF correlation first** to identify the
radical and pin :math:`A_\mu`, then **ALC** to map the rest of the coupling
network and the anisotropy and dynamics. Asymmetry already provides the ALC
route — see :doc:`alc_mode` for the integral-asymmetry field-scan workflow and
its resonance fitting.

References
----------

- I. McKenzie, Annu. Rep. Prog. Chem. Sect. C **109**, 65 (2013).
- I. McKenzie, R. Scheuermann, S. P. Cottrell, J. S. Lord, and I. M. Tucker,
  J. Phys. Chem. B **117**, 13614 (2013).
- F. L. Pratt, Physica B **289–290**, 710 (2000).
- S. J. Blundell, R. De Renzi, T. Lancaster, and F. L. Pratt,
  *Muon Spectroscopy: An Introduction* (Oxford University Press, Oxford, 2022).
- A. D. Hillier, S. J. Blundell, *et al.*, Nat. Rev. Methods Primers **2**, 4 (2022).

.. |muSR| replace:: μSR
