Muon-Fluorine Polarization Models
=================================

Asymmetry includes three polarization components for muon-fluorine entangled
states:

* ``MuF``: analytical two-spin ``mu-F`` polarization
* ``FmuF_Linear``: analytical collinear ``F-mu-F`` polarization
* ``FmuF_General``: numerical powder-averaged ``F-mu-F`` polarization with
  independent distances and bond angle

These components appear under the **Muon-Fluorine** category in the fit-function
builder.

The intended use case is zero-field or weak-field time-domain fitting where the
oscillatory structure arises from entanglement of the muon spin with one or two
nearby ``19F`` nuclei. The functions implemented here are polarization building
blocks, not full experimental asymmetry models by themselves.

Physics Background
------------------

Both papers emphasize that fluorine-containing materials can provide unusually
well-defined muon stopping states. In simple ionic fluorides the muon can pull
two fluorines together into a hydrogen-bond-like linear ``F-mu-F`` center,
whereas in molecular magnets the local chemistry can instead stabilize either a
single-fluorine state or a bent/asymmetric two-fluorine geometry.

The dipolar Hamiltonian used in the implementation is

.. math::

   H_{ij} = \omega_{ij}\left[\mathbf{S}_i\cdot\mathbf{S}_j
   - 3\left(\mathbf{S}_i\cdot\hat{\mathbf{r}}_{ij}\right)
   \left(\mathbf{S}_j\cdot\hat{\mathbf{r}}_{ij}\right)\right],

with dipolar angular frequency

.. math::

   \omega_{ij} = \frac{\mu_0}{4\pi}\,\gamma_i\gamma_j\hbar\,r_{ij}^{-3}.

Distances are entered as Angstrom in the GUI and converted internally to SI
units.

For powder samples, the experimentally relevant longitudinal polarization is
obtained by averaging over orientation. In the general three-spin case this is
done numerically from the dipolar eigenspectrum, following the strategy used in
Lancaster et al.

Measured asymmetry context
--------------------------

The component functions implemented here are typically multiplied by an
empirical relaxation envelope and combined with one or more background terms.

For the molecular-magnet analysis of Lancaster et al., the measured asymmetry
above :math:`T_N` was written as

.. math::

   A(t) = A_0\left[p_1 D_z(t)e^{-\lambda t} + p_2 e^{-\sigma^2 t^2}\right] + A_{bg},

with :math:`p_1 + p_2 = 1`.

For the ionic-fluoride treatment of Brewer et al., the ``FmuF`` polarization was
likewise embedded in a broader phenomenological model containing envelope and
background terms. In Asymmetry, the intended workflow is therefore to combine
the Muon-Fluorine component with existing builder components, for example
``FmuF_Linear * Exponential + Constant``.

``MuF`` (single fluorine)
-------------------------

For one strongly coupled fluorine (Case I in Lancaster et al.), the
longitudinal polarization is

.. math::

   D_z(t) = \frac{1}{6}\left[1 + 2\cos\left(\frac{\omega_d t}{2}\right)
   + \cos(\omega_d t) + 2\cos\left(\frac{3\omega_d t}{2}\right)\right].

Fit parameters:

* ``A`` (%): component amplitude
* ``r_muF`` (A): muon-fluorine distance

Physical scenario:

* Use when the muon is localized near one dominant fluorine and the data show
   the characteristic three-frequency ``mu-F`` pattern.
* This is the relevant model for the ``CuF2(H2O)2(pyz)`` stopping state
   discussed in the PRL paper, where a symmetric site between two fluorines is
   argued to be energetically disfavored.
* It is not intended for cases where two fluorines contribute comparably, or
   where an extra nearby nucleus such as a proton materially affects the
   entangled-state spectrum.

``FmuF_Linear`` (collinear F-mu-F)
----------------------------------

For the classic collinear three-spin center (Brewer et al.),

.. math::

   G_{F\mu F}(t)=\frac{1}{6}\left[3 + \cos(\sqrt{3}\,\omega_d t)
   + \left(1-\frac{1}{\sqrt{3}}\right)
   \cos\left(\frac{3-\sqrt{3}}{2}\,\omega_d t\right)
   + \left(1+\frac{1}{\sqrt{3}}\right)
   \cos\left(\frac{3+\sqrt{3}}{2}\,\omega_d t\right)\right].

Fit parameters:

* ``A`` (%): component amplitude
* ``r_muF`` (A): muon-fluorine distance

Physical scenario:

* Use for the classic linear hydrogen-bond-like ``F-mu-F`` center of ionic
   fluorides.
* This is the correct starting point for materials such as ``LiF``, ``NaF``,
   ``CaF2``, and ``BaF2`` where the muon sits approximately midway between two
   equivalent fluorines.
* Brewer et al. extracted typical ``mu-F`` distances of about ``1.17 A``
   (equivalently ``2r ≈ 2.34 - 2.38 A``) from these fits.
* Do not use this model when the local chemistry suggests inequivalent
   fluorines or a bent bond geometry.

``FmuF_General`` (bent or asymmetric F-mu-F)
--------------------------------------------

For general geometry with two distances and a bond angle, there is no compact
closed form. Asymmetry computes the polarization numerically by:

1. building the full three-spin dipolar Hamiltonian (F1-mu-F2, including F1-F2
   coupling),
2. diagonalizing the 8x8 Hamiltonian for each powder orientation,
3. evaluating

   .. math::

      D_z(t) = \frac{1}{N}\sum_{m,n}\left|\langle m|\sigma_z^\mu|n\rangle\right|^2
      \cos\left[(\omega_m-\omega_n)t\right],

4. averaging over orientation using Gauss-Legendre x uniform Euler-angle
   quadrature.

The geometry-dependent eigenspectrum is cached, so repeated evaluations at the
same ``(r1, r2, theta)`` are significantly faster.

Fit parameters:

* ``A`` (%): component amplitude
* ``r1`` (A): first muon-fluorine distance
* ``r2`` (A): second muon-fluorine distance
* ``theta`` (deg): F-mu-F bond angle

Physical scenario:

* Use when the muon couples to two fluorines but the site is not the symmetric
   linear ``FmuF`` center.
* This corresponds to the distorted two-fluorine stopping state identified in
   ``[Cu(NO3)(pyz)2]PF6`` by Lancaster et al., with fitted values
   :math:`r_1 = 0.106(3)` nm, :math:`r_2 = 0.156(3)` nm, and
   :math:`\theta = 143(1)^\circ`.
* The present implementation assumes exactly three coupled spins
   (``F - mu - F``). It therefore does not cover Case III situations where an
   additional nearby nucleus is essential, such as the proton-coupled
   ``HF2^-`` configuration in ``[Cu(HF2)(pyz)2]ClO4``.

Which model to choose
---------------------

Use ``MuF`` when one fluorine dominates.

Use ``FmuF_Linear`` when two equivalent fluorines form the standard linear
hydrogen-bond-like center.

Use ``FmuF_General`` when two fluorines are present but the geometry is bent or
the two ``mu-F`` distances are inequivalent.

If the data require a fluorine plus some additional nearby nucleus rather than a
pure ``F-mu-F`` three-spin system, none of the current Muon-Fluorine components
is fully adequate; that case needs a dedicated multi-spin model.

Usage Notes
-----------

These polarization functions are envelope-free building blocks. For damped
signals, combine them with other components in a composite expression, for
example:

.. code-block:: text

   FmuF_Linear * Exponential + Constant

or

.. code-block:: text

   FmuF_General * StretchedExponential + Constant

References
----------

1. J. H. Brewer et al., Phys. Rev. B 33, 7813 (1986).
2. T. Lancaster et al., Phys. Rev. Lett. 99, 267601 (2007).
