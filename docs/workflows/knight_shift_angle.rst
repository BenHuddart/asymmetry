Angle-dependent Knight shift
============================

This chapter is a worked example of an angle-resolved Knight-shift
measurement: rotating a single crystal in a transverse field and following
the muon Knight shift :math:`K(\theta)` of each site as the orientation
changes. It is the most direct μSR route to the muon stopping site, because
the anisotropy of :math:`K(\theta)` is fixed by the dipolar coupling tensor
at the site (Amato & Morenzoni 2024, Ch. 5). The data here are synthetic, but
the workflow — per-orientation fitting, conversion to the Knight shift, and a
joint :math:`K(\theta)` fit that resolves the component labelling through
crossings — is exactly the one used on real data. The full feature reference
lives at :ref:`knight-shift`; the per-orientation grouped fit that feeds it is
documented in :doc:`/reference/grouped_time_domain_fitting`.

Physical motivation
-------------------

A muon in an applied field :math:`B` precesses at a frequency shifted from the
bare Larmor value by the local hyperfine field. The fractional shift is the
muon Knight shift,

.. math::

   K = \frac{\nu - \nu_{\mathrm{ref}}}{\nu_{\mathrm{ref}}},

the muon analogue of the NMR Knight shift (Knight 1949). It splits into an
isotropic contact term and a traceless dipolar term; for a crystal rotated
about a principal axis the latter carries the orientation dependence,

.. math::

   K(\theta) = K_{\mathrm{iso}} + K_{\mathrm{ax}}\,\frac{3\cos^2\theta - 1}{2}.

The contact part :math:`K_{\mathrm{iso}}` reports the local spin
susceptibility; the axial part :math:`K_{\mathrm{ax}}` and its sign report the
dipolar geometry, and so pin down where in the unit cell the muon sits. The
axial factor :math:`(3\cos^2\theta - 1)/2` vanishes at the magic angle
(:math:`\theta \approx 54.7^\circ`, and again at :math:`125.3^\circ`), where
every site shows only its contact shift — which is exactly where two sites'
branches can cross and their labels become ambiguous.

The data
--------

The example is a synthetic orientation series of transverse-field runs at
:math:`\theta = 0, 15, \ldots, 165^\circ`, with two inequivalent muon sites.
The two sites are given the same contact shift
(:math:`K_{\mathrm{iso}} = 0.40\%`) and equal-but-opposite axial shifts
(:math:`K_{\mathrm{ax}} = \mp 0.30\%`), so their :math:`K(\theta)` branches
cross twice — at both magic angles — the case the joint fit exists to handle.

Walkthrough
-----------

#. **Load and tag the orientation.** Load the runs and give each one an
   **Angle (°)** value in a logbook column (:ref:`logbook-columns`). This is
   the column the trend panel will use as its x-axis.

#. **Fit each orientation.** Fit the precession frequency at every angle. For a
   site-resolved measurement this is the individual-groups time-domain fit
   (:doc:`/reference/grouped_time_domain_fitting`), which fits each detector
   group separately so the per-site lines are kept apart. Seed each fit from its
   neighbour (chained batch seeding) so each component keeps a stable label
   through the scan and the trend follows one site at a time.

#. **Convert to the Knight shift.** With the fitted frequencies trended, open
   the **Knight shift analysis** window — the **Knight shift window…** button
   in the *Derived parameters* section of the Fit Parameters panel, or
   **Analysis → Knight shift analysis…** — and reference against the
   **Applied field** in the *Conversion* section of its sidebar. Each frequency
   trace becomes a branch in the *Branches* section, converted live as you edit
   the reference or unit. This yields the directly measured shift
   :math:`K_{\mathrm{exp}}`; if the sample's shape and bulk susceptibility are
   known, tick **Lorentz/demag correction**, pick the **Shape** (or **Custom
   N**) and enter **χ (SI)**, to recover the intrinsic :math:`K_\mu` — see
   *Reading the result*, below, for the caveat this correction carries under
   rotation.

   .. figure:: /_generated/screenshots/knight_shift_window.png
      :width: 100%
      :align: center
      :alt: The Knight shift analysis window, with the Applied field
         reference selected, two frequency branches converted, a completed
         joint K(theta) fit in the Model fit section, the Suggest next angle
         section expanded with a computed D-optimal refine suggestion, and
         the fitted curves plus the suggestion's utility band on the K(theta)
         plot.

      The Knight shift analysis window on a two-site angle scan. The sidebar
      reads top to bottom as the pipeline — **Source** (the fitted series
      supplying the frequencies), **Conversion** (reference and unit),
      **Branches** (one :math:`K` trace per converted component, with a count
      of the crossings flagged along the scan), **Model fit** (the joint
      :math:`K(\theta)` fit covered in a later step, shown here already run),
      and **Suggest next angle** (covered further below, shown here already
      expanded with a computed **Refine parameters** suggestion); the plot
      shows both branches against angle with their fitted curves,
      **Crossing markers** on (dashed at the scan intervals where the raw
      component labels can swap), and the suggestion's utility band with the
      suggested angle marked. Converting here does not touch the trend
      table — press **Send K columns to trend table** in the footer to
      publish the :math:`K[\ldots]` columns for plotting and export (the
      joint fit runs here in the window and does not need them published
      first).

#. **Plot against orientation.** Back in the trend panel, select **Angle (°)**
   as the trend x-axis to see the published :math:`K[\ldots]` traces. If the
   scan wraps past one period, the **Fold** control overlays equivalent
   orientations onto a single :math:`180^\circ` period, doubling the effective
   angular sampling — the same fold the analysis window offers as its own
   **Fold 180°** view toggle for inspecting the branches before publishing.

#. **Resolve and fit with a joint** :math:`K(\theta)` **fit.** Back in the
   Knight shift analysis window, use its **Model fit** sidebar section:
   pick a model (``KnightAnisotropy`` for the axial dipolar form used here)
   and press the footer's **Run joint K(θ) fit** button. This fits one
   :math:`K(\theta)` curve per site at once and, at each angle, assigns that
   angle's points one-to-one to the curves they best match (a Hungarian
   matching), iterating until both the curves and the assignment settle. The
   plotted branches realign so each follows a single physical site
   continuously through the crossings, with the per-curve fits overlaid and
   swap markers at the angles where the assignment changes. ``KnightAnisotropy``
   also fits a per-site :math:`\theta_0`, the goniometer/mount misalignment
   between the scale's zero and the crystal's principal axis. A large reduced
   :math:`\chi^2` with :math:`\theta_0` pinned at zero was the old failure
   mode here — a mount that is even slightly off-axis pushes the residual
   misalignment into :math:`K_{\mathrm{iso}}` and :math:`K_{\mathrm{ax}}`
   instead, biasing exactly the parameters that identify the site; fitting
   :math:`\theta_0` absorbs it. If **Scale errors by √χ²ᵣ** is ticked and the
   fit's reduced :math:`\chi^2` still exceeds one after that, the quoted
   uncertainties are inflated accordingly. The Knight shift analysis window
   captured above already shows the result of this step — both branches
   converted and the joint fit run, with swap markers at the angles where
   the assignment changes so each trace continues through the crossings
   along its own site.

#. **Which angle next?** With the joint fit in hand, the window's own
   collapsible **Suggest next angle** section (:ref:`knight-shift`) plans
   the *next* run rather than just fitting the ones already taken. Leave
   **Mode** on **Refine parameters** and **Target** on **All parameters
   (D-optimal)** to see where a new angle would tighten both sites' fitted
   curves at once — for this scan, still coarsely sampled every
   :math:`15^\circ`, the suggestion lands away from the two magic-angle
   crossings, where each site's own curve has the steepest slope (the
   screenshot above shows this suggestion already computed).

   It is also worth asking whether the rotation axis is really aligned
   rather than assuming it, especially given how cleanly the two sites came
   out equal-and-opposite: switch **Mode** to **Test misalignment** and
   click **Suggest** again. This fits the first-harmonic
   ``AngularFourier2`` alternative in the background and reports which
   model the data currently prefer — here, correctly, the aligned
   ``KnightAnisotropy`` fit, by a decisive Akaike weight, since the
   synthetic scan was built from a perfectly aligned axis.

   Finally, **Resolve assignment** targets the crossings themselves: near
   :math:`54.7^\circ` and :math:`125.3^\circ` the classification-EM step
   above had a competing, near-equally-good labelling available (the
   envelope assignment this whole workflow exists to avoid), and this mode
   ranks candidate angles by how well a new run would tell the winning
   assignment apart from that runner-up — typically somewhere between the
   crossings, where the two labellings genuinely disagree, rather than at a
   crossing itself.

Reading the result
------------------

The joint fit recovers the two branches cleanly:
:math:`K_{\mathrm{iso}} \approx 0.40\%` for both sites, with
:math:`K_{\mathrm{ax}} \approx -0.30\%` and :math:`+0.30\%` — the values the
scan was built from. The point of the assignment is visible at the two magic
angles: without it, a fit to the raw label order would track the *envelope* of
the two branches (the upper curve, then the lower) and report a spurious
near-flat shift; with it, each trace continues through the crossing along its
own site, and the fitted :math:`K_{\mathrm{ax}}` carries the real sign and
magnitude. That sign, together with the magnitude of the anisotropy, is what
constrains the candidate stopping site.

For the absolute shift, note that the conversion by itself yields the directly
measured :math:`K_{\mathrm{exp}}`; recovering the intrinsic :math:`K_\mu` needs
the **Lorentz/demag correction** step above, with the sample geometry and bulk
susceptibility as inputs. For a rotating sample that is not itself spheroidal,
remember that the correction assumes a fixed demagnetisation factor :math:`N`
along the field — exact for a sphere, an approximation as the sample turns
otherwise. To turn a :math:`K`–:math:`\chi` pair into a hyperfine coupling, see
the Clogston–Jaccarino discussion at :ref:`knight-shift`.

See also
--------

- :ref:`knight-shift` — the full Knight-shift reference: the two references,
  component crossings, the joint fit, all three angular basis models
  (``KnightAnisotropy``, ``AngularCos2``, and ``AngularFourier2``), and
  *Suggest next angle*.
- :doc:`/reference/suggest_next_point` — the same Bayesian-experimental-design
  acquisition, for a scalar trend rather than a joint angular fit.
- :doc:`/reference/grouped_time_domain_fitting` — the individual-groups
  time-domain fit that produces the per-site frequencies.
- :doc:`temperature_scan_magnetism` — the companion trend workflow, fitting a
  power law to a frequency trend rather than an anisotropy to a shift.

References
----------

1. A. Amato and E. Morenzoni, *Introduction to Muon Spin Spectroscopy:
   Applications to Solid State and Material Sciences*, Lecture Notes in Physics
   Vol. 961 (Springer, Cham, 2024).
2. W. D. Knight, Phys. Rev. **76**, 1259 (1949).
