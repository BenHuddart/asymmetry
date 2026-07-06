.. _fit-workflow-diagnostics:

Fit workflow: asymmetric errors, chaining, and abort
====================================================

Four tools sharpen the fitting workflow: MINOS asymmetric errors, the
:math:`\chi^2` quality verdict, sequential seeding for scans, and a mid-fit
abort. The quality verdict is covered under :ref:`assessing-a-fit`; this page
covers the other three and the persistent fit record.

.. _minos-asymmetric-errors:

Asymmetric (MINOS) errors
-------------------------

The default uncertainty Asymmetry reports is the **symmetric** Hessian error: it
approximates the log-likelihood as a parabola at the minimum and quotes its
curvature, giving a single :math:`\sigma` used as :math:`\pm\sigma`. That
approximation is excellent when the likelihood really is parabolic — which is
most high-statistics fits well away from any bound — and it is what every
downstream surface (trend error bars, exports, propagated composite
uncertainties, the promoted detector balance) uses.

It breaks down in three situations, and these are exactly where **MINOS**
earns its cost. MINOS walks the :math:`\chi^2` profile outward from the best fit
along each parameter until the cost rises by the one-:math:`\sigma` amount,
re-minimising every other parameter at each step, and reports the upward and
downward excursions separately as :math:`+\sigma_+` and :math:`-\sigma_-`:

* **Low statistics.** With few counts the likelihood is visibly skewed and the
  parabola is a poor fit to its shape. The classic case is the static
  Kubo–Toyabe width :math:`\Delta` in a weakly relaxing zero-field run: the
  data constrain an upper bound on :math:`\Delta` far more tightly than a lower
  one, so the honest interval is asymmetric.
* **A parameter against a limit.** A relaxation rate or amplitude bounded at
  zero cannot move below its bound; the downward error is truncated while the
  upward error runs free. A symmetric :math:`\sigma` either overstates the lower
  side or, worse, implies unphysical negative values.
* **Strong correlations.** In a fit with a long, curved valley in parameter
  space — common when an amplitude and a relaxation rate trade off — the
  curvature at the minimum understates the true extent of the valley. MINOS
  follows the valley; HESSE does not.

When none of these hold, MINOS and HESSE agree to a few percent, and the symmetric
number is the right thing to quote.

Running MINOS
~~~~~~~~~~~~~

MINOS is opt-in because it is roughly an order of magnitude slower than the
Hessian: it is a fresh constrained minimisation per parameter per side. Tick
**Asymmetric errors (MINOS)** next to the **Fit** button before fitting. It
applies to single, batch, global, grouped, and count-domain fits alike; for the
forward/backward count fit it is especially informative on the detector balance
:math:`\alpha`, whose error is strongly correlated with the physics amplitude.

When asymmetric errors are present the parameter table shows them inline as
``value +σ₊ / −σ₋`` in place of ``± σ``, and the result summary and the exported
fit report carry both sides. A parameter whose MINOS scan fails (it diverges, or
hits a bound it cannot cross) silently keeps its Hessian error for that row.

**MINOS is a display-only diagnostic.** It does not change the symmetric error
that the rest of the program consumes: trend error bars, GLE/text export of
trends, propagated composite-parameter uncertainties, an equality-linked
follower's inherited error, and the promoted :math:`\alpha` calibration all stay
on the Hessian :math:`\sigma`. This is deliberate — asymmetric intervals are not
closed under the linear error algebra those surfaces use, so folding a
:math:`(+\sigma_+, -\sigma_-)` pair into a quadrature sum or a trend bar would be
a category error. Read the asymmetric interval where it is shown; quote the
symmetric error where the analysis propagates it. musrfit, which pioneered
exposing MINOS in muon fitting, takes the same view: MINOS annotates the
``STATISTIC`` block but the fit's working errors remain HESSE.

.. _chain-from-previous:

Chaining seeds through a scan
-----------------------------

A temperature or field scan that crosses a transition is hard to fit from fixed
starting values: a seed tuned for the low-temperature spectra lands in the wrong
basin once the order parameter has collapsed, and the fit either fails or wanders
into a spurious minimum. The fix is to **chain from the previous run** — fit the
scan in order of the control parameter and start each member from its neighbour's
fitted values, so each fit begins already inside the right basin. This is the
muon-fitting workhorse for scans through :math:`T_\mathrm{c}`.

Asymmetry's batch-series seeding is set under **Analysis ▸ Batch seeding**:

* **Auto** (default) inspects the series. If the members carry a usable
  temperature or field order key spanning a real range over at least three runs —
  an ordered scan — it chains from the previous run, ordered by that key;
  otherwise it leaves each member on its own independent seed. Auto always
  reports which it chose and why in the fit log, so the decision is never silent.
* **Chain from previous run** forces chaining.
* **Independent seeds** forces each member onto its own starting values.

For grouped fits the chained values pass through the normalised-polarisation
contract before reuse: the shared shape parameters (rates, fields, fractions,
phases) carry forward, while each group's amplitude and background are re-pinned
to the contract (amplitude one, background zero) that the per-group scale owns.
Carrying a fitted background into the next run's *normalised* model would be
meaningless, so it is reset by construction. A member that fails to converge does
not poison the chain: the next member falls back to its independent seed and
chaining resumes from there.

.. note::

   "Chain from previous run" is distinct from the global-fit wizard's
   **warm start from single fits**, which seeds a *simultaneous* global fit from
   the separate per-run single fits. Chaining is sequential (run N seeds run
   N+1); the warm start is one-shot (many single fits seed one joint fit).

Aborting a running fit
----------------------

A batch, grouped, or global fit runs on a background thread, and while it runs
the **Fit** button is replaced by a **Stop** button. Stopping is cooperative and
clean: the fit checks for cancellation both between member fits in a series and
within a running minimisation, and on abort it raises out **without recording any
partial result**. The project is left exactly as it was before the fit started,
and the next fit is unaffected. Nothing half-fitted is ever written to a trend or
a project file.

The persistent fit record
--------------------------

Asymmetry does not keep a separate fit-log file. The durable record of a fit is
**structured, and lives in the project**: the latest fit of each
``(dataset, representation)`` is stored on that representation, and the latest
batch on its series — overwritten when you re-fit, exactly the "most recent fit
per dataset" snapshot that WiMDA's ``.fit``/``.bfit`` files hold, but kept inside
the ``.asymp`` project with the full provenance (the quality verdict and, when
run, the MINOS intervals) rather than in a side file.

When you want that record outside the project — to paste into a logbook or grep
across runs — **Analysis ▸ Export fit report…** writes a human-readable block per
dataset's latest fit: model, parameters with their symmetric and (when present)
asymmetric errors, and the :math:`\chi^2_r` with its quality verdict.

See also
--------

* :ref:`assessing-a-fit` — the χ² quality verdict, recorded summary, and pull
  diagnostic.
* :doc:`count_domain_fitting` — the forward/backward :math:`\alpha` fit where
  MINOS on the detector balance is most useful.
* :doc:`global_fit_wizard` — the warm start from single fits, distinct from
  chaining.

References
----------

* S. L. Meyer, *Data Analysis for Scientists and Engineers* (Wiley, New York,
  1975) — the likelihood-profile basis of asymmetric errors.
* F. James and M. Roos, Comput. Phys. Commun. **10**, 343 (1975) — the MINOS
  error definition Asymmetry's backend implements.
* A. Suter and B. M. Wojek, Phys. Procedia **30**, 69 (2012) — musrfit, the
  muon-fitting program that exposes MINOS in this domain.
