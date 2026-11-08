.. _background-ladder:

The background ladder: which stage removes what
===============================================

Background controls act on different quantities and representations. Choose
them according to the observation being analysed: an accidental count level,
a declared reference count spectrum, a physical sample/holder signal, or a
spectral display feature. These are not interchangeable estimates of one rate.
See :ref:`background-correction` for the contribution and scale assumptions
needed for count subtraction and for additive physical modelling advice.

A steady, uncorrelated count rate is one source of background in raw histograms.
It can matter at continuous sources and in long-time pulsed-source data
(Blundell, De Renzi, Lancaster, and Pratt, *Muon Spectroscopy: An Introduction*).
Its effects after a nonlinear asymmetry ratio or a spectral transform depend
on that operation. A physical holder polarisation instead follows its own
relaxation or precession; exposure alone cannot determine its amplitude.

The ladder at a glance
----------------------

.. list-table::
   :header-rows: 1
   :widths: 26 16 30 28

   * - Stage
     - Operates on
     - Removes
     - When
   * - Grouping background mode
       (:ref:`backgrounds <background-correction>`)
     - raw counts, pre-asymmetry / pre-FFT
     - a flat count level or a declared reference spectrum (mutually
       exclusive modes, applied once)
     - enable for supported reduced-data workflows
   * - Count-fit background nuisance
       (:doc:`../count_domain_fitting`)
     - raw counts, in the fit
     - a flat rate fitted *jointly* with the count model — independent of the
       grouping correction
     - when fitting full counts, including their accidental background
   * - σ-clip baseline
       (:ref:`Robust baseline offset <robust-baseline-offset>`)
     - spectrum, post-FFT, display
     - the spectral noise floor (a positive pedestal of redistributed counting
       noise)
     - when measuring peak heights or areas above a true zero
   * - MaxEnt SpecBG
       (:doc:`../fourier_analysis`)
     - spectrum, display, ZF/LF
     - the zero-frequency central peak of a ZF/LF field distribution
     - ZF/LF field distributions with weak satellite structure buried under the
       central peak

Stage 1 — the grouping background mode (pre-asymmetry, pre-FFT)
---------------------------------------------------------------------

The grouping dialog's **Background** selector offers **Fixed values**,
**Range average**, and **Tail fit** for flat count levels, and **Background run**
for a declared count spectrum. The modes are mutually exclusive; a reference
spectrum can be time dependent. Subtraction acts on grouped counts before the
asymmetry ratio. See :ref:`background-correction` for mode selection and
uncertainty assumptions.

Supported asymmetry and grouped Fourier/MaxEnt preparation use the current
count-background operation. Fourier input is rebuilt from the current grouping;
its background status describes that preparation. Grouped/raw count-fit inputs
and the **Individual Groups** display have a separate contract: they retain
full count backgrounds and reject active reference subtraction. They do not
consume the reference-subtracted group sums. See :doc:`../count_domain_fitting`.

*When to enable.* Set a grouping background mode whenever the uncorrelated
rate is non-negligible — routinely for continuous-source data (Range average),
and for pulsed data analysed to long times (Tail fit), where even the small
residual rate biases weak late-time relaxation.

.. _background-ladder-count-fit-trap:

Stage 2 — the count-fit background nuisance (in the fit, on raw counts)
---------------------------------------------------------------------------

Count-domain fitting (:doc:`../count_domain_fitting`) fits a flat background
term ``bg`` jointly with the count model :math:`N_0 e^{-t/\tau_\mu}[1 + s A
P(t)] + \mathrm{bg}`. This term is fitted against the **raw counts**.

.. warning::

   **The grouping background correction never reaches the count fit.** The
   count fit consumes full grouped counts, with deadtime correction only when
   enabled and available. Poisson fitting requires eligible raw observations.
   Active **Background run** subtraction is rejected even if the reference is
   loaded; disable background correction or choose a non-reference mode. A
   user who believes the grouping correction has already removed the flat rate,
   and therefore fixes ``bg = 0``, will bias :math:`N_0` and α: the raw counts
   still contain the full background. Let ``bg`` float (or seed it from the
   grouping value) — do not assume Stage 1 has done the job for the count fit.

This is not a double subtraction; it is the *opposite* trap — a stage that
looks downstream of the grouping correction but is not. The two are
separate estimates of the same flat rate: Stage 1 stores it in the
grouping and applies it to the reduced data; Stage 2 measures it from the raw
counts the fit actually sees.

*When to enable.* Whenever you fit raw counts and want the background
determined from the same data and weighting as the rest of the count model —
the statistically clean route when a separate grouping estimate is unavailable
or untrusted. The fitted value is a legitimate measurement of the steady rate
and can be promoted into the grouping as a ``Fixed`` value (see
:doc:`../count_domain_fitting`).

Stage 3 — the σ-clip baseline (post-FFT, display)
-------------------------------------------------

A power or magnitude spectrum sits on a positive pedestal from the
redistributed counting noise — distinct from the flat *time-domain* rate of
Stages 1–2. The Fourier panel's robust **σ-clip baseline** estimates that
pedestal and subtracts it from the displayed spectrum, leaving sharp peaks
intact (they are rejected as outliers during the estimate). It is a
**display-channel** operation: it never alters the underlying transform, and
its converged width doubles as the noise estimate behind the signal-to-noise
readout. See :ref:`Robust baseline offset <robust-baseline-offset>` in
:doc:`../frequency_finishers` for the σ-clip mechanism and its pitfalls.

*When to enable.* Whenever peak heights or areas must be measured above a true
zero — comparing intensities across runs, integrating a line. It is orthogonal
to Stages 1–2: subtracting the time-domain rate flattens the low-frequency
ramp, while the σ-clip baseline removes the spectral noise floor that remains.

Stage 4 — MaxEnt SpecBG (post-reconstruction, display, ZF/LF)
-------------------------------------------------------------

In zero or longitudinal field the MaxEnt reconstruction returns a **field
distribution** :math:`p(B)` dominated by a strong zero-frequency central peak,
which can bury weak satellite structure. The **Zero-frequency background
(SpecBG)** control (ZF/LF mode only) subtracts a zero-centred pseudo-Voigt
model of that central peak from the *displayed* spectrum. Like the σ-clip
baseline it is **display-only** and never alters the reconstructed spectrum;
unlike it, it targets one specific feature — the ZF/LF central peak — rather
than a broadband pedestal. See :doc:`../fourier_analysis` for the SpecBG
controls.

*When to enable.* Only for ZF/LF MaxEnt field distributions where a central
peak is hiding the structure you care about. It is unrelated to the
transverse-field precession lines that the other stages serve.

Why they coexist
----------------

Each control has its own purpose and input contract. Count subtraction changes
the observations used to form asymmetry; a count-fit nuisance models the full
accidental rate in the fitted counts. Spectral display controls change the
presentation of the transformed or reconstructed data. Applying several controls
does not establish that their physical assumptions hold; inspect the owning
workflow before combining them. A physical sample/holder contribution belongs
in the fit model when it represents the measured polarisation.

See also
--------

* :ref:`Background correction <background-correction>` — the four mutually
  exclusive grouping modes and how to choose among them.
* :doc:`../count_domain_fitting` — the count model and its ``bg`` nuisance term.
* :doc:`../frequency_finishers` — the σ-clip baseline and other post-FFT
  conditioning.
* :doc:`../fourier_analysis` — the FFT background inheritance and MaxEnt SpecBG.

**References**

- S. J. Blundell, R. De Renzi, T. Lancaster, and F. L. Pratt, *Muon
  Spectroscopy: An Introduction* (Oxford University Press, Oxford, 2022) — the
  steady background count from uncorrelated detector hits at continuous
  sources, and its suppression at pulsed sources.
