Copper dynamic Kubo–Toyabe (muon diffusion)
===========================================

This worked example fits a zero-field (ZF) copper run with the
**dynamic Gaussian Kubo–Toyabe** model to measure the field-fluctuation
rate :math:`\nu` — the observable that reports on muon diffusion. It
demonstrates the standard tactic for dynamic-KT fits: **fix the static
width** :math:`\Delta` from a low-temperature reference and **float only**
:math:`\nu`, because :math:`\Delta` and :math:`\nu` are strongly
correlated.

Physical motivation
-------------------

A muon at rest in copper sees a static Gaussian distribution of nuclear
dipolar fields of width :math:`\Delta` (in :math:`\mu\mathrm{s}^{-1}`),
giving the classic static Kubo–Toyabe relaxation with its ⅓ recovery
tail. As temperature rises the muon begins to **hop** between
interstitial sites; the field it samples then *fluctuates* at a rate
:math:`\nu`. In the strong-collision dynamic KT (Hayano *et al.*, PRB 20,
850 (1979)) increasing :math:`\nu` washes out the KT minimum and, once
:math:`\nu \gg \Delta`, **motionally narrows** the relaxation toward
:math:`\exp(-2\Delta^2 t / \nu)`. Tracking :math:`\nu(T)` is a direct
probe of the hop rate and hence the diffusion barrier.

The run
-------

``EMU00020917`` (corpus *Nuclear magnetism and ionic motion → Muon
diffusion and QLCR in copper*, ``Data_hdf5/``) is a ZF copper run at
:math:`T = 200\;\mathrm{K}`, warm enough that the muon is mobile and the
KT minimum is partly filled in.

Set up the grouping as in :doc:`calibration_grouping_emu` (the EMU
Longitudinal preset; :math:`\alpha` is not critical for a ZF relaxation
shape). Then fit in the **Fit** panel, or script it as below.

The model and seeds
-------------------

Use **DynamicGaussianKT**. The parameters are the amplitude ``A0`` (%),
the static width ``Delta`` (:math:`\mu\mathrm{s}^{-1}`), the fluctuation
rate ``nu`` (MHz), an optional longitudinal field ``B_L`` (G), and a
``baseline``.

Two settings matter:

- **Fix** ``baseline = 0``. In ZF the ⅓ tail is part of the KT function
  itself; a free baseline competes with it and drifts to unphysical
  values.
- **Fix** ``Delta = 0.37`` :math:`\mu\mathrm{s}^{-1}` (the static
  copper value, measured from a low-:math:`T` run where the muon is
  frozen) and **float** ``nu``. Floating both is degenerate — the fit
  trades width against rate at essentially the same :math:`\chi^2`.

.. list-table::
   :header-rows: 1
   :widths: 22 18 24 36

   * - Parameter
     - Seed
     - Setting
     - Meaning
   * - ``A0``
     - 20
     - free
     - Asymmetry amplitude (%)
   * - ``Delta``
     - 0.37
     - **fixed**
     - Static Gaussian width (:math:`\mu\mathrm{s}^{-1}`)
   * - ``nu``
     - 1.0
     - free
     - Field-fluctuation / hop rate (MHz)
   * - ``B_L``
     - 0.0
     - fixed
     - Longitudinal field (G), zero here
   * - ``baseline``
     - 0.0
     - **fixed**
     - No free offset in ZF

.. code-block:: python

   from asymmetry.core.io import load
   from asymmetry.core.fitting.engine import FitEngine
   from asymmetry.core.fitting.models import MODELS
   from asymmetry.core.fitting.parameters import Parameter, ParameterSet

   ds = load(".../Muon diffusion and QLCR in copper/Data_hdf5/EMU00020917.nxs")

   model = MODELS["DynamicGaussianKT"]
   params = ParameterSet([
       Parameter("A0", value=20.0, min=0.0),
       Parameter("Delta", value=0.37, fixed=True),   # static width, held
       Parameter("nu", value=1.0, min=0.0),           # the parameter of interest
       Parameter("B_L", value=0.0, fixed=True),
       Parameter("baseline", value=0.0, fixed=True),  # ZF: no free offset
   ])
   result = FitEngine().fit(ds, model.function, params)

   fitted = {p.name: p.value for p in result.parameters}
   print(round(fitted["nu"], 2))                       # 2.40 (MHz)

Expected result
---------------

The fit converges with :math:`A_0 \approx 21.7\,\%` and

.. math::

   \nu \approx 2.40\;\mathrm{MHz} \quad (\text{reduced } \chi^2 \approx 0.80),

a clear sign the muon is hopping at 200 K. As a check, releasing
:math:`\Delta` lands at :math:`\Delta \approx 0.31\;\mu\mathrm{s}^{-1}`,
:math:`\nu \approx 1.6\;\mathrm{MHz}` at the *same* :math:`\chi^2` — the
:math:`\Delta`–:math:`\nu` degeneracy that motivates holding
:math:`\Delta` fixed.

Building a hop-rate trend
-------------------------

Repeat the fit across a temperature scan (holding :math:`\Delta` fixed
for every run) to extract :math:`\nu(T)`. The **Batch** tab and the
parameter-trending panel automate this — see
:doc:`/user_guide/parameter_trending` and the batch playbook in
:doc:`temperature_scan_magnetism`. An Arrhenius fit of :math:`\nu(T)`
then yields the diffusion activation energy.

See also
--------

- :doc:`/user_guide/fit_functions/kubo_toyabe` — full DynamicGaussianKT
  reference and its relation to the Keren and static-KT models.
- :doc:`/user_guide/fit_functions/kubo_toyabe` — the static limit used to fix
  :math:`\Delta`.
