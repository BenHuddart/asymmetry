.. _api-cookbook:

API cookbook
============

Copy-paste recipes for the most common scripting workflows, each linking to the
page that documents it in full. If you are driving Asymmetry from a script or an
agent rather than the GUI, **start here** — every recipe below is a runnable
snippet against the public :mod:`asymmetry.core` API.

.. contents:: Recipes
   :local:
   :depth: 1

.. note::

   Three things trip up almost every first script. They are baked into the
   recipes below, but worth stating up front:

   * **Fit *expressions* use COMPONENTS names, not MODELS names.** Write
     ``Gaussian``/``Exponential`` in an expression, not ``GaussianRelaxation``/
     ``ExponentialRelaxation`` (those are the standalone names in
     :data:`~asymmetry.core.fitting.models.MODELS`). The parser now raises a
     pointing error if you mix them up — see :ref:`components-vs-models`.
   * **MaxEnt wants a** :class:`~asymmetry.core.data.Run`. Pass ``ds.run``, not
     the :class:`~asymmetry.core.data.dataset.MuonDataset`.
   * **Seed parameters by the model's own** ``param_names``. A composite mangles
     names (the first amplitude becomes ``A_1``, the background ``A_bg``); call
     ``model.param_names`` *first* and seed exactly those.


Load a run, group, calibrate α, correct deadtime
------------------------------------------------

:func:`~asymmetry.core.io.load` returns a
:class:`~asymmetry.core.data.dataset.MuonDataset`; its raw histograms live on
``ds.run``. Group detectors, estimate the F/B balance :math:`\alpha`, and form
the asymmetry:

.. code-block:: python

   from asymmetry.core.io import load
   from asymmetry.core.transform import (
       apply_grouping, estimate_alpha, compute_asymmetry,
   )

   ds = load("MUSR00044989.nxs")
   histograms = ds.run.histograms

   # apply_grouping takes 0-based histogram indices.
   backward = apply_grouping(histograms, list(range(0, 32)))   # detectors 1–32
   forward = apply_grouping(histograms, list(range(32, 64)))   # detectors 33–64

   alpha = estimate_alpha(forward, backward)
   asymmetry, error = compute_asymmetry(forward, backward, alpha=alpha)

Deadtime correction (PSI BIN / ROOT raw histograms need it; ISIS NeXus arrives
pre-corrected) is applied to the histograms *before* grouping — the order is
always deadtime → background → grouping → asymmetry. See
:doc:`grouping_calibration` and :doc:`data_processing` for the deadtime APIs and
the error model.


Recompute the asymmetry with a custom α
---------------------------------------

:func:`~asymmetry.core.transform.compute_asymmetry` is the one knob — pass any
``alpha`` to re-form ``(asymmetry, error)`` from the same grouped counts:

.. code-block:: python

   from asymmetry.core.transform import compute_asymmetry

   asymmetry, error = compute_asymmetry(forward, backward, alpha=1.08)

See :doc:`data_processing`.


Fit a model
-----------

Build the model, **read its** ``param_names``, then seed exactly those names. A
composite renames parameters (sequential ``A_1``, ``A_2``, …; background
``A_bg``), so seeding ``"A"`` would silently miss.

.. code-block:: python

   from asymmetry.core.io import load
   from asymmetry.core.fitting import (
       FitEngine, CompositeModel, Parameter, ParameterSet,
   )

   ds = load("data.nxs")

   # In expressions use COMPONENTS names (Gaussian, Exponential, Constant) —
   # NOT the MODELS names (GaussianRelaxation, ...).
   model = CompositeModel.from_expression("Gaussian + Constant")
   print(model.param_names)          # ['A_1', 'sigma', 'A_bg']  <- seed THESE

   params = ParameterSet([
       Parameter("A_1", value=0.2, min=0.0, max=1.0),
       Parameter("sigma", value=0.3, min=0.0),
       Parameter("A_bg", value=0.0),
   ])

   result = FitEngine().fit(ds, model.function, params)
   print(result.reduced_chi_squared, result.parameters["A_1"].value)

See :doc:`fitting`, :doc:`composite_models`, and the
:ref:`COMPONENTS-vs-MODELS naming rule <components-vs-models>`.


MaxEnt spectrum
---------------

MaxEnt is a grouped raw-count algorithm — hand it ``ds.run``, not the dataset,
and construct :class:`~asymmetry.core.maxent.MaxEntConfig` with **keyword**
arguments (its first positional field is ``n_spectrum_points``):

.. code-block:: python

   from asymmetry.core.io import load
   from asymmetry.core.maxent import maxent, MaxEntConfig

   ds = load("data.nxs")
   config = MaxEntConfig(n_spectrum_points=2048, f_max_mhz=5.0,
                         t_min_us=0.1, t_max_us=8.0)
   result = maxent(ds.run, config, cycles=10)        # ds.run, NOT ds
   freqs, spectrum = result.frequencies_mhz, result.spectrum

Passing the ``MuonDataset`` raises a ``TypeError`` telling you to pass
``ds.run``. See :doc:`fourier_analysis`.


Field scan with a period filter
-------------------------------

:func:`~asymmetry.core.transform.build_field_scan` reduces a series of runs to a
sorted scan. Select a period *upstream* with
:func:`~asymmetry.core.io.periods.select_period` for multi-period files, or pass
a ``filter`` predicate to keep one subset when the periods are separate runs:

.. code-block:: python

   from asymmetry.core.io import load
   from asymmetry.core.io.periods import select_period
   from asymmetry.core.transform import build_field_scan

   combined = [load(f) for f in rf_files]               # 2-period red/green files
   red = [select_period(d, "red") for d in combined]
   scan = build_field_scan(red, order_key="field", t_min=0.1, t_max=8.0)
   # scan.x, scan.value, scan.error, scan.run_numbers, scan.excluded

   # Or, when periods/run-types are separate runs in one series:
   scan = build_field_scan(
       runs, order_key="field",
       filter=lambda run: run.metadata.get("period_label") == "red",
   )

See :doc:`alc_mode` and :doc:`data_reduction/index`.


Accurate multi-group transverse-field σ
---------------------------------------

A 2-group forward/backward asymmetry carries a systematic offset in the fitted
TF width. For an accurate per-group σ, fit the lifetime-corrected grouped
**counts** directly with
:func:`~asymmetry.core.fitting.grouped_time_domain.fit_grouped_time_domain`,
sharing the physics parameters across the dataset's detector groups:

.. code-block:: python

   from asymmetry.core.io import load
   from asymmetry.core.fitting import Parameter, ParameterSet
   from asymmetry.core.fitting.grouped_time_domain import (
       build_grouped_time_domain_groups, fit_grouped_time_domain,
       GROUP_NUISANCE_PARAMS,
   )

   ds = load("tf_run.nxs")
   groups = build_grouped_time_domain_groups(ds, t_min=0.1, t_max=8.0)

   def tf_gaussian(t, sigma, frequency):
       ...   # transverse-field polarization P(t; sigma, frequency)

   seed = {g.group_id: ParameterSet([
       Parameter("N0", 1000.0, min=0.0),
       Parameter("background", 0.0),
       Parameter("amplitude", 0.2, min=0.0),
       Parameter("relative_phase", 0.0),
       Parameter("sigma", 0.2, min=0.0),       # physics — shared below
       Parameter("frequency", 1.5, min=0.0),   # physics — shared below
   ]) for g in groups}

   result = fit_grouped_time_domain(
       groups, tf_gaussian,
       global_params=["sigma", "frequency"],        # shared across groups
       local_params=list(GROUP_NUISANCE_PARAMS),    # per-group nuisance
       initial_params=seed,
   )
   print(result.global_parameters["sigma"].value)

See :doc:`grouped_time_domain_fitting`.


Rebin a curve
-------------

:meth:`MuonDataset.rebin <asymmetry.core.data.dataset.MuonDataset.rebin>` returns
a coarser copy with every ``factor`` bins merged (errors propagated):

.. code-block:: python

   coarse = ds.rebin(4)        # 4 bins -> 1; coarse.time/.asymmetry/.error

   # Or the array-level function when you do not have a MuonDataset:
   from asymmetry.core.transform import rebin
   time, values, errors = rebin(ds.time, ds.asymmetry, ds.error, factor=4)

See :doc:`data_reduction/index`.


Asymmetry-domain global (shared-parameter) fit
----------------------------------------------

Fit one model across many asymmetry datasets, sharing some parameters globally
and keeping others local, with
:func:`~asymmetry.core.fitting.fit_global`:

.. code-block:: python

   import numpy as np
   from asymmetry.core.fitting import fit_global, Parameter, ParameterSet

   def model(t, **p):
       return p["amp"] * np.exp(-p["lambda"] * np.asarray(t, dtype=float))

   seed = ParameterSet([
       Parameter("amp", value=0.2, min=0.0, max=1.0),
       Parameter("lambda", value=0.5, min=0.0, max=10.0),
   ])

   result = fit_global(
       datasets,                  # list (or dict) of asymmetry MuonDatasets
       model,
       global_params=["lambda"],  # shared across all datasets
       local_params=["amp"],      # independent per dataset
       initial_params=seed,
   )
   print(result.global_parameters["lambda"].value, result.reduced_chi_squared)

For a model from a composite expression, take the global/local names from
``model_def.param_names`` (mangled, e.g. ``A_1``/``A_bg``). See
:doc:`asymmetry_domain_global_fit`.


Fast-muonium reaction kinetics (pulsed source)
----------------------------------------------

When a transverse-field Mu signal has decayed before the first good bin, the Mu
amplitude and rate are degenerate per run. Share the muonium amplitude across the
concentration series (a slow water reference at ``[x] = 0`` anchors it), then
trend to the rate constant and activation energy:

.. code-block:: python

   from asymmetry.core.fitting import (
       fit_mu_relaxation_series, fit_bimolecular_rate, fit_arrhenius,
   )

   # 2 G Mu datasets at one temperature, ordered by relative concentration.
   relax = fit_mu_relaxation_series(datasets, f_mu=2.78, share_amplitude=True)
   rate = fit_bimolecular_rate(concentrations, relax.lambda_mu, relax.lambda_mu_error)
   print(rate.k_mu, rate.lambda0)            # bimolecular rate, solvent background

   # Repeat per temperature, then:
   arr = fit_arrhenius(temperatures, k_values, k_errors)
   print(arr.activation_energy)              # E_a in kJ/mol

The series **must** include a slow, well-surviving member (deoxygenated water) to
pin the shared amplitude. See :doc:`muonium_kinetics`.


RF-resonance fit (muon–proton)
------------------------------

The ``RFResonanceMuP`` parameter-domain model fits a field-swept RF resonance
(two Lorentzians at the exact-diagonalisation resonance fields). Reduce the RF
observable to a :class:`~asymmetry.core.transform.FieldScan`, then fit it with
:func:`~asymmetry.core.fitting.fit_scan_model`:

.. code-block:: python

   from asymmetry.core.fitting import fit_scan_model

   result = fit_scan_model(
       scan, "RFResonanceMuP",
       initial={"A_mu": 515.0, "A_p": 124.0, "nu_RF": 218.5,
                "ampl1": 0.017, "wid1": 20.0, "ampl2": 0.017, "wid2": 20.0,
                "BG": 0.002},
   )
   print(result.parameters["A_mu"].value, result.parameters["A_p"].value)

Separate the RF-on / RF-off periods *before* building the scan (see the
field-scan recipe above) or the resonance positions bias. See :doc:`alc_mode`.


Penetration depth (incl. Brandt vortex lattice)
-----------------------------------------------

Superconducting depolarisation-rate models live in
:data:`~asymmetry.core.fitting.PARAMETER_MODEL_COMPONENTS` (``SC_SWave``,
``SC_DWave``, ``SC_TwoGap_SS``, the field-dependent ``SC_Brandt_VortexLattice``
/ ``SC_Brandt_VortexLattice_Powder``, …). Fit a σ(B) or σ(T) trend with
:func:`~asymmetry.core.fitting.parameter_models.fit_parameter_model`:

.. code-block:: python

   import numpy as np
   from asymmetry.core.fitting import Parameter, ParameterSet, ParameterCompositeModel
   from asymmetry.core.fitting.parameter_models import fit_parameter_model

   # Field sweep (gauss) of a powder type-II SC; sigma in us^-1, one per run.
   B0 = np.array([100.0, 200.0, 800.0, 1600.0, 3200.0, 6000.0])
   model = ParameterCompositeModel(["SC_Brandt_VortexLattice_Powder"])

   params = ParameterSet([
       Parameter("lambda_ab", value=200.0, min=0.0),   # nm
       Parameter("Bc2", value=20.0, min=0.0),          # tesla
       Parameter("sigma_bg", value=0.0, min=0.0, fixed=True),
   ])
   result = fit_parameter_model(B0, sigma, sigma_err, model, params)
   print(result.parameters["lambda_ab"].value)         # ab-plane depth in nm

See :doc:`sc_penetration_depth`.


.. _cookbook-tf-frequency:

Transverse-field frequency: let it float
----------------------------------------

In a transverse-field (TF) fit the precession line sits at the Larmor frequency
:math:`\nu = \gamma_\mu B / 2\pi` (use
:func:`~asymmetry.core.fourier.units.gauss_to_mhz` to convert the run's field).
It is tempting to **fix** ``frequency`` at the nominal applied field, but the
*true* line can shift — most sharply in the vortex state of a type-II
superconductor below :math:`T_c`, where the diamagnetic response lowers the
internal field. Pinning the line away from its true position pushes the misfit
into the damping term and **inflates the fitted Gaussian** ``sigma`` (~8% on a
real BiSCCO run). Let the frequency float:

.. code-block:: python

   from asymmetry.core.fitting import FitEngine, Parameter, ParameterSet
   from asymmetry.core.fitting.composite import CompositeModel
   from asymmetry.core.fourier.units import gauss_to_mhz

   model = CompositeModel.from_expression(
       "Oscillatory * Gaussian + Constant"
   ).to_model_definition()

   params = ParameterSet([
       Parameter("A_1", value=9.0),
       Parameter("frequency_1", value=float(gauss_to_mhz(ds.field)), fixed=False),  # float it
       Parameter("phase_1", value=0.0),
       Parameter("sigma_1", value=1.0, min=0.0),
       Parameter("A_bg", value=ds.asymmetry.mean()),
   ])
   result = FitEngine().fit(ds, model.function, params)

If you do fix ``frequency`` and the seed sits more than ~2% from
:math:`\gamma_\mu B` implied by the run's ``field`` metadata, the engine emits a
:class:`~asymmetry.core.fitting.FixedFrequencyFieldMismatchWarning` pointing you
back here. The guard stays silent for free frequencies and for zero-/low-field
runs (where :math:`\gamma_\mu B` is not the relevant line). The message is carried
on ``result.warnings`` and, **in the GUI, shown in the fit panel's result box**
beneath the converged line, so the trap is visible without reading the log (see
:ref:`fit-advisory-warnings`).


.. _cookbook-asymmetry-scale:

Match the asymmetry scale (percent vs fraction)
-----------------------------------------------

A loaded :class:`~asymmetry.core.data.dataset.MuonDataset` stores its
``asymmetry`` on the **percent** scale (``×100``), so an amplitude seeded on the
**fraction** scale (``A ∈ [-1, 1]``) is ~100× too small. The fit then either
converges to a degenerate amplitude or parks at the wrong minimum. Seed amplitudes
to match the data you are fitting:

.. code-block:: python

   ds = load("data.nxs")
   print(ds.asymmetry.max())     # ~25 here -> percent scale

   model = CompositeModel.from_expression("Gaussian + Constant").to_model_definition()
   params = ParameterSet([
       Parameter("A_1", value=20.0),   # percent-scale seed, matching ds.asymmetry
       Parameter("sigma", value=0.3, min=0.0),
       Parameter("A_bg", value=-20.0),
   ])
   result = FitEngine().fit(ds, model.function, params)

   # ...or work on the fraction scale explicitly (then seed A_1 in [-1, 1]):
   print(ds.asymmetry_fraction.max(), ds.asymmetry_percent.max())

When the seeded model curve and the data straddle the fraction/percent boundary
(one peak ``≤ 1.5``, the other clearly percent) the engine emits an
:class:`~asymmetry.core.fitting.AsymmetryScaleWarning`. Like the fixed-frequency
guard it is advisory only — it never raises or changes the fit — and its message
is carried on ``result.warnings`` and **surfaced in the GUI fit panel's result
box** (see :ref:`fit-advisory-warnings`). Use
:attr:`~asymmetry.core.data.dataset.MuonDataset.asymmetry_fraction` /
:attr:`~asymmetry.core.data.dataset.MuonDataset.asymmetry_percent` to pick the
scale explicitly.


Parameter trends across runs
----------------------------

Trending fits a model to a parameter extracted per run (vs field, temperature,
or frequency) with the same
:func:`~asymmetry.core.fitting.parameter_models.fit_parameter_model`:

.. code-block:: python

   import numpy as np
   from asymmetry.core.fitting import (
       ParameterCompositeModel, fit_parameter_model, parameter_set_for_model,
   )

   field = np.array([20.0, 50.0, 100.0, 200.0, 400.0, 800.0])
   values = np.array([2.7, 2.2, 1.6, 1.1, 0.7, 0.5])     # e.g. lambda per run
   errors = np.full_like(values, 0.1)

   model = ParameterCompositeModel.from_expression("PowerLaw ⊕ Constant")
   params = parameter_set_for_model(model)               # default seeds
   result = fit_parameter_model(field, values, errors, model, params)
   print(result.success, result.reduced_chi_squared)

``parameter_set_for_model`` (also exported from
:mod:`asymmetry.core.fitting`) returns a default-seeded
:class:`~asymmetry.core.fitting.ParameterSet` for any parameter model. See
:doc:`parameter_trending` and :doc:`diffusion_ballistic_lf`.
