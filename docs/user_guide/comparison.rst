Asymmetry in the μSR Software Landscape
=======================================

This page positions Asymmetry alongside the established μSR analysis
programs — **WiMDA**, **musrfit**, and **Mantid** — and publishes the
team's near-term roadmap. It is the public summary of a more detailed
internal comparison maintained under ``docs/porting/`` in the
Asymmetry repository.

The intended reader is a μSR practitioner who already uses one of
WiMDA, musrfit, or Mantid and wants to know where Asymmetry fits.

For end-to-end analyses analogous to the typical workflows in
those tools, see :doc:`workflows/index`.

The reference programs
----------------------

.. list-table::
   :header-rows: 1
   :widths: 18 22 35 25

   * - Tool
     - Authors / facility
     - Language / stack
     - Strength
   * - **WiMDA**
     - Francis Pratt (ISIS)
     - Object Pascal (Delphi)
     - Legacy ISIS standard; rich frequency-domain tools (MaxEnt,
       moments, eigenvalue spectral estimator)
   * - **musrfit**
     - A. Suter, B. Wojek (PSI)
     - C++ / ROOT / Minuit2
     - Hand-editable ``.msr`` workflow files; largest built-in theory
       library (~34 functions); robust Minuit2 fitting
   * - **Mantid**
     - ISIS + ORNL community
     - C++ core / Python GUI / matplotlib
     - Workspace-based pipeline; unique ALC interface; broadest set
       of specialist muonium fit functions
   * - **Asymmetry**
     - Asymmetry contributors
     - Python / PySide6 / matplotlib
     - Modern Python stack; Fit Wizard (AICc-ranked); composite-model
       expression syntax; interactive parameter trending

At a glance: feature coverage
-----------------------------

Symbols: ✅ present  ◐ partial / stub  ❌ absent  ★ distinctive
strength.

.. list-table::
   :header-rows: 1
   :widths: 30 14 14 14 14

   * - Category
     - WiMDA
     - musrfit
     - Mantid
     - Asymmetry
   * - Multi-format data ingestion (NeXus, PSI BIN, MUD, ROOT)
     - ✅
     - ✅
     - ✅
     - ✅
   * - Deadtime correction
     - ◐
     - ◐
     - ✅
     - ✅
   * - Asymmetry calculation (F/B grouping, α)
     - ✅
     - ✅
     - ✅
     - ✅
   * - Automatic phase calibration
     - ❌
     - ❌
     - ★
     - ❌
   * - Rotating Reference Frame
     - ❌
     - ◐
     - ★
     - ❌
   * - Theory function library
     - ◐ ~12
     - ★ ~34
     - ★ ~15 specialist
     - ◐ ~17
   * - Static Kubo–Toyabe (ZF / LF)
     - ✅
     - ✅
     - ✅
     - ✅
   * - Dynamic Kubo–Toyabe
     - ✅
     - ✅
     - ★
     - ❌
   * - Composite-model expression syntax
     - ❌
     - ◐
     - ◐
     - ★
   * - MIGRAD / MINOS / HESSE
     - ◐ Hessian only
     - ★ full set
     - ✅ via Mantid Fit
     - ◐ Hessian only
   * - Multi-spectrum / global fit
     - ◐ sequential
     - ✅ shared params
     - ✅
     - ✅
   * - Multi-group time-domain fit
     - ◐
     - ✅
     - ✅
     - ✅
   * - Fourier (FFT + apodisation)
     - ✅
     - ✅
     - ✅
     - ✅
   * - MaxEnt frequency reconstruction
     - ★ Burg pole-scan
     - ◐
     - ★ iterative
     - ◐ stub
   * - Spectral moments analysis
     - ★
     - ❌
     - ❌
     - ❌
   * - Interactive parameter trending
     - ◐ table
     - ◐ ``msr2data`` CLI
     - ◐ table
     - ★ panel
   * - Avoided Level Crossing (ALC) workflow
     - ❌
     - ❌
     - ★
     - ✅
   * - Period arithmetic (pulsed data)
     - ◐
     - ❌
     - ★
     - ❌
   * - Logbook / multi-run manager
     - ✅
     - ◐
     - ✅
     - ★
   * - Synthetic data simulation
     - ★
     - ❌
     - ❌
     - ❌
   * - User-defined functions
     - ◐ DLL
     - ◐ C++ plugin
     - ◐ Mantid plugin
     - ◐ via composite syntax
   * - Project files (hand-editable)
     - ❌
     - ★ ``.msr``
     - ◐ ``.mantid``
     - ✅ ``.asymp`` (JSON)
   * - Model-recommendation wizard
     - ❌
     - ❌
     - ❌
     - ★ Fit Wizard

Where Asymmetry leads
---------------------

These are the parts of the workflow where Asymmetry's
implementation is materially richer or more ergonomic than the
alternatives.

**Fit Wizard (AICc-ranked model recommendation).**
  No equivalent in WiMDA, musrfit, or Mantid. The wizard runs a
  curated portfolio of candidate models on the active dataset and
  ranks them by an information-theoretic metric (AICc by default).
  Especially valuable for new users encountering an unfamiliar
  spectrum. See :doc:`fit_wizard`.

**Composite-model expression syntax.**
  Free-form arithmetic over registered components — ``Exponential *
  Oscillatory + Constant``, with fraction-group syntax ``(...){frac}``
  for shared-amplitude bundles. musrfit's ``FUNCTIONS`` block is the
  closest analogue but is more limited; WiMDA and Mantid require
  building composite functions procedurally. See
  :doc:`composite_models`.

**Interactive parameter trending.**
  Per-run fit parameters appear as a sortable trend table that
  drives an integrated parametric-model fit panel (e.g.
  ``SC_TwoGap_SS`` for σ(T) → λ(T) in superconductors). musrfit's
  ``msr2data`` is a CLI-only batch tool; Mantid's results table
  does not perform secondary parametric fits in the same window.
  See :doc:`parameter_trending`.

**Modern PySide6 + matplotlib single-process GUI.**
  musrfit fragments across separate ``musrview``, ``musredit``,
  ``musrWiz``, ``mupp`` processes. Mantid is a heavy install
  (~1 GB). Asymmetry runs in one process with a single ``pip
  install``.

**Schema-versioned JSON project files (``.asymp``).**
  Forward-compatible state serialisation with documented schema
  migrations. ``.mantid`` files are HDF5 binary; ``.msr`` files lack
  a versioned schema. See :doc:`project_files`.

Where the other tools lead
--------------------------

Areas where Asymmetry currently lags. Each item is tracked as a
candidate in the roadmap (see below).

**Theory function breadth (musrfit / Mantid).**
  musrfit ships ~34 built-in theory functions; Mantid adds ~15
  specialist muon-only functions (``Keren``, ``Meier``,
  ``MuonFInteraction``, four ``*Muonium*`` variants,
  ``MuoniumDecouplingCurve``). Asymmetry has ~17 components.
  Tracked as ``theory-library-expansion``.

**Dynamic Kubo–Toyabe (Mantid).**
  Strong-collision dynamic KT — the canonical model for muon
  dynamics through magnetic transitions — is in all three
  reference programs. Asymmetry's :doc:`lf_kubo_toyabe` page
  currently stays in the static regime. Tracked as
  ``dynamic-kubo-toyabe``.

**MaxEnt frequency reconstruction (WiMDA, Mantid).**
  Production MaxEnt — Burg pole-scan in WiMDA, iterative entropy
  maximisation in Mantid — outperforms apodised FFT on short or
  multi-frequency data. Asymmetry has a placeholder stub. Tracked
  as ``maxent-spectrum``.

**MINOS asymmetric error analysis (musrfit).**
  Per-parameter ``+err / -err`` triples from a χ² contour walk.
  Asymmetry currently reports symmetric Hessian errors only.
  Tracked as ``minos-error-analysis``.

**Rotating Reference Frame transform (Mantid).**
  RRF demodulation for high-TF and vortex-lattice studies. Tracked
  as ``rrf-transform``.

**Automatic phase calibration (Mantid).**
  ``CalMuonDetectorPhases`` fits per-detector phases automatically
  from early-time data. Asymmetry treats phase as either manual or
  a fit parameter. Tracked as ``phase-auto-calibration``.

**Period arithmetic for pulsed beams (Mantid).**
  ISIS multi-period data routinely needs sum / difference
  operations before grouping. Tracked as ``period-arithmetic``.

**Synthetic data simulation (WiMDA).**
  WiMDA's ``Simulate.pas`` generates synthetic count histograms
  from a model + parameters. Useful for teaching, fit validation,
  and cross-tool benchmarking. Tracked as ``simulate-mode``.

**Spectral moments analysis (WiMDA).**
  Quick lineshape characterisation in the frequency-domain panel.
  Tracked as ``moments-analysis``.

Roadmap — the next 12 months
----------------------------

The team is actively working through a prioritised list of
gaps. Priority is set by an explicit ``impact × ease`` score
(see ``docs/porting/ROADMAP.md`` in the repository for the full
methodology). Four candidates are scheduled for the next 4 months:

**Now (0–4 months).**

#. **MINOS asymmetric error analysis** — expose iminuit's
   ``Minuit.minos()`` through the fit panel so parameter errors
   include the asymmetric tail when the χ² landscape is
   non-quadratic.
#. **Dynamic Kubo–Toyabe** — strong-collision dynamic KT,
   completing Asymmetry's KT story (currently only static).
#. **Theory library expansion** — port Keren, Abragam, Bessel,
   SpinGlass, Meier, MuoniumDecouplingCurve, and the time-domain
   superconductor vortex-lattice function from musrfit / Mantid.
#. **Simulate mode** — first-class synthetic-data generation
   from any registered model, with a GUI dialog and per-bin
   Poisson noise.

**Next (4–9 months).**

* **MaxEnt frequency reconstruction** (Burg pole-scan first;
  iterative entropy as a second engine).
* **Rotating Reference Frame transform** for high-TF and
  vortex-lattice analyses.
* **Python user-function plugins** via a one-file decorator API
  (lower-friction analogue of musrfit's C++ ``PUserFcnBase``).
* **Automatic phase calibration** for TF datasets.
* **Period arithmetic** for ISIS pulsed-beam multi-period runs.
* **musrfit ``.msr`` import** for cross-tool interoperability.

**Later (9–12+ months).**

* **Spectral moments analysis** in the Fourier panel.

The roadmap is refreshed quarterly. The latest ranked candidate
list lives in ``docs/porting/ROADMAP.md`` in the repository.

References
----------

Software references:

* **WiMDA:** F. L. Pratt, Physica B 289-290, 710 (2000).
* **musrfit:** A. Suter, B. M. Wojek, Phys. Procedia 30, 69 (2012).
  Source: https://bitbucket.org/muonspin/musrfit
* **Mantid:** O. Arnold *et al.*, Nucl. Instrum. Methods A 764, 156
  (2014). Source: https://github.com/mantidproject/mantid

For the underlying physics see Blundell, De Renzi, Lancaster, Pratt
(eds.), *Muon Spectroscopy: An Introduction*, Oxford University
Press, 2022.
