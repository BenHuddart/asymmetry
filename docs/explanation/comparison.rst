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
those tools, see :doc:`/workflows/index`.

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
     - ◐
   * - Rotating Reference Frame
     - ❌
     - ◐
     - ★
     - ✅
   * - Theory function library
     - ◐ ~12
     - ★ ~34
     - ★ ~15 specialist
     - ◐ growing
   * - Static Kubo–Toyabe (ZF / LF)
     - ✅
     - ✅
     - ✅
     - ✅
   * - Dynamic Kubo–Toyabe
     - ✅
     - ✅
     - ★
     - ✅
   * - Composite-model expression syntax
     - ❌
     - ◐
     - ◐
     - ★
   * - MIGRAD / MINOS / HESSE
     - ◐ Hessian only
     - ★ full set
     - ✅ via Mantid Fit
     - ✅ full set
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
     - ✅
   * - Spectral moments analysis
     - ★
     - ❌
     - ❌
     - ✅
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
     - ✅
   * - Logbook / multi-run manager
     - ✅
     - ◐
     - ✅
     - ★
   * - Synthetic data simulation
     - ★
     - ❌
     - ❌
     - ✅
   * - User-defined functions
     - ◐ DLL
     - ◐ C++ plugin
     - ◐ Mantid plugin
     - ✅ Python plugins
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
  spectrum. See :doc:`/reference/fit_wizard`.

**Composite-model expression syntax.**
  Free-form arithmetic over registered components — ``Exponential *
  Oscillatory + Constant``, with fraction-group syntax ``(...){frac}``
  for shared-amplitude bundles. musrfit's ``FUNCTIONS`` block is the
  closest analogue but is more limited; WiMDA and Mantid require
  building composite functions procedurally. See
  :doc:`/reference/composite_models`.

**Interactive parameter trending.**
  Per-run fit parameters appear as a sortable trend table that
  drives an integrated parametric-model fit panel (e.g.
  ``SC_TwoGap_SS`` for σ(T) → λ(T) in superconductors). musrfit's
  ``msr2data`` is a CLI-only batch tool; Mantid's results table
  does not perform secondary parametric fits in the same window.
  See :doc:`/reference/parameter_trending`.

**Modern PySide6 + matplotlib single-process GUI.**
  musrfit fragments across separate ``musrview``, ``musredit``,
  ``musrWiz``, ``mupp`` processes. Mantid is a heavy install
  (~1 GB). Asymmetry runs in one process with a single ``pip
  install``.

**Schema-versioned JSON project files (``.asymp``).**
  Forward-compatible state serialisation with documented schema
  migrations. ``.mantid`` files are HDF5 binary; ``.msr`` files lack
  a versioned schema. See :doc:`/reference/project_files`.

Where the other tools lead
--------------------------

Areas where Asymmetry currently lags. Each item is tracked as a
candidate in the roadmap (see below).

**Theory function breadth (musrfit / Mantid).**
  musrfit ships ~34 built-in theory functions; Mantid adds ~15
  specialist muon-only functions (``Keren``, ``Meier``,
  ``MuonFInteraction``, four ``*Muonium*`` variants,
  ``MuoniumDecouplingCurve``). Asymmetry's component library is
  smaller and still filling in specialist forms (``Keren``,
  ``Abragam``, ``Bessel``, ``SpinGlass``, ``Meier``). Tracked as
  ``theory-library-expansion``.

**Automatic phase calibration (Mantid).**
  Mantid's ``CalMuonDetectorPhases`` fits *every* detector phase
  automatically from early-time data in one step. Asymmetry
  estimates per-group FFT phases and can fit per-group phases inside
  the MaxEnt reconstruction, but has no equivalent one-click
  full-detector auto-calibration. Tracked as
  ``phase-auto-calibration``.

**musrfit ``.msr`` project import.**
  Asymmetry reads its own ``.asymp`` projects but cannot yet import
  a musrfit ``.msr`` file. Cross-tool interoperability is tracked as
  ``msr-import``.

Shipped since the first landscape survey
----------------------------------------

Much of the original gap list has since landed and is documented in
the reference manual: MINOS asymmetric errors
(:ref:`minos-asymmetric-errors`), dynamic Kubo–Toyabe
(:doc:`/reference/fit_functions/kubo_toyabe`), the MaxEnt
reconstruction and Burg pole-scan (:doc:`/reference/fourier_analysis`,
:doc:`/reference/frequency_finishers`), the rotating-reference-frame
transform (:doc:`/reference/rotating_frame`), spectral moments
(:doc:`/reference/spectral_moments`), synthetic-data simulation
(:doc:`/reference/simulation`), period arithmetic and RF-μSR
resonance (:doc:`/reference/alc_mode`), Python user-function plugins
(:doc:`/reference/user_functions`), and negative-muon analysis
(:doc:`/reference/negative_muon_analysis`).

Roadmap — the remaining gaps
----------------------------

The team works through a prioritised list of the gaps that remain.
Priority is set by an explicit ``impact × ease`` score (see
``docs/porting/ROADMAP.md`` in the repository for the full
methodology). The near-term candidates are:

* **Theory library expansion** — continue porting specialist forms
  (Keren, Abragam, Bessel, SpinGlass, Meier, MuoniumDecouplingCurve,
  and the time-domain superconductor vortex-lattice function) from
  musrfit / Mantid.
* **Automatic phase calibration** — a one-step full-detector
  auto-phase for TF datasets, analogous to Mantid's
  ``CalMuonDetectorPhases``.
* **musrfit ``.msr`` import** — read a musrfit project file for
  cross-tool interoperability.

The roadmap is refreshed quarterly. The latest ranked candidate
list lives in ``docs/porting/ROADMAP.md`` in the repository.

References
----------

Software references:

* **WiMDA:** F. L. Pratt, Physica B **289–290**, 710 (2000).
* **musrfit:** A. Suter and B. M. Wojek, Phys. Procedia **30**, 69 (2012).
  Source: https://bitbucket.org/muonspin/musrfit
* **Mantid:** O. Arnold *et al.*, Nucl. Instrum. Methods A **764**, 156
  (2014). Source: https://github.com/mantidproject/mantid

For the underlying physics see S. J. Blundell, R. De Renzi, T. Lancaster,
and F. L. Pratt, *Muon Spectroscopy: An Introduction* (Oxford University
Press, Oxford, 2022).
