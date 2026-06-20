Analysis Workflows
==================

These chapters walk through end-to-end μSR data analyses and
demonstrate the Asymmetry GUI and Python API in lock-step. The first
four are reproducible walkthroughs on real instrument runs from the
WiMDA muon-school corpus, each with the model, seeds, and expected
numbers; the remainder use synthetic versions of canonical experiments
grounded in textbook archetypes (Blundell *et al.* 2022; Amato &
Morenzoni 2024).

The narratives are the recommended onboarding path for users who
have collected real data and want to see how to analyse it. They
cross-reference the reference pages elsewhere in this user guide
for full feature documentation.

New users should start with :doc:`calibration_grouping_emu`.

.. toctree::
   :maxdepth: 2

   calibration_grouping_emu
   dynamic_kt_copper
   photomusr_silicon_periods
   alc_scan_tcnq
   temperature_scan_magnetism
   superconductor_penetration_depth
   lf_decoupling_dynamics

Quick chooser
-------------

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Workflow
     - Typical experiment
     - Key Asymmetry features
   * - :doc:`calibration_grouping_emu`
     - Set up EMU detector grouping and calibrate α from a TF run
       (onboarding)
     - Grouping window, Estimate α, visual Detector Layout Editor,
       dead-time
   * - :doc:`dynamic_kt_copper`
     - Measure the muon hop rate ν in copper from a ZF dynamic-KT fit
     - DynamicGaussianKT, fixed Δ / floated ν, motional narrowing
   * - :doc:`photomusr_silicon_periods`
     - Compare light-ON vs light-OFF muonium dynamics in a photo-µSR run
     - Red/green RG box, ``select_period`` / ``load(period=…)`` API
   * - :doc:`alc_scan_tcnq`
     - Locate an ALC resonance from a field-stepped scan
     - ALC mode, Build Scan, baseline + peak fit, integral asymmetry
   * - :doc:`temperature_scan_magnetism`
     - Locate Tc and measure the order parameter of a ferromagnet
       or antiferromagnet
     - Logbook sort, Fit Wizard, parameter trending, power-law fit
   * - :doc:`superconductor_penetration_depth`
     - Extract λ(T) from TF μSR in the vortex state of a
       superconductor
     - Multi-run TF fits, parameter trending panel, SC gap models
   * - :doc:`lf_decoupling_dynamics`
     - Measure the local-field width Δ in a nonmagnetic host and
       distinguish static from dynamic field distributions
     - LF-KT model, global fit with shared Δ, decoupling diagnostic

Further workflows
-----------------

The internal porting workspace
(``docs/porting/practical-workflows/workflow-catalogue.md``)
catalogues additional workflows that Asymmetry partially or
fully supports — F–μ–F identification, paramagnetic Knight shift,
spin-glass freezing, muonium-radical hyperfine, and others.
ALC resonance scans have a worked example above
(:doc:`alc_scan_tcnq`) and a full feature reference in
:doc:`/reference/alc_mode`. Workflows that Asymmetry does **not**
yet support (e.g. multi-period arithmetic, LEM depth profiling) are
catalogued there too and surface as candidates in
``docs/porting/ROADMAP.md``.
