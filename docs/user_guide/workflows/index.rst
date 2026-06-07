Analysis Workflows
==================

These chapters walk through end-to-end μSR data analyses on
synthetic versions of canonical experiments. Each is grounded in a
textbook archetype (Blundell *et al.* 2022; Amato & Morenzoni 2024)
and demonstrates the Asymmetry GUI and Python API in lock-step.

The narratives are the recommended onboarding path for users who
have collected real data and want to see how to analyse it. They
cross-reference the reference pages elsewhere in this user guide
for full feature documentation.

.. toctree::
   :maxdepth: 2

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
ALC resonance scans are covered as a reference feature in
:doc:`/user_guide/alc_mode`. Workflows that Asymmetry does **not**
yet support (e.g. multi-period arithmetic, LEM depth profiling) are
catalogued there too and surface as candidates in
``docs/porting/ROADMAP.md``.
