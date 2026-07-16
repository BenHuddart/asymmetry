Analysis workflows
==================

These chapters walk through end-to-end μSR data analyses and
demonstrate the Asymmetry GUI and Python API in lock-step. Almost all
are reproducible walkthroughs on **real instrument runs from the WiMDA
muon-school corpus** — several on the very datasets behind published
papers — each with the model, seeds, and expected numbers; the
Knight-shift chapter (and a few illustrative subsections) use synthetic
versions of canonical experiments grounded in textbook archetypes
(Blundell *et al.* 2022; Amato & Morenzoni 2024).

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
   global_fit_ionic_motion
   knight_shift_angle
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
     - Compare light-ON vs light-OFF muonium dynamics in a photo-μSR run
     - Red/green RG box, ``select_period`` / ``load(period=…)`` API
   * - :doc:`alc_scan_tcnq`
     - Locate an ALC resonance from a field-stepped scan
     - ALC mode, Build Scan, baseline + peak fit, integral asymmetry
   * - :doc:`temperature_scan_magnetism`
     - Locate Tc and measure the order parameter of a ferromagnet
       or antiferromagnet (real EuO / Ni / molecular-magnet data)
     - Logbook sort, Fit Wizard, parameter trending, power-law fit
   * - :doc:`global_fit_ionic_motion`
     - Extract an ion-hopping activation energy from LF triplets in a
       solid electrolyte (real Li₇La₃Zr₂O₁₂ data)
     - Global (batch) fit with shared parameters, Keren model, axis
       transforms, Arrhenius trend
   * - :doc:`knight_shift_angle`
     - Find the muon site from the angle dependence of the Knight
       shift in a rotated single crystal
     - Knight shift analysis window, Angle axis + Fold, Joint K(θ) fit, KnightAnisotropy, Suggest next angle
   * - :doc:`superconductor_penetration_depth`
     - Extract λ(T) from TF μSR in the vortex state of a
       superconductor (real BiSCCO / LiFeAs / Re₆Zr data)
     - Multi-run TF fits, parameter trending, multi-series overlay,
       MaxEnt vortex lineshape, SC gap models
   * - :doc:`lf_decoupling_dynamics`
     - Distinguish static from dynamic local fields with an LF scan
       (real Ca₃Co₂O₆ data)
     - LF exponential fits, axis transforms (1/λ vs B² Redfield),
       static-KT decoupling diagnostic

Further workflows
-----------------

The internal porting workspace
(``docs/porting/practical-workflows/workflow-catalogue.md``)
catalogues additional workflows that Asymmetry partially or
fully supports — F–μ–F identification, spin-glass freezing,
muonium-radical hyperfine, and others.
ALC resonance scans have a worked example above
(:doc:`alc_scan_tcnq`) and a full feature reference in
:doc:`/reference/alc_mode`; the angle-dependent Knight shift has a
worked example above (:doc:`knight_shift_angle`) and a feature
reference at :ref:`knight-shift`. Workflows that Asymmetry does **not**
yet support (e.g. multi-period arithmetic, LEM depth profiling) are
catalogued there too and surface as candidates in
``docs/porting/ROADMAP.md``.
