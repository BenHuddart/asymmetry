Analysis workflows
==================

Start with :doc:`/getting_started/quickstart` for a complete analysis using
synthetic data. These chapters then show how to approach particular
experiments, mainly through the graphical interface, with selected Python
examples. For scripting recipes, see :doc:`/reference/cookbook` and the
:doc:`/reference/agent_workflow` command-line workflow.

Use :doc:`example_data` to find the raw runs behind the figures and fitted
values. The :doc:`knight_shift_angle` chapter and illustrations marked as
synthetic describe their inputs in place.

If you have collected data, :doc:`calibration_grouping_emu` explains grouping
and calibration before you move to the chapter closest to your experiment.
The linked reference pages describe the controls and their limits in detail.

.. toctree::
   :maxdepth: 2

   example_data
   calibration_grouping_emu
   dynamic_kt_copper
   photomusr_silicon_periods
   alc_scan_tcnq
   temperature_scan_magnetism
   global_fit_ionic_motion
   fmuf_entangled_states
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
       (instrument setup)
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
   * - :doc:`fmuf_entangled_states`
     - Identify an F–µ–F entangled state and measure the muon–fluorine
       bond length (real PTFE data)
     - FmuF models, ZF beating signature, sub-MHz FFT
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

The :doc:`feature reference </reference/index>` describes the supported
analysis operations and their limits, including F–μ–F identification,
spin-glass fitting, and muonium/radical models. ALC resonance scans have a
worked example above (:doc:`alc_scan_tcnq`) and a reference in
:doc:`/reference/alc_mode`; the angle-dependent Knight shift has a worked
example (:doc:`knight_shift_angle`) and a reference at :ref:`knight-shift`.

Proposed workflows and unfinished work belong in
`GitHub issues <https://github.com/BenHuddart/asymmetry/issues>`_.
