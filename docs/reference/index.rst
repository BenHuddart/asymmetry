Reference
=========

This is the feature-by-feature reference for Asymmetry: what each part of the
program does, the options it exposes, and the physics behind it. If you are new,
start with :doc:`/getting_started/index`; if you have data and want a worked
end-to-end analysis, see :doc:`/workflows/index`. If you are scripting or
driving Asymmetry from an agent, the :doc:`cookbook` collects copy-paste recipes
that link out to the pages below.

Asymmetry is scriptable: almost everything in the GUI has a Python equivalent.
The cookbook gathers short, copy-paste recipes for common tasks, each linking
out to the reference page that explains it in full.

.. toctree::
   :maxdepth: 1
   :caption: Scripting

   cookbook

Everything starts with getting runs into the program and keeping track of them —
loading the supported file formats, organising and annotating runs in the
logbook, and saving a whole analysis (data, reduction, fits and trends) to a
project file.

.. toctree::
   :maxdepth: 2
   :caption: Data

   loading_data
   logbook
   project_files

Before any fit, the raw detector histograms are turned into an asymmetry:
detectors balanced and calibrated, backgrounds handled, time-zero located, bad
detectors excluded, and pulsed-beam periods mapped. These choices feed every
later result, so each page is explicit about when its method applies — and when
it does not.

.. toctree::
   :maxdepth: 2
   :caption: Reduction and calibration

   grouping_calibration
   detector_grouping
   data_reduction/index
   data_processing
   run_arithmetic
   exclusions

The heart of the program: fitting a model to the asymmetry as a function of
time. These pages run from building and judging a single fit, through the
composite-model and wizard machinery, to the grouped, global and count-domain
variants for when one forward–backward fit is not enough.

.. toctree::
   :maxdepth: 2
   :caption: Time-domain fitting

   fitting
   assessing_a_fit
   fit_workflow_diagnostics
   composite_models
   fit_functions/index
   fit_wizard
   global_fit_wizard
   grouped_time_domain_fitting
   asymmetry_domain_global_fit
   count_domain_fitting
   user_functions

When a signal is easier to read as a spectrum than a decay — several precession
lines, or a vortex-lattice field distribution — these pages cover the Fourier
transform and its apodisation, fitting directly in the frequency domain, the
maximum-entropy estimator, and the spectral moments that summarise a lineshape.

.. toctree::
   :maxdepth: 2
   :caption: Frequency-domain analysis

   fourier_analysis
   frequency_domain_fitting
   frequency_finishers
   spectral_moments
   radical_correlation

A μSR study is rarely a single run. These pages cover following a fitted
parameter across a temperature, field or angle scan and fitting a physical model
to the resulting trend — muonium kinetics, diffusion autocorrelation, the Knight
shift, and the superconducting penetration depth.

.. toctree::
   :maxdepth: 2
   :caption: Parameter trending

   parameter_trending
   muonium_kinetics
   diffusion_ballistic_lf
   sc_penetration_depth

Techniques that step outside the standard time-domain fit: avoided-level-crossing
field scans, the rotating reference frame, full vector polarisation, and the
lifetime-based analysis of negative muons.

.. toctree::
   :maxdepth: 2
   :caption: Specialised analyses

   alc_mode
   rotating_frame
   vector_polarization
   negative_muon_analysis

Asymmetry can generate synthetic data from any model it can fit — useful for
planning a measurement, rehearsing a fitting strategy, or learning the program
away from the beamline.

.. toctree::
   :maxdepth: 1
   :caption: Simulation

   simulation

How the desktop application is laid out, and the conveniences that speed up
everyday work — the docks and panels, and the small workflow aids the rest of
this reference assumes you can find.

.. toctree::
   :maxdepth: 2
   :caption: The graphical interface

   gui_usage
   workflow_conveniences
