Reference
=========

This is the feature-by-feature reference for Asymmetry: what each part of the
program does, the options it exposes, and the physics behind it. If you are new,
start with :doc:`/getting_started/index`; if you have data and want a worked
end-to-end analysis, see :doc:`/workflows/index`. If you are scripting or
driving Asymmetry from an agent, the :doc:`cookbook` collects copy-paste recipes
that link out to the pages below.

Find a feature
--------------

If you know the task but not the page, this table maps the common goals to the
reference page that documents them and, where it applies, the place in the
graphical interface where the feature lives.

.. list-table::
   :header-rows: 1
   :widths: 48 30 22

   * - I want to…
     - Reference page
     - In the GUI
   * - Load ISIS, PSI, or ROOT μSR files
     - :doc:`loading_data`
     - File → Open Data File(s)…
   * - Browse, sort, and annotate runs as a logbook
     - :doc:`logbook`
     - Data Browser
   * - Group runs for batch fitting and organisation
     - :doc:`gui_usage`
     - Data Browser → context menu
   * - Save a whole analysis session to reopen later
     - :doc:`project_files`
     - File → Save Project
   * - Calibrate α from a transverse-field run
     - :doc:`grouping_calibration`, :doc:`data_reduction/alpha_calibration`
     - Grouping window → Estimate α
   * - Define forward and backward detector groups
     - :doc:`detector_grouping`
     - Detector Layout editor
   * - Locate or override time-zero (t0)
     - :doc:`data_reduction/t0_search`
     - Grouping window → Find t0
   * - Subtract a background before forming the asymmetry
     - :doc:`data_reduction/backgrounds`, :doc:`data_reduction/background_ladder`
     - Grouping window
   * - Rebin histograms or set the fit time-range
     - :doc:`data_reduction/binning`, :doc:`data_processing`
     - Grouping window → Binning
   * - Drop a dead or hot detector
     - :doc:`data_reduction/detector_exclusion`
     - Grouping window → Exclude Detectors
   * - Map a multi-period pulsed-source file
     - :doc:`data_reduction/period_mapping`
     - Grouping window → Map periods…
   * - Co-add or co-subtract runs (e.g. light-on − light-off)
     - :doc:`run_arithmetic`
     - Data Browser → context menu
   * - Fit a model to a single asymmetry spectrum
     - :doc:`fitting`
     - Fit dock → Single tab
   * - Decide which model to fit
     - :doc:`fit_wizard`
     - Fit dock → Fit Wizard…
   * - Fit one model across a run series
     - :doc:`global_fit_wizard`, :doc:`asymmetry_domain_global_fit`
     - Fit dock → Batch tab → Global Wizard…
   * - Judge whether a fit can be trusted
     - :doc:`assessing_a_fit`
     - Fit results
   * - Build a multi-component model
     - :doc:`composite_models`
     - Build Fit Function dialog
   * - Look up a fit-function form and its parameters
     - :doc:`fit_functions/index`
     - Build Fit Function dialog
   * - Fit each detector group separately (Knight shift)
     - :doc:`grouped_time_domain_fitting`
     - Individual Groups → Multi-Group Fit
   * - Fit raw counts to free α or a single histogram
     - :doc:`count_domain_fitting`
     - Multi-Group Fit → Fit target
   * - Add a custom fit or trend function
     - :doc:`user_functions`
     - Build Fit Function → user function
   * - View a Fourier spectrum of the asymmetry
     - :doc:`fourier_analysis`
     - Frequency-domain workspace
   * - Fit lines directly in the frequency domain
     - :doc:`frequency_domain_fitting`
     - Fourier panel
   * - Read a muon hyperfine coupling from a radical
     - :doc:`radical_correlation`
     - Fourier panel → Correlation (radical)
   * - Trend a fitted parameter across a scan
     - :doc:`parameter_trending`
     - Fit Parameters panel
   * - Decide where to measure next during a scan
     - :doc:`suggest_next_point`
     - Model Fit dialog → Suggest next point
   * - Fit λ(T) and superfluid density
     - :doc:`sc_penetration_depth`
     - Fit Parameters panel
   * - Build an ALC / repolarisation field scan
     - :doc:`alc_mode`
     - Integral scan mode
   * - Demodulate fast TF precession (rotating frame)
     - :doc:`rotating_frame`
     - F-B Asymmetry plot → RRF
   * - Generate synthetic runs to plan or rehearse
     - :doc:`simulation`
     - File → Simulate Preset
   * - Find a menu, panel, or keyboard shortcut
     - :doc:`gui_usage`
     - —

Scripting
---------

Asymmetry is scriptable: almost everything in the GUI has a Python equivalent.
The cookbook gathers short, copy-paste recipes for common tasks, each linking
out to the reference page that explains it in full.

.. toctree::
   :maxdepth: 1
   :caption: Scripting

   cookbook

Data and projects
-----------------

Everything starts with getting runs into the program and keeping track of them —
reading the supported file formats, organising and annotating runs in the
logbook, and saving a whole analysis (data, reduction, fits, and trends) to a
project file so a session can be reopened exactly as it was left.

.. toctree::
   :maxdepth: 1
   :caption: Data and projects

   loading_data
   logbook
   project_files

Reduction and calibration
-------------------------

Before any fit, the raw detector histograms are turned into an asymmetry:
detectors balanced and calibrated, backgrounds handled, time-zero located, bad
detectors excluded, and pulsed-beam periods mapped. Every quantity here feeds
*all* downstream analysis — a weak α estimate or a missed background degrades
every fit made from the data — so each page is explicit about when its method
applies and when it does not. The grouping walkthrough is the guided tour; the
individual reduction pages below document each correction in isolation, and
:doc:`exclusions` disentangles the several "exclude" controls that recur across
the program.

.. toctree::
   :maxdepth: 1
   :caption: Reduction and calibration

   grouping_calibration
   detector_grouping
   data_processing
   data_reduction/alpha_calibration
   data_reduction/backgrounds
   data_reduction/background_ladder
   data_reduction/binning
   data_reduction/t0_search
   data_reduction/detector_exclusion
   data_reduction/period_mapping
   run_arithmetic
   exclusions

Setting up and running fits
---------------------------

The heart of the program: fitting a model to the asymmetry as a function of
time. These pages run from building and judging a single fit, through the
composite-model machinery and the fit-workflow diagnostics, to the grouped,
global, and count-domain variants for when one forward–backward fit is not
enough.

.. toctree::
   :maxdepth: 1
   :caption: Setting up and running fits

   fitting
   assessing_a_fit
   fit_workflow_diagnostics
   composite_models
   grouped_time_domain_fitting
   asymmetry_domain_global_fit
   count_domain_fitting
   user_functions

Fit wizards
-----------

When you are unsure which model to reach for, the wizards fingerprint the data
and recommend one. The single-fit wizard works from one spectrum; the global-fit
wizard reasons across an ordered run series and proposes which parameters to
share and which to let vary.

.. toctree::
   :maxdepth: 1
   :caption: Fit wizards

   fit_wizard
   global_fit_wizard

Fit-function library
--------------------

The catalogue of fit components, mirroring the component picker in the Build Fit
Function dialog: the relaxation and oscillation envelopes, the Kubo–Toyabe
family, muonium and nuclear-dipolar (F–μ–F) models, backgrounds, and the
frequency-domain lines.

.. toctree::
   :maxdepth: 2
   :caption: Fit-function library

   fit_functions/index

Frequency-domain analysis
-------------------------

When a signal is easier to read as a spectrum than a decay — several precession
lines, or a vortex-lattice field distribution — these pages cover the Fourier
transform and its apodisation, fitting directly in the frequency domain, the
maximum-entropy estimator, the spectral moments that summarise a lineshape, and
the correlation spectrum that reads a muoniated radical's hyperfine coupling.

.. toctree::
   :maxdepth: 1
   :caption: Frequency-domain analysis

   fourier_analysis
   frequency_domain_fitting
   frequency_finishers
   spectral_moments
   radical_correlation

Trending and downstream models
------------------------------

A μSR study is rarely a single run. These pages cover following a fitted
parameter across a temperature, field, or angle scan and fitting a physical model
to the resulting trend — muonium kinetics, field-dependent transport, the
superconducting penetration depth, and other consumers of a parameter series.

.. toctree::
   :maxdepth: 1
   :caption: Trending and downstream models

   parameter_trending
   suggest_next_point
   muonium_kinetics
   diffusion_ballistic_lf
   sc_penetration_depth

Specialised modes
-----------------

Techniques that step outside the standard time-domain fit: integral-asymmetry
field scans for avoided-level-crossing work, the rotating reference frame, full
vector polarisation, and the lifetime-based analysis of negative muons.

.. toctree::
   :maxdepth: 1
   :caption: Specialised modes

   alc_mode
   rotating_frame
   vector_polarization
   negative_muon_analysis

Simulation
----------

Asymmetry can generate synthetic data from any model it can fit — useful for
planning a measurement, rehearsing a fitting strategy, or learning the program
away from the beamline.

.. toctree::
   :maxdepth: 1
   :caption: Simulation

   simulation

The graphical interface
-----------------------

How the desktop application is laid out, and the conveniences that speed up
everyday work — the docks and panels, and the small workflow aids the rest of
this reference assumes you can find.

.. toctree::
   :maxdepth: 1
   :caption: The graphical interface

   gui_usage
   workflow_conveniences
