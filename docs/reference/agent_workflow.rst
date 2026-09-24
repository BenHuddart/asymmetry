Scripting and AI-agent workflow
================================

A μSR temperature or field scan starts the same way every time: survey what
is in the folder, calibrate the detector balance, reduce to asymmetry, screen
a model, fit the scan, and read the trend. Doing that from the GUI is a guided
interactive session; doing it from a shell script, a notebook, or an AI coding
agent needs a command that returns the same answers without a window to
click through. The ``asymmetry`` command-line workflow is that command: a
thin layer of subcommands over the pure-core :mod:`asymmetry.core.workflow`
façade, one per step of the analysis above, each readable by a person and
parseable by a program.

It is meant to be driven by a person from a shell, or by an AI coding agent —
Claude Code or Codex are the two tested so far — through the packaged
``asymmetry-analysis`` skill, which teaches an agent the same workflow, the
decision rules an analyst applies at each step, and the summary a
spectroscopist would recognise at the end.

**In scope**: preliminary forward–backward asymmetry analysis of zero-field
(ZF), transverse-field (TF) and longitudinal-field (LF) runs — a temperature
scan or a field scan at one geometry, named or numbered periods in a
multi-period run, Fourier spectra, integral-asymmetry ALC/QLCR field scans,
and a simultaneous group of runs with shared fitted parameters. Input may be
ISIS NeXus (``.nxs``) or PSI (``.bin``, ``.mdu``), using one forward group
against one backward group. "Preliminary" is the operative word: the workflow produces a
defensible first pass — the right model family, a trend, and flagged runs —
not a publication analysis.

**Out of scope.** The workflow does not improvise around any of these; the
skill tells an agent to say so and stop rather than force a fit:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Case
     - Why it is out of scope
   * - Count-domain fitting
     - ``reduce`` only produces asymmetry, not per-detector counts with an
       N₀ and relaxation term.
   * - Multi-group / orientation-resolved analysis
     - The workflow reduces exactly one forward/backward detector pair, not
       several groups fit together.
   * - Maximum-entropy spectra
     - ``fourier`` provides an FFT and peak table, not maximum entropy
       reconstruction.
   * - Negative-muon (μ⁻) elemental analysis
     - Gamma spectra and elemental lines are not asymmetry data.
   * - Rotating-reference-frame or RF-resonance runs
     - Data modulated at a reference frequency needs a different demodulation
       step than forward–backward asymmetry.
   * - A series of simultaneous groups
     - ``fit-global`` fits one group jointly and ``trend`` reads that group's
       run-local parameters, but no command yet repeats the coupled fit over
       every temperature and trends the *shared* parameters.
   * - A trend of fitted trend parameters
     - ``trend --model`` fits one stored series; a law fitted across several
       such fits (an Arrhenius law through rate constants each fitted at one
       temperature) has no command.
   * - A fragment of a published multi-field campaign
     - No self-contained scan in the survey and fields the files do not
       record cannot be reconstructed from what is on disk. Two complete
       scans at two recorded fields are analysable even with a gap in run
       numbers between them.

A folder the tool can *load* is not automatically a folder the tool can
*analyse* — loadability is not scope.

For the underlying Python API this CLI wraps, see the :doc:`cookbook`; for the
physics behind each step, see the workflow chapters linked from
:doc:`/workflows/index` and section 5 of the skill itself.

Installation
------------

The CLI ships as part of the ``asymmetry`` package; the ``agent`` extra adds
the two optional dependencies the workflow needs for headless plots and NeXus
loading (matplotlib and h5py):

.. code-block:: bash

   pipx install "asymmetry[agent] @ git+https://github.com/BenHuddart/asymmetry.git"

Add the ``hdf4`` extra for legacy HDF4-container ``.nxs`` files, or ``root``
for ROOT files, alongside ``agent`` in the same bracket
(``asymmetry[agent,hdf4]``). The prebuilt desktop application does **not**
put this CLI on ``PATH`` — it is a separate, self-contained bundle — so
agent-driven use always needs this ``pip``/``pipx`` install even on a machine
that already has the desktop app.

Install the skill for an agent, then verify it:

.. code-block:: bash

   asymmetry skill install --agent claude   # or --agent codex
   asymmetry skill check
   asymmetry skill uninstall --agent claude

For developing the skill itself, ``skill install --link`` symlinks the packaged
skill instead of copying it, so edits in a checkout reach the agent without
reinstalling:

.. code-block:: bash

   asymmetry skill install --agent claude --link

A linked install writes no manifest — nothing is written into the package
directory — and needs none: the symlink resolves to the packaged skill, which is
both the proof this command made it (so ``install`` may replace it and
``uninstall`` may remove it, the link only, never what it points at) and the
reason it is always current. ``skill check`` reports it as ``linked
(development)``. A symlink pointing anywhere else is refused without ``--force``.

``skill install`` copies the packaged skill into the agent's usual skill
directory — ``~/.claude/skills/asymmetry-analysis/`` for Claude, or
``~/.agents/skills/asymmetry-analysis/`` for Codex — and stamps a
``.asymmetry-skill.json`` manifest beside ``SKILL.md`` recording the
installed version, so a later ``skill check`` can tell a stale copy from a
current one and ``uninstall`` can tell a directory this command wrote from
one that just happens to be in the way. Pass ``--project`` to install under
``./.claude`` or ``./.agents`` in the current directory instead of the home
directory, or ``--into DIR`` to install (or remove) at an explicit path.
``skill check`` reports, for every installed agent, whether the CLI is on
``PATH``, matplotlib imports, at least one file-format loader extra is
present, and whether each installed skill's manifest matches the running
``asymmetry`` version — rerun ``skill install`` after upgrading the package
if it does not.

**Restart the agent** (or start a new session) after installing or upgrading
the skill — an already-running agent has already scanned its skills
directory and will not pick up a change made mid-session.

Common options
--------------

Every subcommand below shares the same conventions:

- ``--json`` emits a machine-readable payload on stdout instead of the human
  table; every payload carries ``"schema": 1`` and ``"asymmetry_version"``.
  Omit it for the default human-readable table, which is usually easier to
  read directly.
- ``--plot`` (on ``reduce``, ``integral-scan``, ``wizard``, ``fit``,
  ``fit-global``, ``fit-series``, ``trend`` and ``fourier``) writes one or
  more headless PNGs into the work directory's
  ``plots/`` — see `Plots`_.
- ``--workdir`` overrides the work directory, which otherwise defaults to
  ``./asymmetry-work`` — in the directory the command is run from, not in the
  data folder. One work directory holds one data folder's session, so this is
  the flag to reach for when a second folder is analysed from the same
  project: ``--workdir asymmetry-work-<name>``, passed to every command on
  that folder. Pointing a command at a directory that already holds a
  different folder's session is a user error, not a silent merge.
- ``--verbose`` (on the main ``asymmetry`` command, before the subcommand)
  prints every warning as Python's own multi-line traceback-style block.
  Without it, a repeated warning — the fit wizard's ``AsymmetryScaleWarning``
  from candidate seeding, in particular — is collapsed to one line on stderr
  per distinct warning rather than once per occurrence.
- **Exit codes**: ``0`` on success, ``1`` on a user error (one line on
  stderr — a bad run number, a missing recipe), ``2`` on an internal error
  (a full traceback, because that is a bug worth reporting).

Commands
--------

``survey``
~~~~~~~~~~

List the runs in a folder with their metadata, scans and calibration
candidates — always the first command run against a new folder.

.. code-block:: text

   asymmetry survey [-h] [--json] [--workdir WORKDIR] folder

Writes ``survey.json`` into the work directory. Groups runs into scans by
(instrument, field) ordered by temperature and by (instrument, temperature)
ordered by field, so the structure of a multi-scan folder is visible without
reading every file. When the logged sample temperature departs from the
setpoint by more than 0.3 K and 1 % on any run, a ``TEMPERATURE:`` line names
those runs (``temperature_departures`` in ``--json``): they were not at their
setpoint, and the setpoint-grouped scans that contain them are provisional.
The line lists the runs in consecutive blocks of similar offset with each
block's range (``T log − T/K``), so a block sitting several kelvin away from the
rest stands out as its own measurement.
Each alpha-calibration candidate is listed with its own
measured alpha, and where alpha moves by more than 10 % between consecutive
candidates in run order — a sample change, a moved detector, a second
instrument — the survey prints an ``ALPHA STEP`` line naming the two runs: no
single run calibrates such a folder, and each block of runs is reduced with a
calibration run from inside it (``alpha_steps`` in ``--json``). ``T/K`` is the temperature setpoint and ``T log/K`` the
logged sample temperature where the file records one (``-`` where it does
not, as in these simulated files and in PSI ``.bin`` files); a cryostat can
leave the two several kelvin apart, and a series can be ordered by either:

.. code-block:: console

   $ asymmetry survey runs
   8 run(s) in runs — SIM

   run  T/K    T log/K  B/G     geom  prec    orient        hist  periods  points  events   dt   title
   ---  -----  -------  ------  ----  ------  ------------  ----  -------  ------  -------  ---  -------------------------------------
   101  5.00   -        100.00  TF*   larmor  Longitudinal  8     1        500     2000123  no   Calibrant T=5.0 K B=100.0 G
   102  10.00  -        0.00    ZF    -       Longitudinal  8     1        500     1999876  no   Sample T=10.0 K B=0.0 G
   ...
   107  60.00  -        0.00    ZF    -       Longitudinal  8     1        500     2000456  no   Sample T=60.0 K B=0.0 G
   108  2.00   -        110.00  -     none    Longitudinal  8     1        500     1999544  no   Sample T=2.0 K B=110.0 G (decoupling)

   Alpha-calibration candidates:
     run 101 (best) [measured] alpha 1.2500: precession at the Larmor frequency of the recorded 100 G (SNR 93)

   Scans:
     temperature scan, SIM, ZF, B = 0 G: 6 runs, 10 to 60 K (run 102 -> 107)

.. _agent-workflow-precession:

Measured transverse-field precession
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A run file's own account of its applied-field geometry cannot be trusted. ISIS
EMU files from 2024 record no field state at all; other ISIS files stamp ``TF``
on longitudinal decoupling runs and on the zero-field runs of a scan that opened
with a weak-TF calibration. A survey built on that metadata alone reports "no
calibration candidates" for a folder holding a perfectly good transverse-field
run, and mislabels a decoupling scan.

So the survey *measures* instead. Every run with a recorded non-zero field is
reduced under the default reduction settings and fingerprinted with the same
spectral reading the fit wizard shortlists candidate models from
(:func:`~asymmetry.core.fitting.fit_wizard.fingerprint_spectrum`), and its
dominant line is compared with the Larmor frequency of the recorded field,
γ\ :sub:`μ`/2π × B. The ``prec`` column reports the verdict:

.. list-table::
   :header-rows: 1
   :widths: 12 88

   * - ``prec``
     - Meaning
   * - ``larmor``
     - A line at SNR ≥ 10 within 25 % of the Larmor frequency of the recorded
       field. The field is transverse, and ``geom`` reads ``TF*`` — the ``*``
       marks a geometry the spectrum decided rather than the file.
   * - ``other@<MHz>``
     - A line somewhere else, printed with its frequency: the muon precesses
       in a field that is not the applied one — an internal field in an
       ordered magnet below its transition, muonium in a weak transverse field
       (1.394 MHz/G), or the critical field inside a type-I superconductor's
       normal domains. This says nothing about the applied field's direction,
       so ``geom`` falls back to the file.
   * - ``none``
     - No line above SNR 10. The applied field is not precessing the muon, so
       the file's ``TF`` stamp is **refuted** and ``geom`` reads ``-``
       (``geometry_source`` ``"refuted"``): the run is either a longitudinal
       measurement or a transverse one with no resolvable line, and the file's
       claim is evidence for neither. A fixed-temperature field scan of such
       runs is longitudinal decoupling.
   * - ``-``
     - Not measured: the field is zero (nothing to look for), or its Larmor
       frequency is above the record's Nyquist frequency. ``precession_note`` in
       the JSON says which.

The thresholds are :data:`~asymmetry.core.workflow.survey.PRECESSION_SNR_FLOOR`
(10) and :data:`~asymmetry.core.workflow.survey.LARMOR_FREQUENCY_TOLERANCE`
(0.25). They were set on measured data: genuine 20 G and 100 G transverse-field
runs score SNR 89–418 and land 3–14 % above the nominal Larmor frequency, while
the longitudinal decoupling runs ISIS stamps ``Transverse`` at 40–120 G score
about 3, and ordered-state runs put their line a factor of 4 to 40 away.

A dominant line that completes fewer than two cycles in the record (the fit
wizard's own
:data:`~asymmetry.core.fitting.fit_wizard.MIN_CYCLES_IN_EFFECTIVE_WINDOW`)
away from the Larmor frequency is the relaxation leaking into the lowest
frequency bins, not precession — on the corpus it sat near 0.1 MHz in every
weak-field and decoupling run. For such a run the survey reports the line the
fingerprint's damped-line scan found instead, which reaches the heavily damped
lines a windowed FFT misses (a muonium line in a 2 G field, the normal-domain
line of a type-I superconductor), and ``none`` when that scan found nothing. A
slow line *at* the Larmor frequency — a weak-TF calibration completing only a
few cycles — is kept.

**Calibration candidates follow from the same measurement**, from two sources,
each named in the candidates block:

- ``[measured]`` — precession at the Larmor frequency of a recorded field inside
  the weak-TF window
  (:data:`~asymmetry.core.data.calibration.WEAK_TF_FIELD_RANGE_GAUSS`), with the
  SNR quoted. The best candidate is the strongest such line.
- ``[metadata]`` — no measurement was possible, and
  :func:`~asymmetry.core.data.calibration.classify_tf_calibration_run` decides
  from the file's own transverse-field evidence alone.

Where a measurement *was* possible it decides: a file's ``TF`` stamp on a run
with no precession in it is not a calibration candidate. ``asymmetry alpha``
applies the same rule and prints its own ``precession`` line, so the two
commands never disagree about whether a run will calibrate alpha.

Each run's ``survey.json`` row carries ``n_periods``, ``total_events``,
``geometry``, ``geometry_source``
(``field``, ``measured``, ``refuted``, ``file`` or ``none``), ``precession``,
``precession_frequency_mhz``, ``precession_snr``, ``precession_larmor_mhz`` and
``precession_note``.

``total_events`` is the gross count summed over every raw detector histogram
in the default (first) period. The human table shows the same value under
``events``. Use it to choose between otherwise comparable science runs; do not
rank candidate spectra by precession SNR, which can favour a calibration run
or a warmer, narrower line.

How scans are grouped
^^^^^^^^^^^^^^^^^^^^^

A scan's key is the **instrument** and the held quantity — never the geometry.
Two instruments in one folder are two campaigns and must never merge into one
scan (an EMU and a MUSR scan of the same sample, say). Geometry, by contrast, is
measured per run, so a scan that resolves only in part is still one scan: a
transverse-field scan taken through a magnetic transition resolves above it,
where the sample is paramagnetic, and not below.

Each :class:`~asymmetry.core.workflow.survey.ScanGroup` therefore reports
``instrument`` alongside ``geometry``, which is the members' single agreed
geometry — and ``None`` with a ``geometry_note`` tallying them when they
disagree:

.. code-block:: console

   Scans:
     temperature scan, EMU, mixed geometry, B = 100 G: 21 runs, 340 to 380 K (run 124269 -> 124249)
         geometry: TF measured on 12 of 21 runs; 9 unresolved

That note is itself a finding: it says where in the scan the measurement could
settle the geometry and where it could not.

``alpha``
~~~~~~~~~

Estimate the forward/backward detector balance α on one run — the same
per-run estimate the GUI's **Estimate α** button computes.

.. code-block:: text

   asymmetry alpha [-h] --run RUN [--json] folder

``alpha`` takes no ``--workdir`` and writes nothing to disk; it loads the
named file directly and prints the estimate, whether the run is a suitable
calibration candidate, and a warning when it is not:

.. code-block:: console

   $ asymmetry alpha runs --run 101
   Run 101 (SIM00000101.nxs)
     alpha           : 1.2521
     method          : per_run_estimate
     calibration run : yes — precession at the Larmor frequency of the recorded 100 G (SNR 93)
     precession      : at the Larmor frequency: yes — 1.363 MHz (SNR 93) against a Larmor 1.355 MHz

``reduce``
~~~~~~~~~~

Reduce runs to forward/backward asymmetry and cache them in the work
directory.

.. code-block:: text

   asymmetry reduce [-h] --runs RUNS [--alpha ALPHA] [--alpha-from ALPHA_FROM]
                    [--period RED|GREEN|N] [--deadtime {off,from_file}]
                    [--rebin REBIN] [--tmin TMIN]
                    [--tmax TMAX] [--plot] [--json] [--workdir WORKDIR]
                    folder

``--runs`` takes ranges and commas (``102-107``, ``102-105,107``).
``--alpha-from RUN`` estimates alpha on that run and uses it; ``--alpha X``
sets it directly; without either, alpha defaults to 1.0. ``--deadtime
from_file`` applies each file's own per-detector deadtime values — the
default is ``off``, matching the GUI's fresh-run default. Writes
``reduced/<run>.npz`` (time, asymmetry, error) and ``reduced/<run>.json``
(metadata, settings and a cache digest) per run, plus ``manifest.json``, and
``plots/reduced-<run>.png`` per run with ``--plot``. Results are cached on a
digest of the source file, the grouping and the reduction settings, so
re-running ``reduce`` on unchanged runs is cheap.

``--period red``, ``--period green`` or ``--period N`` selects one period
before alpha calibration and reduction. The choice is part of the cache
digest. ISIS photo-μSR commonly records light-ON as red and light-OFF as
green, but the experiment notes and spectra remain the authority. Since the
cache is keyed by source run number, use separate work directories if two
periods of the same run must coexist.

For photo-μSR, use the weak-TF run only to obtain alpha, then reduce the same
science run's light-ON and light-OFF periods into separate work directories.
Fit the dark period first when its amplitude is needed to stabilise an
early-time light-period rate. Run order may visualise a rate sequence, but it
is not a substitute for injected carrier density or laser delay; do not report
a density exponent or carrier lifetime unless those x values were supplied from
the experiment's record (``--order`` with ``--x``) and fitted with
``trend --model``.

.. code-block:: console

   $ asymmetry reduce runs --runs 102-107 --alpha-from 101 --deadtime from_file --plot
   run  T/K    B/G   points  A(0)/%   err/%  alpha   deadtime
   ---  -----  ----  ------  -------  -----  ------  --------
   102  10.00  0.00  500     8.537    2.366  1.2521  file
   ...
   107  60.00  0.00  500     -11.301  2.321  1.2521  file

``wizard``
~~~~~~~~~~

Screen a reduced run against the fit wizard's candidate models — the same
engine behind the GUI's single-fit wizard.

.. code-block:: text

   asymmetry wizard [-h] --run RUN [--geometry {ZF,TF,LF}] [--scope PRESET]
                    [--include C,D] [--exclude C,D] [--tmin TMIN]
                    [--tmax TMAX] [--plot] [--json] [--workdir WORKDIR]
                    folder

``--geometry`` overrides every other source. Without it the geometry comes from
the folder's ``survey.json`` when one exists — including a geometry the survey
*measured* from Larmor precession (see :ref:`agent-workflow-precession`) — and
otherwise from this run's own metadata; the header line names the source
(``user``, ``survey``, ``field``, ``file`` or ``none``). Pass it whenever the
reduced spectrum tells you something the survey could not, such as a
longitudinal decoupling run the file stamps ``TF``. ``--scope`` restricts the
candidate families to a preset (``auto``, ``zf-static-magnetism``,
``tf-knight-precession``, ``tf-superconductor``, ``lf-dynamics``,
``fluoride-fmuf``, ``muonium-radical``, ``all``) when the physics is already
known. ``--include`` adds time-domain components the preset leaves out (for
example ``Oscillatory`` for a precession line in an LF run, as in a type-I
superconductor's intermediate state) and ``--exclude`` drops components the
physics rules out (``VortexLattice,VortexLatticePowder`` for a magnet in TF);
exclude wins over include, the header line lists both (``scope lf-dynamics
+Oscillatory``), and an unknown component name is refused with the full list.
``--tmin``/``--tmax`` screen a window of the run, and the recipe written keeps
it. Below the ranked table the report prints ``Spectral lines`` (every line
the spectral search detected, with its SNR) and ``Recommended fit`` (the
recommended model's fitted values), so a precession frequency found while
screening is on the page, not only in the stored recipe. A detected line the
recommendation does not fit is followed by a ready ``asymmetry recipe`` command
started at that frequency. Lines that complete fewer than two cycles in the
record's informative window — relaxation leaking into the lowest bins — are
not listed, by the survey's rule.
Writes ``wizard/<run>.json`` (the full screening payload:
recommendation, ranked candidate table, narrative) and
``recipes/wizard-<run>.json`` (the fit recipe built from the recommended
candidate's fitted values — see `The fit recipe`_), plus
``plots/wizard-<run>.png`` (data, recommended curve, residuals) with
``--plot``:

.. code-block:: console

   $ asymmetry wizard runs --run 102 --plot
   Run 102 — geometry ZF (from survey), scope auto

   Recommendation: exp_constant
     Recommended: Exponential + Constant by AICc (medium confidence).
     confidence : medium
     verdict    : structured

      key           title                  category  AICc   chi2_red  params
   -  ------------  ---------------------  --------  -----  --------  ------
   *  exp_constant  Exponential + Constant  General  487.7  0.969     3

``recipe``
~~~~~~~~~~

Write a fit recipe for a model expression, bypassing the wizard — for the model
the physics calls for when the wizard does not recommend it, or has no template
for it.

.. code-block:: text

   asymmetry recipe [-h] --expression EXPRESSION --name NAME [--run RUN]
                    [--initial NAME=VALUE] [--fix NAME=VALUE] [--tmin TMIN]
                    [--tmax TMAX] [--json] [--workdir WORKDIR]
                    folder

``--run`` seeds the amplitudes, background and applied field from that reduced
run (the same seeding every fit surface uses); ``--initial`` moves a start
value, ``--fix`` holds one (and pins it, so a series never re-seeds it), and
``--tmin``/``--tmax`` set the fit window. The command writes
``recipes/<name>.json`` and prints every parameter — the names a repeated
component is numbered with are otherwise easy to guess wrong:

.. code-block:: console

   $ asymmetry recipe runs --name mu --run 101 \
         --expression "Oscillatory * Exponential + Oscillatory * Exponential" \
         --fix frequency_1=2.79 --fix frequency_3=0.0279
   mu — Oscillatory * Exponential + Oscillatory * Exponential, seeded from run 101

   parameter    start      min     max  state
   -----------  ---------  ------  ---  -----
   A_1          ...        0.0000  inf  free
   frequency_1  2.790000   0.0000  inf  fixed
   phase_1      0.000000   -inf    inf  free
   Lambda_2     ...        0.0000  inf  free
   A_3          ...        0.0000  inf  free
   frequency_3  0.027900   0.0000  inf  fixed
   phase_3      0.000000   -inf    inf  free
   Lambda_4     ...        0.0000  inf  free

An unknown component is refused with the nearest names, and a parameter the
expression does not have with the list it does.

``fit``
~~~~~~~

Fit one reduced run with a recipe.

.. code-block:: text

   asymmetry fit [-h] --run RUN --recipe RECIPE [--fix NAME=VALUE] [--free NAME]
                 [--tmin TMIN] [--tmax TMAX] [--plot] [--json] [--workdir WORKDIR]
                 folder

``--recipe`` takes a name in the work directory's ``recipes/`` (no path, no
``.json``) or an explicit path. ``--fix NAME=VALUE`` holds a parameter
(repeatable); ``--free NAME`` releases one the recipe holds fixed. ``fit``
reads the reduced run and the recipe but does not persist its own result —
only ``plots/fit-<run>.png`` with ``--plot`` — so it is the quick way to check
a hand-edited recipe converges on one run before spending a whole series on
it (see `Hand-editing a recipe`_).

``fit-global``
~~~~~~~~~~~~~~

Fit two or more reduced runs in one coupled objective with named parameters
fitted once across every run:

.. code-block:: text

   asymmetry fit-global [-h] --runs RUNS --recipe RECIPE [--fix NAME=VALUE]
                        [--free NAME] --shared P,Q [--field-param NAME]
                        [--strategy {joint,profiled,least_squares}]
                        [--order QUANTITY] [--x RUN=VALUE,...] [--name NAME]
                        [--plot] [--json] [--workdir WORKDIR]
                        folder

``--shared P,Q`` is a true shared fit, unlike ``fit-series --global``.
``--field-param B_L`` seeds that parameter from each run's recorded field and
holds it for that run, which is the usual structure of an LF decoupling
triplet. Other parameters remain run-local. ``--order`` and ``--x`` work as
for `fit-series`_ (the default is ``run``): the printed table lists every run
along that axis with its run-local parameters, and the stored
``series/<name>.json`` carries them as a trend table, so ``trend`` reads and
fits a simultaneous fit exactly as it does a series. The shared values and
their uncertainties are stored beside it; ``--plot`` writes one fit plot per
run. The command fits one group at a time—repeat it for each temperature when
analysing a sequence of triplets.

``fit-series``
~~~~~~~~~~~~~~

Fit a recipe across a scan of reduced runs, chained along the scan order —
the command that actually produces a trend.

.. code-block:: text

   asymmetry fit-series [-h] --runs RUNS --recipe RECIPE [--fix NAME=VALUE]
                        --order QUANTITY [--x RUN=VALUE,...] [--tmin TMIN]
                        [--tmax TMAX] [--global P,Q]
                        [--start RUN] [--name NAME] [--plot] [--json]
                        [--workdir WORKDIR]
                        folder

``--order`` names the scan quantity the series is ordered and trended along:
``temperature`` (the setpoint), ``sample_temperature_logged``, ``field`` or
``run``, read from each run's file. Any other name orders the series along a
quantity the files do not record — a concentration, a degrader foil count, a
magnet current — whose value for every run is given with ``--x``, for
example ``--order concentration --x 101=0,102=0.25,103=0.5``. A run
with no value, a value for a run outside the series, or a file-recorded
quantity given values by hand is refused.
``--start RUN`` chains outward from that run in both directions instead of
from the first run in scan order — see `Series fitting`_ for why this
matters. ``--global P,Q`` pins those parameters at their recipe value for
every run rather than fitting them (see `The fit recipe`_). Writes
``series/<name>.json`` (per-run results, a trend table, and quality flags —
the default name is ``series-<recipe stem>``), plus a PNG per run
(``plots/<name>/<run>.png``) and one trend PNG per free parameter
(``plots/<name>-trend-<param>.png``) with ``--plot``.

.. code-block:: console

   $ asymmetry fit-series runs --runs 102-107 --recipe wizard-102 \
         --order temperature --start 102 --name zf-scan --plot
   run  temperature  chi2_red  verdict  flags
   ---  -----------  --------  -------  --------------------------------
   102  10.000       0.969     good     -
   ...
   107  60.000       1.004     good     large_rel_err, spurious_reseeded

   flagged  : 1 of 6 run(s) — none were dropped

``trend``
~~~~~~~~~

Print, export or fit the parameter trend of a stored series or simultaneous
fit.

.. code-block:: text

   asymmetry trend [-h] --series SERIES [--csv CSV] [--plot] [--model EXPR]
                   [--param PARAM] [--xmin XMIN] [--xmax XMAX]
                   [--initial NAME=VALUE] [--fix NAME=VALUE] [--exclude RUNS]
                   [--json] [--workdir WORKDIR]
                   folder

Reads ``series/<name>.json`` and prints the scan variable and every fitted
parameter with its uncertainty, per run, with the quality flags carried
through. ``--csv PATH`` also writes the table as CSV. ``--plot`` writes one
PNG per free parameter (``plots/<series>-trend-<param>.png``):

.. code-block:: console

   $ asymmetry trend runs --series zf-scan --plot
   run  x   A_1       A_1_err   Lambda    Lambda_err  flags
   ---  --  --------  --------  --------  ----------  --------------------------------
   102  10  19.7178   2.06445   0.143275  0.0228251   -
   ...
   107  60  0.198726  0.232033  1.06863   1.67424     large_rel_err, spurious_reseeded

``--model EXPR --param NAME`` fits a parameter-vs-x law to one trend column —
the same fit, seeding and extra starts as the desktop
:doc:`parameter trending <parameter_trending>` dialog. The expression uses the
trend-model components: ``OrderParameter`` for a precession frequency or
internal field below a transition, ``Arrhenius`` for an activated rate,
``Redfield`` for a relaxation rate against longitudinal field, the ``SC_*``
gap models for a superconducting σ(T), ``Linear`` and sums such as
``Redfield + Constant``. ``--xmin``/``--xmax`` bound the fit in the trend's x
units, ``--fix NAME=VALUE`` holds a law parameter and ``--initial NAME=VALUE``
moves a start value. When χ²\ :sub:`r` is above 1 the error column holds the
errors scaled by √χ²\ :sub:`r` (the unscaled ones follow), and a ``NOTE``
appears when ``--param`` is one of several components of the same kind in the
series' model (``Lambda_1`` beside ``Lambda_2``): a law describes one physical
rate or line, not one of two that split it between them. Excluding a run is the analyst's call: every run with a
value enters unless ``--exclude RUNS`` names it, and the output lists both the
runs left out (with the reason) and the flagged runs that were fitted. The fit
is stored in ``series/<name>.json`` under ``trend_fits``, and ``--plot``
draws the curve over the points it rests on. On the simulated scan, whose rate
was generated as 0.10 + 0.004 T:

.. code-block:: console

   $ asymmetry trend runs --series scan --model Linear --param Lambda
   ...
   Fit of Linear to Lambda against temperature: 5 point(s), chi2_red 0.285
   parameter  value     error
   ---------  --------  --------
   m          0.004150  0.000646
   b          0.099391  0.022419
   flagged but fitted: 102 (large_rel_err); 103 (large_rel_err); ...

``audit``
~~~~~~~~~

List the numbers in a draft summary that no command printed.

.. code-block:: text

   asymmetry audit [-h] [--workdir WORKDIR] [--json] draft

Every other command appends what it printed to ``cli-output.log`` in the work
directory it used (commands without a ``--workdir``, such as ``alpha``, log to
the default one when it exists). ``audit`` extracts each number from the draft
and reports the ones that appear in none of those logs, with the line they sit
on. A number written with *d* decimals matches any printed value it rounds
from, so a clean audit means every number appears in *some* output, not that it
is the right one; a multiple or a significance (``10×``, ``4.3σ``) matches only
when a command printed that exact token. What it catches is the arithmetic an
analyst does in prose — percentage changes, ratios, unit conversions,
differences between printed columns — which the agent skill's number rule
forbids.

``integral-scan``
~~~~~~~~~~~~~~~~~

Build time-integral asymmetry versus field (or temperature/run order) and
optionally fit a field-scan expression. This is the ALC/QLCR path:

.. code-block:: text

   asymmetry integral-scan [-h] --runs RUNS [--name NAME] [--alpha ALPHA]
                           [--alpha-from ALPHA_FROM] [--period RED|GREEN|N]
                           [--tmin TMIN] [--tmax TMAX]
                           [--method {integral,differential}]
                           [--order {field,temperature,run}] [--model MODEL]
                           [--initial NAME=VALUE] [--fix NAME=VALUE]
                           [--baseline MODEL] [--baseline-regions LO:HI,...]
                           [--plot] [--json] [--workdir WORKDIR] folder

For example, ``--model "LorentzianLCR + Cubic"`` fits an off-zero resonance
and background together. Alternatively, ``--baseline Cubic
--baseline-regions 2000:2600,4500:5000 --model LorentzianLCR`` determines the
background only from non-resonant regions before fitting the corrected scan.
The scan points, excluded runs, fit parameters and uncertainties are stored in
``scans/<name>.json``.

Report the resonance field, width, amplitude and uncertainties the command
prints. Do not use shell arithmetic or a literature formula to turn them into
a numeric hyperfine coupling and present it as command output; keep that
relationship qualitative unless Asymmetry itself emitted the derived value.

``fourier``
~~~~~~~~~~~

Transform one reduced run and report quantitative peaks:

.. code-block:: text

   asymmetry fourier [-h] --run RUN [--name NAME]
                     [--window {none,hann,cosine,gaussian,lorentzian}]
                     [--padding PADDING] [--tmin TMIN] [--tmax TMAX]
                     [--phase PHASE] [--filter-tau FILTER_TAU]
                     [--fmin FMIN] [--fmax FMAX] [--peaks PEAKS]
                     [--plot] [--json] [--workdir WORKDIR] folder

The command stores numerical arrays in ``spectra/<name>.npz`` and settings,
resolution and the peak table in ``spectra/<name>.json``. Zero padding makes
the plotted curve smoother but does not improve the reported resolution,
which is set by the selected time window. This is an FFT, not MaxEnt.
Peaks are detected on the whole spectrum and then restricted to
``--fmin``/``--fmax``, so a narrow zoom around a line keeps the line in the
table (the noise floor is the spectrum's, never the zoomed band's). Peak
detection is deliberately conservative: when nothing passes it, the command
lists the band's strongest maxima with their height over the noise floor,
headed "candidates, not detections" (``candidate_maxima`` in ``--json``) — a
weak line to confirm or refute with a time-domain fit started at that
frequency. Inspect the spectrum PNG for weak shoulders as well as reading the
table. Heed an
``ApodisationEarlySignalWarning``: use an unwindowed physical ``--tmax`` crop,
or a ``lorentzian`` window whose ``--filter-tau`` matches the damping rate,
when a symmetric taper removes the early-time signal.

Use a science run—not the weak-TF alpha-calibration run—for the representative
FFT. In a temperature series, the coldest high-statistics science run is the
usual first choice when splitting or broadening is expected. If several runs
share that temperature, use their event counts to choose the highest-statistics
one rather than the first run number. For semiconductor/shallow-donor or
muonium data, inspect around the Larmor line for a central peak and symmetric
satellites. When Fourier structure is the experiment's point, call the result
an FFT rather than MaxEnt and include its window, resolution, peak table and
visual PNG inspection in the final report; a generic time-domain fit does not
replace them.

``skill``
~~~~~~~~~

Install, check or remove the ``asymmetry-analysis`` agent skill — see
`Installation`_ above for ``install``, ``check`` and ``uninstall`` in detail.

.. code-block:: text

   asymmetry skill [-h] {install,check,uninstall} ...

``info``
~~~~~~~~

Show metadata for a single data file, independent of any work directory —
the same summary the GUI's file inspector shows, useful for checking one
file directly without surveying a whole folder. ``--json`` adds the run
number, the point count and the file's metadata beside that summary (without
the loader's verbatim NeXus field tree, which would bury it).

.. code-block:: text

   asymmetry info [-h] [--json] file

The work directory
-------------------

``survey``, ``reduce``, ``integral-scan``, ``wizard``, ``recipe``,
``fit-global``, ``fit-series`` and ``fourier`` persist their state in ``./asymmetry-work/``;
``fit`` and ``trend`` read it and add only what
``--plot`` (and ``trend --csv``) asks for. ``alpha`` and ``info`` are
stateless — they load a file, print, and write nothing — and ``skill`` writes
into the agent's own skill directory instead.

The directory is resolved against **the directory the command is run from** —
the project the analysis lives in — and never against the data folder, which
is routinely a read-only share or an archive and is not somewhere an analyst
would want cached spectra and plots to appear. It is deliberately not hidden:
the plots, the recipes and the stale sessions in it are all things a person
has to find.

The work directory holds:

.. code-block:: text

   asymmetry-work/
     manifest.json          # asymmetry version, folder, settings, run list
     survey.json            # output of `survey`
     reduced/<run>.npz      # time, asymmetry, error
     reduced/<run>.json     # run metadata + reduction settings + cache digest
     wizard/<run>.json      # screening payload: recommendation, narrative, recipe
     recipes/<name>.json    # a fit recipe (model + parameters + window)
     series/<name>.json     # per-run results, trend table, quality flags, trend fits
     scans/<name>.json      # integral-scan points, exclusions and optional fit
     spectra/<name>.npz     # Fourier frequency, real part and magnitude
     spectra/<name>.json    # Fourier settings, resolution and peak table
     plots/*.png            # headless PNGs written by --plot
     cli-output.log         # every command's printed output, for `audit`

so a later command picks up a reduced spectrum, a recipe, or a series without
reloading or recomputing it, and an agent (or a shell script run in several
steps) has state between invocations without a long-lived process — the same
role a future MCP server would hold in memory instead. A reduced entry is
keyed on a digest of the source file's identity (size, mtime, and the SHA-256
of the whole file), the resolved grouping payload, the reduction settings and
the work-directory schema; an entry whose digest no longer matches its inputs
is recomputed, never trusted stale, and a reduced run or series written by an
older asymmetry is refused with a message to reduce or fit it again. The whole directory is safe to delete — every command
rebuilds whatever it needs from the original data files and, for ``fit``,
``fit-global`` and ``fit-series``, the recipe.

One directory, one data folder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Everything under the work directory is keyed on the **run number** alone, so
two data folders whose run numbers overlap would overwrite each other's
spectra, recipes and series inside one directory. The manifest therefore
records the folder the session was opened for, as an absolute resolved path,
and every command checks it before reading or writing anything
(:meth:`asymmetry.core.workflow.workdir.WorkDir.bind`). A directory that
already belongs to another folder is refused:

.. code-block:: console

   $ asymmetry survey /data/nickel
   asymmetry: /work/asymmetry-work belongs to /data/ptfe; for /data/nickel pass --workdir asymmetry-work-<name>

``survey`` and ``reduce`` write the manifest, so the first of them run against
a fresh directory claims it; a directory with no manifest yet is unclaimed.
The same folder named relatively, absolutely, or through a symlink is one
folder — both paths are resolved before they are compared.

The fit recipe
---------------

A recipe is the only contract between screening and fitting: small JSON that
``wizard`` writes, that a person or an agent may hand-edit, and that ``fit``,
``fit-global`` and ``fit-series`` consume.

.. code-block:: json

   {
     "schema": 1,
     "asymmetry_version": "0.19.0",
     "expression": "Exponential + Constant",
     "model": { "...CompositeModel.to_dict()..." },
     "parameters": [
       {"name": "A_1", "value": 19.7, "min": null, "max": null, "fixed": false},
       {"name": "Lambda", "value": 0.14, "min": 0.0, "max": null, "fixed": false},
       {"name": "A_bg", "value": 0.0, "min": null, "max": null, "fixed": false}
     ],
     "t_min": null,
     "t_max": 12.0,
     "rebin": 1,
     "source": {"wizard_run": 102, "template_key": "exp_constant"},
     "pinned": []
   }

Two rules govern what a recipe carries, settled once the wizard's own
per-run bounds turned out not to travel across a scan:

- **A recipe's bounds are the model's static defaults**, not the wizard's
  per-run search window. The wizard's bounds (a peak window, 0.5–2× a seeded
  amplitude, Nyquist and duration caps) describe the one run it screened;
  carrying a frequency window measured at 300 K into a fit at 380 K clamps
  the fit at a bound instead of letting it follow the physics. The *values*
  are still carried — a wizard fit's numbers are the best available starting
  point — only the bounds are reset to what typing the same expression into
  the desktop application would give.
- **Run-bound values are re-seeded per run in a series unless pinned.** A
  value that describes a run rather than the physics — an applied field, a
  spectral peak — is re-seeded from each run's own record as
  ``fit-series`` chains along the scan. A value named in ``pinned`` (written
  by ``--fix``, ``--global``, or a hand-edit to this list) is never moved,
  because it was chosen rather than measured. This is distinct from a
  parameter's own ``fixed`` flag, which also covers a parameter the model or
  the wizard holds fixed by default and which *is* still re-seeded per run.

Hand-editing a recipe
~~~~~~~~~~~~~~~~~~~~~

To swap one component, edit three things: ``expression``, the matching entry
in ``model``'s component list, and the parameter list (name and starting
value) — for example ``Exponential`` → ``Gaussian`` renames ``Lambda`` to
``sigma``. Write the result to ``recipes/<name>.json`` and pass
``--recipe <name>``; run ``asymmetry fit --run N --recipe <name>`` on one run
first to check it converges before spending a whole series on it.

``--global P,Q`` on ``fit-series`` **pins** those parameters at the recipe's
value for every run rather than fitting them jointly: ``fit_asymmetry_series``
is block-separable. Use ``fit-global --shared P,Q`` for a true simultaneous
fit of one run group.

Series fitting
---------------

``fit-series`` chains: each run's fit seeds from its nearest already-fitted
neighbour along the scan order, rather than every run starting cold from the
recipe's own values. ``--start RUN`` picks where the chain begins and lets it
grow outward in both directions, instead of always starting from the first
run in scan order — the recommended flow is to screen the run with the
clearest structure (``asymmetry wizard --run N``) and then start the series
there (``fit-series --start N``), so every fit warm-starts from a neighbour
nearer the run the recipe actually describes. Chaining from one end of a scan
with a recipe fitted at the other end can lose the runs in between entirely.

Every per-run result carries quality flags, read from ``fit_result_summary``:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Flag
     - Meaning
   * - ``failed``
     - The minimiser did not converge. The row is not a result.
   * - ``large_rel_err``
     - A free parameter's σ/value is large — the data barely constrained it.
   * - ``bound_pinned``
     - A free parameter sat on a bound.
   * - ``spurious_reseeded``
     - The fit landed on the spurious branch (amplitude collapse or frequency
       jump) near a transition, whether or not a reseed rescued it.

**Nothing is dropped.** A flagged run still appears in ``series/<name>.json``
and in the ``trend`` table, with its flags attached, so the full scan is
always visible; a flagged run's value is simply not a result to quote, which
is a judgement made when writing up the summary, not something the tool
does for you by omitting the row.

Plots
-----

Each PNG framed by ``--plot`` follows the same two rules:

- **The informative window.** μSR errors grow with time and are capped at
  100%, so an unframed plot's y-axis follows the noisy tail rather than the
  signal. ``reduce`` and ``fit`` plots frame the x-range on the same
  SNR-truncated analysis window the fit wizard's own fingerprint and peak
  detector use, and set the y-range from what actually ends up on screen — a
  cropped or bunched panel says so in a small note in the corner.
- **Display bunching.** A panel with more than a few hundred points is
  bunched to about 400 points for display only; the underlying fit or reduced
  record is untouched.

``trend`` plots frame the y-axis on the **clean** (unflagged) points, so one
wildly flagged run cannot squash every other point onto the axis edge; a
flagged point outside that range is still drawn, clamped to it, with a
distinct marker, and counted in the frame note, rather than silently moved or
dropped.

The skill
---------

The packaged ``asymmetry-analysis`` skill tells an agent to work through
seven steps: survey the folder; decide alpha (and never guess it); reduce the
scan; confirm the geometry of every non-zero-field scan from the survey's
measured precession and the reduced spectrum rather than the file's stamp;
screen one run with the
wizard; fit the scan as a chained series; read the trend; then write the
summary. Alongside the steps, it carries the decision rules an analyst
applies — which model family the physics calls for versus what AICc ranks
highest, what every fitted amplitude physically means, what to do when many
runs are flagged — and a cost table so an agent screens one or two runs
rather than the whole scan.

The summary template has fixed headings — Experiment, Calibration, Model,
Results, Excluded or flagged runs, What this suggests, Files — and one
governing rule: **every number in the summary must come from a command's
output.** A temperature, field, frequency, rate, exponent, run number, alpha,
χ² or count that no command printed does not go in; a textbook or literature
value may be discussed if it is clearly attributed as such, never presented
as a result of the session.

Anyone evaluating an agent against this skill — checking that it makes
sensible choices on real data — should start with the rubrics and runner
under ``tools/agent_eval/``: ``tools/agent_eval/rubrics/<dataset>.md`` records
the expected findings for a dataset, and ``tools/agent_eval/run_eval.py``
drives an agent against a copied dataset folder and reports the transcript
and work directory for scoring against the rubric.

The evaluation target is host-specific: use Sonnet through Claude Code and
``gpt-5.6-luna`` through Codex. Record both the host and exact model with each
result so the two evaluation tracks remain comparable without being conflated.

Limits and what is not there yet
----------------------------------

- No headless ``.asymp`` project export — the workflow produces reduced
  data, recipes, series and plots in the work directory, not a project file
  the desktop application can reopen.
- No batch-over-groups global fit. ``fit-global`` couples one group of runs;
  a temperature series of LF triplets currently requires one invocation per
  temperature and manual collation of the stored shared results.
- No maximum-entropy transform; ``fourier`` is an FFT.
- No MCP server. The work directory is the mechanism that gives an agent
  state between separate command invocations in its place.
- Warnings raised inside the fit wizard's worker processes are not collapsed
  by ``_collapse_repeated_warnings`` (which only replaces
  ``warnings.showwarning`` in the parent process), so a ``wizard`` run can
  still print several identical multi-line warning blocks on stderr on
  platforms that spawn worker processes. They are benign noise from seeding
  trial models during the candidate search.
