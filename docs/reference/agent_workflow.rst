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

**In scope**: preliminary, time-domain forward–backward asymmetry analysis of
zero-field (ZF), transverse-field (TF) and longitudinal-field (LF) runs — a
temperature scan or a field scan at one geometry, from ISIS NeXus (``.nxs``)
or PSI (``.bin``, ``.mdu``) files, one forward group against one backward
group. "Preliminary" is the operative word: the workflow produces a
defensible first pass — the right model family, a trend, and flagged runs —
not a publication analysis.

**Out of scope.** The workflow does not improvise around any of these; the
skill tells an agent to say so and stop rather than force a fit:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Case
     - Why it is out of scope
   * - ALC / avoided-level-crossing resonance
     - The analysis is integral asymmetry *versus field*, fit as a resonance
       line shape; there is no command for that.
   * - Count-domain fitting
     - ``reduce`` only produces asymmetry, not per-detector counts with an
       N₀ and relaxation term.
   * - Multi-group / orientation-resolved analysis
     - The workflow reduces exactly one forward/backward detector pair, not
       several groups fit together.
   * - Fourier or maximum-entropy spectra
     - There is no transform command; everything here is a time-domain fit.
   * - Negative-muon (μ⁻) elemental analysis
     - Gamma spectra and elemental lines are not asymmetry data.
   * - Rotating-reference-frame or RF-resonance runs
     - Data modulated at a reference frequency needs a different demodulation
       step than forward–backward asymmetry.
   * - Muonium chemistry / reaction rates
     - Rates versus concentration across samples are not a spin-relaxation
       trend.
   * - Simultaneous multi-field fits
     - ``fit-series --global`` pins a parameter at one value across the
       series; it does not fit several runs jointly, so it cannot represent a
       decoupling triplet that needs a shared parameter fit across fields.
   * - A fragment of a published multi-field campaign
     - Disjoint run-number blocks with large gaps and no self-contained scan
       cannot be reconstructed from what is on disk.

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
- ``--plot`` (on ``reduce``, ``wizard``, ``fit``, ``fit-series`` and
  ``trend``) writes one or more headless PNGs into the work directory's
  ``plots/`` — see `Plots`_.
- ``--workdir`` overrides the work directory, which otherwise defaults to
  ``<folder>/.asymmetry``. Passing a different one deliberately starts a
  separate session against the same data; passing it by mistake silently
  loses whatever an earlier command in the default location already wrote.
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
(geometry, field) ordered by temperature and by (geometry, temperature)
ordered by field, so the structure of a multi-scan folder is visible without
reading every file:

.. code-block:: console

   $ asymmetry survey runs
   8 run(s) in runs — SIM

   run  T/K    B/G     geom  prec    orient        hist  points  dt   title
   ---  -----  ------  ----  ------  ------------  ----  ------  ---  -------------------------------------
   101  5.00   100.00  TF*   larmor  Longitudinal  8     500     no   Calibrant T=5.0 K B=100.0 G
   102  10.00  0.00    ZF    -       Longitudinal  8     500     no   Sample T=10.0 K B=0.0 G
   ...
   107  60.00  0.00    ZF    -       Longitudinal  8     500     no   Sample T=60.0 K B=0.0 G
   108  2.00   110.00  TF    none    Longitudinal  8     500     no   Sample T=2.0 K B=110.0 G (decoupling)

   Alpha-calibration candidates:
     run 101 (best) [measured]: precession at the Larmor frequency of the recorded 100 G (SNR 93)

   Scans:
     temperature scan, ZF, B = 0 G: 6 runs, 10 to 60 K (run 102 -> 107)

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
   * - ``other``
     - A line at SNR ≥ 10 somewhere else: the muon is precessing in an internal
       field that beats the applied one, as in an ordered magnet below its
       transition. This says nothing about the applied field's direction, so
       ``geom`` falls back to the file.
   * - ``none``
     - No line above SNR 10. The applied field is not precessing the muon, so it
       is not transverse — in a field scan, longitudinal decoupling. A row
       reading ``geom TF`` with ``prec none`` is a file stamp the data refutes.
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

Each run's ``survey.json`` row carries ``geometry``, ``geometry_source``
(``field``, ``measured``, ``file`` or ``none``), ``precession``,
``precession_frequency_mhz``, ``precession_snr``, ``precession_larmor_mhz`` and
``precession_note``.

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
                    [--deadtime {off,from_file}] [--rebin REBIN] [--tmin TMIN]
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
                    [--plot] [--json] [--workdir WORKDIR]
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
known. Writes ``wizard/<run>.json`` (the full screening payload:
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

``fit-series``
~~~~~~~~~~~~~~

Fit a recipe across a scan of reduced runs, chained along the scan order —
the command that actually produces a trend.

.. code-block:: text

   asymmetry fit-series [-h] --runs RUNS --recipe RECIPE [--fix NAME=VALUE]
                        --order {temperature,field,run} [--global P,Q]
                        [--start RUN] [--name NAME] [--plot] [--json]
                        [--workdir WORKDIR]
                        folder

``--order`` names the scan quantity the series is ordered and trended along.
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

Print (or export) the parameter trend of a stored series.

.. code-block:: text

   asymmetry trend [-h] --series SERIES [--csv CSV] [--plot] [--json]
                   [--workdir WORKDIR]
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
file directly without surveying a whole folder.

.. code-block:: text

   asymmetry info [-h] file

The work directory
-------------------

Every command above reads and writes ``<folder>/.asymmetry/``:

.. code-block:: text

   .asymmetry/
     manifest.json          # asymmetry version, folder, settings, run list
     survey.json            # output of `survey`
     reduced/<run>.npz      # time, asymmetry, error
     reduced/<run>.json     # run metadata + reduction settings + cache digest
     wizard/<run>.json      # screening payload: recommendation, narrative, recipe
     recipes/<name>.json    # a fit recipe (model + parameters + window)
     series/<name>.json     # per-run results, trend table, quality flags
     plots/*.png            # headless PNGs written by --plot

so a later command picks up a reduced spectrum, a recipe, or a series without
reloading or recomputing it, and an agent (or a shell script run in several
steps) has state between invocations without a long-lived process — the same
role a future MCP server would hold in memory instead. A reduced entry is
keyed on a digest of the source file's identity (size, mtime, and a hash of
its leading bytes), the resolved grouping payload, and the reduction
settings; an entry whose digest no longer matches its inputs is recomputed,
never trusted stale. The whole directory is safe to delete — every command
rebuilds whatever it needs from the original data files and, for ``fit`` and
``fit-series``, the recipe.

The fit recipe
---------------

A recipe is the only contract between screening and fitting: small JSON that
``wizard`` writes, that a person or an agent may hand-edit, and that ``fit``
and ``fit-series`` consume.

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
is block-separable and cannot share a fitted parameter across runs, so this
is not a simultaneous fit (see `Limits and what is not there yet`_).

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

Limits and what is not there yet
----------------------------------

- No headless ``.asymp`` project export — the workflow produces reduced
  data, recipes, series and plots in the work directory, not a project file
  the desktop application can reopen.
- No simultaneous multi-field fits. ``fit-series --global`` pins a shared
  parameter's value rather than fitting it jointly across runs (see
  `Hand-editing a recipe`_); a decoupling measurement that genuinely needs a
  joint fit across fields is out of scope (see the scope table above).
- No MCP server. The work directory is the mechanism that gives an agent
  state between separate command invocations in its place.
- Warnings raised inside the fit wizard's worker processes are not collapsed
  by ``_collapse_repeated_warnings`` (which only replaces
  ``warnings.showwarning`` in the parent process), so a ``wizard`` run can
  still print several identical multi-line warning blocks on stderr on
  platforms that spawn worker processes. They are benign noise from seeding
  trial models during the candidate search.
