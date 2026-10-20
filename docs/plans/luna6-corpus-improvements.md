# GPT-6 Luna corpus improvement loop

Continue unattended on `codex/luna6-corpus-improvements` until a fresh full
24-case gate passes on Codex / `gpt-6-luna`. Keep rubrics fixed and retain every
first attempt. General skill guidance and reusable CLI behavior are the allowed
improvements; no run-number, expected-answer or corpus-name dispatch rules.

The baseline expansion is committed as `95edcb9`. Its detailed record is
[agent-cli-skill-luna6-expansion-2026-10-19.json](agent-cli-skill-luna6-expansion-2026-10-19.json):
7/16 scientific passes, 86/100 Musts, 11/16 exact-copy checks and 3/16 combined.
The previous eight-case gate is historical, not a contemporaneous full wave.
The earlier raw `/tmp` runs are no longer present on this host;
their committed score/evidence record remains the baseline. New waves keep
their own fresh reports and transcripts and do not reuse those missing runs.
The high-TF AFM literal pass has an incorrect superconducting interpretation;
that warning must be resolved as well as the literal rubric failures.

## Operating rules

- Commit each round of improvements before its evaluation wave; keep the CLI,
  skill and rubrics frozen while the wave runs. Score the completed reports
  before starting the next round, recording Must/Should evidence, number and
  boundary checks, model/host, commit, skill/rubric hashes, and first attempts.
- Run up to three eval cases concurrently with all BLAS thread counts pinned
  to one. Check available memory and memory PSI before launching; reduce load
  if pressure rises. Never overlap broad tests with live evaluations. Cap
  pytest at two workers. Inspect live processes before resuming work to avoid
  launching duplicates after an interruption.
- Keep corpus copies and raw results in `/tmp`, outside the repository. Obtain
  the original corpus only through `ASYMMETRY_CORPUS_ROOT`.
- The baseline full validation has four GUI sizing failures in unchanged GUI
  code; record these honestly. Do not expand this task into GUI redesign.
- Quiet heartbeat `gpt-6-luna-corpus-improvements` wakes this chat every thirty
  minutes initially, now every ten minutes to shorten idle gaps. Inspect this
  plan and current processes, continue real work, and
  notify only on meaningful scored rounds, completion or required input.
  Disable it once the objective is achieved.

## Round one

Implementation and evaluation complete:

- Skill checkpoints distinguish integral polarisation recovery, rate
  decoupling, resonances, magnetic broadening, superconducting Gaussian widths,
  calibration samples, temperature endpoints and KT control parameters.
- `trend` prints individual reliable cold/next/warm rate or width values and
  names the KT width/dynamics comparison. Exponential TF damping is directed
  to a Gaussian recipe rather than silently used in a gap law.
- `integral-scan --bound NAME=LO:HI` exposes validated, stored search intervals
  for broad or narrow lines; bound-pinned widths prompt a window/baseline check.
  RF contrast guidance preserves its LF experimental context.
- `survey` distinguishes the field-scan observables before suggesting a law.
- Final-report guidance explicitly forbids character changes after audit.

Validation: focused checks 240 passed in 7.68 s; fast tier 5,890 passed,
10 skipped, one xfailed in 104.67 s. Structural, Ruff and skill validation
passed. Sphinx built with 85 existing warnings. The standard gate, using two
workers and scratch XDG directories, finished with 9,903 passed, 25 skipped,
one xfailed and the same four GUI sizing failures as the baseline in 450.85 s:
`test_resting_columns_fit_a_narrow_inspector_dock`,
`test_narrow_columns_keep_their_character_widths`,
`test_both_columns_fit_without_scroll_at_default_size`, and
`test_default_width_fits_the_t0_line_on_a_fifteen_detector_run`.
No GUI source was changed; the standard gate is not green. Logs are
`/tmp/asymmetry-luna6-round1-{focused,fast,docs,validation}.log`.

Run the thirteen cases with
baseline science or exact-copy failures (including the AFM scope warning):

`plateau-redfield`, `sn-critical-field`, `euo-psi`, `maleic-mu-kinetics`,
`copper-diffusion`, `afm-high-tf-mdu`, `lifeas-psi`, `trsb-re6zr`, `basics`,
`corannulene-alc`, `benzene-rf`, `benzene-repolarisation`, `benzene-alc`.

Round one is committed as `ea920f9`. Its 13-case wave completed in
`/tmp/asymmetry-luna6-improvements/round1` with Codex / `gpt-6-luna`, three
concurrent cases and BLAS threads pinned to one. The first three cases entered
model execution successfully. `wave-metadata.json` records the commit, skill
and rubric hashes; console log `/tmp/asymmetry-luna6-round1-wave.log`.
The supervising exec session is `98755`; inspect host processes and per-case
`cost.json`/transcripts before relaunching after any interruption. No round-one
reports were scored at launch. Keep CLI/skill frozen until the wave completes.

Final score: thirteen reports scored in
[luna6-corpus-round1.json](luna6-corpus-round1.json), with 73/83 Musts,
6/13 scientific passes, 12/13 exact-copy passes and 5/13 combined passes.
Sn deletes one plot-file entry after audit, so its exact-copy check fails.
EuO repairs its baseline missing
paramagnetic rate: it used pre-t0 background correction and a separate
warm-ZF exponential series, reporting roughly constant relaxation without
inventing a critical divergence. AFM passes and resolves the earlier physics
scope warning: it explicitly uses magnetic TF scope and reports two lines,
rate changes and limits without a superconducting gap law.
Copper fails the TF envelope-change Must: early-window fits give Gaussian
at the EMU cold point and indistinguishable envelopes at the warm points,
while the report omits that comparison. Its cold/next ZF comparison and
exact-copy reporting now pass. Maleic acid regresses on the converged-slope
Must: it prints the fitted slope but then explicitly withdraws it as a
reaction-rate result. LiFeAs still fails the Gaussian-width Must: the new
sample-class caution led it to decline superconducting-model comparison and
retain exponential damping. Its final reply does pass exact-copy. It also
reads the personal skill path, violating the stated boundary. That path is a
symlink to the packaged source and has the frozen staged hash, so it introduces
no different skill content; the boundary violation is still recorded separately.
Benzene RF and repolarisation now pass: LF RF context and the actual integral
recovery observable are both repaired. Corannulene recovers the cold broad
resonance with a wider local polynomial window, but fails to interpret the
hot sharpening as molecular reorientation. TRSB now compares Delta and nu,
but a coupled-fit reversal leads it to reject a candidate TRSB headline.
Basics repairs foil saturation at 13–15, but still treats the Ag calibration
as ZF with a temperature axis and replaces the Ag TF frequency/phase exercise
with integrated asymmetry. Benzene ALC still fails by withholding both liquid
ring-proton fits and all solution fitted centres. Twenty width overrides were
used; one solution centre is stable across nearby windows with chi2_red 1.642,
but the two-width coverage rule makes the CLI call it range-limited. Distinguish
a qualified fitted centre from a demonstrated isolated resonance and baseline.
Scoring records include the
frozen hashes, final-summary hashes, actual command evidence and Should gaps.
Tests/logs use `/tmp/asymmetry-luna6-round1-*`. Once targeted cases pass,
run all 24 from fresh projects and score the entire wave; do not pool latest
passing attempts and call that a fresh gate.

Next-round candidates (do not edit CLI/skill while this wave runs): require
LF geometry and appropriate sample physics before recommending an integral
MuRepolarisation model; two-field TF/control pairs are not LF recovery scans.
Both copper and maleic acid spent effort on physically mismatched recovery
fits after round-one guidance. Make envelope comparison explicit even when
one shape wins throughout or warm points cannot distinguish shapes; resolve
window/baseline dependence rather than forcing the expected change. Distinguish
a converged empirical slope with uncertainty and model limitations from a
validated intrinsic kinetic constant, preserving the former as a fitted result.
Allow provisional physical-model comparisons with clearly conditional
interpretation when material context is incomplete; distinguish testing a
superconducting width model from asserting that broadening proves a vortex
lattice. Review why host skills remain discoverable despite the runner's
skip-host-skill-discovery option, before a fresh confirmation wave.
All thirteen first attempts have now been scored; round two may proceed.

Additional round-two candidates: choose the observable before satisfying scan
coverage (TF field scans require frequency/phase or amplitude, not the integral
of an oscillation). For known nonmagnetic calibration samples, test an
unrecorded weak TF rather than interpreting a header-ZF line with a powder
magnetic model. When a small KT onset reverses under nuisance sharing, compare
the static KT times exponential baseline and dynamic model with plausible
sharing choices before declaring it absent. ALC width comparisons should
include the conditional molecular-motion interpretation as well as fit values.

Isolation research: official Codex documentation describes per-skill disable
overrides (`skills.config`), whereas the current skip-host option did not
prevent the LiFeAs read. Sources checked:
https://learn.chatgpt.com/docs/config-file/config-reference and
https://learn.chatgpt.com/docs/build-skills. Test any runner isolation change
separately after this frozen wave; do not silently change its protocol.

## Round two

Implementation complete after all round-one results were scored:

- Survey guidance checks TF/LF geometry before suggesting recovery models;
  TF frequency/phase/amplitude observables are named explicitly.
- Envelope summaries retain indeterminate warm points. Conditional Gaussian
  and gap-model comparisons no longer require a firm material identification.
- Stable non-bound line centres may be reported as conditional fit estimates
  after window/background checks. Two-width coverage limits baseline certainty;
  it is no longer described as proof that no dip exists. Poor fits retain their
  mismatch caveats. Resolved-line gates and bound-width rejection are unchanged.
- Molecular ALC comparisons include a conditional reorientation/anisotropy
  interpretation; concentration slopes may be apparent fitted rate coefficients.
- Skill guidance addresses nonmagnetic calibration headers, unrelated sensor
  drift, static/dynamic KT and nuisance-sharing comparisons, and final copying.
- The runner explicitly names the project skill and disables personal copies
  through per-skill config on both turns. The scientific prompt, two-turn image
  review and rubrics are unchanged; record the runner hash with new waves.

Validation so far: 268 focused tests passed in 7.39 s, structural and lint
passed (one import-order issue repaired), skill quick validation passed.
The GPT-6 Luna isolation smoke at `/tmp/asymmetry-luna6-round2-isolation-smoke`
completed exit0 and read the project copy. It is not a scientific score.
Fast tier passed: 5,892 passed, 10 skipped, one xfailed in 102.92 s. Docs built
with 85 existing warnings. Standard validation first stopped at a leftover
import-spacing lint failure before tests; repaired and restarted. The completed
gate has 9,905 passed, 25 skipped, one xfailed and the same four known GUI sizing
failures as round one in 453.56 s, using two workers and scratch XDG paths.
No new failures appeared; the standard gate remains non-green for those GUI
failures. Log: `/tmp/asymmetry-luna6-round2-validation.log`.
Other logs: `/tmp/asymmetry-luna6-round2-{focused,fast,docs}.log`. No corpus
evals overlapped validation; round one had finished exit0 before it began.

Round two is committed as `a5047ae`. Its nine targeted cases are running with
Luna and three concurrent cases: `sn-critical-field`, `maleic-mu-kinetics`,
`copper-diffusion`, `lifeas-psi`, `trsb-re6zr`, `basics`, `corannulene-alc`,
`benzene-alc`, and `afm-high-tf-mdu` (physics-scope regression check).
Result root `/tmp/asymmetry-luna6-improvements/round2` contains first attempts
and `wave-metadata.json` with frozen commit and skill/runner/rubric hashes.
Exec session `19319`, console `/tmp/asymmetry-luna6-round2-wave.log`; inspect
processes and outputs before relaunching. LifeAs, TRSB and Basics started
successfully. First round-two report scored in
[luna6-corpus-round2.json](luna6-corpus-round2.json): Basics fails with 5/9
Musts, while exact-copy and boundary checks pass. Freeze CLI/skill until all
nine finish; score each final report against the unchanged rubrics.

Round-three candidate confirmed from Basics evidence: `_measured` censors
entire trend rows when only a nuisance damping parameter has `bound_pinned`.
(`large_rel_err` alone is already retained with its error bar.) At 13–15 foils
the amplitude errors are small, yet those
rows disappear from the step analysis and it incorrectly reports an
intermediate 9–11-foil plateau. Preserve parameter-specific reliability while
retaining whole-fit exclusions for actual failure/reseed/amplitude pathology.
Also print uncertainty-aware cold-pair comparisons: current endpoint wording
lets the agent call statistically compatible rates different. The Ag
frequency/phase observable is repaired, but MUSR calibration weak-TF context
and irrelevant sensor-temperature treatment remain wrong. The quartz line
amplitude series also failed; inspect frequency seeding before changing it.

Three round-two reports are now scored: 19/24 Musts, one scientific/combined
pass, two exact-copy passes and three boundary passes. LiFeAs repairs the
Gaussian-width requirement (8/8 Musts), with conditional gap-model fits and
separate campaigns. Its report quotes earlier real fits before a later t0
revision while describing final reduction settings; retain this provenance
limitation. Inspection confirms a further general plotting problem: nuisance
flags remove the warm sigma points from axis scaling, leaving them clipped
below the frame. The agent reads that clipping as poor model agreement.
Parameter-specific reliability must also govern plotting and its limits.
TRSB improves to 6/7 Musts: both KT forms recover the cold width increase and
the control rate lacks an onset, but the report stops at additional static
broadening rather than identifying candidate spontaneous fields/TRSB at Tc.
Its exact-copy check fails because the reply adds "sample temperatures" after
"logged" in one table row. Both cases read only the staged skill with frozen
hash `5a03b792caf69290af6284bf80974e8ac4d35b3f123c670a466b578adb936af3`;
round-two isolation is working so far. All nine first attempts remain frozen;
benzene ALC, copper and maleic acid are running, with three more queued.
Host process inspection confirms one wave and three model subprocesses;
about 22 GiB is available with no swap/PSI pressure. Do not relaunch live runs.

Benzene ALC is also scored: 5/7 Musts, exact-copy and boundary pass. All
seventeen solution centres are now reported after width-bound expansion,
repairing the previous withholding of every centre. However sample identities
(aqueous, hexane, 20mM aqueous) disappear, and the report applies the
cross-sample field-reference warning to valid within-sample temperature trends.
The upper ordinary-scan ring line is fitted at 29538.289717 G, non-bound and
consistent with the broad fit, but omitted in favour of discussing a failed
high-field differential fit. Keep ordinary and differential evidence distinct;
large chi2 requires a mismatch caveat and stability check rather than erasing
all quantitative results. One differential fit's quoted chi2 belongs to an
earlier variant (3.076 versus final LF-only 4.410); a general provenance check
must bind each reported estimate to its own calibration/model/run subset.
Four reports are scored so far: 24/31 Musts, one scientific/combined pass,
three exact-copy passes, four boundary passes. Corannulene has started in the
freed slot; copper and maleic acid remain live. No source changes yet.

Copper is scored and now passes all six Musts, exact-copy and boundary checks.
It uses the full retained TF window and explicitly reports Gaussian cold versus
exponential warm envelopes, repairing the round-one omission. Its EMU ZF nu
cold-end upturn remains clear. Literal pass does not imply complete analysis:
ARGUS TF was mechanically fitted with the ZF KT recipe then abandoned, and the
LF scans remain weakly modelled. General model selection must match geometry
for every scan, rather than reusing one era's ZF recipe indiscriminately.
Five of nine reports scored: 30/37 Musts, two scientific/combined passes,
four exact-copy passes, five boundary passes. Three live cases are maleic acid,
corannulene and Sn; AFM is still queued. Host has 22 GiB available, zero swap
and zero memory PSI. Source remains frozen at round-two implementation.

Maleic acid is scored: 4/5 Musts, boundary pass, exact-copy fail. The final
report again withdraws the converged concentration slope as a rate constant
rather than retaining an apparent fitted coefficient with units and caveats.
The new CLI advice was printed explicitly, so merely repeating it in another
NOTE is unlikely to solve this. Consider making the result's quantity and units
part of the structured/primary fit output and reconciling the final report's
conclusions with its own retained estimates. Geometry is now appropriate:
no physically mismatched integral recovery fit was attempted. The copy failure
removes "in 78294" from one table cell and a final period from the last plot
entry after audit. Nuisance-bound flags also clip warm Mu-rate points despite
only the diamagnetic damping being pinned, reinforcing the shared reliability
fix. Six reports scored: 34/42 Musts, two scientific/combined passes,
four exact-copy passes and six boundary passes. Corannulene, Sn and AFM are
live; about 21 GiB available, no swap or memory pressure. Keep source frozen.

Corannulene and Sn are scored, leaving only AFM live. Sn retains all five
Musts and now passes exact-copy. Corannulene regresses to 5/10 Musts: hot
local centres remain good, but cold local Linear-background fits pin their
widths and no broader-bound/curved-baseline investigation follows. The cold
centre, comparative width/field and molecular-reorientation interpretation are
therefore absent. Dense ALC scans also remain labelled at nominal temperature
without an explicit logged offset, and the non-polynomial whole-background
reason is missing. Bound-pinned local fits should trigger a general sequence
of bounds, window and baseline checks before concluding no stable centre.
Eight reports scored: 44/57 Musts, three scientific/combined passes, six
exact-copy passes and eight boundary passes. One host wave and one AFM model
process remain; about 20 GiB available with no swap or memory pressure.
No CLI/skill changes until AFM completes and is scored.

Round two is complete (supervising session 19319 exited 0; host inspection
shows no remaining eval/model processes). Final result: 49/62 Musts,
4/9 scientific passes, 7/9 exact-copy passes, 4/9 combined and 9/9 boundary.
Same nine cases in round one: 52/62 Musts, 2/9 scientific, 8/9 copy and 1/9
combined. Scientific case count improved but Must coverage regressed, mainly
corannulene and Basics; this is not an all-pass gate. AFM retains all five
Musts and the corrected magnetic scope without a superconducting-gap fit.
All first attempts, host/model and frozen hashes are in the completed score
record. The CLI/skill freeze is now lifted for round three.

## Round three

Implement general repairs before launching the next Luna wave:
parameter-specific pinned/large-error diagnostics for summaries and plots;
uncertainty-aware endpoint differences; primary fitted quantity/units for
concentration slopes; a report reconciliation step preserving qualified
estimates, sample identities, calibration/model versions and final audited
text. Investigate broader local bounds/curved baselines for broad ALC dips
without corpus-specific branches. Keep the rubrics and two-turn protocol fixed.
Run focused and fast checks with at most two workers on this now-quiet host,
then the required standard gate (known four GUI sizing failures remain).
Commit all round-three changes before starting its evaluation wave.

Round-three implementation is in the working tree (not committed yet).
The shared per-parameter quality reader retains precise measurements when a
nuisance term is pinned/uncertain; summaries, law diagnostics and plot framing
use it. Trend rows now retain bound names under workdir schema 5, deliberately
requiring old caches/series to be regenerated. Cold endpoint comparisons print
the difference, combined error and consistency verdict. Linear rate fits store
supplied-axis units and prominently report a determined apparent coefficient.
Survey prints the unrecorded weak-TF alternative for a zero-field header with
a line, and local broad-dip guidance compares curved baselines as well as bounds.
The skill's blanket "flagged runs are never results" instruction was a direct
contradiction and is replaced by per-parameter interpretation. Report guidance
now reconciles versions, sample identity, qualified estimates and final copying.

Focused checks: 339 passed in 13.74 s (before the final move of the coefficient
label into the converged-result branch; rerun that focused check before gate).
Structural passed. Lint initially requested formatting for three files; fixed,
full lint passed. Skill validation passed. Docs built with the same 85 warnings;
a docs prose update followed. Fast tier is running in exec session 58290 with
exactly two workers, log /tmp/asymmetry-luna6-round3-fast.log. A prior fast
invocation was rejected by argparse because -n needed a -- separator; no tests
ran in that rejected invocation. No eval wave is live. Finish fast, then standard
validation with two workers/scratch XDG paths, record the known GUI failures
honestly, commit and launch a fresh targeted Luna wave. Inspect actual state
before resuming; do not duplicate tests or evaluations.

Fast tier completed: 5,897 passed, 10 skipped, one xfailed in 104.99 s,
with two workers. The moved coefficient check passed separately; docs rebuilt
with 85 warnings and lint is clean. Standard validation is now running in
exec session 41829, log /tmp/asymmetry-luna6-round3-validation.log, two workers,
BLAS pinned to one and scratch /tmp/asymmetry-luna6-round3-xdg directories.
Do not launch another test pool or any eval until it finishes. Next wave is
prepared but NOT launched: /tmp/asymmetry-luna6-round3-wave.sh, six cases
(basics, TRSB, corannulene, maleic acid, benzene ALC and LiFeAs plotting
regression). It requires committed source and a fresh round3 output directory.
Record gate results and commit the whole improvement round before running it.

Round-three validation completed: 339 focused passes in 13.74 s, the final
coefficient-result branch check passed separately (1.34 s), and fast tier
5,897 passed / 10 skipped / one xfailed in 104.99 s. Structural, full Ruff
and skill validation passed; docs rebuilt with the same 85 warnings.
Standard gate finished with 9,910 passed, 25 skipped, one xfailed and exactly
the four known GUI sizing failures in 453.53 s. No new failures appeared;
validation is not green. No GUI source was changed. Logs are
/tmp/asymmetry-luna6-round3-{focused,coefficient,fast,docs,validation}.log.
Scratch /tmp currently has about 12 GiB free; memory has no PSI/swap pressure.
Commit the source now, then start the prepared six-case wave with three
concurrent Luna cases and one BLAS thread. Source/skill must remain frozen
until that wave finishes. Keep all first attempts and score each final reply.

Round three is committed as `d70581e` and its six-case wave is now LIVE:
`/tmp/asymmetry-luna6-improvements/round3`, supervising exec session `90769`,
console `/tmp/asymmetry-luna6-round3-wave.log`. Basics, TRSB and corannulene
started successfully and read the staged project skill. There is exactly one
wave process and three model processes; no tests are running. Prelaunch memory
was 21 GiB available with no swap/PSI pressure. The frozen skill hash is
`584f7f167f99f5acf97434e2bf7c224226f3bd7adf3a1faf5171b549196de134`.
Runner, rubrics, two-turn review protocol and host-skill isolation are unchanged;
`wave-metadata.json` records all hashes and the committed source. All six are
fresh first attempts, not retries of earlier output directories. Keep CLI and
skill frozen until all finish; score completed final replies while others run.
Next: score in docs/plans/luna6-corpus-round3.json, investigate failures, make
and commit the next general improvement round before another wave. A fresh
all-24 gate still follows only after targeted failures pass; do not pool passes.

First round-three report scored: corannulene improves from 5/10 to 8/10 Musts,
with exact-copy and boundary pass. Local Cubic backgrounds recover the cold
15401.6 G centre/1137.8 G width, and the report now gives logged ALC temperatures
and conditional thermal-motion averaging of anisotropic hyperfine interactions.
Remaining gaps: no explicit absence of a narrow cold 7 kG line (it instead
emphasizes an unconstrained broad trough), and no explicit explanation that one
polynomial cannot describe the whole-scan background. General next-round work
should distinguish broad baseline curvature/undetermined width from a measured
narrow resonance, and make local-versus-whole-background limitations visible.
Hot fits quote real but unscaled errors despite printed scaled alternatives;
recorded as a limitation. Source remains frozen; Basics, TRSB and maleic acid
are live, with benzene ALC and LiFeAs pending. One wave/three model processes,
20 GiB available, no swap or PSI pressure. Score completed final replies only.
