# GPT-6 Luna corpus improvement loop

The user resumed the improvement/evaluation loop on 20 October, with a hard
stop at **07:00 Europe/London on 21 October 2026 (06:00 UTC)**. After each
completed wave score actual replies, make general CLI/skill and eval-harness
repairs, validate and commit before another wave. Scientific rubrics remain
fixed. Trial twelve concurrent cases with one wizard worker each and numerical
library threads one; resource measurements must determine whether to keep it.
Do not launch a full gate after 04:30 UTC. The supervised evaluation process
tree must stop by the deadline; retain interruption records for unfinished
attempts. Pause the follow-up by the deadline or on a genuinely passing fresh
full24 gate. No run-number, expected-answer or corpus-name dispatch rules.

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
- For confirmation4, trial twelve eval cases concurrently, with wizard fit
  workers limited to one per case and all BLAS thread counts pinned to one.
  Confirmation3 used six cases/two wizard workers; earlier waves used three cases. Check available memory and CPU/memory PSI
  before launching and sample the process tree while running; reduce load
  if pressure rises. Never overlap broad tests with live evaluations. Cap
  pytest at two workers. Inspect live processes before resuming work to avoid
  launching duplicates after an interruption.
- Keep corpus copies and raw results in the persistent external cache
  `/home/ben/.cache/asymmetry-luna6-overnight-20261020`, outside the repository.
  `/tmp` was used historically and lost evidence across a restart. Obtain
  the original corpus only through `ASYMMETRY_CORPUS_ROOT`.
- The baseline full validation has four GUI sizing failures in unchanged GUI
  code; record these honestly. Do not expand this task into GUI redesign.
- Quiet heartbeat `gpt-6-luna-corpus-improvements` wakes this chat every thirty
  minutes initially, now every five minutes to shorten idle gaps. Inspect this
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

Basics and TRSB are scored. TRSB now passes all seven Musts, exact-copy and
boundary checks: it explicitly identifies candidate TRSB from the small ZF
width onset, stable control rate and LF decoupling, retaining model/sharing
sensitivity rather than claiming proof. Basics improves from 5/9 to 8/9 Musts:
MUSR unrecorded weak-TF context/sensor logging, usable Mu response amplitudes,
alpha blocks and Ag t0 test are all repaired. The foil CLI now correctly prints
saturation over 13–15 beginning at 13, but the final report omits the numeric
plateau coordinate despite tabulating those rows; this remains a report
completeness failure. Its final reply duplicates the sentence excluding run
18894, so exact-copy fails. All three completed reports obey the boundary.
Partial totals: 23/26 Musts, 1/3 scientific/combined passes, 2/3 copy and
3/3 boundary. Maleic acid, benzene ALC and LiFeAs are live. About 19 GiB
available, zero swap and PSI; do not change frozen CLI/skill until they finish.

Maleic acid and benzene ALC are scored and both now pass every scientific
Must (5/5 and 7/7). Maleic retains the apparent concentration coefficient with
its proper relative units and limitations, rather than withdrawing it; it
also computes slopes from further composition groups without inventing an
Arrhenius activation energy. Benzene retains both ordinary ring-proton fitted
centres separately from the failed differential control and preserves the
three solution identities/within-sample field trends. Maleic exact-copy passes;
benzene fails only by moving one closing backtick in a scan label after audit.
All five completed reports satisfy the tool boundary. Partial totals: 35/38
Musts, 3/5 scientific passes, 3/5 exact-copy, 2/5 combined and 5/5 boundary.
LiFeAs is the only live case; no source changes until it finishes. Memory
20 GiB available, no swap/PSI pressure. Preserve scientific pass versus
exact-copy failure as separate dimensions; never substitute the audited file
for the actual final reply to force a copy pass.

Round three is complete; session 90769 exited 0 and host inspection shows no
remaining eval/model/test processes. LiFeAs retains all eight Musts plus
exact-copy/boundary, with a measured Gaussian-width curve whose warm level
begins at logged 16.8973 K; it still declines an SC gap comparison (Should).
Final six-case score: 43/46 Musts, 4/6 scientific passes, 4/6 exact-copy,
3/6 combined and 6/6 boundary. Same six round-two cases: 33/46 Musts,
1/6 scientific, 4/6 exact-copy and 1/6 combined. TRSB, maleic and benzene ALC
science gaps are repaired. Remaining scientific gaps are report completeness
in Basics (saturation coordinate) and corannulene (narrow-line comparison and
whole-background explanation); Basics and benzene also fail literal copying.

## Round four

The CLI's measured outputs now contain the missing main estimates. Make a
focused, general skill/documentation improvement: preserve numeric coordinates
for reported plateaus/onsets in prose; distinguish no resolved narrow line
from unconstrained broad curvature; explain local-baseline limitations; keep
long reports concise enough to copy without duplicating or reformatting text.
Reconcile remaining blanket flag wording in the summary template with the new
per-parameter contract. This is documentation-only, so validate skill and
Sphinx docs rather than rerunning the already completed standard/core gates
(AGENTS.md explicitly exempts doc/comment-only follow-up edits). Commit before
a fresh targeted wave covering Basics, corannulene and benzene exact-copy;
then fresh all24 only if those pass. Keep rubrics/protocol unchanged.

Round-four changes are skill/owning-documentation only. The workflow contract
and summary template now retain numeric plateau/onset coordinates explicitly,
compare resolved narrow components separately from unconstrained broad troughs,
and explain whole-scan versus local baseline suitability. Long exploratory
tables stay in the CLI products; decisive values and all provenance/windows
remain in the report. Final copying forbids Markdown cleanup or merging drafts
and repeated sentences. Remaining summary-template blanket flag wording is
reconciled with per-parameter reliability. No CLI, runner or rubric changed.
Skill validation passed; Sphinx built with the same 85 warnings. Round-three
core/standard validation remains the applicable gate under AGENTS.md's
explicit documentation-only follow-up exception. Commit before fresh three-case
wave (Basics, corannulene, benzene ALC); score all before the next change/gate.

Round four is committed as `2741b45`; its three-case Luna wave is LIVE under
`/tmp/asymmetry-luna6-improvements/round4`, supervising exec session `64476`,
console `/tmp/asymmetry-luna6-round4-wave.log`. All three cases (Basics,
corannulene, benzene ALC) started and read the staged project skill. Source and
skill are frozen until all finish; one wave/three model processes, no tests.
Prelaunch memory was 20 GiB available, no swap/PSI; /tmp had 10 GiB free.
Frozen skill hash `95486cb2e648256bfeb8f410c4146e7fa4a3c981d47dd35e834639c12c87ecf8`;
runner/protocol/rubrics unchanged and hashes recorded in wave-metadata.json.
Score actual final replies in docs/plans/luna6-corpus-round4.json. Helper
/tmp/score-luna6-round4.py records manual decisions; it is not a judge.
If these three pass all scientific, exact-copy and boundary checks, launch a
fresh full 24-case confirmation wave without pooling earlier results. Include
the AFM magnetic-scope check. Otherwise make general improvements, commit and
evaluate again. No full gate has run yet and the objective is not complete.

First two round-four reports scored: Basics now passes 9/9 Musts, exact-copy
and boundary, explicitly retaining the 13–15 saturation range/from-13 start.
Benzene exact-copy also passes, but science regresses to 6/7 Musts because the
second solid line is missed. Its local window 21450–21750 G puts the real broad
feature near an edge; the fit locks a spurious narrow B0 at 21743.269249 G
(at bound). It correctly withholds that failed identification but stops there.
Current CLI prominently handles width bounds, not bound-centre/window-edge
recovery when chi2 is modest. A general next step should widen/shift nearby
windows to include both flanks and compare local baselines before concluding
there is no line. No corpus-specific window/centre dispatch is allowed.
Partial scores: 15/16 Musts, 1/2 scientific and combined, 2/2 copy/boundary.
Corannulene remains live; keep source frozen until its final reply is scored.
About 19 GiB available, no swap or PSI pressure. Full24 is still not launched.

Round four is complete; supervising session 64476 exited 0 and actual host
inspection shows no remaining eval/model/test processes. Final score: 24/26
Musts, 1/3 scientific passes, 3/3 exact-copy, 1/3 combined and 3/3 boundary.
Corannulene improves to 9/10 Musts, explicitly resolving the cold narrow-line
comparison. It still omits why one polynomial cannot follow the whole-scan
background. Its local model definition incorrectly satisfies the audit's broad
explanation regex; repair that general false positive. Both failures are now
fully recorded in luna6-corpus-round4.json. Source freeze is lifted. Next:
general bound-centre window-recovery advice and a genuine background-limitation
audit check, focused tests/fast/docs, commit, then two targeted fresh cases.
No full24 gate yet. Host has 19 GiB available, zero swap use/PSI and 9 GiB free
in /tmp. Standard validation retains the four documented GUI sizing failures.

## Round five

General CLI and skill repairs: a centre at a search bound now prints an explicit
window-recovery sequence even when chi-squared is modest (inspect full scan,
shift/widen around the feature, include both flanks, compare baselines/seeds).
The existing background-explanation audit now requires a limitation rather
than accepting a sentence that merely defines a local polynomial. Regression
cases cover that false positive and legitimate alternative explanations.
Skill and owning Sphinx page explain both behaviors; no rubric or runner edits.
Focused checks: 296 passed in 7.05 s. An initial edit placed the centre note in
the wrong helper; focused failures caught it, the note was moved into
_poor_fit_note and the complete focused pair passed. Full lint, structural and
skill validation pass; docs built with 85 existing warnings. Fast tier is in
progress with two workers; do not launch an eval until it finishes. No repeat
of the task's completed standard gate for this small CLI-output/audit repair;
its four existing GUI sizing failures remain unresolved and explicitly recorded.

Round-five fast tier completed: 5,902 passed, 10 skipped and one xfailed in
104.21 s with two workers. Focused, fast, lint, structural, skill and docs checks
are complete; no new failures. Standard validation remains the round-three
record with four known GUI sizing failures. Commit this improvement round,
then launch fresh corannulene-alc and benzene-alc cases using gpt-6-luna and
fixed rubrics/two-turn review. Prelaunch memory 18 GiB available, zero swap
use/PSI; /tmp has 8.9 GiB free. No evaluations overlapped the tests.

Round five is committed as `31bff2a`; the two-case Luna wave is LIVE under
`/tmp/asymmetry-luna6-improvements/round5`, exec session `49577`, console
`/tmp/asymmetry-luna6-round5-wave.log`. Host inspection confirms one wave, two
model processes and no tests. Both case runners started successfully. Freeze
CLI/skill until both finish and are scored. Metadata records frozen skill
`0f9857b9c93ef7ed0b352cef0cbfe8d768f98e623883006081b9f2b038061d92`,
runner/rubric hashes and first-attempt source commit. Record scores in
`docs/plans/luna6-corpus-round5.json`; `/tmp/score-luna6-round5.py` is a manual
record helper, not a judge. If both pass science, exact-copy and boundary,
launch a fresh `--set all` 24-case confirmation wave with the same committed
source; score every case, including AFM physics scope. Otherwise improve again
and commit before the next targeted wave. No full confirmation has run yet.

First round-five report scored: corannulene regresses to 4/10 Musts while
exact-copy and boundary pass. It applies MuRepolarisation to all LF scans and
stops after the failed whole-scan fits, never investigating local ALC lines.
The new centre-bound/background guidance is therefore never reached. The
report's numbers verify against actual non-audit CLI outputs, but the main
resonance science is absent. Next general candidate: expose non-monotonic
local-dip investigation independently of the initially requested recovery model;
a failed global model must not be interpreted as absence of another observable.
Benzene remains live in its review turn; keep CLI/skill frozen. One wave/one
model, no tests, 18 GiB available, zero swap use/PSI, 8 GiB free /tmp. Do not
launch a confirmation gate from these results.

Round five completed; host inspection shows no remaining eval/model/test
processes. Benzene passes all 7/7 Musts and boundary, including the
two converged broad solid lines at 19478.2/21474.6 G with stated mismatch.
Final two-case totals: 11/17 Musts, 1/2 scientific, 1/2 exact-copy,
0/2 combined and 2/2 boundary. Benzene deletes `plots/` from the solution plot
directory in its final reply, so literal copying fails. Corannulene's 4/10 regression prevents a full24 launch. Freeze
lifted only after both first attempts scored. Next round should make local dip
investigation available even when the initial whole-scan model is recovery;
the current core only finds next-dip windows for existing LCR models. This is
an observable/model-selection gap, not a reason for corpus-specific rules.

## Round six

Implementing a model-independent shape diagnostic in the pure core. Significant
interior minima (five combined errors of prominence) produce exploratory local
windows. These are measured minima, not fitted resonance fields or widths;
quarter-prominence widths keep windows near trough bottoms. Monotonic recovery
and sub-error excursions produce no candidates. The CLI prints LOCAL DIP
CANDIDATES when no resonance model was selected, even if a recovery fit failed,
and asks for nearby-window/Linear/Cubic baseline investigation before concluding
no line is resolved. Existing resonance and RF-hyperfine models keep their own
line diagnostics. No rubric/runner/protocol changes or corpus-name dispatch.

Focused checks: 320 passed in 8.38 s; structural/lint/skill checks passed.
Fast tier (two workers) and docs are running; no evaluation overlaps them.
An exploratory development probe on stored scan points outside the repository
confirmed that the new diagnostic exposes local structure in both extended
scans while the short monotonic recovery produces none. This is development
evidence, not a new eval score and not data supplied to a Luna agent. The task's
standard gate remains 9910 passes with four known GUI sizing failures; do not
call it green or repeat it as an inner-loop test. Commit round six before fresh
corannulene plus benzene exact-copy regression cases, then full24 only if both
pass all required dimensions. Raw evidence remains outside the repository.

Round-six validation completed: 320 focused passes (8.38 s), 5905 fast passes,
10 skipped/one xfailed (104.13 s), lint/structural/skill pass and docs built with
85 existing warnings. CLI smoke on stored high-field data without a requested
resonance model prints LOCAL DIP CANDIDATES and local-window commands; output
/tmp/asymmetry-luna6-round6-shape-smoke.json. This is development validation,
not an eval score. No broad tests overlapped evals. Standard gate remains the
recorded four existing GUI failures. Host is quiet, 18 GiB available, zero swap
use/PSI and 8 GiB scratch free. Commit then run the two prepared targeted cases.

Round six committed as `a24043a`, wave LIVE in
`/tmp/asymmetry-luna6-improvements/round6`, exec session `63872`, console
`/tmp/asymmetry-luna6-round6-wave.log`. Two cases: corannulene-alc and
benzene-alc (science/copy regression). Both started successfully; host shows
one wave/two model processes and no tests. Freeze CLI/skill until both finish
and are scored. wave-metadata.json records commit, skill/runner/rubric hashes,
model/host and unchanged two-turn protocol. Scores go in
`docs/plans/luna6-corpus-round6.json`; `/tmp/score-luna6-round6.py` is a manual
record helper. Inspect live processes and costs before any relaunch. If both
pass science/copy/boundary, launch fresh full24 with `--set all` and the same
source; include AFM magnetic scope and never pool historical passes. Otherwise
make general improvements and commit before the next wave. Do not repeat broad
tests while the eval wave runs. Standard validation still has four known GUI
failures; the objective is not complete and no full24 gate has run yet.

First round-six report scored: benzene 6/7 Musts, exact-copy and boundary pass.
Both solid lines and ordinary liquid lines remain fitted, but all seventeen
solution widths remain at default bounds and no representative recovery is
attempted. The report explicitly withholds every solution resonance pending
local-window/baseline work; the solution Must therefore fails. It also extends
cross-campaign field-reference caution to within-sample temperature trends and
misreads one same-feature cross-fit comparison as comparing the two distinct
ordinary lines. These are recorded science limitations, not number inventions;
all numbers verify against actual non-audit CLI output. Corannulene is still
live in review; do not change CLI/skill until it finishes and is scored. Host
has 17 GiB available, zero swap use/PSI, 7.2 GiB scratch free. No full24 launch.

Round six complete; no eval/model/test processes remain. Final 14/17 Musts,
0/2 scientific/combined, 1/2 exact-copy and 2/2 boundary. Corannulene recovers
8/10 Musts: the new shape diagnostic triggers local ALC fits and the whole-
background explanation. But it reuses a narrow hot window for the broad cold
feature, returning centres below the required range and unresolved widths;
thermal anisotropy-averaging headline remains missing. Copy fails by changing
14,828.1 G to 14.83 kG after audit. The actual reply is scored. Number checks
now explicitly exclude draft displays as well as audit output. Frozen wave
fully scored before edits. Next: distinguish fit-search width bounds from
resolved-line flank coverage; offer concrete wider-window/Cubic investigation
for truncated lines and independent windows for different conditions. The
span/4 automatic width ceiling is a geometric resolution criterion currently
used as a hard fitting prior, truncating broad but potentially fitted widths.
Test any change on synthetic broad lines and keep resolved-line criteria fixed.
No full24 launch until targeted scientific/copy/boundary failures pass.

## Round seven

General repair in progress: automatic LCR width searches extend to half the
fitted span, while the unchanged resolved-line gate still requires two widths
on both flanks, significant amplitude and a non-bound converged fit. Search
bounds no longer impose that geometric resolution criterion as a hard prior.
Explicit positive bounds remain supported. The CLI's insufficient-coverage
note prints a tripled-window Cubic check and explains that different conditions
need independent windows when one feature broadens. Skill and owning docs
retain conditional estimates without calling partial coverage a resolved line.

Focused checks: 321 passed in 8.39 s; lint/structural/skill pass. Fast tier
is running with two workers; docs built with 85 existing warnings. Development
probes on stored points (outside repo, not supplied to eval agents) recover a
non-bound broad cold fit in wider windows and a non-bound solution width with
the expanded default. Probes are diagnostic development evidence, not scores;
raw script /tmp/probe-luna6-round7.py. Synthetic tests verify broad-width
recovery and continued rejection as an isolated resolved line when flanks are
missing. No eval is live and no broad tests overlap one. Standard remains the
recorded four pre-existing GUI failures; do not call it green. Finish fast,
commit, then run fresh two-case targeted wave before any full24 confirmation.

Round-seven validation complete: 321 focused passes (8.39 s), 5906 fast passes,
10 skipped/one xfailed (104.49 s), lint/structural/skill pass and docs 85 existing
warnings. Host inspection confirms no eval/model/test processes before launch;
17 GiB available, zero swap use/PSI and 7.2 GiB free /tmp. No broad validation
repeat; standard remains the recorded four known GUI failures. Commit these
changes, then run both targeted cases with fixed rubrics/two-turn protocol.

Round seven committed as `dcac043`; two-case gpt-6-luna wave LIVE under
`/tmp/asymmetry-luna6-improvements/round7`, exec session `67269`, console
`/tmp/asymmetry-luna6-round7-wave.log`. Both cases started; host shows one
wave/two model processes, no tests. Freeze CLI/skill until both finish and are
scored. Metadata records frozen source/skill/runner/rubric hashes and unchanged
protocol. Scores go in `docs/plans/luna6-corpus-round7.json`; manual record
helper `/tmp/score-luna6-round7.py` is not a judge. Check actual processes/costs
before any launch; never duplicate live runs. Score actual final replies with
Must/Should evidence, number-rule excluding audit/draft displays, boundary,
exact-copy and physics limitations. If both targeted cases pass all required
dimensions, launch a fresh `--set all` full24 wave from this source, including
AFM magnetic scope; do not pool historical passes. Otherwise improve generally,
test appropriately and commit before the next wave. Standard validation still
has four known GUI sizing failures. No full24 gate has run; objective incomplete.

First round-seven report scored: benzene passes 7/7 literal Musts, exact-copy
and boundary. Expanded width searches recover non-bound solution dip estimates
(e.g. aqueous 350 K width 62.77 ± 2.30 G). Both ordinary ring fitted centres
are reported with uncertainties/fit quality; high mismatch and qualitative
overcaution do not add an unstated validated-line requirement to that Must.
This differs from round-six solution fits whose bound-pinned widths were all
withdrawn as resonance results. Record limitations: ring stability checks not
demonstrated, within-sample reference caution overextended, separate solid
windows not a joint width assignment, and differential mean-asymmetry control
remains weak/unresolved. Literal pass is not a claim of complete physics.
Corannulene remains live; keep CLI/skill frozen. Host has 16 GiB available,
zero swap/PSI and 6.3 GiB scratch free. Do not launch full24 until it is scored.

Round seven COMPLETE: 17/17 Musts, 2/2 scientific, exact-copy, combined and
boundary. Corannulene recovers cold 15306.1 ± 27.1 G / width 870.45 ± 75.4 G
with a wider Cubic baseline, explicit cold/hot width/field comparison and
conditional molecular-reorientation/anisotropy interpretation. Whole-background
limitations and absent radical model are explicit. Remaining Should/quality
caveats are recorded, not silently promoted to new Musts. Both actual replies
match their audited files. No code/skill changes before fresh confirmation.
Prepared `/tmp/asymmetry-luna6-confirmation1-wave.sh` uses --set all, 24 fresh
projects, gpt-6-luna, max3 cases, BLAS1 and same protocol/source. Total staged
corpus data is 1.76 GiB; 6.3 GiB scratch free before launch is sufficient with
headroom for logs/products. Inspect processes again, commit scores, then launch.
Score every full-gate first attempt, including AFM physics scope; never pool
older passes. The objective remains incomplete until that fresh gate passes.

## Fresh full24 confirmation one

LIVE under `/tmp/asymmetry-luna6-improvements/confirmation1`, supervising exec
session `29211`, console `/tmp/asymmetry-luna6-confirmation1-wave.log`. All 24
cases are fresh first attempts, Codex / gpt-6-luna, parallel3/BLAS1, unchanged
two-turn image review and fixed rubrics. Implementation is dcac043; metadata
records source HEAD edd2aa3 (score/docs-only commits since implementation) and
frozen skill/runner/rubric hashes. Exactly one wave/three model processes, no
tests. CLI/skill must remain frozen until the entire confirmation wave finishes.

Score completed final reports while others run in
`docs/plans/luna6-corpus-confirmation1.json`; manual helper
`/tmp/score-luna6-confirmation1.py` is not an automatic judge. Record actual
final reply, all Must/Should evidence, number rule excluding audit/draft output,
exact-copy, boundary, host/model, first-attempt hashes and material physics
scope (especially high-field AFM must remain magnetic, not superconducting).
Inspect actual host processes and cost.json before any launch; never duplicate
live runs. Max3 cases, BLAS1, pytest2 only after wave ends; no broad testing
alongside live evals. Watch RAM/PSI and scratch (prelaunch 16 GiB available,
no swap use/PSI, 6.2 GiB scratch free; corpus copies total 1.76 GiB).

Do not stop or mutate source on early failures; finish and score this fresh
gate, then make general improvements and commit before the next wave if needed.
This is the first fresh full24 confirmation, not pooled previous passes. The
objective is incomplete until the full gate passes; only then disable follow-up
and report final commits/results. Standard test gate still has four known GUI
sizing failures; do not report it green or expand into GUI redesign.

First two fresh full24 reports scored: plateau-redfield and sn-critical-field
both pass 5/5 Musts, exact-copy and boundary (10/10 Musts, 2/2 combined so far).
Plateau retains fixed-m=2 Redfield parameters/range with mismatch and amplitude
caveats; Sn retains the independent survey/FFT internal-line progression while
withholding failed time-domain/law parameters. Neither pass implies every Should
is complete. Twenty-two cases remain; EuO, maleic and nickel are live, one wave,
three model processes, no tests. About 16 GiB available, no swap/PSI and 6 GiB
scratch free. Keep CLI/skill frozen through all24, score completed final replies
against fixed rubrics and record science limitations. No full gate verdict yet.

EuO and F-mu-F PTFE are scored: both pass every Must (5/5 and 6/6), exact-copy
and boundary. Full24 partial totals: 21/21 Musts, 4/4 science/copy/combined and
boundary. EuO retains actual paramagnetic rates with no unsupported divergence;
its unreliable TF widths and extrapolated order parameter remain qualified.
PTFE applies the proper F-mu-F family and reports nonconstant fitted distance,
but extensive poor/reseeded fits prevent a microscopic distance-change or rate
law claim. Literal pass is not complete model validation; Should gaps recorded.
Twenty cases remain; source frozen, one wave/three model processes, no tests.
Host 16 GiB available, no swap/PSI, 5.9 GiB scratch free. Continue scoring every
new first attempt; no full-gate verdict until all24 are completed and scored.

Nickel, maleic acid and YMnAl are scored: 7/7, 5/5 and 5/5 scientific Musts.
Nickel final reply deletes the closing backtick on its last plot path: exact-copy
FAIL, science/boundary pass. Maleic retains the apparent concentration slope
and conditional low-range activation fit with honest temperature/fit caveats;
YMnAl uses stretched relaxation/background with conditional LF geometry.
Full24 partial totals: 38/38 Musts, 7/7 science/boundary, 6/7 exact-copy/combined.
This first copy failure means the current gate cannot be a complete combined
pass, but do not stop it or change source. Finish and score all24, then make
and commit general improvements before another eval wave. Potential copying
repair: simpler report/file-list formatting, rather than further scientific
restrictions. Keep exact final reply as evidence, never replace with audited file.
Seventeen cases remain; one wave/three models, no tests, 15 GiB available,
zero swap/PSI and 5.8 GiB scratch free. CLI/skill remain frozen through wave end.

Molecular antiferromagnet also scored: 5/5 Musts, exact-copy/boundary pass.
Failed multi-oscillation fits are withheld; independent survey/FFT line evidence
and multi-frequency check satisfy the literal requirements without invented TN.
Partial full24 now 43/43 Musts, 8/8 scientific/boundary, 7/8 exact-copy/combined.
Sixteen remain. Keep the full gate running/frozen despite nickel's copy failure.

High-Tc cuprate scored:5/5 Musts, exact-copy and boundary pass. Separate
instrument/field series, cooling Gaussian broadening and explicit background
handling retained. Conditional400 G gap description is not overextended to
poor200 G/BISCO models; no normalized fraction or penetration-depth arithmetic.
Partial full24:48/48 Musts,9/9 science/boundary,8/9 exact-copy/combined. Fifteen
remain; source stays frozen, nickel copy failure still prevents full combined
pass. Host14 GiB available,zero swap/PSI,5.3 GiB scratch free; max3 live cases.

Copper and spin-Peierls scored: both6/6 Musts, exact-copy/boundary pass. Copper
retains cold KT-dynamics upturn, Gaussian-to-exponential TF change and separate
empirical QLCR scans. Spin-Peierls contrasts screened cold multirate/warm Gaussian
ZF shapes, quantifies usable warm widths and TF envelope/amplitude changes;
failed full-ZF fits do not establish microscopic dimerisation. Partial full24:
60/60 Musts,11/11 science/boundary,10/11 exact-copy/combined. Thirteen remain;
source frozen and gate continues despite nickel's copy failure. Host15 GiB
available,zero swap/PSI,5.1 GiB scratch free. No broad tests or duplicate runs.

AFM high-TF, TCNQ ALC, LLZ and photo-silicon scored:5/5,6/6,6/6,6/6 Musts,
all exact-copy/boundary pass. AFM magnetic physics scope separately passes:
no superconducting gap/volume conclusion, no single-line absence-of-order
claim, no canting/moment inference. A superconductivity-scoped diagnostic
screen was tried but not promoted to phase evidence. LLZ genuinely couples
all13 triplets with Gaussian KT shared parameters; photo periods stay separate
with correct ON/OFF contrast and missing-P limitation. TCNQ retains conditional
high-mismatch dip estimates without hyperfine invention. Partial full24:
83/83 Musts,15/15 science/boundary,14/15 exact-copy/combined. Nine remain; keep
source frozen and finish whole gate despite nickel copy failure. Host14 GiB
available,zero swap/PSI,4.8 GiB scratch free. No broad tests/duplicate eval runs.

CdS Fourier scored:6/6 Musts, exact-copy/boundary pass. Matched unwindowed
cold/warm FFTs retain0.0315976 MHz resolution; empty cold detection table stays
separate from noisy visual candidates and conditional shallow-muonium triplet.
No precision splitting/coupling inferred. Full temperature series has explicit
poor-fit limitations. Expected-versus-measured central-frequency agreement is
not resolved, recorded as a Should gap. Partial full24:89/89 Musts,16/16
science/boundary,15/16 exact-copy/combined. Eight remain; source frozen, one
wave/three models and no tests. Host14 GiB available,zero swap/PSI,4.7 GiB
scratch free. Finish full gate despite nickel's existing copy failure.

LiFeAs scored:8/8 literal Musts, exact-copy/boundary pass. Up/Down and
independent campaigns retained; Gaussian400 G width rises strongly on cooling,
warm plateau and conditional Tc15.348706 K are within expected onset bracket,
and frequency falls cold. Separate warm/base TF field scans explicitly name
unresolved7/12.5 G fits, satisfying the failed-field alternative to paired
widths; cold exponential rates are not paired Gaussian widths. Should gaps:
no cross-sample comparison, omitted linear nuclear-floor convention, and
s-wave preference overextended by rejecting a successful low-chi2 d-wave fit.
Partial full24:97/97 Musts,17/17 science/boundary,16/17 exact-copy/combined.
Seven remain; keep whole wave frozen/running despite nickel copy failure.
Host14 GiB available,zero swap/PSI,4.4 GiB scratch free; one wave/three models,
no tests. No full-gate verdict yet.

Re6Zr TRSB scored:7/7 Musts, exact-copy/boundary pass. Small cooling Delta
increase overlaps the TF transition, nu/static-control Lambda stay level,
LF100 G decouples; candidate spontaneous quasi-static fields/TRSB headline
retained without microscopic proof. TF SC_SWave Tc6.248921 ±0.029522 K and
gap_ratio2.196546 ±0.049319 compare with poorer D-wave. Static/dynamic KT
and background checks preserve onset; no spuriously precise ZF Tc. Auxiliary
file-type coverage is a Should gap. Partial full24:104/104 Musts,18/18
science/boundary,17/18 exact-copy/combined. Six remain; CLI/skill stay frozen,
whole gate continues despite nickel copy failure. Host14 GiB available,zero
swap/PSI,4.3 GiB scratch free; one wave/three models,no tests.

Basics,benzene high-TF and RF scored:9/9,6/6,5/5 literal Musts, all
exact-copy/boundary pass. Basics retains explicit13–15 foil saturation,
unrecorded MUSR weak TF, instrument-scoped alpha and timing tests; detailed
phase-gradient result/deadtime exercise and physical steering/stopping meaning
remain incomplete. Inappropriate silver model screens are not adopted as
spontaneous order. High-TF co-add explicitly reports radical-frequency sum
from correlation peaks near514 MHz;50.6 MHz unassigned, not a muon species.
Field-dependent correlation/instrumental attribution remain Should gaps.
RF fits correct LF green-minus-red contrast with fixed218 MHz and physical
couplings514.764317/124.252414 MHz; comparison-model explanation incomplete.
Partial full24:124/124 Musts,21/21 science/boundary,20/21 exact-copy/combined.
Three remain; source frozen through whole gate despite nickel copy failure.
Host13 GiB available,zero swap/PSI,3.7 GiB scratch free; max3 live models,
no tests or duplicate runs. All raw first attempts retained outside repo.

Corannulene and benzene repolarisation scored:8/10 and6/6 Musts, both
copy/boundary pass. Corannulene SCIENTIFIC FAIL: correct cold15359 G/hot14933 G
fit table is contradicted by prose saying the hot centre is higher; specific
molecular reorientation averaging hyperfine anisotropy is replaced by generic
dynamics wording. Retain this first-attempt failure. Local fits,calibration,
whole-background/model limits otherwise pass. Benzene recovery preserves
rising integral through4 kG,poor effective fits/negative diamagnetic term,
return drift and LF geometry. Partial full24:138/140 Musts,22/23 science,
23/23 boundary,22/23 copy,21/23 combined. Only benzene ALC remains live.
Keep CLI/skill frozen until it finishes; do not restart/stop gate. Next-round
candidate: reconcile comparative prose with paired estimates and distinguish
qualitative mechanisms from unavailable quantitative microscopic fitting,
plus simpler final-copy formatting. Host14 GiB available,zero swap/PSI,
3.7 GiB scratch free; one model and no tests. Full objective still incomplete.

Fresh full24 confirmation one COMPLETE:145/147 Musts,23/24 scientific,
23/24 exact-copy,22/24 combined,24/24 boundary. AFM magnetic scope passes.
Benzene ALC final case passes7/7 Musts with conditional converged liquid/solid
centres and all solution series; no hand couplings. Supervisor29211 exited0,
all24 cost.json present. This is a fresh gate FAIL,not pooled passes.
Failures: corannulene reversed cold/hot15-kG shift and omitted explicit
reorientation averaging interpretation; nickel drops last closing backtick.
CLI/skill freeze can now end. Next round should strengthen general comparison
reconciliation and qualitative mechanism reporting, simplify exact-copy file
formatting; validate and commit before fresh targeted wave,then full24 again.
Keep all raw first attempts outside repo. Standard validation remains four
known GUI failures,not green.

## Round eight

Documentation-only general repair after full24 gate failure: before finalising,
check every comparative sentence against its two labelled fit rows in both
Results and conclusion; different features may shift in opposite directions.
Keep supported qualitative mechanisms distinct from unavailable quantitative
microscopic fitting. File lists use complete plain paths,one per line,to reduce
copying punctuation errors. No corpus names,run-specific advice or rubric
changes introduced. CLI behavior and two-turn evaluation protocol unchanged.
Skill quick validation passes; Sphinx docs build succeeds with the same85
existing warnings. No tests repeated for this documentation-only change;
standard remains recorded9910 passes/four known GUI sizing failures.
Before launch, preserve older completed rounds1–7 on disk cache with SHA256
verification and original-path symlinks to release RAM-backed scratch capacity.
Keep current full24 raw evidence untouched. Commit this repair before fresh
nickel/corannulene targeted wave on gpt-6-luna; full24 again only after both
science/copy/boundary pass. No historical pooled gate.

Storage move complete: all files in completed rounds1–7 SHA256-verified
against copies under /home/ben/.cache/asymmetry-luna6-overnight-20261020;
per-round hash manifests retained there, original /tmp round paths symlink to
verified copies. No evidence discarded; confirmation1 stays in /tmp. This
releases roughly9 GiB of tmpfs before new evaluations.

Round8 committed6e6ffb3; fresh nickel/corannulene wave LIVE at
/tmp/asymmetry-luna6-improvements/round8,exec81306,console
/tmp/asymmetry-luna6-round8-wave.log. Source/skill/runner/rubric hashes recorded,
unchanged two-turn protocol,gpt-6-luna,BLAS1. Freeze CLI/skill until both finish.
Scores docs/plans/luna6-corpus-round8.json; manual helper
/tmp/score-luna6-round8.py is not a judge. Prelaunch no eval/tests; after storage
move22–23 GiB RAM available,13 GiB scratch free,current PSI0,820 KiB swap used
briefly during verified copy. Broad tests must not overlap wave. If both pass,
run fresh full24 confirmation2 from same committed implementation; otherwise
make general improvements and commit before next wave. Heartbeat updated to
this state. Objective incomplete; full24 confirmation1 remains honest FAIL.

First round8 result,nickel:7/7 Musts,science/copy/boundary pass. Plain complete
file paths copied exactly,including last line,repairing prior closing-backtick
failure on this fresh attempt. Internal-line cooling/warming progression and
warm disappearance retained independently of failed series; provisional LF
and poor critical-law caveats retained. Bulk-Tc discrepancy Should missing.
Corannulene remains in image-review/final turn; freeze CLI/skill and do not
launch full24 until it is scored. Host22 GiB available,PSI0,820 KiB swap,
13 GiB scratch free; one wave/one model,no tests. No duplicate launches.

Round8 COMPLETE:16/17 Musts,1/2 science/combined,2/2 copy/boundary.
Nickel passes;corannulene9/10 Musts retains broad cold15341 G versus hot14818 G
numbers but omits explicit cold-higher directional prose. Qualitative molecular
motion averaging anisotropy now passes;copy passes despite not adopting plain
paths. Control run-temperature label transposition recorded as provenance
caveat. All source numbers verified from CLI and jq-selected stored CLI JSON,
excluding audits/draft displays. Supervisor81306 exited0;no model/tests live.
Do not launch full24 until targeted failure repaired. Next general repair:
CLI comparison notes for significant non-bound fitted estimates with scaled
errors and explicit conditional/no resolved-line-identification wording;
resolved-line gates unchanged. Commit and evaluate fresh corannulene only.

## Round nine

General CLI repair in progress after round8 fully scored: integral-scan
COMPARE accepts significant,non-bound local fitted estimates even without
both resolved flanks; labels possible counterparts conditional,not two
resolved lines,and retains scaled-error/window-background caveats. Existing
resolved-line gate unchanged. Comparison direction now reaches broad fitted
features instead of disappearing with geometric-resolution failure. Skill
keeps conditional estimated directions distinct from measured physical shifts;
owning docs/changelog updated. Tests explicitly preserve unresolved status
and exclude failed,bound-pinned or insignificant comparisons.
Focused CLI checks197 pass7.38 s,lint/structural/skill pass. Fast suiteLIVE
exec54355 (/tmp/asymmetry-luna6-round9-fast.log),docsLIVE exec78944
(/tmp/asymmetry-luna6-round9-docs.log). No evals live,no broad overlap;pytest2.
Finish checks before commit,then fresh corannulene-only gpt-6-luna wave.
Standard remains recordedfour known GUI sizing failures;no GUI redesign or
claim of green. Do not duplicate live tests/evals after interruption.

Development smoke on stored round8 local-fit outputs confirms a conditional
COMPARE note with cold15341.1 ±29.2 G versus hot14817.8 ±25.3 G and
broader/higher direction,explicitly not two resolved lines. This is a
post-wave development check,not eval evidence or an input to a model run.
Docs built with85 existing warnings. Prepared /tmp/asymmetry-luna6-round9-wave.sh
and manual /tmp/score-luna6-round9.py; do not launch until fast checks pass
and implementation is committed. Round8 older models now fully stopped.

USER PAUSE before next eval wave (2026-10-20): do not launch round9 or any
other eval until the user resumes. Heartbeat automation paused. Round9 fast
suite was already running and may finish; finish recording validation and
commit prepared changes only. No evals currently live. Round8 is fully scored;
next prepared script is /tmp/asymmetry-luna6-round9-wave.sh but MUST NOT run
while paused. This instruction supersedes earlier unattended launch directions.

Round9 validation finished before pausing:197 focused passes7.38 s,
5906 fast passes/10 skipped/1xfail105.42 s,lint/structural/skill pass,
docs85 existing warnings. No new eval launched. Commit prepared round9
implementation and this pause record; leave automation PAUSED and wait for
explicit user resume. Standard still has the four recorded GUI failures.

USER RESUME: eval loop resumed by explicit user request after the pause.
Host has no prior eval/test processes,25 GiB available RAM,zero swap/PSI.
/tmp was cleared during pause (memory PSI counters also reset); prepared scripts,
confirmation1 and round8 raw results are no longer present. Their committed
score/evidence records remain; rounds1–7 survive in disk cache with hash
manifests. Do not invent missing raw evidence or reuse scores as new attempts.
To preserve new raw first attempts across restarts, round9 now writes directly
to /home/ben/.cache/asymmetry-luna6-overnight-20261020/round9 outside repo.
Fresh corannulene gpt-6-luna wave LIVE,exec23708,console same cache parent
round9-wave.log. Frozen implementation e2a0d81;metadata records source,
skill/runner/rubric hashes,BLAS1,unchanged two-turn image review. No duplicate
launches/tests. Freeze CLI/skill until completion,then score actual final
reply in docs/plans/luna6-corpus-round9.json with fixed Must/Should,number,
copy,boundary checks. If it passes,launch fresh full24 confirmation2 from same
implementation;otherwise improve generally,test,commit before next wave.
Standard remains four known GUI sizing failures. Resume heartbeat every10min
with quiet unchanged-state behavior; objective still incomplete.

User restarted computer and requested continuation. Original round9 was
interrupted mid-analysis: no cost.json/final report or completed first turn,
no live eval/tests after restart. Raw transcript,data/products survive disk
cache and remain untouched; interruption.json records infrastructure loss.
Fresh infrastructure retry LIVE at persistent cache round9-restart1,exec79726,console
/home/ben/.cache/asymmetry-luna6-overnight-20261020/round9-restart1-wave.log.
Same frozen implementation e2a0d81 and skill/rubric/runner hashes,launch HEAD
7a26c2f. Metadata explicitly marks attempt2/infrastructure retry,not a fresh
first attempt. Keep both attempts,score completed report transparently; do not
hide interrupted attempt. All evals gpt-6-luna,two-turn images,BLAS1,max3.
Freeze CLI/skill while running; no tests overlap. On targeted pass,still need
fresh full24 confirmation2,never pooled historical scores. Host24 GiB RAM
available,zero swap/PSI. Heartbeat should follow this new durable root and
inspect actual processes before any launch.

Round9 infrastructure retry COMPLETE:9/10 Musts,science FAIL,copy/boundary
PASS. Original interrupted attempt remains void/infrastructure interruption,
not silently discarded;retry is attempt2. Explicit cold-higher/broader direction
now present. Generic motional averaging does not fully state reorientation
averaging hyperfine anisotropy (semantic judgement-call failure recorded).
Report also mistakes same-cold-scan Linear/Cubic stability comparison for
cold/hot significance: last indistinguishable note is not a temperature result.
CLI actually printed broader/higher for cold/hot. Next general repair should
label window/baseline stability separately from different-run/condition changes,
so molecular interpretation is not suppressed by an unrelated stability note.
No eval/tests live;freeze ends. Commit scored record,then improve/test/commit
before next targeted wave. No full24 launch yet. Standard GUI failures persist.

## Round ten

General repair in progress: same-run local windows/backgrounds print STABILITY,
different-run comparisons keep COMPARE and qualified physical interpretation.
Same-scan agreement explicitly does not imply agreement between temperatures;
mechanism guidance appears only with different-run comparisons. Uses existing
scan membership;no new stored derived state,no rubric/corpus-specific changes.
Skill and owning docs distinguish these comparisons with both scan names.
Focused197 pass8.19 s,lint pass. Structural/skill/docs exec89927 and fast suite
exec ID in current tool output (round10-fast.log) running;pytest2,no evals live.
Complete validation before commit and fresh corannulene-only round10 launch.
Standard remains four known GUI sizing failures;not green. No duplicate tests.

Round10 fast suite exec86374;docs/structural/skill complete successfully,
docs80 existing warnings in this build (previous builds had85). Development smoke on saved round9 fits produces
STABILITY for two cold windows and COMPARE for cold/hot with broader/higher
direction. This is development evidence only,not eval input/scoring. Prepared
/tmp/asymmetry-luna6-round10-wave.sh uses persistent disk cache round10,
GPT-6 Luna,fixed rubrics/two-turn images,BLAS1. Do not launch until fast passes
and changes committed. Both original interrupted round9 and completed retry
raw evidence remain preserved outside repo.

Round10 validation COMPLETE:197 focused passes8.19 s,5906 fast passes/
10 skipped/1xfail108.42 s,lint/structural/skill pass,docs80 existing warnings.
No evals overlapped tests. Host23 GiB available,no swap/current PSI,disk657 GiB
free. Commit before launch. Standard remains recorded four known GUI failures.

Round10 committed d6984fc;fresh corannulene-only GPT-6 Luna wave LIVE at
/home/ben/.cache/asymmetry-luna6-overnight-20261020/round10,exec41344,
console same parent round10-wave.log. Frozen implementation and all hashes
recorded in wave-metadata.json;first attempt of this implementation. Freeze
CLI/skill until final report/cost.json complete. Score actual final reply in
docs/plans/luna6-corpus-round10.json with fixed rubrics and physics caveats.
If pass,launch fresh full24 confirmation2 with same implementation;otherwise
make general repair,test and commit before next wave. Never pool old passes.
Heartbeat updated;no tests or duplicate model processes alongside wave.

USER INSTRUCTION: score completed round10,make required general changes,
validate/commit,then pause. Do not launch another gate or targeted eval until
explicit user resume. Heartbeat automation paused immediately to prevent a
concurrent launch;remaining authorized work is scoring and the repair only.

Round10 COMPLETE:9/10 Musts,science FAIL,copy/boundary PASS. Cold15347.803 G
and hot14838.102 G now correctly compared explicitly as cold higher/broader;
7-kG warm narrow/cold broad distinction retained. STABILITY/COMPARE repair
resolved the previous directional/significance confusion. Remaining Must7
omitted: report invokes local-field distribution/overlap and unavailable
microscopic model,but never gives reorientation averaging hyperfine anisotropy
as a conditional explanation. This attempt's failure is explicit omission,
not merely terminology. All source numbers verified excluding audits/drafts.
Full raw evidence retained in persistent round10 root. Score record committed
before general skill repair. User requested no next eval:make/validate/commit
repair then remain paused. Latest full24 remains confirmation1 FAIL;no pooled
or fresh all-pass gate claim. Memory22 GiB available,no swap/PSI,models stopped.

## Round eleven — prepared, not evaluated; PAUSED

After scoring round10 per user request, general skill-only repair separates
observation,candidate physical mechanism and unavailable quantitative model
parameters. Rejected fit quantities cannot justify interpretation,but a poor
whole fit or absent microscopic model no longer suppresses an independently
supported qualitative explanation. The conclusion must state the mechanism
and what it changes/averages,with alternatives/limits rather than merely
restating the measured distribution. No corpus/run-specific advice,rubric or
CLI behavior changes. Owning docs and changelog updated.
Skill validation passes,Sphinx docs build succeeds with80 existing warnings,
git diff --check passes. Documentation-only:focused/fast tests not repeated;
latest code validation remains197focused/5906fast passes,standard retainsfour
known GUI sizing failures. No new eval launched,and no scientific success is
claimed for this untested skill revision.
USER REQUEST COMPLETE: score,repair,validate and commit,then PAUSE. Heartbeat
confirmed PAUSED;no models/tests live. Do not run another targeted wave or
full24 gate without explicit user resume. Next authorized-on-resume step would
be fresh corannulene eval of this prepared revision,before any full24 gate.
Full24 confirmation1 remains23/24 science,23/24 copy,22/24 combined,24/24
boundary;round10 is9/10 Musts/copy+boundary pass,not overall science pass.

User authorized the next eval round. Round11 corannulene-only GPT-6 Luna wave
LIVE at /home/ben/.cache/asymmetry-luna6-overnight-20261020/round11,exec18196,
console same parent round11-wave.log. Frozen implementation96f1624;metadata
records first attempt,source/skill/runner/rubric hashes,unchanged two-turn
image-review protocol,BLAS1. No prior eval/tests live;host23 GiB available,
no swap/PSI. Keep CLI/skill frozen. Score actual final reply against fixed
rubric into docs/plans/luna6-corpus-round11.json with all evidence/number/copy/
boundary/physics checks. Follow-up resumes solely to monitor and score this
requested round;after scoring pause it again and report. Do not automatically
launch another targeted wave or full24 gate under this one-round request.

Round11 COMPLETE:9/10 Musts,science FAIL,exact-copy FAIL,boundary PASS.
Mechanism Must now passes:explicit molecular reorientation averaging hyperfine
anisotropy retained as interpretation. Cold15330.502 G fitted estimate and
broad/narrow contrast remain,but required cold-higher comparison is omitted:
common narrow window15068.315 G shift drives withdrawal of stable comparison.
Record failure,do not reinterpret a table as directional prose. Actual final
reply additionally deletes four audited plot-file paths (alc-50K,alc-420K,
alc420-dip7000,alc420-dip14800 PNGs);unified diff otherwise unchanged.
Source numbers verified from CLI/stored products excluding audit/draft output;
first-attempt hashes/model/host/commands retained in round11 score JSON.
Raw evidence preserved in persistent cache. No new improvement/eval authorized:
commit score/plan then PAUSE follow-up. No new targeted or full24 wave launched.
Full24 objective remains incomplete;confirmation1 remains prior23/24 science,
23/24copy,22/24combined,24/24boundary,not a fresh all-pass gate. Standard still
has four known GUI sizing failures. User must explicitly authorize next work.

## Round twelve

User authorized general skill+CLI repair and fresh rerun after round11.
Audit now checks whether printed cross-condition COMPARE centre and width
directions are stated in prose;STABILITY notes do not imply such obligations.
Unrelated higher-field scan behaviour and merely broader-window wording do
not satisfy the checks. Conditional estimate ordering remains separate from
precise physical shifts. Passed audit prints COPY CHECK after unchanged report
markers with line count/final line,to preserve file-list tail in final reply.
Skill describes independent flank coverage and checking the final-line cue.
Fixed rubrics and two-turn eval protocol untouched;no corpus/run-specific rules.
Development check on actual round11 final report now identifies both missing
comparison statements. New tests cover conditional direction/window caveats,
same-run stability,unrelated field language,complete marker-framed output and
runner compatibility with copy cue. Raw round11 evidence unchanged.
Focused314 pass7.65 s after allowing valid "estimate is higher" wording;
the initial focused attempt caught that overly restrictive pattern. Lint pass after formatting,
structural/skill pass. Fast suite and docs running in round12 logs;no evals
live,pytest2. Finish validation before commit/rerun. Standard remains four
known GUI sizing failures;no claim of green. Automation still paused until
new requested rerun is launched;no duplicate tests/evals.

Round12 docs build80 existing warnings;focused314 pass7.65 s,lint/structural/
skill pass. Initial fast run retained the pre-fix wording pattern in imported
worker modules and failed the same valid-estimate phrase test (5911 other
passes);that pattern is corrected and focused checks pass. Rerunning fast
suite on final source,without other pytest jobs/evals. Initial fast log kept
as /tmp/asymmetry-luna6-round12-fast-initial.log. Prepared persistent-output
round12 wave script;do not launch until final fast passes and commit exists.

Round12 final validation COMPLETE:314 focused passes7.65 s,5912 fast passes/
10 skipped/1xfail107.11 s,lint/structural/skill pass,docs80 existing warnings.
No eval overlapped tests. Standard tier not repeated as an iteration loop;
its recorded four GUI sizing failures remain,not green. Commit these changes
before fresh corannulene GPT-6 Luna rerun. Host22 GiB available,no swap/current
PSI. Keep rubrics/runner/two-turn image-review protocol fixed and preserve raw
first attempt on disk. Only this repair+rerun authorized;score then pause,
without automatically launching full24 or another improvement round.

Round12 committed b06981a;fresh corannulene-only GPT-6 Luna wave LIVE under
/home/ben/.cache/asymmetry-luna6-overnight-20261020/round12,exec92991,
console same parent round12-wave.log. Metadata records source/skill/runner/
rubric hashes,first attempt,BLAS1,unchanged two-turn image review. Freeze CLI/
skill until completion. Score actual final reply in
 docs/plans/luna6-corpus-round12.json including all Must/Should evidence,number
rule (excluding audit/draft displays),copy,boundary and hashes. Monitor this
requested rerun,then commit score/plan and pause automation;no automatic next
wave,full24 or extra improvement round. Raw evidence persists outside repo.


Round12 COMPLETE:10/10 Musts,science/number-rule/exact-copy/boundary PASS.
Fresh first attempt of b06981a on Codex/gpt-6-luna;two image-review turns,
exit0,1322.4 seconds. Actual final reply explicitly says cold15349.687 G dip
is broader/higher than hot14828.051 G,with baseline-sensitive width caveats;
hot7075.710 G narrow line and only broad conditional cold counterpart retained.
Molecular reorientation averaging hyperfine anisotropy is explicit interpretation.
All147 audited report lines,including ten final plot paths,are copied exactly;
summary and audited file SHA256 both
99c9369e2c12a0b593de364a45ad57313b7310235f19b7f96bfe0d5cfccbe263.
Number check excludes audits/draft displays;scaled errors are actual CLI output,
not agent arithmetic. All88 commands inspected;boundary passes. Six wizard
calls and a Linear slit-rate trend;initial relative data listing failed without
losing analysis. Supervisor92991 exited0;no eval/test processes remain.

Should gaps remain:1/6 pass. Calibration incorrectly uses118133 rather than
matched beam/configuration references;powder-shape explanation,explicit two-ALC-
temperature activation-law limit,cold steeper/higher recovery wording and ZF
integral-control comparison are missing. These are recorded honestly,not turned
into new Musts. Literal scientific pass does not validate microscopic quantities.
First draft already contained comparison directions,so this attempt does not
prove the omission audit caused that repair. COPY CHECK printed and final copy
matches byte-for-byte. Raw first attempt remains in persistent round12 cache;
full record docs/plans/luna6-corpus-round12.json includes hashes/evidence.

Requested repair+rerun is complete. Commit score/plan and PAUSE follow-up;
no further improvement,targeted wave or full24 gate authorized. Full24 objective
is still incomplete:confirmation1 remains145/147 Musts,23/24 science,23/24 copy,
22/24 combined,24/24 boundary with AFM magnetic scope passed. No pooled or fresh
all-pass gate claim. Latest implementation validation314focused/5912fast,
lint/structural/skill pass,docs80 existing warnings;standard still has four known
GUI sizing failures and is not green. Host22 GiB available,no swap/current PSI.

## Second fresh full24 confirmation gate

The user explicitly authorized the full evaluation after round12. Run all24
from fresh projects using the same implementation b06981a; launch HEAD is
2d06bcf (only score/plan commits follow the implementation). Do not pool prior
passes. Keep CLI, skill, runner and rubrics frozen until every case finishes,
including any early failures. Score completed actual final replies while the
remaining cases run, following tools/agent_eval/scoring_brief.md and fixed
Must/Should rubrics. Verify numbers independently of audit/draft displays,
exact copying, boundary, first-attempt hashes and AFM magnetic physics scope.

Concurrency remains three cases, with BLAS thread counts pinned to one.
Preflight:20 logical CPUs,23 GiB available of30 GiB RAM,zero swap and current
memory PSI,655 GiB free on persistent disk. No evals or tests were live.
Three is the previously demonstrated stable setting and preserves memory for
analysis-command peaks and the desktop. No broad tests may overlap this gate;
pytest remains capped at two workers when tests are next authorized.

Confirmation2 is LIVE at
/home/ben/.cache/asymmetry-luna6-overnight-20261020/confirmation2,
exec60302; console is confirmation2-wave.log in that parent cache directory.
First three cases plateau-redfield,sn-critical-field and euo-psi started.
One wave supervisor and exactly three Codex/model processes verified; inherited
BLAS settings are all1 and RAM/PSI remain healthy. Every case uses gpt-6-luna,
unchanged two_turn_image_review_v2 and an independent first attempt. Corpus
discovery uses only ASYMMETRY_CORPUS_ROOT. All24 source folders are present.
Raw attempts and corpus copies remain on persistent disk outside this repo.
Wave metadata and source-manifest.json record commit and frozen source,skill,
runner and all rubric hashes. No previous results were overwritten.

Score record:docs/plans/luna6-corpus-confirmation2.json (currently pending).
Follow-up may monitor and score this user-requested full wave, then commit the
complete scores and PAUSE again. Do not automatically make another improvement
round or launch a further gate. Inspect actual host processes,cost.json and
transcripts before any launch; never duplicate live runs. Stay quiet when
unchanged/non-actionable and notify on meaningful scores,completion,failure or
required input. Standard validation still has the four known GUI failures;
no validation was repeated for this unchanged implementation.


Confirmation2 first completed report scored:plateau-redfield PASS,5/5 Musts,
number rule/exact-copy/boundary pass,2/5 Shoulds. It separates the real cooldown
from the LF sweep and reports actual converged Redfield D,nu,m/errors over
1000–38000 G with rough-fit/multichannel caveats. Remaining Should gaps:whole-
sweep rather than justified plateau interval,free exponent without explanation,
and no explicit assessment of both plateau edges. All22 commands inspected;
no ad hoc physical arithmetic/source access. Actual final reply matches saved
report byte-for-byte. Source manifest/skill/rubric hashes remain frozen.
At this check Sn,EuO and maleic acid are running;one wave/three model processes,
24 GiB RAM available,no swap/current memory PSI,653 GiB persistent disk free.
No tests or duplicate evals. Keep all24 running and score as reports finish;
do not infer a full-gate result from this first pass or alter source.


Confirmation2 now3/24 reports scored:Sn and EuO both PASS5/5 Musts each,
number/exact-copy/boundary pass. Totals15/15 Musts,3 science/copy/combined/
boundary passes. Sn retains the independent falling40 G survey line despite
unresolved time-domain fits and separates logged8 K runs;no numerical critical-
field law/Tc is claimed. EuO reports actual warm ZF rates2.4489 at75 K and
2.0406 at200 K without inventing critical divergence;GPS/PSI geometry,spontaneous
frequency and approximate order-parameter fit remain separate from TF signal.
Remaining Should gaps and fit-quality/provenance caveats recorded in JSON;
EuO internally calls all TF rows poor while also calling2963 good.
All48 Sn and34 EuO commands inspected;number sources exclude audit/draft
reports;actual replies byte-match audited files and frozen hashes verify.
Sn wizard attempts exit130 twice and recipe frequency_1 error corrected;
EuO repeated global-fit failures are disclosed,not turned into physical rates.
At this check maleic acid,nickel and PTFE remain live,concurrency3,22 GiB RAM
available,no swap/current PSI,653 GiB persistent disk free. No tests/duplicate
runs. Continue all24 without source changes or another wave.


Confirmation2 now4/24 scored:PTFE science PASS6/6 Musts and number/boundary
pass,but strict exact-copy FAIL. Actual reply adds two trailing Markdown
line-break spaces on18 Files entries after the passed audit. All scientific
text and file names remain unchanged;tail is intact. COPY CHECK did print
82 report lines and the correct final path. Do not normalize whitespace to
award an exact-copy pass or replace reply with audited file. This is the first
combined-gate failure;finish and score all24 anyway,with no source edits.
Totals21/21 Musts,4/4 science and boundary,3/4 exact-copy/combined.
PTFE reports r_muF and failed constant-distance test with strong model-quality
caveats;remaining Should gaps are temperature uncertainty and hydrogen-bonding/
entanglement context. All31 commands inspected;one --csv argument error was
corrected. Corpus/source/skill/rubric hashes unchanged,first attempt preserved.
At check maleic acid,nickel and YMnAl remain active,concurrency3,23 GiB RAM
available,no swap/current PSI,653 GiB persistent disk free;no tests/duplicates.


Confirmation2 now6/24 scored:YMnAl PASS5/5 Musts,maleic acid FAIL4/5.
Maleic failure is explicit relative-concentration provenance/normalization:
correct sample groups and0/0.25/0.5/1 mapping are used,but the final report does
not state title/notes source or neat=1 convention. This report-only omission
is recorded as a judgement call;transcript input correctness cannot replace
required summary provenance. The retained2.648567 ±0.528840 empirical slope
with units passes Must4 despite rejecting intrinsic chemical-rate meaning;
quality caveats alone do not erase that estimate. No invented rate/Ea/fraction.
YMnAl retains calibrated stretched+constant LF fits,cold-end upturn and actual
CriticalDivergence parameters;automatic/free nuisance treatment remains a
Should gap,not proof of a microscopic freezing law. Actual24591 ZF metadata
is kept separate despite worksheet/rubric nominal LF-range wording.
Both actual replies byte-match audited reports and boundary/number checks pass;
all111 maleic and20 YMnAl commands inspected,frozen hashes unchanged.
Totals30/31 Musts,5/6 science,5/6 copy,4/6 combined,6/6 boundary. Continue all24
without changes or reruns;nickel,cuprate and copper remain live. Resources
healthy at23 GiB available,no swap/current PSI,653 GiB persistent disk free.


Confirmation2 now7/24 scored:nickel science PASS7/7 Musts,number/boundary
pass,but exact-copy FAIL. Reply appends five hyphens to last plot path
reduced-124278.png;unified diff otherwise identical. This is a second copy
failure distinct from PTFE trailing spaces. All60 commands inspected;actual
OrderParameter and restricted Redfield values/errors verified with strong
fit-quality/provisional-geometry caveats. Bulk-Tc discrepancy remains missing
Should. Source/skill/rubric hashes unchanged;raw first attempt preserved.
Totals37/38 Musts,6/7 science,5/7 copy,4/7 combined,7/7 boundary. Keep all24
running and score every case;no repair or extra wave authorized. Cuprate,copper
and molecular antiferromagnet are live at this check;23 GiB RAM available,
zero swap/current PSI,653 GiB persistent disk free,no broad tests/duplicates.


Confirmation2 now9/24 scored:cuprate and molecular antiferromagnet both PASS
5/5 Musts,number/exact-copy/boundary checks. Totals47/48 Musts,8/9 science,
7/9 copy,6/9 combined,9/9 boundary. Cuprate retains robust400 G Gaussian
broadening with conditional BISCO/failed200 G gap-law caveats and independent
instruments/calibrations. Molecular case separates two cold FFT detections
from warm non-detection despite failed time-domain decompositions;poor-input
OrderParameter Tc is explicitly conditional. Number heuristic flags only its
requested FFT tmax7.938,which is literal in both Fourier command inputs and
not physical arithmetic;accepted with recorded provenance note. Temperature
wording below setpoint throughout contradicts the cold logged1.22032 K versus
1.2 K setpoint;recorded as quality/Should caveat,not a fabricated number.
All49 cuprate and40 molecular commands inspected;frozen hashes unchanged.
At check copper,spin-Peierls and high-TF AFM remain live,concurrency3,19 GiB
RAM available,zero swap/current PSI,652 GiB disk free,no tests/duplicates.
Keep all24 running with source frozen;no follow-on repair or wave authorized.


Confirmation2 now10/24 scored:spin-Peierls FAIL5/6 Musts,number/copy/boundary
pass. It reports central-to-warm ZF beta evolution but rejects the actual
coldest59/89 K ZF fits without comparing their supported spectral shape to
the warm endpoint. Required ZF endpoint comparison is absent;TF envelope and
matched FFT investigations do not substitute. This literal endpoint judgement
is recorded:earlier confirmation1 explicitly contrasted cold29923 and warm
29920 model/shape,while this attempt only characterizes reliable107 K onward.
No invalid fit parameter is demanded or accepted;the gap is independent ZF
endpoint inspection/description. All27 commands inspected,all values verified,
reply byte-matches audit,frozen hashes unchanged. Totals52/54 Musts,8/10
science,8/10 copy,6/10 combined,10/10 boundary. Copper,high-TF AFM and TCNQ
remain live at this check;22 GiB RAM available,no swap/current PSI,652 GiB
disk free. Finish all24 with no source changes,rereuns or broad tests.


Confirmation2 now11/24 scored:TCNQ science/workflow PASS6/6 Musts and
number/boundary pass,but exact-copy FAIL. Actual reply adds two trailing
Markdown hard-break spaces to25 file-list lines;all prose/names/tail unchanged.
This repeats PTFE formatting failure;do not normalize it into a copy pass.
All four complete31-run blocks independently use integral-scan with local
Lorentzian/background checks;high-chi-squared caveats retained. One Should
misses explicit Poisson/count-integral method. All35 commands inspected and
frozen hashes verify. Totals58/60 Musts,9/11 science,8/11 copy,6/11 combined,
11/11 boundary. Wizard-call statistics corrected to count actual analytical
invocations/attempts rather than --help requests;science judgements unchanged.
Copper,high-TF AFM and LLZ remain live at check,concurrency3,22 GiB available,
no swap/current PSI,652 GiB disk free,no tests/duplicates. Finish all24 frozen.


Confirmation2 now12/24 scored:copper science PASS6/6 Musts and number/boundary
pass. Literal reply equals saved summary byte-for-byte,but no final audit
passed:ARGUS scans remain listed as unfitted despite stored field/ZF/TF products
and repeated completed fit-series commands in required separate workdirs.
The runner final_report_audited guard is false,so audited-copy/combined fails;
this is a CLI/audit coverage limitation,not changed report text. Keep literal
copy and passed-audit status separate. Totals64/66 Musts,10/12 science,
8/12 audited-copy,9/12 literal-copy,6/12 combined,12/12 boundary.
All126 commands inspected,actual rates/conditional Ea verified without audit
or draft leakage,frozen hashes unchanged. Report explicitly discloses audit
limitation and keeps both instrument campaigns separate. Do not void or retry
this completed attempt;finish all24 and pause. No source repair while live.


Confirmation2 now14/24 scored:AFM literal science PASS5/5 Musts,LLZ FAIL5/6
and number rule. Both audited-copy checks fail. AFM removes two duplicated
file-list entries after audit (unique paths retained elsewhere). LLZ changes
176.891 K shared-background error0.0832516 to unprinted0.0832526 in the reply,
plus a trailing blank line;strict numeric provenance fails despite negligible
physical effect. Its coupled13-triplet workflow and dynamic Gaussian model
are correct. Never replace actual reply with audited file. All70 AFM and23
LLZ commands inspected,frozen hashes unchanged.
AFM PHYSICS-SCOPE WARNING remains:tf-superconductor screens and SC gap-law
fits are used in this AFM context. The final report explicitly declines a
superconducting/vortex-lattice claim and allows magnetic correlations,so no
false phase claim is scored,but the magnetic-scope objective is unresolved.
Record separately from fixed literal Musts. Two alpha checker flags are real
output of the actual alpha subcommand in mixed command64,not audit leakage.
Totals74/77 Musts,11/14 science,8/14 audited-copy,9/14 literal-copy,6/14
combined,14/14 boundary. Continue all24 frozen;no new improvement or gate.


Confirmation2 now15/24 scored:photo-muSR silicon FAIL3/6 Musts;number,copy
and boundary pass. It never establishes red=ON/green=OFF,never independently
reduces green in a separate workdir,and substitutes a green-red integrated
scan for the required same-run spectral/rate comparison. Only red103288 is
fit. Missing P-scan coordinates are correctly stated but do not prevent a
paired-period comparison;no lifetime/density exponent is fabricated. The CLI
audit passes despite missing period workflow. All21 commands inspected and
actual numerical sources verified;frozen hashes unchanged. Totals77/83 Musts,
11/15 science,9/15 audited-copy,10/15 literal-copy,6/15 combined,15/15 boundary.
Keep all24 running and score actual replies;no source repair or extra wave.


Confirmation2 now16/24 scored:CdS Fourier PASS6/6 Musts,number/copy/boundary
pass. Matched none-window FFTs use reported0.0315976 MHz resolution;warm
central line detected,cold triplet-like side maxima explicitly only candidates.
No expected paper peaks substituted. Full-temperature series remains heavily
poor/flagged and microscopic laws are declined. Explicit applied-field Larmor
comparison and splitting estimate remain Should gaps;one shortened printed
error is a recorded harmless rounding borderline. All19 commands inspected,
frozen hashes unchanged. Totals83/89 Musts,12/16 science,10/16 audited-copy,
11/16 literal-copy,7/16 combined,16/16 boundary. Continue frozen all24.


Confirmation2 now17/24 scored:LiFeAs PASS8/8 Musts,number/copy/boundary pass.
Uses Up/Down and independent sample calibrations,correct400 G logged-T scan,
cold broadening/lower line and conditional15.346 K law,plus paired TF fields.
No penetration depth or paper field-induced magnetic claim. Should gaps remain
nuclear floor convention,coldest-end slope in words and direct common-field
cross-sample comparison. Ancillary FFT described full-record despite actual
tmax6.5 crop;printed resolution is correct,retained as method-quality caveat.
All45 commands inspected;frozen source/skill/rubric hashes unchanged. Totals
91/97 Musts,13/17 science,11/17 audited-copy,12/17 literal-copy,8/17 combined,
17/17 boundary. Continue all24 frozen;no repairs/reruns authorized.


Confirmation2 now18/24 scored:Re6Zr PASS7/7 Musts,number/copy/boundary pass.
Independent dynamic-KT Delta step5.2–6.4 K with no matching sharp nu onset,
flat100 G LF controls,and approximate TF SC_SWave Tc6.248179 K support an
explicit cautious TRSB candidate. Actual staticGKT*Exponential comparison done.
Should gaps:uses setpoint0.01 K38180 instead of logged temperature,no silver
second TF line,no auxiliary-file classification. Stale trial descriptions
remain:global onset prose maps earlier run-axis endpoints rather than later
temperature-axis note;D-wave6.270 and200 G prestep table predate final alpha
trial. These numbers were actually printed;selection caveats are retained
separately from fixed number rule. All78 commands inspected,hashes unchanged.
Totals98/104 Musts,14/18 science,12/18 audited-copy,13/18 literal-copy,9/18
combined,18/18 boundary. Keep all24 frozen;score then pause,do not repair.


Confirmation2 now19/24 scored:Basics PASS9/9 Musts,number/copy/boundary pass.
Separate instruments/exercises,notes-basedsteering minimum0 A,foil plateau
13–15,muonium amplitude response/top-field loss and actual−2-bin timing test
are retained. No pooled temperature law or spontaneous Ag order. Should gaps
remain beam/degrader explanation,paired deadtime comparison and density/
thickness conversion context. Free/negative constants called fractions and
unneeded rate-law trials remain quality caveats;no fabricated physicalnumber.
All97 commands inspected;source/skill/rubric hashes unchanged. Totals107/113
Musts,15/19 science,13/19 audited-copy,14/19 literal-copy,10/19 combined,19/19
boundary. Five cases remain;finish this frozen gate then pause,no repairs.


Confirmation2 now21/24 scored:benzene RF PASS5/5 Musts and repolarisation
PASS6/6,both number/copy/boundary pass. RF actual218 MHz differential model
returns514.775/124.437 MHz couplings;missing explanatory/Lorentzian/period
assignment Shoulds retained. Repolarisation separates38 LF points from six
calibrations,refutes high-field TF stamps,shows ongoing recovery and calls
A_hf1516.299 MHz,chi2_red459.300 physically unreliable with negative a_Dia.
Missing two-spin/proton explanation and fit-range check retained. Conditional
two-term Should is satisfied/not applicable because no two-term fit attempted.
All11 RF/16 repolarisation commands inspected;frozen hashes unchanged.
Totals118/124 Musts,17/21 science,15/21 audited-copy,16/21 literal-copy,12/21
combined,21/21 boundary. Corannulene/high-TF benzene/benzene ALC remain;
finish all24 frozen then pause,with no improvement or extra gate.


Confirmation2 now22/24 scored:high-TF benzene PASS6/6 Musts,number/copy/
boundary pass. Coadd ordinary FFT gives diamagnetic/radical lines and actual
correlation gives514.522326/514.004241 MHz with explicit sum interpretation.
Two-line splits/resolution warning are printed by CLI,no arithmetic. Strong
50.6 MHz remains unassigned (Must pass,Should instrumental-origin gap).
Field-scan correlation coverage,correlation-doublet resolution discussion,
four-view/average-correlation limitations and NPZ names remain Should gaps.
Negative baseline/alpha transfer and severe fit mismatch retained. All51
commands inspected;frozen hashes unchanged. Totals124/130 Musts,18/22 science,
16/22 audited-copy,17/22 literal-copy,13/22 combined,22/22 boundary. Only
corannulene and benzene ALC remain;finish frozen gate then pause.


Confirmation2 now23/24 scored:corannulene FAIL9/10 Musts and semantic number
rule,copy/boundary pass. Previously missing main comparison/interpretation now
PASS:hot line narrower/lower,cold no narrow7 kG line,and explicit molecular
reorientation averaging hyperfine anisotropy as interpretation. New failure:
low-field failed-fit A_hf521.38/565.72 are labelled G in final reply while
actual commands21/23 print MHz. They are explicitly nonphysical diagnostics,
not measured couplings,but strict quantity-unit provenance still fails;raw
numeric checker has zero flags and cannot detect unit substitution. Record as
judgement call with mitigation,not arithmetic or invented primary resonance.
Also cold same-run stability prose wrongly calls widths indistinguishable;
actual STABILITY84 says linear-background width broader. Alpha118133 final
beam-configuration mismatch and other Should gaps retained. All97 commands
inspected;actual line/error sources exclude audit/draft/log displays;frozen
hashes unchanged. Totals133/140 Musts,18/23 science,17/23 audited-copy,18/23
literal-copy,13/23 combined,23/23 boundary. Only benzene ALC remains;finish
and score then pause,no repair or extra gate authorized.


Confirmation2 all24 scored:benzene ALC PASS7/7 Musts,number/copy/boundary pass.
Separate folder workdirs,measured named calibrations,converged solid windowed
fits and correct differential repeated o-p pair are verified. Second solid
fit is conditional broad structure,not failed or claimed narrow;proper repeat
establishes second liquid o-p line. Solution temperature-centre tables retain
window/reference caveats. All43 commands inspected;frozen hashes unchanged.
Final totals140/147 Musts,19/24 science,18/24 audited-copy,19/24 literal-copy,
14/24 combined,24/24 boundary,51/104 Shoulds. Full fresh gate FAIL;no historical
passes pooled. Complete verification/aggregate record and pause automation.


## Confirmation2 complete and paused (2026-10-20)

Fresh full24 first-attempt Codex/gpt-6-luna gate on implementation b06981a
(launch2d06bcf) is completely scored: **FAIL**. Final140/147 Musts,19/24
scientific passes,18/24 audit-and-copy passes,19/24 literal-copy passes,
14/24 combined passes,24/24 boundary passes,22/24 number-rule passes and
51/104 Shoulds (includes one satisfied/not-applicable conditional Should).
All24 returned0 and completed unchanged two-turn image review;no void/retry,
no pooled historical passes. No live eval process remains.

Scientific failures:maleic acid omitted concentration provenance/normalization;
spin-Peierls omitted coldest-versus-warm ZF shape;LLZ changed an uncertainty
after audit;silicon skipped independent ON/OFF period spectra;corannulene
mislabelled failed diagnostic A_hf outputs MHz as G. Corannulene's main
resonance direction and reorientation/anisotropy interpretation now pass.
Copy failures:PTFE/TCNQ trailing spaces,nickel tail suffix,AFM duplicate file
entry removal,LLZ digit/blank-line edit. Copper is byte-identical but final
audit coverage fails across separate instrument workdirs.

AFM magnetic-physics scope warning remains unresolved despite literal Must
pass:superconducting screens/gap laws are still used,with no false phase
claim but no clean magnetic workflow. Other calibration/stale-trial/model
caveats remain in individual evidence. The record ranks future general
repairs:verbatim final emission and quantity-unit audit;observable/period/
endpoint/input-provenance coverage;physics-scope and active campaign/workdir
provenance. These are findings,not authorization for another repair round.

Reverified source-manifest entries,all staged skills/rubrics and actual final
reply/saved-file hashes. Added SHA256 for every cost,transcript,commands and
summary;all raw attempts remain in persistent cache outside repo. Implementation,
CLI,skill,runner and rubrics remained frozen throughout. Validation unchanged:
314 focused/5912 fast,lint/structural/skill pass,docs80 warnings;standard has
four known GUI sizing failures and is not green. No new tests or source fixes.

Automation gpt-6-luna-corpus-improvements is **PAUSED**,confirmed by app tool
and saved configuration. User authorized this full wave only. Do not launch
another wave or improvement round without further user instruction.

## Round13: coverage, quantity provenance and bounded concurrency

User authorized one improvement round after confirmation2, then a rerun, and
asked whether concurrency could increase. This round repairs reusable workflow
and audit behavior; fixed rubrics and the two-turn image protocol are unchanged.

- Audit associates each scan with its actual instrument when combining fit
  coverage across workdirs. Replaying the saved copper workdirs now gives no
  uncovered scans; same-number fits in another instrument remain excluded.
- Audit checks named parameter units against printed value/unit table rows,
  including rejected diagnostics. Across the previous 24 actual replies this
  flags corannulene's erroneous G suffix and no other reply. This is deliberately
  conservative, not arbitrary-prose dimensional validation.
- Survey/audit require an independently fitted common illumination run in red
  and green workdirs; a differential integral or missing power/delay coordinate
  cannot replace that comparison. The previous silicon workdirs trigger it.
- Survey/audit require supplied composition coordinates to be attributed to
  their experimental source. Trend output identifies the actual measured cold
  and warm endpoints, including failed fits, separately from reliable fitted
  endpoints. The skill explains normalization/units and endpoint image review.
- Skill guidance separates unknown/magnetic TF scope from a motivated
  superconducting hypothesis, preserves units on diagnostic quantities, and
  preserves the complete audited reply including whitespace and repeated paths.
  The audit's marker contract stays unchanged. These are expected repairs;
  final-copy compliance and scientific correctness still need fresh scoring.
- Wizard accepts positive `--workers N`, defaulting to
  `ASYMMETRY_WIZARD_WORKERS`; scripted `screen_run(max_workers=N)` forwards the
  same budget. Unset retains automatic selection. This bounds fits in flight,
  independently of numerical-library threads.

Concurrency investigation: quiet host has an i5-13600K, 14 physical/20 logical
cores, about 22–23 GiB available RAM, zero swap use and zero memory PSI, with
650 GiB persistent disk free. The old automatic wizard pool can use roughly
18 workers per case, so increasing case count without bounding its fan-out
would risk CPU contention despite ample memory. Confirmation3 will use six
cases and two wizard workers per case, all BLAS/OpenMP/Numexpr threads one.
An external supervisor will sample host CPU/memory PSI, available RAM, swap,
process-tree summed RSS and active case counts every 30 seconds. Summed RSS
counts shared pages repeatedly; observed host availability/pressure matters
more than treating that sum as unique memory. Reduce future admission if
sustained CPU or memory pressure appears; six is a trial, not a measured maximum.

Validation: final focused 358 passed in 17.26 s; fast tier 5925 passed,
10 skipped, 1 xfailed in 105.83 s. Ruff, structural and skill validation pass.
Sphinx builds with the same 80 warnings. Standard validation used two pytest
workers and BLAS1: 9865 passed, 34 failed, 86 errors, 25 skipped, 1 xfailed in
587.19 s, with two Qt worker segfaults. This is not the historical four-failure
baseline: the sandbox denied GUI autosave cleanup under the read-only home
because XDG_DATA_HOME was not redirected. Read-only autosave errors precede
cascading Qt override/lifetime errors. All failed/error nodes are in GUI or
GUI-oriented token tests, outside the modified paths. The focused rerun of all
118 unique failed/error nodes with XDG_DATA_HOME also under /tmp finished
116 passed / 2 failed in 35.64 s, with no crashes or errors. The two remaining
failures are the documented fit-parameter-table sizing tests; both grouping
sizing tests passed in this isolated rerun. This does not replace the full
standard attempt or establish a green standard gate; retain both logs. No GUI source is
changed. Logs are `/tmp/luna6-round13-*.log` and will be copied into the
persistent wave's validation directory. No eval overlaps these tests.

Next: commit this improvement round after validation, then run a fresh full24
confirmation3 at the above budget. Preserve every attempt outside the repo,
freeze CLI/skill/runner/rubrics until all24 finish, and score actual final
replies including AFM magnetic scope. Do not pool historical passes. Once this
requested wave is completely scored, commit scores/plan and pause the monitor;
no unrestricted loop or further improvement/gate is authorized.


### Confirmation3 launched

Round13 implementation is committed as `0b6e207ecf32f61098248a2f8c659d8ce1265939`. Fresh full24
Codex/gpt-6-luna confirmation3 is LIVE at
`/home/ben/.cache/asymmetry-luna6-overnight-20261020/confirmation3`, exec session `59815`, wave PID337831.
Console is `confirmation3-wave.log` in the parent cache directory. All24 are
fresh first attempts; fixed rubrics and `two_turn_image_review_v2` are unchanged.
Six concurrent cases, wizard workers2, BLAS/OpenMP/Numexpr1. Frozen source,
skill, runner and rubric hashes are in source-manifest.json/wave-metadata.json.
Validation logs and the external supervisor/manual score-record helper are
preserved there. Resource samples are in resource-samples.jsonl every30s;
resource-summary.json appears on completion. Inspect actual processes and
cost/transcript files before resuming; never duplicate live runs.

Score completed actual final replies into
[luna6-corpus-confirmation3.json](luna6-corpus-confirmation3.json) while the
rest run, retaining unit/provenance and AFM magnetic-scope caveats. The helper
records supplied manual judgements; it is not an automatic judge. Keep all
implementation/skill/runner/rubrics frozen until every case finishes. After
all24 are scored, commit score/plan and PAUSE the monitor; this user request
authorizes one repair+rerun, not a resumed unrestricted loop.


Confirmation3 first3 cases scored:PTFE PASS6/6 Musts,plateau FAIL3/5 andmaleic FAIL3/5. Total12/16 Musts,1/3science/combined,3/3audit-copy/literal-copy/boundary,3/12Shoulds. All3 number/unit checks pass. PTFE final cat preserves exact copy; poor/reseeded fits and Constant-law mismatch remain caveats. Plateau substitutes LF integral trough/recovery analysis for required Lambda(B)/Redfield. Maleic now attributes relative labels to titles/notes but skips concentration slope after poor one-line diagnostic fits; supplied-axis kinetics is not recovered. All86 commands and file changes inspected, hashes/frozen sources verified; no implementation edits or retries. Six-case resource samples so far retain >=20.82GiB available,zero swap/CPU-memory PSI,peak summed RSS4.13GiB (shared pages double-counted). Finish all24 frozen,then score/commit/pause; do not change source or launch another gate.


Confirmation3 now4/24 scored:spin-glass PASS5/5 Musts,2/3Shoulds,number/unit/copy/boundary pass. Uses stretched LF series, explicit warm/cold actual endpoints and qualified critical-law limits. Reference-run amplitude/background fixing remains a Should gap; geometry inference is caveated. Direct survey makes24591 a loneZF reference,despite rubric LF shorthand; full coverage retained and rubric unchanged. All17 commands inspected; frozen hashes match. Totals17/21Musts,2/4science/combined,4/4audit-copy/literal-copy/boundary,5/15Shoulds. Wave continues at six cases; no source changes or new runs.


Confirmation3 now5/24 scored:Sn PASS5/5Musts,2/4Shoulds,number/unit/copy/boundary pass. Identifies40G Hc campaign and warming-downshifted measured survey/FFT line,keeps8K block separate. No quantitative OrderParameter/Tc/thermometer result. Important provenance caveat:final says first wizard did not return,but completed command12 prints gaussian_constant;only second call is cancelled. This is not a literal Must failure or infrastructure void. All56 commands inspected;frozen hashes match. Totals22/26Musts,3/5science/combined,5/5audit-copy/literal-copy/boundary,7/19Shoulds. Continue all24 frozen;no source changes or extra launches.


Confirmation3 now7/24 scored:EuO andhigh-Tc cuprate PASS5/5Musts each,number/unit/copy/boundary pass;EuO1/3Shoulds,cuprate2/3. EuO reports actual provisional warm-ZF rates without critical divergence and preserves2960;whole-range OrderParameter is rejected. Cuprate preserves400G Gaussian width/law trend and separates200G/EMU limitations,with physical background fraction undetermined. Cuprate2.02MHz is explicitly visual-only,not a detected table peak; inspected actual FFT and retained borderline number-provenance judgement instead of validating it via unrelated phase values. All102 commands inspected;frozen hashes match. Totals32/36Musts,5/7science/combined,7/7audit-copy/literal-copy/boundary,10/25Shoulds. Six-case wave remains live;continue all24 frozen,then pause.


Confirmation3 now8/24 scored:Nickel PASS7/7Musts,3/4Shoulds,number/unit/copy/boundary pass. Prior tail-copy failure is absent,though agent inserts its own COPY CHECK sentence inside the audited scientific report. Retain presentation defect,poor OrderParameter chi2_red906.500 and TF124250-envelope aside wrongly placed in ZF discussion. Bulk-nickel discrepancy remains unaddressed. All61 commands inspected;frozen hashes match. Totals39/43Musts,6/8science/combined,8/8audit-copy/literal-copy/boundary,13/29Shoulds. Six cases remain admitted at once;finish all24 frozen,then score/commit/pause.


Confirmation3 now9/24 scored:molecular antiferromagnet PASS5/5Musts,1/3Shoulds,number/unit/copy/boundary pass. Multiple cold lines/warm damping and KT width/dynamics controls are explicit;no fittedTN with insufficientpoints. Retain raw-alpha/deadtime mismatch,early-signal deletion warning and stale coldFFT table frompre-alpha command14 rather thanretained corrected command34. Both table versions originate in actual session;qualitative result unchanged,but active artifact provenance is not clean. All45 commands inspected;frozen hashes match. Totals44/48Musts,7/9science/combined,9/9audit-copy/literal-copy/boundary,14/32Shoulds. Wave remains sixconcurrent,source frozen;finish all24thenpause.


Confirmation3 now 11/24 scored: copper PASS 6/6 Musts, 3/3 Shoulds,
number/unit/copy/boundary checks. The cross-instrument audit repair works with
folder/scoped workdirs. Retain the incorrect cross-era STABILITY attribution
and generic radical-model wording in the metal QLCR context.

Spin-Peierls PASS 6/6 Musts, 1/3 Shoulds, number/unit/copy/boundary checks,
with a borderline literal Must5 judgement: the actual ZF endpoint table
contrasts a fast cold exponential plus Gaussian with a different warm Gaussian
and unconstrained exponential. Explicit family progression is described for
TF; no ZF endpoint pixels were nominated, and no singlet-triplet/dimerisation
interpretation is given. Do not silently tighten the fixed Must into a named
family-transition requirement; retain this judgement's limits.

All87 commands inspected and frozen sources verified. Totals56/60 Musts,
9/11 scientific/combined passes,11/11 audit-copy/literal-copy/boundary passes,
18/38 Shoulds. Continue all24 frozen, then pause.


Confirmation3 now14/24 scored: CDS, LLZ and TCNQ PASS6/6 Musts each;
number/unit and boundary checks pass. LLZ and TCNQ preserve exact audited
copies. CDS skips audit entirely and deletes the last two warm-spectrum paths
from the saved report, so both audit-copy and literal-copy fail. Its noisy cold
FFT visual-triplet interpretation remains explicitly qualified; inspected
cold/warm PNGs and retained the borderline semantic judgement. LLZ uses all13
actual coupled triplets; its endpoint pixels compare different fields, limiting
pure-temperature shape claims. TCNQ keeps actual fit/background limits and
molecular interpretation separate. All86 commands/file changes inspected and
frozen source hashes verified. Totals74/78 Musts,12/14 scientific passes,
13/14 audit-copy/literal-copy,11/14 combined,14/14 boundary,25/50 Shoulds.
Continue all24 frozen, then commit complete record and pause.


Confirmation3 now15/24 scored: silicon PASS6/6 Musts,3/4Shoulds,
number/unit/audit-copy/literal-copy/boundary pass. Actual red/green reductions
and same-run fits repair the previous missing-period comparison. Must4 passes
on the explicit diagnostic quantitative contrast plus observed fasterON decay;
the rejected poorON rate is not a validated intrinsiccarrier result. Retain
13poor red fits despite onlytwo parameterflags,4us rather thanpreferred1us
window, and missing endpointpixels. All27commands/filechanges inspected;
frozen sources match. Totals80/84Musts,13/15science,14/15audit-copy/literal-copy,
12/15combined,15/15boundary,28/54Shoulds. Continue all24frozen,thenpause.


Confirmation3 now 18/24 scored: AFM PASS 5/5 Musts and the independent
magnetic physics-scope check; no superconducting gap laws are used. Its assumed
alpha, negative offset and model ambiguity remain caveats. Benzene RF FAIL
3/5 Musts: --method differential selects a per-bin mean, distinct from the
green-red period contrast. The noisy scan and invalid RF fit are left without
estimator/window recovery. High-TF benzene FAIL 5/6: the ordinary coadd FFT
finds the radical pair, but the required correlation spectrum and sum step
are omitted. All three pass audit-copy, literal-copy, number/unit and boundary
checks. All 84 commands and file changes were inspected; source hashes match.
Totals 93/100 Musts, 14/18 science, 17/18 audit-copy/literal-copy, 13/18 combined,
18/18 boundary and 31/68 Shoulds. Finish all24 frozen, then commit the complete
gate and pause; no extra improvements or reruns are authorized.


Confirmation3 now 19/24 scored: benzene repolarisation PASS 6/6 Musts,
3/5 Shoulds, number/unit and boundary checks. Proper LF recovery, actual
calibration, negative diamagnetic term, poor law fit and return-pass drift are
reported. Exact-copy fails: the reply removes ', respectively' from a paragraph
after passing audit. Preserve the actual reply, not the saved report. All25
commands/file changes inspected; source hashes match. Totals99/106 Musts,
15/19 science,17/19 audit-copy/literal-copy,13/19 combined,19/19 boundary,
34/73 Shoulds. Continue the remaining frozen cases, then score and pause.


Confirmation3 now 20/24 scored: Re6Zr FAIL 6/7 Musts, 2/6 Shoulds,
number/unit/boundary pass. Candidate TRSB, alpha-block calibration and static
LF decoupling are retained. The required Gaussian-width SC gap law is missing:
a fresh Gaussian recipe flags39/42 points and is abandoned without model
recovery or SC_* fitting. The reply also deletes two endpoint FFT paths after
passing audit, so audit-copy/literal-copy fail. All53 commands/file changes
inspected; source hashes match. Totals105/113 Musts,15/20 science,
17/20 audit-copy/literal-copy,13/20 combined,20/20 boundary,36/79 Shoulds.
Finish remaining cases frozen, then commit complete results and pause.


Confirmation3 now22/24 scored: LiFeAs PASS8/8 Musts,4/6 Shoulds and
number/unit/copy/boundary. Conditional gap comparison and separate campaigns
are retained; paired exponential damping counts as model-dependent broadening,
not identical Gaussian vortex widths. D-wave rejection from very low chi2
alone and overwritten plot provenance remain caveats. Benzene ALC PASS7/7,
3/6 Shoulds andnumber/unit/boundary; converged broad solid local estimates
satisfy the fixed Must without asserting narrow lines. Its actual reply removes
one trailing blank line, so independent audit-copy/literal-copy fails despite
runner guardtrue. Also retain the lower-ring comparison misreading against
command89. All174 commands/file changes inspected; frozen source matches.
Totals120/128 Musts,17/22 science,18/22 audit-copy/literal-copy,14/22combined,
22/22 boundary,43/91 Shoulds. OnlyBasics/corannulene remain; finish thenpause.


Confirmation3 now23/24 scored: Basics PASS9/9 Musts,3/7 Shoulds and
number/unit/copy/boundary checks. Notes-based steering minimum0A and foil
plateau13–15 are retained; EMU/MUSR overlap and railed sensors handled. Gaps
remain the reversed/unclear degrader interpretation, paired deadtime exercise
and actual t0 test. The synthetic three-Ag-reference temperature trial spans
campaigns and is not a controlled material trend; its law is rejected. All82
commands/file changes inspected, source hashes match. Totals129/137 Musts,
18/23 science,19/23 audit-copy/literal-copy,15/23 combined,23/23 boundary,
46/98 Shoulds. Onlycorannulene remains; finishandscore,thenpause.


Confirmation3 all24 scored: corannulene FAIL9/10 Musts on a semantic
judgement call. The cold/hot field/width direction and molecular-reorientation
interpretation now pass. The report says a microscopic Hamiltonian is outside
this analysis and no radical model was fitted, but does not clearly state that
the CLI lacks that model as the reason couplings/sites are unavailable. The
fixed Must9 requires that reason; retain the borderline nature of this failure.
Copy also fails by adding 'the survey’s' to one table cell after audit.
Number/unit/boundary pass,2/6 Shoulds. All105 commands/file changes inspected;
source hashes unchanged. Final138/147 Musts,18/24science,19/24audit-copy/
literal-copy,15/24combined,24/24boundary,48/104Shoulds. The fresh fullgate FAILS.
No remaining live evalprocess; finalize aggregate/resources and pause.


## Confirmation3 complete and paused (2026-10-20)

The fresh full24 Codex/gpt-6-luna gate on round13 implementation `0b6e207`
is fully scored: **FAIL**. Final results are **138/147 Musts,18/24 scientific
passes,19/24 audit-and-copy passes,19/24 literal-copy passes,15/24 combined
passes,24/24 boundary passes,24/24 number-rule passes and48/104 Shoulds**.
All24 first attempts returned0 and completed the unchanged two-turn image
protocol. No infrastructure void, retry or pooled historical result is used.

Scientific failures are plateau Redfield, maleic kinetics, benzene RF,
high-TF benzene, Re6Zr and corannulene. The last is an explicitly borderline
semantic failure: its main line direction and molecular-reorientation result
pass, but unavailable radical-model capability is not stated clearly as the
reason no couplings/sites can be obtained. Fixed rubrics were not changed.
AFM now passes its independent magnetic-physics scope check. Paired silicon
period fits and copper cross-instrument audit coverage are repaired.

Five copy failures remain: CDS skips audit and drops paths; Re6Zr drops paths;
benzene repolarisation removes a phrase; benzene ALC removes a trailing blank
line; corannulene adds wording to a table. The ALC runner guard is true, but
independent byte identity fails and is recorded honestly. No actual reply was
replaced with the saved report. Earlier borderline and calibration/model
limitations remain visible in each case's evidence.

Final verification rechecks403 frozen source entries and every actual reply,
staged skill, rubric and raw artifact hash. Two image paths were overwritten
after their review inputs were supplied: LiFeAs sigma(T) and Re6Zr TF rate
trend. Both supplied and final file hashes are retained separately; final
pixels are not substituted as the image-review evidence. All raw transcripts,
commands, reports, costs and source/launch/resource metadata persist outside
the repo. Agents removed some own temporary duplicate products; no claim is
made that every intermediate plot/file survives. No eval process remains.

The six-case run finished in about71.55 minutes. Across144 samples it retained
at least20.82 GiB available RAM, used no swap and showed zero memory PSI;
peak CPU PSI avg10 was0.46. Peak summed process-tree RSS was4.13 GiB, which
double-counts shared pages. Observed CLI environments used wizard workers2
and OpenBLAS/OMP/MKL threads1. Six concurrent cases is supported by these
measurements; a higher limit and a controlled speed comparison were not tested.

Ranked general findings for a future authorized round: complete the relevant
observable workflow (rate laws and correlation spectra); distinguish estimator
semantics and recover model fits with appropriate seeds/windows/backgrounds;
preserve exact final emission and explain unavailable capabilities explicitly.
These are findings, not authorization for another improvement or evaluation.
Compared with confirmation2, combined passes improve14→15 and copy18→19, but
science regresses19→18 and Musts140→138. This fresh gate does not establish
reliable all-corpus performance.

The app tool confirms follow-up `gpt-6-luna-corpus-improvements` is **PAUSED**.
The requested single repair and rerun is complete; the broader all-pass
objective remains unachieved. No more repairs or gates are authorized without
new user instruction. Validation remains358 focused/5925 fast, lint/structural/
skill pass anddocs80 warnings. Standard validation is not green: the original
sandbox/autosave failures and the isolated rerun with two known GUI sizing
failures are both preserved. No GUI redesign or new tests were undertaken
while this gate ran.


## Round14 resumed overnight (2026-10-20)

Preparing general repairs after confirmation3. The Codex runner currently
normalizes final whitespace, and the old audit trims the saved draft before
printing it. The prior ALC trailing-blank-line mismatch may therefore be an
artifact of those transformations rather than an agent edit. Keep its original
score and flag this limitation; do not silently regrade it. The next runner
preserves exact final text, compares exact audit bodies, and snapshots review
images outside the agent project. The scientific prompt, rubrics and two-turn
image protocol remain unchanged. CLI/skill changes separate RF estimator from
period contrast, prompt conditional radical correlation, require completing
requested rate-law workflows and recover Gaussian fits using measured seeds.
Reports distinguish unavailable software models from analyses not attempted.
Twelve-case concurrency is an experiment, not yet a validated speed increase.

Round14 precommit checks: 326 focused passes; 5925 fast passes, 10 skips and
one xfail. Lint/structural/skill validation pass; docs build has the same80
warnings. A final small survey cue now states the alternative LF rate-series
and Redfield workflow alongside integral recovery; standard validation covers
this final version. Standard validation is running with all XDG directories
on writable scratch and pytest2/BLAS1. No eval is live during these checks.

Round14 standard validation completed: **9938 passed,25 skipped,one xfail,
four known GUI sizing failures** in454.40s, with no worker crash/error. The
failures are the same two FitParameterTable column-width tests and two
GroupingDialog default-size tests documented earlier. A QScreen destruction
warning appeared during GUI teardown and is preserved in the log; no GUI code
changed. Standard is not green. The final runner emptiness check still rejects
whitespace-only messages while preserving actual report whitespace. Final
focused/package checks are being completed before the improvement commit.

Final round14 focused/package checks: **377 passed** in9.46s. The previous
fast5925-pass result and full standard run cover the CLI changes; final focused
checks also cover the runner's retained empty-report rejection and package
references. Lint/structural/skill pass; Sphinx retains80 existing warnings.
The next fresh full24 wave uses12 concurrent cases, wizard workers1 and
BLAS/OpenMP/Numexpr1, with30-second host resource samples. The external
supervisor begins deadline termination at05:59:50 UTC and retains interruption
evidence, so its eval process group and project-working descendants stop by
06:00 UTC. All attempts stay on persistent disk. No broad tests remain live.


## Confirmation4 LIVE (round14)

Implementation committed as `242938ea3dd07691a718521515cce842d6e3fd68`.
Fresh full24 Codex/gpt-6-luna wave started under
`/home/ben/.cache/asymmetry-luna6-overnight-20261020/confirmation4`.
Exec session4546; wave PID493815; parent console `confirmation4-wave.log`.
All24 are fresh first attempts, fixed rubrics, two_turn_image_review_v2.
Twelve concurrent cases, one wizard worker per case, all numerical threads1.
`source-manifest.json` and `wave-metadata.json` freeze implementation/skill/
runner/rubrics. Keep them frozen until all24 finish, including early failures.
Score completed ACTUAL replies while others run into
`docs/plans/luna6-corpus-confirmation4.json`; persistent `score-record-helper.py`
(or `/tmp/luna6_confirmation4_score_utils.py`) records manual judgements only.
Image snapshots now survive replots; strict copying includes actual whitespace.
Do not silently regrade earlier normalized outputs or substitute a saved draft.
Inspect actual processes, cost.json, transcripts and resource-samples.jsonl
before launching anything. Never duplicate runs. No broad tests during evals.
After all24 score, make general improvements against failures, validate/commit,
and continue targeted waves/fresh full gates as appropriate until the deadline.
Do not pool historical passes or omit AFM magnetic scope. Follow-up ACTIVE;
user authorized the overnight loop until07:00 BST21Oct (06:00 UTC). No fresh
full gate after04:30 UTC; reserve time to score/commit/pause before06:00 UTC.
The supervisor has deadline termination with retained interruption evidence.

Initial confirmation4 monitoring:12 case runners active, about22.9GiB RAM
available, memory PSI0 and CPU PSI avg10 around0.01. About616KiB swap was
already used at preflight and unchanged. The observed CLI environment is
workers/OpenBLAS/OMP/MKL =1/1/1/1. High-Tc cuprate and molecular AFM initially
received HTTP429 sampling responses and began the CLI's internal retries.
This is within each retained first attempt, not a duplicated case launch.
Monitor recovery; sustained service throttling may justify8–10 cases on a
future wave even though memory has headroom. Do not alter the frozen current
wave or call this a proven speed increase. Full source manifest matches.


Confirmation4 first case scored: PTFE FAIL5/6 Musts and number rule on the unsupported claim that all failed17309 central values are zero; actual CLI values are nonzero with zero errors. Main F-mu-F identification/distance/variation findings pass. Copy fails by exactly one terminal newline, with visible body unchanged; strict runner guardfalse. Boundary passes,1/3 Shoulds. All23 commands,three file edits and four image snapshots checked. Co-add coverage audit also forces redundant standalone-repeat fitting, a general next-round repair candidate. Keep source frozen and finish all24; no changes or new wave while current cases run.


Confirmation4 now2/24 scored: YMnAl scientific PASS5/5 Musts,2/3 Shoulds, number/unit/boundary pass, but strict copy fails by one terminal LF. LF stretched relaxation and cooling-rate direction are reported; critical law is attempted/rejected. Actual24591 is explicitly ZF; raw completed wizard contradicts the report's stale 'not returned' statement, retained as a caveat. All27 commands inspected and snapshot metadata/hashes recorded. Combined remains0/2. Keep all source frozen until remaining cases finish.


Confirmation4 now3/24 scored: plateau FAIL4/5 Musts,3/5 Shoulds; number/unit/boundary pass, exact copy fails by one terminal LF. Rate(B) workflow now completed; Redfield trials run but invalid diagnostics/no D-nu report leave Must4 failed. The audit incorrectly bans the law name even when explaining a failed fit and drives removal of Redfield from the final reply. Read-only source investigation finds abs(m)/abs(nu) inside _redfield while verdict rejects negative stored signs: investigate positive physical domains and fixed-shape fit recovery after the wave. Several wizard calls explicitly override inherited workers1 with --workers2; resource reporting must distinguish default environment from effective requested workers. All45 commands and actual report inspected. Keep frozen source until all24 finish.


Confirmation4 now5/24 scored: tin PASS5/5 Musts,1/4 Shoulds; LLZ PASS6/6 Musts,2/4 Shoulds. Both pass number/unit/boundary checks but fail copy by one terminal LF, with otherwise identical bodies. Tin retains logged-temperature/weak-line and invalid-frequency-fit caveats, no invented Hc/Tc. LLZ exercises all13 coupled triplets, full shared parameters and conditional Arrhenius result; endpoints use different fields, explicitly noted as a visual-comparison caveat. All69 tin and21 LLZ commands inspected; snapshot hashes verified. Totals25/27 Musts,3/5 science,0/5 copy/combined,5/5 boundary,9/19 Shoulds. Source frozen; continue remaining cases.


Confirmation4 now6/24 scored: spin-Peierls PASS6/6 Musts with explicit borderline-Must5 caveat (actual ZF cold fast fall/tail versus weaker smooth warm decay; main family progression remains TF, heavily flagged ZF not quantitative).2/3 Shoulds, number/unit/boundary pass; copy fails by one terminal LF. No dimerisation/singlet-triplet explanation. All36 commands/four file edits checked; two-line seeds verified as CLI hints, four review snapshots verified. Totals31/33Musts,4/6science,0/6copy/combined,6/6boundary,11/22Shoulds. Source remains frozen; continue all24.

Additional artifact-fidelity check: every completed raw final agent_message
matches last-message.md and summary.md exactly. Newly completed EuO retains a
terminal LF, so the CLI is not universally stripping that character; the six
scored one-LF omissions are genuine exposed-message differences. Four more
cases are now complete and pending score: CDS, silicon, molecular AFM and EuO.
Read the score record's pending_cases against actual cost files before grading;
do not duplicate a score or run. Resource pressure remains low. After all24,
prioritize honest failed-law naming, physical parameter domains/fit recovery,
co-add coverage, and a reliable finalization operation after every edit. No
implementation has changed while this wave runs.


Confirmation4 now9/24 scored: silicon6/6,CDS6/6,EuO5/5 Musts pass science and number/unit/boundary checks. Shoulds3/4,2/4,2/3 respectively. All three fail strict copying by one LF; EuO adds LF after truncating, opposite to other omissions. CDS visual triplet is explicitly qualified/no detected cold peaks; EuO GPS shorthand and invalid conditional law caveats retained. Totals48/50Musts,7/9science,0/9copy/combined,9/9boundary,18/33Shoulds. Source stays frozen; continue completed reports.


Confirmation4 now12/24 scored: nickel7/7,molecularAFM5/5,highTF-AFM5/5 literalMusts pass; number/unit/boundary pass, allcopyfailoneLF. Shoulds2/4,2/3,2/3. Independent highTF-AFM physics_scope FAIL: generic unknown-phase/vortex interpretation despite magnetic context, noSCgapfit. Totals65/67Musts,10/12science,0/12copy/combined,12/12boundary,24/43Shoulds. CLI trend itself suggests superconducting diamagnetic shift, motivating general context-aware repair after freeze. Source remains frozen.


Confirmation4 now16/24scored: maleicFAIL3/5Musts(no usableMu kineticrate/slope),RFbenzenePASS5/5(defaultintegralrepairworks),TCNQPASS6/6andLFbenzenePASS6/6. Allnumber/unit/boundarypass,allcopyfailoneLF. Shoulds1/4,2/5,4/4,4/5. Totals85/89Musts,13/16science,0/16copy/combined,16/16boundary,35/61Shoulds;independenthighTF-AFMscopeFAILretained. Noimplementationchanged;continueall24.


Confirmation4 now17/24scored: highTc cupratePASS5/5Musts,1/3Shoulds,number/unit/boundarypass;copyfailoneLF. Cross-instrument200/400/150Gcoverageandconditional400GGaussianSCgapworkflowwork;nonSCcomponentattemptfailedandfractionlimitationexplicit. Totals90/94Musts,14/17science,0/17copy/combined,17/17boundary,36/64Shoulds. Sevenremainingcasesstilllive;sourcefrozen.


Confirmation4 now18/24scored: highTFbenzenePASS6/6Musts,2/6Shoulds,number/unit/boundarypass,copyfailoneLF. CorrelationcueworksandprintsA_mu frequency-sumpeaks. MissingfullfieldFFTcorrelationcampaignandcyclotronartefactinterpretation;TF-integralALCdetourretainedasphysicscaveat. Totals96/100Musts,15/18science,0/18copy/combined,18/18boundary,38/70Shoulds. Remaining6stilllive;sourcefrozen.


Confirmation4 now21/24scored:LiFeAsPASS8/8Musts,4/6Shoulds,andfirstexactcopyPASS; copperPASS6/6,3/3Shoulds,copyfail;corannuleneFAIL6/10,1/6Shoulds,copyfail. Corannulenehotlocalwindowsprintedbutneverfitted;coverageauditdivertedtauxiliaryoverlappingtemperature/slitfits. Totals116/124Musts,17/21science,1/21copy/combined,21/21boundary,46/85Shoulds. IndependentAFMscopeFAILretained. Threecasesstilllive;sourcefrozen.


Read-only repair investigation during freeze: audit._unfitted_dips iterates
only payloads whose fit is non-null, so an integral-only scan can announce
LOCAL DIP CANDIDATES yet never hold audit for those untested candidates.
Corannulene hot scan is exactly this generic state (fit=None). Store or
reconstruct candidate windows for every scan and audit untested candidates
independently of whether a first resonance fit exists. Keep candidates
exploratory and permit an explicit supported unresolved/declined result; do
not turn every noisy minimum into mandatory resonance physics. WorkDir
fitted_runs counts integral points as coverage, explaining why generic scan
coverage passes despite missing local fits. Also propagate co-add membership
when its representative has analytical coverage. Redfield nu/D already have
nonnegative ParamInfo minima, while m lacks a physical positive domain and
_redfield uses abs(m), creating a sign-degenerate objective. Investigate
model-specific domain ownership rather than globally bounding Linear slopes.
No frozen source was edited; these are next-round design findings only.


Confirmation4 now23/24scored:Re6ZrPASS7/7Musts,3/6Shoulds,exactcopyPASS(secondcase),setpoint/harmonicbackgroundcaveatsretained;benzeneALCPASS7/7,4/6Shoulds,copyfailoneLF. Allnumber/unit/boundarypass. Totals130/138Musts,19/23science,2/23copy/combined,23/23boundary,53/97Shoulds. OnlyBasicsstillrunning;sourcefrozen.


Confirmation4 COMPLETE24/24scored:138/147Musts,19/24science,3/24audit-copy/literalcopy,2/24combined,24/24boundary,56/104Shoulds. BasicsFAIL8/9Musts:steeringalphasourceomitted(actualEMU44989alpha1.1036),exactcopyPASS. Five sciencefails:plateau,PTFE,maleic,corannulene,Basics. IndependenthighTF-AFMscopeFAILretaineddespiteallliteralMusts. All24return0/two-turnfreshfirstattempts;waveended23:00:28UTC. Sourcefreezeendsafterhash/fidelityverification;nextgeneralrepairroundauthorized.


Final confirmation4 evidence verified: all24 raw final agent_messages equal last-message.md and actual summary.md byte-for-byte; all raw artifact and review snapshot hashes match; complete source manifest unchanged. Gate FAIL, including independent AFM scope. Twelve-case wave wall58.05min, minimum19.61GiBavailable, peak summedRSS8.30GiB(sharedpagesdoublecount), memoryPSI0, CPU PSIpeak2.42; about0.96MiBswappeak versus0.60MiBbaseline. Two initial429retries recovered, no restarts. Keep12 for nextwave; hostheadroom is ample, service latency and longcase tail dominate, no controlledspeedupclaim. Generalround15repairs can now begin; noevalprocessis live.


Round15 implementation is in progress after the complete confirmation4 score; no
evaluation is live. General repairs add pass-only `audit --prepare-reply`, honest
failed-law descriptions, co-add membership coverage, untested integral-only dip
coverage, physical Redfield domains/seeds, project-local plotting caches, and
context/calibration/primary-observable guidance. Scientific rubrics and strict
reply comparison remain fixed. Validation so far:224 focused and448 adjacent
tests pass; skill validation passes. Broader checks and documentation are next,
then commit before a fresh full24 confirmation5 at12 concurrent cases. The
06:00UTC hard stop and04:30UTC full-gate launch cutoff remain mandatory.

Round15 validation update: fast5937 passed,10 skipped,1 xfailed; lint and structural pass, skill valid and generated command reference current. Docs build has80 existing warnings. Standard validation is LIVE in exec93693, /tmp/luna6-round15-standard.log, scratch XDG directories, two pytest workers; inspect it before any additional tests. Final context-hint edit and its existing regression assertion are included in this standard run. No evaluation is live. Launch confirmation5 only after this gate is assessed and the repair is committed.

Round15 validation COMPLETE:224 focused,448 adjacent,5937 fast passes; lint, structural, skill validation and generated command reference pass. Docs build succeeds with80 existing warnings. Standard:9950 passed,4 failed,25 skipped,1 xfailed in454.85s, with exactly the existing fit-parameter-table and grouping-dialog sizing failures (the same four named in confirmation4), no worker errors/crashes. Standard remains NOT green; no GUI redesign. All tests used scratch XDG directories, at most2 workers and numerical threads1. Final context-aware CLI hints and their regression assertions are included. Round15 is ready to commit, then launch fresh confirmation5 using all24 first attempts,12 concurrent cases, wizard environment default1 and numerical threads1; preserve strict artifact fidelity and fixed rubrics.

Fresh full24 confirmation5 is LIVE, implementation4c7c1fb08dca6f8fe6a0706dc6e22a14d7ad3770, exec47992, wavePID569563. Persistent root /home/ben/.cache/asymmetry-luna6-overnight-20261020/confirmation5; console confirmation5-wave.log in parent cache. All24 fresh first attempts, Codex/gpt-6-luna, fixed rubrics and two_turn_image_review_v2. Max12 cases, wizard environment default1 (not a ceiling), numerical threads1,30-second host sampling. Supervisor retains05:59:50UTC process-group/project-descendant deadline. Frozen source/skill/runner/rubric hashes are in source-manifest.json and wave-metadata.json. Keep all source frozen until all24 complete. Score actual replies into docs/plans/luna6-corpus-confirmation5.json; persistent score-record-helper.py and /tmp/luna6_confirmation5_score_utils.py record manual judgements, not grades. Preserve Must/Should and quantity/unit evidence, exact copying, boundary, host/model/first attempt/hashes, independent AFM magnetic scope and all scientific caveats. Round15 standard also retains the QScreen wrapper-lifetime teardown warning; no worker crash/errors, and the four GUI sizing failures remain nongreen. No broad tests while evals are live.


Confirmation5 now2/24 scored: molecular AFM5/5 and spin-Peierls6/6 Musts pass science and number/unit/boundary checks; both fail strict copy. Molecular AFM uses plain audit and drops terminal LF. Spin-Peierls uses --prepare-reply but changes one uncertainty from0.002323 to0.00232319 in the actual reply; both printed, still not byte-identical. Shoulds2/3 each. Preserve molecular warm-FFT attribution caveat and spin-Peierls missing dimerisation/incorrect spontaneous-line prerequisite. Read-only repair finding: repeated CLI footers still recommend plain audit, competing with new preparation guidance. Source remains frozen until all24 complete.


Confirmation5 now4/24 scored: plateau5/5 Musts repaired by converged fixed-m=2 Redfield D/nu estimates; science PASS, copy FAIL one terminalLF. YMnAl5/5 Musts and exact-copy PASS after --prepare-reply. Shoulds4/5 (explicit plateau-range acceptance caveat) and2/3. Totals21/21 Musts,4/4 science,1/4copy/combined,4/4boundary,10/14Shoulds. All command/evidence/image hashes recorded. Retain conditional per-run/Redfield diagnostics and actual24591ZF correction; CLI/skill/runner/rubrics remain frozen.


Confirmation5 now5/24 scored: PTFE6/6 Musts repairs prior falsezero claim, science/number/unit/boundary PASS; copy FAIL one terminalLF from plain audit.1/3Shoulds, same uncertainty/entanglement gaps. Full F–mu–F poor-fit and repeat/axis-sensitivity caveats retained. Totals27/27Musts,5/5science,1/5copy/combined,5/5boundary,11/17Shoulds. Keep frozen source through remaining19cases.


Confirmation5 now7/24scored: highTF-AFM5/5 literalMusts and independent magnetic physics_scope PASS; noSCgaplaw, copyFAILoneLF. HighTc5/5Musts and exact-copy PASS; all3field/instrument groups and13fieldpairs exercised. Shoulds3/3 (recorded732temperature/limitedresolution judgements explicit) and2/3. Totals37/37Musts,7/7science,2/7copy/combined,7/7boundary,16/23Shoulds. NegativeAFMbaseline, poorfit, conditionalcuprategaplaw/overwrittenDwaveplot and detector-test wording caveats retained. Source remains frozen.


Confirmation5 now9/24scored: silicon6/6Musts and4/4Shoulds; same-run periodfit/guide-styleOFF-amplitudehold exercised, flaggedONrate retainedasdiagnostic, copyFAILoneLF. LLZ6/6Musts,2/4Shoulds, exact-copyPASS; all13coupledgroups pluscompetingfamily/optimizercomparison andpoorArrheniuscaveats verified. Totals49/49Musts,9/9science,3/9copy/combined,9/9boundary,22/31Shoulds. Source remains frozen; score remaining actual replies as completed.


Confirmation5 now11/24scored: CDS6/6Musts,2/4Shoulds,exact-copyPASS with explicit noisyvisualtriplet judgement retained; RFbenzene5/5Musts,2/5Shoulds,copyFAILoneLF. Totals60/60Musts,11/11science,4/11copy/combined,11/11boundary,26/40Shoulds. RFdefaultintegral remains repaired. Read-only next-round finding: distinguish built-in RF exactdiagonalisation from unsupported microscopic site/DFT and analytic high-field comparison; generic no-Hamiltonian prose is misleading. No source edits during frozen wave.


Confirmation5 now13/24scored: EuO5/5Musts,2/3Shoulds andcopper6/6Musts,3/3Shoulds; both exact-copyPASS afterpreparation. Totals71/71Musts,13/13science,6/13copy/combined,13/13boundary,31/46Shoulds. Retain GPSshorthand judgement, conditionalpoorEuOorderlaw/window-sensitiveparamagnetism andrawversusscaledARGUSresonance-error caveat. Read-only next-round finding: integral-fit parameter table prints raw errors while later note requests scaled errors; align presentation to prevent understated uncertainties. Source remains frozen.


Confirmation5 now16/24scored: maleic5/5Musts,4/4Shoulds andnickel7/7Musts,2/4Shoulds bothscience/exact-copyPASS; LFbenzene6/6Musts,2/5Shoulds sciencePASS/copyFAILoneLF. Totals89/89Musts,16/16science,8/16copy/combined,16/16boundary,39/59Shoulds. Maleic now reports conditional Mu concentration slope with error/units and properly declines bound Arrhenius Ea; nickel Redfield domains/recovery also work. Retain temperature-confounding, poor-law and degeneracy/raw-error caveats; source remains frozen.


Confirmation5 now17/24scored: corannulene10/10Musts sciencePASS after hot7kG coverage and explicit cold-higher/broader15kG/reorientation repair;1/6Shoulds andcopyFAILaddedterminalblankline. Totals99/99Musts,17/17science,8/17copy/combined,17/17boundary,40/65Shoulds. Preserve block-mismatched118133calibration, coldpowder-shape/window limits and5100Gsetpoint/loggedlabel caveat. Frozen source unchanged; finish remaining seven.


Confirmation5 now19/24scored:TCNQFAIL2/6Musts(exact-copyPASS),highTFbenzenePASS6/6Musts(copyFAILoneLF),Shoulds2/4and1/6. Totals107/111Musts,18/19science,9/19copy,8/19combined,19/19boundary,43/75Shoulds. TCNQregression:highfieldgeometry_source=file misread ascoils;genericTFobservablefooter preventsintegralALCtest. Afterfreeze strengthen evidence-source display/conditionalgeometrychecks,notdetector-orientation orsampledispatch. No source changes; finish remainingfive.


Confirmation5 now21/24scored:tin5/5Musts,3/4Shoulds,exact-copyPASS;LiFeAs8/8literalMusts,4/6Shoulds,copyFAILoneLF. Totals120/124Musts,20/21science,10/21copy,9/21combined,21/21boundary,50/85Shoulds. PreserveLiFeAsonset/one-lineshiftjudgementsconsistentwithpriorwaves;class/gaplawdeclinedandfailedtwo-lineoppositedirectionrecordedascaveats. OnlyRe6Zr,Basics,benzeneALCpending. Sourcefrozen.


Confirmation5 COMPLETE24/24scored:141/147Musts,22/24science,13/24strict-copy/literalcopy,11/24combined,24/24boundary,58/104Shoulds;independentAFMmagneticscopePASS. Re6ZrFAIL5/7(LFgeometry/decouplingdeclined),BasicsPASS9/9steeringalpharepaired,benzeneALCPASS7/7withsecondsolidline. AllthreeexactcopyPASS. Two sciencefailsTCNQ/Re6Zraregeometry/evidencesynthesis;copyfailuresmostlyterminalLFplusoneprecisionrewrite. RetainallcasecaveatsincludingbenzeneALCcrosssamplemislabel. Waveended00:12:41UTC;verifyhashes/resourcesbeforegeneralround16repairs.


Confirmation5 final verification: all24rawfinal agent_messages equal last-message.md andactualsummary.md bytes;403frozen source entries,allskill/rubric/rawartifact andsuppliedreviewimagehashesverified. 1 nominatedimagepaths changedafterreview; suppliedandcurrenthashesretainedseparately. FreshgateFAIL(22/24science,13/24copy,11/24combined),AFMscopePASS. Twelve-case wavewall48.53min,minavailable21.01GiB,peakprocesssumRSS5.74GiB(sharedpagesdoublecount),memoryPSI0,CPUavg10peak0.36,swap~0.96MiBunchanged. Keep12,notacontrolledspeedupclaim. Noevalprocesslive; freezeends. Generalround16canrepairgeometryevidence/conditionaldiagnostics,consistentpreparedreplyguidanceincludingharness,andquantity/error/capabilityprovenance. Fixedrubricsandtwo-turnimage-reviewsemanticsremainunchanged;deadline06:00UTC/fullgatecutoff04:30UTC.


Round16 general repair in progress after complete confirmation5; no eval live.
CLI/skill/harness finalization now consistently names --prepare-reply after
last edits. Survey adds source column and scan-level evidence tallies, plus
conditional highfield integral/decoupling tests without treating detector-bank
orientation as applied field or relabelling all unresolvedTF. Integral-fit
tables put chi2-scaled errors first with raw errors alongside, JSON unchanged.
Skill separates supported exact RF diagonalisation from unavailable site/ALC
models and permits bounded competing hypotheses without material identity.
Fixed scientific rubrics, strict byte copying and two-turn image-review
semantics remain unchanged. Focused tests, docs and validation precede commit
and a fresh full24 confirmation6 at12 concurrent cases before cutoff.

Round16 final focused438 andfast5940passed (10skip/1xfail), lint/structural,
skillvalidation andgeneratedreferencecheckpass;docsbuild80existingwarnings.
Resonance notes, comparison notes and fit tables now share one uncertainty
scaling function (never shrink errors below raw); detection SNR remains based
on raw fit errors and is labelled accordingly. Final standard validation is
LIVE in /tmp/luna6-round16-standard.log, exec37560. Inspect actual host
processes before any tests/launch. Scratch XDG homes, two pytest workers and
numericalthreads1; no eval live. Only after standard is assessed should round16
be committed and confirmation6 launched. No GUI code or scientific rubrics changed.


Round16 validation COMPLETE:438 focused and5940 fast passed (fast10skip/1xfail); lint, structural, skill validation and generated command reference pass; docs build80existingwarnings. Standard9953passed,4failed,25skipped,1xfailed in453.43s: exactly the two documented fit-parameter-table and two grouping-dialog sizing failures, no worker errors/crashes. QScreen wrapper-lifetime teardown warning retained. Standard remains NOT green; no GUI redesign. Scratch XDG homes, pytest2, numericalthreads1. Ready to commit general source/provenance/finalization/error-presentation repairs, then fresh full24 confirmation6 at12 concurrent cases with fixed scientific rubrics and two-turn image-review semantics.


Fresh full24 confirmation6 LIVE at /home/ben/.cache/asymmetry-luna6-overnight-20261020/confirmation6; implementation5fb689af19ae652deceb4f7a45d1aec5ceaa246c, exec87556, wavePID639182, supervisor639173. Parent console confirmation6-wave.log. All24 fresh first attempts Codex/gpt-6-luna, unchanged scientific rubrics and two_turn_image_review_v2 semantics, strict raw-byte actual replies. Max12 cases, wizard environment default1 (not ceiling), numericalthreads1,30s resource samples;05:59:50UTC supervisor termination/interruption evidence retained. Freeze CLI/skill/runner/rubrics through ALL24 despite early failures. Source403entries and skill/runner/rubric hashes stored in manifest/metadata. Round16 validation logs copied persistently. Score actual replies into luna6-corpus-confirmation6.json; helper records manual judgements, not automatic grades. No broad tests alongside evals. Deadline/cutoff unchanged.


Confirmation6 now1/24 scored:spinPeierls6/6Musts,2/3Shoulds,science/number/unit/boundaryPASS,strictcopyFAILone-wordpost-preparationrewrite(the→its). ActualZFendpointshapesandall26runcoverageverified;dimerisation/singlet-tripletinterpretationstillmissing. Source remains frozen through all24;recorded snapshot/hash/caveats retained.


Confirmation6 now2/24scored:PTFE6/6Musts,1/3Shoulds,science/number/unit/boundary/exactcopyPASS with --prepare-reply. Totals12/12Musts,2/2science,1/2copy/combined,2/2boundary,3/6Shoulds. Preservepoor/flaggedcouplingfitsandcontradictorycold-upturnprosecaveat;sourcefrozen,remainingcasescontinue.


Confirmation6 now4/24scored:YMnAl5/5Musts,1/3Shoulds andmolecularAFM5/5Musts,2/3Shoulds bothscience/number/unit/boundary/exactcopyPASS. Totals22/22Musts,4/4science,3/4copy/combined,4/4boundary,6/12Shoulds. PreserveconditionalLF/24591ZF,failedmolecularOrderParameter/noTc,temperatureandcritical-lawgaps. Frozenremainingcasescontinue.


Confirmation6 now5/24scored:LLZ6/6literalMusts,2/4Shoulds butscienceFAILglobalnumberrule andcopyFAIL afterprepared0.0832477uncertaintydigitsubstituted0.0832472inactualreply. BoundaryPASS;coupled13groupworkflowandconditionalLFcontrastverified. Totals28/28literalMusts,4/5science,3/5copy/combined,5/5boundary,8/16Shoulds. FreezeallsourceuntilALL24complete.


Confirmation6 now7/24 scored:TCNQ repaired6/6Musts,4/4Shoulds andscience/copy/number/boundaryPASS. HighTF-AFM5/5literalMusts,2/3Shoulds,copy/number/boundaryPASS butindependentmagneticscopeFAIL. NewmaterialAFM8Talias:rebin20lowersFFTNyquistto1024MHz,originalsurveyline1084.788MHzbecomes963MHz;agentcallsitapplied-fieldline. Investigategeneralanti-aliasbinningguardafterfreeze. Totals39/39literalMusts,6/7science,5/7copy/combined,7/7boundary,14/23Shoulds. Combined here is the literal science/number/copy metric; independent AFM scope remains a separate failed objective gate. Source frozen.


Confirmation6 now8/24scored:CDS6/6Musts,2/4Shoulds,science/number/unit/copy/boundaryPASS. Noisycoldvisualtripletjudgementretained;noalgorithmiccoldpeaksandpoorfitprecisionexplicit. Totals45/45literalMusts,7/8science,6/8copy/combined,8/8boundary,16/27Shoulds;independentAFMscopeFAILstillseparate. Sourcefrozen.


Confirmation6 now9/24scored:plateau5/5Musts,4/5Shoulds,science/number/unit/copy/boundaryPASS. Redfieldm2fixedrestricted5000–38000GconvergeswithscaledD/nuerrors;approximateplateau-windowjudgementretained,edgeinterpretationmissing. Totals50/50literalMusts,8/9science,7/9copy/combined,9/9boundary,20/32Shoulds. IndependentAFMscopeFAILremains;sourcefrozen.


Confirmation6 now11/24 scored:RFbenzene5/5Musts,2/5Shoulds andsilicon6/6Musts,3/4Shoulds bothscience/number/unit/copy/boundaryPASS. RFexact-diagonalisation/scalederrorrepairused;siliconONfixed-amplituderatecorrectlyrejected,2μsguidewindowdeviationrecorded. Totals61/61literalMusts,10/11science,9/11copy/combined,11/11boundary,25/41Shoulds;independentAFMscopeFAILretained. Frozenwavecontinues.
