"""Tests for :mod:`asymmetry.cli._numbers` and the ``audit`` command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from asymmetry import cli
from asymmetry.cli._numbers import unverified_numbers
from asymmetry.core.workflow.workdir import SCHEMA
from tests.core.conftest import ALL_RUNS, SCAN_RUNS, ZF_RUNS

_LOG = """$ asymmetry fit-series runs --runs 102-106
run  temperature  chi2_red
102  10.000       0.969
Tc 69.1731 ± 0.0541, frequency 30.19 MHz, SNR 3.1x the noise floor
"""


def _manifest(folder: str, instrument: str | None = None) -> str:
    """A work directory's manifest binding it to *folder* and *instrument*."""
    return json.dumps({"schema": 4, "folder": folder, "instrument": instrument})


def test_printed_values_verify_at_the_precision_they_are_written() -> None:
    draft = "Tc = 69.17 ± 0.05 K from runs 102–106; the line at 30.2 MHz; chi2 0.97."
    assert unverified_numbers(draft, _LOG) == []


@pytest.mark.parametrize(
    ("draft", "flagged"),
    [
        # A conversion, a significance and a multiple are arithmetic.
        ("the field is 223.4 G", ["223.4"]),
        ("a 4.3σ effect", ["4.3σ"]),
        ("rises 10× on cooling", ["10×"]),
        # A percentage or a "factor of" is a ratio of printed values.
        ("errors of 32 % throughout", ["32 %"]),
        # ... but a decimal percentage is an asymmetry in its unit.
        ("A(0) of 69.17 %", []),
        ("a factor of 3 increase", ["3"]),
        ("rises roughly 3.5-fold", ["3.5-fold"]),
        ("about 4 times faster", ["4 times"]),
        ("agree within 2 combined errors", ["2 combined errors"]),
        # A multiple a command printed verbatim is fine.
        ("the candidate at 3.1x the noise floor", []),
        # A negative printed value verifies a negative in the draft.
        ("offset -0.969", []),
    ],
)
def test_derived_numbers_are_flagged(draft: str, flagged: list[str]) -> None:
    assert [entry.text for entry in unverified_numbers(draft, _LOG)] == flagged


def test_audit_reads_the_output_every_command_logged(
    workflow_folder: Path, tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.chdir(tmp_path)
    cli.main(["reduce", str(workflow_folder), "--runs", str(SCAN_RUNS[0])])
    log = (tmp_path / "asymmetry-work" / "cli-output.log").read_text(encoding="utf-8")
    assert log.startswith(f"$ asymmetry reduce {workflow_folder}")
    printed = capsys.readouterr().out
    assert printed.strip() in log

    draft = tmp_path / "summary.md"
    draft.write_text(f"Run {SCAN_RUNS[0]} was reduced; its A(0) is 999.25 %.\n", encoding="utf-8")
    cli.main(["audit", str(draft)])
    out = capsys.readouterr().out
    assert "'999.25'" in out
    assert f"'{SCAN_RUNS[0]}'" not in out
    assert "Then run audit again" in out
    # The audit's own report is not logged, so it can never verify itself.
    assert "999.25" not in (tmp_path / "asymmetry-work" / "cli-output.log").read_text(
        encoding="utf-8"
    )


def test_audit_without_any_log_says_what_to_do(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    draft = tmp_path / "summary.md"
    draft.write_text("nothing\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        cli.main(["audit", str(draft)])
    assert exc.value.code == 1
    assert "No cli-output.log" in capsys.readouterr().err


def test_bulk_arrays_in_the_log_verify_nothing() -> None:
    log = "time: [" + ", ".join(f"{0.016 * i:.6f}" for i in range(2000)) + "]\nA(0) 22.516\n"
    # 22.94 lies among the time bins, but no one read it there.
    assert [entry.text for entry in unverified_numbers("A(0) = 22.94 %", log)] == ["22.94"]
    assert unverified_numbers("A(0) = 22.52 %", log) == []


def test_the_vocabulary_of_a_law_no_fit_established_is_flagged() -> None:
    from asymmetry.cli._numbers import unsupported_laws

    log = """$ asymmetry trend runs --series tf --model CriticalDivergence --param Lambda
Fit of CriticalDivergence to Lambda against temperature, over the points' span 69 .. 200: 9 point(s)
LAW NOT ESTABLISHED (Tc is at a bound): CriticalDivergence does not describe this trend.
$ asymmetry trend runs --series zf --model OrderParameter --param frequency
Fit of OrderParameter to frequency against temperature, over the points' span 1.5 .. 68: 20 point(s)
Converged, with its physical parameters determined: report them.
"""
    draft = "The rate rises: critical slowing down. The critical exponent is 0.44."
    # OrderParameter converged, so its vocabulary stands; CriticalDivergence never did.
    assert unsupported_laws(draft, log) == [("CriticalDivergence", "critical slowing")]
    # A law fitted once without success and once with is established.
    retried = log + log.replace("LAW NOT ESTABLISHED (Tc is at a bound)", "Converged")
    assert unsupported_laws(draft, retried) == []
    # A law never fitted is not judged.
    assert unsupported_laws("an activation energy", log) == []
    # A sentence denying the law does not lean on it.
    assert unsupported_laws("No critical slowing down was established.", log) == []


def test_a_multiple_verifies_only_as_a_whole_printed_token() -> None:
    log = "candidate 13x the noise floor"
    assert [entry.text for entry in unverified_numbers("a rate 3× the warm value", log)] == ["3×"]
    assert unverified_numbers("a candidate 13× the noise floor", log) == []
    assert unverified_numbers("62 % of the amplitude", "relaxing 62% of it") == []


def test_hedged_ratios_and_differences_are_always_listed() -> None:
    log = "B0 78.18 G\nB0 80.09 G\nBwid 12.25\nBwid 12.83\nnu 2\n"
    draft = (
        "The fields agree to within about 2 G and the widths differ by 0.6 G; "
        "the rates sit a factor of ~2 apart. B0 = 78.18 G."
    )
    assert [entry.text for entry in unverified_numbers(draft, log)] == ["2", "0.6", "2"]


def test_audit_names_the_runs_of_a_surveyed_scan_that_no_fit_covers(
    workflow_folder: Path, tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.chdir(tmp_path)
    folder = str(workflow_folder)
    cli.main(["survey", folder])
    cli.main(["reduce", folder, "--runs", ",".join(str(run) for run in ZF_RUNS)])
    cli.main(
        [
            "recipe",
            folder,
            "--expression",
            "Exponential + Constant",
            "--name",
            "relax",
            "--run",
            str(SCAN_RUNS[0]),
        ]
    )
    cold, warm = ZF_RUNS[:3], ZF_RUNS[3:]
    series = ["--recipe", "relax", "--order", "temperature"]
    capsys.readouterr()
    cli.main(["fit-series", folder, "--runs", ",".join(map(str, cold)), *series, "--name", "cold"])
    # A result command closes by naming what is still unfitted, before the audit step.
    assert "Still unfitted: 1 measurement(s)" in capsys.readouterr().out
    draft = tmp_path / "summary.md"
    # The folder's files carry deadtimes this reduction left off, which a draft must say.
    draft.write_text(
        f"{len(ALL_RUNS)} runs, SIM .nxs; ZF scan {min(ZF_RUNS)} – {max(ZF_RUNS)}; "
        "the reduction left deadtime off.\n\nFiles\nplots/review.png\n",
        encoding="utf-8",
    )
    capsys.readouterr()

    cli.main(["audit", str(draft), "--json"])
    (scan,) = json.loads(capsys.readouterr().out)["unfitted_scans"]
    assert scan["unfitted_runs"] == list(warm)

    cli.main(["fit-series", folder, "--runs", ",".join(map(str, warm)), *series, "--name", "warm"])
    assert "Still unfitted" not in capsys.readouterr().out
    cli.main(["audit", str(draft)])
    # Every scan is fitted, but neither series has been read with trend yet.
    assert "Series no trend command has read: cold" in capsys.readouterr().out
    for name in ("cold", "warm"):
        cli.main(["trend", folder, "--series", name])
    capsys.readouterr()
    cli.main(["audit", str(draft)])
    out = capsys.readouterr().out
    assert "your final message is this summary itself" in out
    # The passed text is printed whole, for the reply to copy rather than retype.
    assert f"----- BEGIN -----\n{draft.read_text()}\n----- END -----" in out
    assert (
        "COPY CHECK: 4 report lines; terminal newline: yes; the final line is 'plots/review.png'"
        in out
    )


@pytest.mark.parametrize(
    ("draft", "missing"),
    [
        ("The fit table lists conditional centres and widths.", 2),
        ("The cold fitted centre is higher in field, but window-sensitive.", 1),
        ("The cold feature is broader, with a model-mismatch caveat.", 1),
        ("The broader-window fit is not a stable centre comparison.", 2),
        ("The field scan rises at higher fields; its width is uncertain.", 2),
        ("The cold estimate is higher in field and broader; the shift is conditional.", 0),
    ],
)
def test_audit_requires_the_printed_cross_condition_directions(draft: str, missing: int) -> None:
    from asymmetry.cli._numbers import unstated

    note = "COMPARE: conditional fitted estimates; this one is broader and higher in field.\n"
    assert len(unstated(draft, note)) == missing
    # Alternative windows of one measurement impose no condition-change claim.
    assert unstated(draft, note.replace("COMPARE:", "STABILITY:")) == []


def test_report_provenance_and_coupled_fits_are_required(
    workflow_folder: Path, tmp_path: Path
) -> None:
    from dataclasses import replace

    from asymmetry.cli.commands.audit import report_obligations
    from asymmetry.core.workflow.survey import _scan_groups, survey_folder
    from asymmetry.core.workflow.workdir import RunSelection, WorkDir

    survey = survey_folder(workflow_folder)
    rows = [
        replace(survey.runs[1], run_number=201 + 3 * i + j, temperature=temperature, field=field)
        for i, temperature in enumerate([160.0, 180.0])
        for j, field in enumerate([0.0, 5.0, 10.0])
    ]
    workdir = WorkDir(tmp_path / "work")
    workdir.write_manifest(RunSelection(workflow_folder, None))
    workdir.write_survey(replace(survey, runs=rows, scans=_scan_groups(rows)[0]).to_dict())
    missing = report_obligations([workdir.root], "A draft at 6 K; Figure 6.")
    assert any("6 runs" in message for message in missing)
    assert any("COUPLED FIELD GROUPS: 2" in message for message in missing)
    for index, members in enumerate([[201, 202, 203], [204, 205, 206]]):
        workdir.write_series(
            str(index),
            {"kind": "series", "trend": {"rows": [{"key": str(run)} for run in members]}},
        )
    text = "6 runs, SIM .nxs, two temperature groups: 201-203 and 204-206."
    assert any("COUPLED FIELD GROUPS: 2" in m for m in report_obligations([workdir.root], text))
    # One global fit over both temperatures forces one set of dynamics on them.
    union = {"kind": "global", "trend": {"rows": [{"key": str(run)} for run in range(201, 207)]}}
    workdir.write_series("union", union)
    assert any("COUPLED FIELD GROUPS: 2" in m for m in report_obligations([workdir.root], text))
    for index in range(2):
        # Any simultaneous fit clears the gate, whichever model it shares.
        workdir.write_series(str(index), workdir.read_series(str(index)) | {"kind": "global"})
    assert report_obligations([workdir.root], text) == []


def test_a_temperature_scan_used_only_for_alpha_is_sent_to_a_series_fit() -> None:
    from asymmetry.cli.commands.audit import _unfitted_report, _UnfittedScan
    from asymmetry.core.workflow.survey import ScanGroup

    def scan(axis: str, runs: list[int]) -> ScanGroup:
        return ScanGroup(
            axis=axis,
            instrument="SIM",
            geometry="TF",
            geometry_note="",
            temperature=None if axis == "temperature" else 40.0,
            field=100.0 if axis == "temperature" else None,
            runs=runs,
            values=[float(index) for index, _ in enumerate(runs)],
        )

    report = _unfitted_report(
        [
            _UnfittedScan(
                "data", scan("temperature", [11, 12, 13, 14]), [11, 12, 13, 14], {12}, set()
            ),
            _UnfittedScan("data", scan("field", [21, 22]), [21, 22], {12}, set()),
            # A short scan's run that measured alpha has done its job.
            _UnfittedScan("data", scan("field", [12, 51]), [12], {12}, set()),
            _UnfittedScan("data", scan("field", [13, 31]), [13], {12}, set()),
        ]
    )

    assert "never fitted. Alpha was measured on run 12" in report
    assert "asymmetry fit-series data --runs 11-14 --recipe wizard-12" in report
    # Runs the survey found no line in are sent to a relaxation fit, not the wizard.
    lineless = _unfitted_report(
        [_UnfittedScan("data", scan("temperature", [41, 42, 43, 44]), [43, 44], set(), {43, 44})]
    )
    assert "asymmetry recipe data --expression 'Exponential + Constant' --run 43" in lineless
    assert "wizard" not in lineless
    # A field scan of unresolved geometry whose runs precess is fitted in time, not integrated.
    unresolved = ScanGroup(
        axis="field",
        instrument="SIM",
        geometry=None,
        geometry_note="",
        temperature=300.0,
        field=None,
        runs=[61, 62, 63, 64],
        values=[50.0, 100.0, 150.0, 200.0],
    )
    runs = [61, 62, 63, 64]
    precessing = _unfitted_report(
        [_UnfittedScan("data", unresolved, runs, set(), set(), frozenset(runs))]
    )
    assert "asymmetry wizard data --run 61" in precessing
    assert "asymmetry integral-scan" not in precessing
    integrated = _unfitted_report([_UnfittedScan("data", unresolved, runs, set(), set())])
    assert "asymmetry integral-scan data --runs 61-64" in integrated
    # A short scan's runs appear once, and not again when a longer scan lists them.
    assert report.endswith(
        "short scans of 2-3 runs, not fitted: runs 21-22 — fit them where they bear on the "
        "question."
    )


def test_a_run_fitted_on_its_own_counts_as_fitted(
    workflow_folder: Path, tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.chdir(tmp_path)
    folder = str(workflow_folder)
    cli.main(["survey", folder])
    cli.main(["reduce", folder, "--runs", ",".join(str(run) for run in ZF_RUNS)])
    cli.main(
        [
            "recipe",
            folder,
            "--expression",
            "Exponential + Constant",
            "--name",
            "relax",
            "--run",
            str(SCAN_RUNS[0]),
        ]
    )
    for run in ZF_RUNS:
        cli.main(["fit", folder, "--run", str(run), "--recipe", "relax"])
    draft = tmp_path / "summary.md"
    draft.write_text("A draft.\n", encoding="utf-8")
    capsys.readouterr()

    cli.main(["audit", str(draft), "--json"])
    assert json.loads(capsys.readouterr().out)["unfitted_scans"] == []


_WAVE_LOG = """AICc 2073.4  2035.8  37.441
frequency_1 moves from 813.57601 to 813.54580 MHz
frequency 1.92645  survey_line_mhz 1.94529  sigma 0.020831
alpha 1.2373 (estimated:24563)  alpha 1.2320  0.50 0.52
r_muF 1.22 1.25  A(0) 16.42 0.05  bound 0.1
series ionic-11 written; recipe wizard-12; 4 unresolved; 1.3 1.2 0.2 0.3 0.6
(Δt = 0.0123 µs); runs 9031 9051; nu 16.3 3.8 MHz; offset +0.34 K
"""


@pytest.mark.parametrize(
    ("draft", "flagged"),
    [
        # Differences written in prose, hedged or not (Haiku 5.5 wave 1, 2026-10-07).
        ("prefers the two-line model by about 37", ["37"]),
        ("the line falls by 0.03 MHz on warming", ["0.03"]),
        ("they agree with these survey lines to about 0.02 MHz", ["0.02"]),
        ("the alphas differ by about 0.5 %", ["0.5 %"]),
        # A comparison named after the number, over a whole range.
        ("about 4 points better on AICc", ["4"]),
        ("values 1.22–1.25 Å, a 1.2–1.3% spread", ["1.2", "1.3%"]),
        ("the 0.2–0.3 % difference is unresolved", ["0.2", "0.3 %"]),
        # A relative error is a ratio; an absolute error on an asymmetry is not.
        ("stable to ±0.6 %", ["0.6 %"]),
        ("A(0) = 16.42 ± 0.05 %", []),
        # A Δ-quantity verifies only as printed.
        ("ΔAICc about 11", ["11"]),
        ("a timing offset Δt = 0.0123 µs", []),
        # Ordinary values, ranges and bounds stay quiet.
        ("runs 9031–9051", []),
        ("ν falls from 16.3 to 3.8 MHz", []),
        ("they agree, and ν reaches 16.3 MHz", []),
        # "By" a temperature is a time, not a change.
        ("the line disappears by 16.3 K", []),
        ("it falls only slightly, by about 0.03 MHz", ["0.03"]),
        ("Delta falls and reaches its plateau by about 16.3 K", []),
        ("Lambda sits at its 0.1 lower bound", []),
        # A unit conversion of a printed value (Haiku 5.5 wave 15, Sonnet 5.5).
        ("a line at 3.8 MHz, which would be a field of about 16 G", ["16"]),
        ("3.8 MHz corresponds to roughly 16 G", ["16"]),
        ("this corresponds to run 9031", []),
        # Roundings and restatements of printed values (round 2, waves 11-15).
        ("A(0) 16.42 (±0.05) at 16.3 K", []),
        ("errors are 1σ; none differ at > 3σ", []),
        ("a 2× LorentzianLCR + Cubic fit", []),
        ("the line sits at about 9030 G", []),
        ("the total is about 16.6", ["16.6"]),
        # Stacked hedges (wave 16).
        ("the lines differ from the survey by up to about 0.02 MHz", ["0.02"]),
        ("the shift is about 1.3 %", ["1.3 %"]),
        ("the spread is only about 40 G", ["40"]),
        ("logged offsets of +0.34 K", []),
        ("a 47-point AICc difference", ["47-point"]),
        # A whole-number percentage naming an asymmetry restates it; elsewhere it is a ratio.
        ("A(0) is about 16 % throughout", []),
        ("the amplitude stays near 16 %", []),
        ("the rate falls about 16 % on warming", ["16 %"]),
        ("the line sits at 9.03 kG", []),
        ("the line sits at 9028 G", ["9028"]),
        ("a 2× faster rate", ["2×"]),
    ],
)
def test_wave_derived_numbers_are_flagged(draft: str, flagged: list[str]) -> None:
    assert [entry.text for entry in unverified_numbers(draft, _WAVE_LOG)] == flagged


_WAVE2_LOG = """rate 3.2e-08  candidate 3.3x the noise floor  2.9x
field 5000 G  step 200 G  run 91500  A_bg 3.56102  dchi2 274.851 480.2
Delta 0.3505 0.2706 0.08  frequency 0.0326399  AICc 4241.2 8423.2
"""


@pytest.mark.parametrize(
    ("draft", "flagged"),
    [
        # Notation: separators, powers of ten and multiple signs (Haiku 5.5 wave 2).
        ("a rate of 3.2 × 10⁻⁸ s", []),
        ("SNR 3.3× and 2.9 times the noise floor", []),
        ("beats the Gaussian by about 4,200 in AICc", ["4,200"]),
        # A context-derived number verifies only as a long verbatim token.
        ("the asymmetry falls by 5000 G", []),
        ("the 91500 discrepancy is the only one", []),
        ("by a margin of 3.6 AICc", ["3.6"]),
        ("the frequency shift of about 0.03 MHz", ["0.03"]),
        ("preferred by 274–480 in chi2", ["274", "480"]),
        ("Delta at 0.3505 drops by about 0.08", ["0.08"]),
        # A step between two printed levels, printed by chance (Haiku, 2026-10-21).
        ("Delta is 0.3505 and 0.2706; the step is about 0.08", ["0.08"]),
        ("Lambda falls to 0.2706 by 5000 G", []),
        ("| Tc (K) | 0.3505 | ± 0.08 |", []),
        ("| T_c | 0.3505 K | ± 0.08 |", []),
        ("not reliable to better than ±0.2 MHz", ["0.2"]),
        # Ratios in words.
        ("the rate falls by roughly a factor of six", ["factor of six"]),
        ("the line is about five and a half times its error", ["five and a half times"]),
        ("amplitudes four to six times their errors", ["four to six times"]),
        ("amplitudes 4 to 6 times their errors", ["4 times", "6 times"]),
        ("SNR 3.3 to 2.9 times the noise floor", []),
        ("one condition repeated five times.", []),
        ("the scan drifted by about 3 K", ["3"]),
        ("amplitudes add up to several times the asymmetry", []),
        ("lines 2.5 resolution elements apart", ["2.5 resolution elements"]),
        # A hedged round spread is one digit, whatever grid value it repeats (Sonnet, 2026-10-21).
        ("the centre moves by about 200 G with window", ["200"]),
        ("the centre is stable to about 200 G across windows", ["200"]),
        ("the field steps by 200 G", []),
        ("the whole change is about three percent of Delta", ["three percent"]),
    ],
)
def test_wave2_notation_and_contexts(draft: str, flagged: list[str]) -> None:
    assert [entry.text for entry in unverified_numbers(draft, _WAVE2_LOG)] == flagged


def test_a_coupling_quoted_without_its_printed_relation_is_named() -> None:
    from asymmetry.cli._numbers import unstated

    log = "The correlation peak is the muon hyperfine coupling A_mu = nu_1 + nu_2, the sum\n"
    bare = "The correlation peak gives A_μ ≈ 514 MHz."
    assert unstated(bare, log) and not unstated(bare, "no fourier here\n")
    for stated in ("A_μ = ν₁ + ν₂ ≈ 514 MHz", "A_mu, the sum of the two lines, is 514 MHz"):
        assert unstated(stated, log) == []
    # Only a draft that quotes the coupling owes its relation.
    assert unstated("No correlation peak is quoted.", log) == []


_HELD_NOTE = (
    "NOTE: on runs 20888, 20896-20897 the relaxation is too slow over the fitted window to "
    "tell from the constant — the free fit ran its amplitude and A_bg off in opposite signs — "
    "so A_bg was held at 0 and the fit repeated. Say in the summary that it was held there.\n"
)
_BACKGROUND_NOTE = (
    "NOTE: the fit converged at chi2_red 40.1: over a long range the background may rise or "
    "step where no polynomial can follow — a fit that cannot is not a result, and the "
    "summary should say that is why — or the range holds more dips than the model.\n"
)


@pytest.mark.parametrize(
    ("log", "silent", "stated"),
    [
        (
            "NOTE: … the zero-field relaxation is from fields static on the muon time scale, "
            "decoupled as a static distribution is. A rate pinned at zero here is that result\n",
            # The model's name is not the link the longitudinal field makes.
            "The zero-field data fit a static Kubo–Toyabe; the 100 G rates are pinned.",
            "The 100 G longitudinal field decouples the relaxation: the fields are static.",
        ),
        (_HELD_NOTE, "Runs 20888-20897 give A_1 near 24 %.", "A_bg was held at 0 on 20888."),
        (
            "NOTE: A_1 is smallest inside the scan — 5.3 at 0 (run 4) … and rises on both "
            "sides: a minimum, not a step.\n",
            "A_1 changes between -0.5 and 0.75 A.",
            "A_1 is smallest near 0 A and rises on both sides.",
        ),
        (
            "frequency moves from 5.38 to 5.39 MHz on warming, a shift of 0.01 ± 0.002 MHz — "
            "the line sits lower at low temperature — while staying near one field: a shift "
            "of the line (a Knight shift, or a superconductor's diamagnetic shift below Tc).\n",
            # Naming the shift is not saying which way the line moved.
            "The frequency moves from 5.38 to 5.39 MHz, a small shift near the applied field.",
            "The line shifts lower in frequency below Tc, by 0.01 ± 0.002 MHz.",
        ),
        (
            _BACKGROUND_NOTE,
            # That the fit fails is not the reason it fails.
            "The full-range LorentzianLCR + Cubic fit fails.",
            "The full-range fit is not quoted: one cubic cannot follow both dips.",
        ),
    ],
)
def test_a_printed_request_the_draft_leaves_out_is_quoted_back(
    log: str, silent: str, stated: str
) -> None:
    from asymmetry.cli._numbers import unstated

    (message,) = unstated(silent, log)
    assert log.strip()[:60] in message
    assert unstated(stated, log) == []


@pytest.mark.parametrize(
    ("draft", "explained"),
    [
        ("The polynomial describes the local non-resonant field dependence.", False),
        ("The whole-scan recovery fit failed; local cubic fits describe the dips.", False),
        ("A single polynomial cannot follow the stepped background across the scan.", True),
        ("The background is not described by one cubic over the full range.", True),
        ("The background rises or steps where no polynomial can follow.", True),
    ],
)
def test_background_explanation_requires_a_limitation(draft: str, explained: bool) -> None:
    from asymmetry.cli._numbers import unstated

    assert (unstated(draft, _BACKGROUND_NOTE) == []) is explained


def test_audit_names_a_notes_scan_that_no_fit_covers(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    workdir = tmp_path / "asymmetry-work"
    workdir.mkdir()
    (workdir / "cli-output.log").write_text("$ asymmetry survey data\n", encoding="utf-8")
    scan = {
        "instrument": "SIM",
        "temperature": 295.0,
        "field": 100.0,
        "source": "notes",
        "template": "Steering <x> A",
        "quantity": "steering",
        "runs": [11, 12, 13],
        "values": [-1.0, 0.0, 1.0],
    }
    survey = {"schema": SCHEMA, "folder": "data", "runs": [], "scans": [], "notes_scans": [scan]}
    (workdir / "survey.json").write_text(json.dumps(survey), encoding="utf-8")
    (workdir / "manifest.json").write_text(_manifest("data"), encoding="utf-8")
    draft = tmp_path / "summary.md"
    draft.write_text("A draft.\n", encoding="utf-8")

    cli.main(["audit", str(draft)])
    out = capsys.readouterr().out
    assert 'SIM runs 11-13, notes "Steering <x> A" (steering): not fitted: runs 11-13' in out
    assert (
        "asymmetry fit-series data --runs 11-13 --recipe wizard-11 --order steering "
        "--x 11=-1,12=0,13=1 --start 11"
    ) in out
    assert "do not reply yet: act on each item above" in out


def test_a_correlation_spectrum_without_its_plain_transform_is_named(tmp_path: Path) -> None:
    from asymmetry.cli.commands.audit import _correlations_without_lines

    spectra = tmp_path / "wd" / "spectra"
    spectra.mkdir(parents=True)
    for name, axis, run in [
        ("run-7-correlation", "hyperfine_coupling", 7),
        ("run-8-correlation", "hyperfine_coupling", 8),
        ("run-8", "frequency", 8),
    ]:
        (spectra / f"{name}.json").write_text(json.dumps({"axis": axis, "run": run}))
    assert _correlations_without_lines([tmp_path / "wd"]) == [(tmp_path / "wd", 7)]


def test_a_tesla_field_line_without_a_two_line_fit_is_named(tmp_path: Path) -> None:
    from asymmetry.cli.commands.audit import _untested_doublets

    root = tmp_path / "wd"
    for folder in ("spectra", "fits", "series"):
        (root / folder).mkdir(parents=True)
    spectrum = {
        "axis": "frequency",
        "run": 686,
        "field_gauss": 60000.0,
        "peak_analysis": {"peaks": [{"frequency_mhz": 813.59}]},
    }
    (root / "spectra" / "run-686.json").write_text(json.dumps(spectrum))

    def fit(**frequencies: float) -> str:
        return json.dumps(
            {"expression": "...", "fit": {"success": True, "parameters": frequencies}}
        )

    (root / "fits" / "one-686.json").write_text(fit(frequency=813.59))
    assert _untested_doublets([root]) == [(root, 686, 813.59)]
    # A second line at the first's harmonic is not a two-line test.
    (root / "fits" / "wizard-686.json").write_text(fit(frequency_1=813.6, frequency_3=1627.2))
    assert _untested_doublets([root]) == [(root, 686, 813.59)]
    (root / "fits" / "two-line-686.json").write_text(fit(frequency_1=813.60, frequency_3=813.54))
    assert _untested_doublets([root]) == []


def test_a_dip_a_scan_announced_and_no_fit_holds_is_named(tmp_path: Path) -> None:
    from asymmetry.cli.commands.audit import _unfitted_dips

    scans = tmp_path / "wd" / "scans"
    scans.mkdir(parents=True)
    window = {"centre": 21400.0, "x_min": 19950.0, "x_max": 22950.0}

    def store(name: str, runs: list[int], centre: float, width: float, windows=()) -> None:
        fit = {
            "success": True,
            "parameters": {"B0": centre, "Bwid": width},
            "next_dip_windows": list(windows),
        }
        (scans / f"{name}.json").write_text(json.dumps({"runs": runs, "fit": fit}))

    store("whole", [1, 2, 3], 19475.0, 300.0, [window])
    assert _unfitted_dips([tmp_path / "wd"]) == [(tmp_path / "wd", "whole", window)]
    # Another scan's line there, or a broad line stretched across the window, holds nothing.
    store("other-scan", [4, 5, 6], 21474.0, 255.0)
    store("broad", [1, 2, 3], 21000.0, 3600.0)
    assert _unfitted_dips([tmp_path / "wd"]) == [(tmp_path / "wd", "whole", window)]
    store("local", [1, 2, 3], 21474.0, 255.0)
    assert _unfitted_dips([tmp_path / "wd"]) == []


def test_a_hold_is_restated_when_unprinted_numbers_are_listed_too(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.chdir(tmp_path)
    workdir = tmp_path / "asymmetry-work"
    workdir.mkdir()
    (workdir / "cli-output.log").write_text("$ asymmetry survey data\n", encoding="utf-8")
    scan = {
        "instrument": "SIM",
        "temperature": 295.0,
        "field": 100.0,
        "source": "notes",
        "template": "Steering <x> A",
        "quantity": "steering",
        "runs": [11, 12, 13],
        "values": [-1.0, 0.0, 1.0],
    }
    survey = {"schema": SCHEMA, "folder": "data", "runs": [], "scans": [], "notes_scans": [scan]}
    (workdir / "survey.json").write_text(json.dumps(survey), encoding="utf-8")
    (workdir / "manifest.json").write_text(_manifest("data"), encoding="utf-8")
    draft = tmp_path / "summary.md"
    draft.write_text("A draft quoting 999.25 G.\n", encoding="utf-8")

    cli.main(["audit", str(draft)])
    out = capsys.readouterr().out
    assert "'999.25'" in out
    assert out.rstrip().endswith("then run audit again.")


def test_each_survey_is_held_against_its_own_work_directory(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    # Run 11 is fitted in one folder's work directory; the other folder's run 11
    # is a different run, and its notes scan is still unfitted.
    monkeypatch.chdir(tmp_path)
    scan = {
        "instrument": "SIM",
        "temperature": 295.0,
        "field": 100.0,
        "source": "notes",
        "template": "Steering <x> A",
        "quantity": "steering",
        "runs": [11, 12, 13],
        "values": [-1.0, 0.0, 1.0],
    }
    for name, notes in (("asymmetry-work", []), ("asymmetry-work-b", [scan])):
        root = tmp_path / name
        (root / "fits").mkdir(parents=True)
        (root / "cli-output.log").write_text("$ asymmetry survey data\n", encoding="utf-8")
        survey = {"schema": SCHEMA, "folder": name, "runs": [], "scans": [], "notes_scans": notes}
        (root / "survey.json").write_text(json.dumps(survey), encoding="utf-8")
        (root / "manifest.json").write_text(_manifest(name), encoding="utf-8")
    for run in (11, 12, 13):
        (tmp_path / "asymmetry-work" / "fits" / f"x-{run}.json").write_text(
            json.dumps(
                {
                    "run": run,
                    "expression": "Exponential",
                    "fit": {"success": True, "parameters": {}},
                }
            ),
            encoding="utf-8",
        )
    draft = tmp_path / "summary.md"
    draft.write_text("A draft.\n", encoding="utf-8")

    cli.main(["audit", str(draft)])
    assert 'notes "Steering <x> A" (steering): not fitted: runs 11-13' in capsys.readouterr().out

    # A second work directory bound to the same folder holds its fits for both,
    # though it never surveyed the folder itself.
    (tmp_path / "asymmetry-work" / "survey.json").unlink()
    (tmp_path / "asymmetry-work" / "manifest.json").write_text(
        _manifest("asymmetry-work-b"), encoding="utf-8"
    )
    cli.main(["audit", str(draft)])
    assert "not fitted" not in capsys.readouterr().out

    # Two instruments in one folder reuse run numbers: one's fits hold none of the other's runs.
    for name, instrument in (("asymmetry-work", "EMU"), ("asymmetry-work-b", "MUSR")):
        (tmp_path / name / "manifest.json").write_text(
            _manifest("asymmetry-work-b", instrument), encoding="utf-8"
        )
    cli.main(["audit", str(draft)])
    assert "not fitted: runs 11-13" in capsys.readouterr().out


def test_review_edge_cases_of_the_audit(tmp_path: Path) -> None:
    from asymmetry.cli.commands.audit import _untested_doublets
    from asymmetry.core.workflow.workdir import WorkDir

    # A spectrum stored before field_gauss was recorded has no tesla-field line to test.
    spectra = tmp_path / "old" / "spectra"
    spectra.mkdir(parents=True)
    old = {"axis": "frequency", "run": 5, "peak_analysis": {"peaks": [{"frequency_mhz": 813.6}]}}
    (spectra / "run-5.json").write_text(json.dumps(old))
    assert _untested_doublets([tmp_path / "old"]) == []

    # A comma-grouped number a command printed in that form verifies.
    assert unverified_numbers("beats it by about 4,200 in AICc", "AICc gap 4,200\n") == []
    assert unverified_numbers("an AICc gap of 4,200", "AICc gap 4,200\n") == []
    assert unverified_numbers("ahead with 4,200 events", "AICc gap 4,200\n") == []

    # A precession series with a held frequency does not fit a lineless run it failed on.
    root = tmp_path / "wd"
    (root / "series").mkdir(parents=True)
    survey = {"schema": SCHEMA, "runs": [{"run_number": 7, "precession": "none"}]}
    (root / "survey.json").write_text(json.dumps(survey))
    series = {
        "kind": "series",
        "expression": "Oscillatory * Exponential + Constant",
        "free_params": ["A_1", "phase", "Lambda", "A_bg"],
        "trend": {"rows": [{"key": "7", "flags": ["amplitude_exceeds_data"]}]},
    }
    (root / "series" / "held.json").write_text(json.dumps(series))
    assert WorkDir(root).fitted_runs() == set()


@pytest.mark.parametrize(
    ("draft", "flagged"),
    [
        ("A_hf = 512.35 MHz", False),
        ("The rejected A_hf values were 512.35 and 613.28 G.", True),
        ("A_hf = 512.35 G", True),
        ("B0 = 512.35 G", False),
        ("An unrelated field is 512.35 G", False),
        ("A_hf = 5.1235 × 10² G", True),
        ("A_hf = 5.1235e2 MHz", False),
    ],
)
def test_named_parameter_units_are_not_verified_by_a_bare_number(draft, flagged):
    from asymmetry.cli._numbers import unverified_units

    log = "parameter value unit error\nA_hf 512.3456 MHz 0.1\nA_hf 613.2781 MHz 0.2\nB0 512.3456 G 0.3\n"
    assert bool(unverified_units(draft, log)) is flagged


def test_all_instrument_survey_uses_each_instruments_fit_coverage(tmp_path):
    from asymmetry.cli.commands.audit import unfitted_scans

    roots = [tmp_path / "overview", tmp_path / "other"]
    scan = {
        "instrument": "MUSR",
        "axis": "temperature",
        "temperature": None,
        "field": 0.0,
        "geometry": "ZF",
        "geometry_note": "",
        "runs": [11, 12, 13, 14],
        "values": [1.0, 2.0, 3.0, 4.0],
        "notes": "",
        "samples": [],
    }
    for root, instrument in zip(roots, ("EMU", "MUSR"), strict=True):
        (root / "fits").mkdir(parents=True)
        (root / "manifest.json").write_text(_manifest("same-data", instrument))
    (roots[0] / "survey.json").write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "folder": "same-data",
                "runs": [],
                "scans": [scan],
                "notes_scans": [],
            }
        )
    )
    assert unfitted_scans(roots)[0]
    for run in scan["runs"]:
        (roots[0] / "fits" / f"wrong-{run}.json").write_text(json.dumps({"run": run}))
    assert unfitted_scans(roots)[0]  # Same-number fits in the other instrument do not count.
    for run in scan["runs"]:
        (roots[1] / "fits" / f"right-{run}.json").write_text(json.dumps({"run": run}))
    assert unfitted_scans(roots) == ([], [])


def test_composition_axis_requires_its_experimental_source():
    from asymmetry.cli._numbers import unstated

    log = "COMPOSITION AXIS: state concentration provenance and normalization."
    assert unstated("The relative concentration is shown in a table.", log)
    assert not unstated("Relative concentrations came from the run titles; full = 1.", log)


@pytest.mark.parametrize("ending", ["\n", "\n\n", ""])
def test_prepare_reply_only_removes_terminal_line_endings_after_a_pass(
    tmp_path: Path, monkeypatch, capsys, ending: str
) -> None:
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "asymmetry-work"
    root.mkdir()
    (root / "cli-output.log").write_text("$ asymmetry info data\n")
    draft = tmp_path / "summary.md"
    body = "Measured signal.  \n\nFiles\nplots/result.png"
    draft.write_text(body + ending)
    cli.main(["audit", str(draft)])
    assert draft.read_text() == body + ending
    capsys.readouterr()
    cli.main(["audit", str(draft), "--prepare-reply"])
    out = capsys.readouterr().out
    assert draft.read_bytes() == body.encode()
    assert out == body
    draft.write_text("Unprinted result 98765.4321.\n")
    with pytest.raises(SystemExit, match="1"):
        cli.main(["audit", str(draft), "--prepare-reply"])
    assert draft.read_text().endswith("\n")
    assert "----- BEGIN -----" not in capsys.readouterr().out


@pytest.mark.parametrize(
    "sentence",
    [
        "The Redfield fit failed because its exponent was negative.",
        "Redfield was rejected as unresolved.",
        "The activation energy is not established.",
    ],
)
def test_failed_law_descriptions_do_not_claim_a_supported_mechanism(sentence: str) -> None:
    from asymmetry.cli._numbers import unsupported_laws

    log = "Fit of Redfield to Lambda against field\nLAW NOT ESTABLISHED\n"
    assert unsupported_laws(sentence, log) == []
    assert unsupported_laws("The Redfield law explains the rate.", log) == [
        ("Redfield", "redfield")
    ]


def test_an_integral_only_scan_requires_its_first_local_dip_test(tmp_path: Path) -> None:
    import numpy as np

    from asymmetry.cli.commands.audit import _unfitted_dips
    from asymmetry.core.transform import FieldScan
    from asymmetry.core.workflow.integral_scan import field_scan_payload

    root = tmp_path / "wd"
    scans = root / "scans"
    scans.mkdir(parents=True)
    x = np.linspace(2000, 5000, 61)
    scan = FieldScan(
        x=x,
        value=0.2 - 0.04 / (1 + ((x - 3500) / 180) ** 2),
        error=np.full_like(x, 0.0005),
        run_numbers=list(range(61)),
        order_key="field",
        method="integral",
    )
    payload = {"runs": list(range(61)), "scan": field_scan_payload(scan), "fit": None}
    (scans / "measured.json").write_text(json.dumps(payload))
    ((directory, name, window),) = _unfitted_dips([root])
    assert (directory, name) == (root, "measured")
    assert window["centre"] == pytest.approx(3500)
    # A failed local test is still a test; it must be reported as unresolved,
    # not repeatedly fitted until a noisy candidate becomes a claimed line.
    fit = {
        "success": False,
        "parameters": {},
        "next_dip_windows": [],
        "x_min": window["x_min"],
        "x_max": window["x_max"],
    }
    (scans / "local.json").write_text(json.dumps(payload | {"fit": fit}))
    assert _unfitted_dips([root]) == []


def test_adjacent_cold_excess_is_not_replaced_by_distant_warm_comparison() -> None:
    from asymmetry.cli._numbers import unstated

    log = "COLD-END EXCESS: nu is higher at the coldest run 21 than the next run 22."
    assert unstated("nu is higher at 5 K than at 100 K.", log)
    assert not unstated("The coldest nu is higher than the next temperature point.", log)


def test_conditional_coupling_needs_its_estimate_and_error() -> None:
    from asymmetry.cli._numbers import unstated

    log = "CONDITIONAL COUPLING: A_hf = 1516.298738 ± 80.450183 MHz, chi2_red 459.300."
    assert unstated("A_hf is not established because the fit is poor.", log)
    assert not unstated(
        "A_hf = 1516.298738 ± 80.450183 MHz is model-conditioned; the fit is poor.", log
    )


@pytest.mark.parametrize(
    "draft",
    [
        "A_mu = 123.4(5) MHz, conditional on the poor fit.",
        "A_μ = 123.4 ± 0.5 MHz, conditional on the poor fit.",
        "Aµ was 123 ± 1 MHz in this model.",
        "The fitted A_mu was 123.4 +/- 0.5 MHz.",
    ],
)
def test_conditional_coupling_accepts_common_value_and_error_spellings(draft) -> None:
    from asymmetry.cli._numbers import unstated

    log = "CONDITIONAL COUPLING: A_mu = 123.412345 ± 0.512345 MHz, chi2_red 9.000."
    assert not unstated(draft, log)
    assert unstated("A_mu could not be interpreted.", log)


def test_report_acquisition_and_sample_identities_are_not_reference_labels(
    workflow_folder: Path, tmp_path: Path
) -> None:
    from dataclasses import replace

    from asymmetry.cli.commands.audit import report_obligations
    from asymmetry.core.workflow.survey import survey_folder
    from asymmetry.core.workflow.workdir import RunSelection, WorkDir

    survey = survey_folder(workflow_folder)
    scan = replace(survey.scans[0], samples=("Salt/water", "Salt/hexane"))
    workdir = WorkDir(tmp_path / "work")
    workdir.write_manifest(RunSelection(workflow_folder, None))
    workdir.write_survey(replace(survey, scans=[scan]).to_dict())
    missing = report_obligations([workdir.root], "Two magnet references, cold and warm.")
    assert any("SIM .nxs" in message for message in missing)
    assert sum("recorded sample" in message for message in missing) == 2
    stated = report_obligations([workdir.root], "SIM .nxs: Salt/water and Salt/hexane.")
    assert not any(
        "recorded sample" in message or "acquisition provenance" in message for message in stated
    )


def test_muonium_identity_is_not_an_unrelated_failed_species_fit() -> None:
    from asymmetry.cli._numbers import unstated

    log = "MUONIUM: runs 10-12 have a fast line."
    assert unstated("The muonium model failed on the Ag control.", log)
    assert not unstated("Muonium amplitude falls with the instrument frequency response.", log)


@pytest.mark.parametrize(
    "description",
    [
        "No site assignment was attempted; fields were not converted into couplings.",
        "A hyperfine model was not attempted because the scan is incomplete.",
    ],
)
def test_unattempted_extraction_does_not_state_a_cli_capability_limit(description):
    from asymmetry.cli._numbers import unstated

    log = "No radical ALC or hyperfine model is available."
    assert unstated(description, log)
    assert not unstated("The CLI lacks a radical ALC model, so no couplings are extracted.", log)
    assert not unstated("A radical ALC model is not implemented in this CLI.", log)


def test_same_field_muonium_does_not_owe_a_frequency_response_claim():
    from asymmetry.cli._numbers import unstated

    assert not unstated(
        "Muonium envelope rates are compared across composition at the same field.",
        "MUONIUM KINETICS: runs 10-12 precess at one field.",
    )


@pytest.mark.parametrize(
    "draft",
    [
        "A_bg is about 8% of the fitted relaxing-plus-constant signal.",
        "The amplitude is 8.2% of the total.",
        "The fitted fraction is 8.2%.",
        "A relative error of 8.2%.",
    ],
)
def test_relative_percentages_need_a_printed_percentage_not_an_amplitude(draft):
    log = "A_bg 8.2 0.1\nA_1 94.0 0.2\n8 runs\n"
    assert unverified_numbers(draft, log)
    assert not unverified_numbers(draft, log + "relative fraction 8.2%\n8% of total\n")


def test_absolute_percent_asymmetry_remains_a_printed_quantity():
    assert not unverified_numbers("A_bg is 8.2% and the amplitude is about 8%.", "A_bg 8.2 0.1\n")


@pytest.mark.parametrize("name", ["nu", "ν"])
@pytest.mark.parametrize("number", ["0.000000", "10.257125"])
def test_dimensionless_exponent_value_and_error_cannot_acquire_rate_units(name, number):
    from asymmetry.cli._numbers import unverified_units

    log = (
        "parameter  value  error  unscaled error  unit\n"
        "---------  -----  -----  --------------  ----\n"
        "nu  0.000000  10.257125 (at bound)  3.955147  dimensionless\n"
    )
    assert unverified_units(f"Rejected exponent {name} has error {number} μs⁻¹.", log)
    assert not unverified_units(f"Exponent {name} = {number} dimensionless.", log)


@pytest.mark.parametrize(
    ("draft", "flagged"),
    [
        ("Holding m fixed, nu = 2 MHz.", False),
        ("Holding m fixed, nu = 2 dimensionless.", True),
        ("With nu = 2 MHz, m = 2 dimensionless.", False),
        ("With nu = 2 MHz, m = 2 MHz.", True),
    ],
)
def test_unit_check_binds_each_number_to_the_nearest_named_parameter(draft, flagged):
    from asymmetry.cli._numbers import unverified_units

    log = "parameter value unit error\nm 2 dimensionless 0.1\nnu 2 MHz 0.1\n"
    assert bool(unverified_units(draft, log)) is flagged


def test_parameter_uncertainty_uses_units_after_value_or_errors():
    from asymmetry.cli._numbers import unverified_units

    for row in ["A_hf  512.3456  MHz  0.123456", "A_hf  512.3456  0.123456  MHz"]:
        log = "parameter  value  error  unit\n" + row + "\n"
        assert unverified_units("A_hf uncertainty 0.123456 G", log)
        assert not unverified_units("A_hf uncertainty 0.123456 MHz", log)


def test_flagged_adjacent_parameter_is_not_replaced_by_an_unrelated_upturn():
    from asymmetry.cli._numbers import unstated

    log = "ADJACENT ENDPOINT: nu at run 1; next measured run 2 is bound-pinned."
    assert unstated("TF sigma has a cold upturn; nu rises at high temperature.", log)
    assert not unstated(
        "The next adjacent nu fit is bound-pinned, so its upturn is uncertain.", log
    )


def test_extreme_logger_is_not_a_temperature_confound_without_validation():
    from asymmetry.cli._numbers import unstated

    log = "EXTREME TEMPERATURE OFFSET: run 17. Check its channel."
    assert unstated("The high logger value confounds the foil curve.", log)
    assert not unstated("The logger is railed and irrelevant to this calibration.", log)
    assert not unstated("The sample temperature was independently validated.", log)
