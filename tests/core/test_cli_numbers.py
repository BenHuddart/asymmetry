"""Tests for :mod:`asymmetry.cli._numbers` and the ``audit`` command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from asymmetry import cli
from asymmetry.cli._numbers import unverified_numbers
from tests.core.conftest import SCAN_RUNS, ZF_RUNS

_LOG = """$ asymmetry fit-series runs --runs 102-106
run  temperature  chi2_red
102  10.000       0.969
Tc 69.1731 ± 0.0541, frequency 30.19 MHz, SNR 3.1x the noise floor
"""


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
    draft.write_text("A draft; the reduction left deadtime off.\n", encoding="utf-8")
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
    assert "----- BEGIN -----\nA draft; the reduction left deadtime off.\n----- END -----" in out


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
field 5000 G  run 91500  A_bg 3.56102  dchi2 274.851 480.2
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
    survey = {"folder": "data", "runs": [], "scans": [], "notes_scans": [scan]}
    (workdir / "survey.json").write_text(json.dumps(survey), encoding="utf-8")
    (workdir / "manifest.json").write_text(json.dumps({"folder": "data"}), encoding="utf-8")
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
    survey = {"folder": "data", "runs": [], "scans": [], "notes_scans": [scan]}
    (workdir / "survey.json").write_text(json.dumps(survey), encoding="utf-8")
    (workdir / "manifest.json").write_text(json.dumps({"folder": "data"}), encoding="utf-8")
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
        survey = {"folder": name, "runs": [], "scans": [], "notes_scans": notes}
        (root / "survey.json").write_text(json.dumps(survey), encoding="utf-8")
        (root / "manifest.json").write_text(json.dumps({"folder": name}), encoding="utf-8")
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
        json.dumps({"folder": "asymmetry-work-b"}), encoding="utf-8"
    )
    cli.main(["audit", str(draft)])
    assert "not fitted" not in capsys.readouterr().out


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
    survey = {"runs": [{"run_number": 7, "precession": "none"}]}
    (root / "survey.json").write_text(json.dumps(survey))
    series = {
        "kind": "series",
        "expression": "Oscillatory * Exponential + Constant",
        "free_params": ["A_1", "phase", "Lambda", "A_bg"],
        "trend": {"rows": [{"key": "7", "flags": ["amplitude_exceeds_data"]}]},
    }
    (root / "series" / "held.json").write_text(json.dumps(series))
    assert WorkDir(root).fitted_runs() == set()
