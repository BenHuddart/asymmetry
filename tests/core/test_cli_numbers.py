"""Tests for :mod:`asymmetry.cli._numbers` and the ``audit`` command."""

from __future__ import annotations

from pathlib import Path

import pytest

from asymmetry import cli
from asymmetry.cli._numbers import unverified_numbers
from tests.core.conftest import SCAN_RUNS

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


def test_hedged_ratios_and_differences_are_always_listed() -> None:
    log = "B0 78.18 G\nB0 80.09 G\nBwid 12.25\nBwid 12.83\nnu 2\n"
    draft = (
        "The fields agree to within about 2 G and the widths differ by 0.6 G; "
        "the rates sit a factor of ~2 apart. B0 = 78.18 G."
    )
    assert [entry.text for entry in unverified_numbers(draft, log)] == ["2", "0.6", "2"]
