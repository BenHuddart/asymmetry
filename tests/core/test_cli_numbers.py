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
