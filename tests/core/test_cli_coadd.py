"""``reduce --coadd`` and ``fourier --correlation``: a co-added run, and the radical spectrum."""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from asymmetry import cli
from asymmetry.core.data.combine import (
    combine_runs,
    reduce_combined_run,
    runs_with_dataset_metadata,
)
from asymmetry.core.fourier.correlation import DEFAULT_CORR_ORDER, breit_rabi_pair
from asymmetry.core.io import load
from asymmetry.core.workflow.workdir import WorkDir
from tests.core.conftest import SCAN_RUNS

#: The synthetic radical: a Breit–Rabi line pair beside the diamagnetic line.
RADICAL_FIELD_G = 3000.0
RADICAL_A_MHZ = 400.0
RADICAL_RUNS = (201, 202)


def _radical_signal(time_us: np.ndarray) -> np.ndarray:
    """Diamagnetic precession plus the radical's two lines, in percent."""
    nu12, nu34 = breit_rabi_pair(RADICAL_FIELD_G, RADICAL_A_MHZ)
    diamag = 0.01355342 * RADICAL_FIELD_G
    lines = np.cos(2.0 * np.pi * nu12 * time_us) + np.cos(2.0 * np.pi * nu34 * time_us)
    return 5.0 * np.cos(2.0 * np.pi * diamag * time_us) + 8.0 * np.exp(-time_us / 1.5) * lines


def _write_radical_runs(folder: Path) -> None:
    """Two runs of the radical on 1 ns bins (a 500 MHz Nyquist) in *folder*."""
    pytest.importorskip("h5py")
    from asymmetry.core.io.nexus_writer import write_nexus_v1
    from asymmetry.core.simulate import BUILTIN_TEMPLATES, simulate_run

    template = dataclasses.replace(
        BUILTIN_TEMPLATES["ideal_continuous_fb"],
        key="test_radical",
        n_bins=5000,
        t0_bin=100,
        instrument_name="SIM",
    ).build()
    template.metadata.update(
        {"temperature": 298.0, "field": RADICAL_FIELD_G, "field_state": "TF", "title": "radical"}
    )
    for seed, run_number in enumerate(RADICAL_RUNS):
        run = simulate_run(
            template,
            _radical_signal,
            total_events=2.0e7,
            seed=seed,
            run_number=run_number,
            title="radical",
        )
        write_nexus_v1(run, folder / f"SIM{run_number:08d}.nxs")


def _reduce_coadd(folder: Path, workdir: Path, runs: str, capsys) -> dict:
    cli.main(
        ["reduce", str(folder), "--runs", runs, "--coadd", "--json", "--workdir", str(workdir)]
    )
    return json.loads(capsys.readouterr().out)


def _scan_pair(folder: Path, tmp_path: Path) -> Path:
    """A private copy of the first two scan runs, free to be touched."""
    copy = tmp_path / "runs"
    copy.mkdir()
    for run_number in SCAN_RUNS[:2]:
        name = f"SIM{run_number:08d}.nxs"
        shutil.copy2(folder / name, copy / name)
    return copy


def test_a_coadd_is_the_count_level_sum_stored_under_its_first_member(
    workflow_folder: Path, tmp_path: Path, capsys
) -> None:
    workdir = tmp_path / "wd"
    payload = _reduce_coadd(workflow_folder, workdir, f"{SCAN_RUNS[0]},{SCAN_RUNS[1]}", capsys)

    [entry] = payload["entries"]
    assert entry["run_number"] == SCAN_RUNS[0]
    assert entry["members"] == list(SCAN_RUNS[:2])
    assert WorkDir(workdir).reduced_runs() == [SCAN_RUNS[0]]

    loaded = [load(str(workflow_folder / f"SIM{run:08d}.nxs")) for run in SCAN_RUNS[:2]]
    expected = reduce_combined_run(combine_runs(runs_with_dataset_metadata(loaded)))
    stored = WorkDir(workdir).reduced(SCAN_RUNS[0])
    np.testing.assert_allclose(stored.time, expected.time)
    np.testing.assert_allclose(stored.asymmetry, expected.asymmetry)
    np.testing.assert_allclose(stored.error, expected.error)


def test_a_coadd_is_named_by_reduce_and_by_every_command_that_reads_it(
    workflow_folder: Path, tmp_path: Path, capsys
) -> None:
    workdir = tmp_path / "wd"
    spec = f"{SCAN_RUNS[0]}-{SCAN_RUNS[2]}"
    note = f"Run {SCAN_RUNS[0]} is co-added from runs {SCAN_RUNS[0]}-{SCAN_RUNS[2]}."
    cli.main(["reduce", str(workflow_folder), "--runs", spec, "--coadd", "--workdir", str(workdir)])
    assert note in capsys.readouterr().out

    cli.main(
        ["fourier", str(workflow_folder), "--run", str(SCAN_RUNS[0]), "--json"]
        + ["--workdir", str(workdir)]
    )
    captured = capsys.readouterr()
    assert json.loads(captured.out)["run"] == SCAN_RUNS[0]
    assert note in captured.err


def test_a_coadd_is_reduced_again_when_a_member_file_changes(
    workflow_folder: Path, tmp_path: Path, capsys
) -> None:
    folder = _scan_pair(workflow_folder, tmp_path)
    workdir = tmp_path / "wd"
    spec = f"{SCAN_RUNS[0]}-{SCAN_RUNS[1]}"
    assert _reduce_coadd(folder, workdir, spec, capsys)["entries"][0]["recomputed"] is True
    assert _reduce_coadd(folder, workdir, spec, capsys)["entries"][0]["recomputed"] is False

    second = folder / f"SIM{SCAN_RUNS[1]:08d}.nxs"
    os.utime(second, ns=(second.stat().st_atime_ns, second.stat().st_mtime_ns + 10**9))
    assert _reduce_coadd(folder, workdir, spec, capsys)["entries"][0]["recomputed"] is True


def test_a_coadd_needs_two_runs(workflow_folder: Path, tmp_path: Path, capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        _reduce_coadd(workflow_folder, tmp_path / "wd", str(SCAN_RUNS[0]), capsys)
    assert exc.value.code == 1
    assert "--coadd sums two or more runs" in capsys.readouterr().err


def test_runs_that_cannot_be_summed_are_a_user_error_naming_the_mismatch(
    workflow_folder: Path, tmp_path: Path, capsys
) -> None:
    folder = _scan_pair(workflow_folder, tmp_path)
    _write_radical_runs(folder)
    with pytest.raises(SystemExit) as exc:
        _reduce_coadd(folder, tmp_path / "wd", f"{SCAN_RUNS[0]},{RADICAL_RUNS[0]}", capsys)
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert f"Co-add of runs {SCAN_RUNS[0]}, {RADICAL_RUNS[0]}" in err
    assert "different detector counts" in err


@pytest.fixture
def coadded_radical(tmp_path: Path, capsys) -> tuple[Path, Path]:
    """The two radical runs, co-added into a work directory."""
    folder = tmp_path / "radical"
    folder.mkdir()
    _write_radical_runs(folder)
    workdir = tmp_path / "wd"
    _reduce_coadd(folder, workdir, f"{RADICAL_RUNS[0]}-{RADICAL_RUNS[1]}", capsys)
    return folder, workdir


def _correlation(folder: Path, workdir: Path, *extra: str) -> list[str]:
    return ["fourier", str(folder), "--run", str(RADICAL_RUNS[0]), "--correlation", *extra] + [
        "--workdir",
        str(workdir),
    ]


def test_the_correlation_spectrum_of_a_coadded_radical_peaks_at_the_sum_of_its_lines(
    coadded_radical: tuple[Path, Path], capsys
) -> None:
    folder, workdir = coadded_radical
    cli.main(_correlation(folder, workdir, "--json"))
    payload = json.loads(capsys.readouterr().out)

    assert payload["axis"] == "hyperfine_coupling"
    assert payload["correlation"] == {"field_gauss": RADICAL_FIELD_G, "order": DEFAULT_CORR_ORDER}
    assert payload["name"] == f"run-{RADICAL_RUNS[0]}-correlation"
    strongest = max(payload["peak_analysis"]["peaks"], key=lambda peak: peak["amplitude"])
    nu12, nu34 = breit_rabi_pair(RADICAL_FIELD_G, RADICAL_A_MHZ)
    assert strongest["frequency_mhz"] == pytest.approx(nu12 + nu34, abs=2.0)


def test_the_correlation_table_reports_couplings(
    coadded_radical: tuple[Path, Path], capsys
) -> None:
    folder, workdir = coadded_radical
    cli.main(_correlation(folder, workdir))
    out = capsys.readouterr().out
    assert f"Run {RADICAL_RUNS[0]} correlation spectrum at {RADICAL_FIELD_G:g} G" in out
    assert "A_mu/MHz" in out


def test_the_correlation_refuses_a_coadd_whose_member_changed_since_it_was_reduced(
    coadded_radical: tuple[Path, Path], capsys
) -> None:
    folder, workdir = coadded_radical
    second = folder / f"SIM{RADICAL_RUNS[1]:08d}.nxs"
    os.utime(second, ns=(second.stat().st_atime_ns, second.stat().st_mtime_ns + 10**9))
    with pytest.raises(SystemExit) as exc:
        cli.main(_correlation(folder, workdir))
    assert exc.value.code == 1
    assert "changed since it was reduced" in capsys.readouterr().err
