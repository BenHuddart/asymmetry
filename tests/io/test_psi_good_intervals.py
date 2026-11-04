"""Tagged observations distinguish file bounds, storage, and aligned selection."""

import shutil
import struct
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from asymmetry.core.io import load
from asymmetry.core.io.psi import _FE_HEADER
from asymmetry.core.project.profiles import (
    ProfileFingerprint,
    profile_from_payload,
    resolve_effective_grouping,
)
from asymmetry.core.simulate import reduce_run_to_dataset
from asymmetry.core.transform.grouping import resolve_alignment, sum_aligned_counts
from tests.io.test_psi_loader import _write_psi_bin
from tests.io.test_psi_mdu_contract import mdu_bytes, mdu_tag

pytestmark = pytest.mark.io
FIXTURES = Path(__file__).parents[1] / "fixtures" / "psi_root"


@pytest.fixture
def boundary_file(tmp_path):
    """Author BIN/MDU byte records independently of the production decoders."""

    def write(kind, last=60, first=16, terminal_zero=False):
        path = tmp_path / ("boundary.bin" if kind == "BIN" else "boundary.mdu")
        tags = np.stack([100 + np.arange(128), 300 + 3 * np.arange(128)]).astype("<i4")
        if terminal_zero:
            tags[:, -1] = 0
        if kind == "BIN":
            _write_psi_bin(
                path,
                labels=[b"Forw", b"Back"],
                counts=tags,
                t0_values=[4, 7],
                first_good_values=[first, first],
            )
            data = bytearray(path.read_bytes())
            struct.pack_into("<2h", data, 522, last, last)
        else:
            # q=3 maps markers 1/2 to anchors 5/8, 8 to first 26,
            # and 20 to last 60. M3 omits its declared terminal sample.
            resolution, code, high, statistics = {
                "M3": (5, 11, 128, 200),
                "T4": (3, 100, 127, 200),
                "T5": (3, 100, 127, 328),
            }[kind]
            records = [
                mdu_tag("Forw", high=high, t0=1, first=8, last=20),
                mdu_tag("Back", high=high, t0=2, first=8, last=20),
            ]
            data = bytearray(
                mdu_bytes(
                    kind,
                    tags=records,
                    histo_resolution=resolution,
                    code=code,
                    num_bytes_statistics=statistics,
                    bin_size=4,
                )
            )
            header = list(_FE_HEADER.unpack(data[: _FE_HEADER.size]))
            header[13] = high + 1
            data[: _FE_HEADER.size] = _FE_HEADER.pack(*header)
            start = _FE_HEADER.size + (1972 if kind == "T5" else 1012) + statistics
            for i in range(2):
                samples = np.append(tags[i], np.int32(987654)) if kind == "M3" else tags[i]
                offset = start + i * (60 + 4 * (high + 1)) + 60
                data[offset : offset + samples.nbytes] = samples.tobytes()
        path.write_bytes(data)
        return path, tags

    return write


@pytest.mark.parametrize("last,stop", [(60, 58), (127, 125), (128, 125)])
@pytest.mark.parametrize("terminal_zero", [False, True])
def test_bin_inclusive_bounds_keep_raw_identity(boundary_file, last, stop, terminal_zero):
    path, tags = boundary_file("BIN", last=last, terminal_zero=terminal_zero)
    dataset = load(path)
    run = dataset.run
    assert [(h.good_bin_start, h.good_bin_end) for h in run.histograms] == [
        (16, min(last, 127)),
        (16, min(last, 127)),
    ]
    for h, counts in zip(run.histograms, tags, strict=True):
        np.testing.assert_array_equal(h.counts, counts)
    # The backward anchor is three bins later: its upper bound limits F.
    raw = np.arange(16, stop)
    alignment = resolve_alignment(run.histograms, [0], [1])
    np.testing.assert_array_equal(
        sum_aligned_counts(run.histograms, [0], alignment)[16:stop], tags[0, raw]
    )
    np.testing.assert_array_equal(
        sum_aligned_counts(run.histograms, [1], alignment)[16:stop], tags[1, raw + 3]
    )
    assert (run.grouping["first_good_bin"], run.grouping["last_good_bin"]) == (19, stop + 2)
    forward_id, backward_id = run.grouping["forward_group"], run.grouping["backward_group"]
    assert run.grouping["groups"][forward_id] == [2]
    assert run.grouping["groups"][backward_id] == [1]
    f, b = tags[1, raw + 3].astype(float), tags[0, raw].astype(float)
    np.testing.assert_allclose(dataset.asymmetry, 100 * (f - b) / (f + b))
    positive = (f > 0) & (b > 0)
    np.testing.assert_allclose(
        dataset.error[positive],
        200 * np.sqrt(f[positive] * b[positive] / (f[positive] + b[positive]) ** 3),
    )
    np.testing.assert_allclose(dataset.time, (raw - 4) * run.histograms[0].bin_width)
    reduced = reduce_run_to_dataset(run)
    np.testing.assert_allclose(reduced.time, dataset.time, rtol=1e-14, atol=1e-16)
    np.testing.assert_array_equal(reduced.asymmetry, dataset.asymmetry)
    np.testing.assert_allclose(reduced.error, dataset.error, rtol=1e-14)


@pytest.mark.parametrize(
    "kind,anchors,first,last,shift",
    [
        ("M3", [7, 11], 35, 80, 4),
        ("T4", [5, 8], 26, 60, 3),
        ("T5", [5, 8], 26, 60, 3),
    ],
)
def test_mdu_marker_bounds_select_tagged_observations(
    boundary_file, kind, anchors, first, last, shift
):
    path, tags = boundary_file(kind)
    dataset = load(path)
    run = dataset.run
    assert [h.t0_bin for h in run.histograms] == anchors
    assert [(h.good_bin_start, h.good_bin_end) for h in run.histograms] == [(first, last)] * 2
    for h, counts in zip(run.histograms, tags, strict=True):
        np.testing.assert_array_equal(h.counts, counts)
    raw = np.arange(first, last - shift + 1)
    align = resolve_alignment(run.histograms, [0], [1])
    np.testing.assert_array_equal(sum_aligned_counts(run.histograms, [0], align)[raw], tags[0, raw])
    np.testing.assert_array_equal(
        sum_aligned_counts(run.histograms, [1], align)[raw], tags[1, raw + shift]
    )
    reduced = reduce_run_to_dataset(run)
    np.testing.assert_allclose(dataset.time, reduced.time, rtol=1e-14, atol=1e-16)
    np.testing.assert_allclose(dataset.asymmetry, reduced.asymmetry)
    assert dataset.time.size == last - shift - first + 1


@pytest.mark.parametrize("last,points", [(0, 0), (3, 1), (128, 125)])
def test_zero_and_single_bin_intersections_preserve_acquisition(boundary_file, last, points):
    path, tags = boundary_file("BIN", first=0, last=last)
    dataset = load(path)
    assert dataset.time.size == points
    for h, counts in zip(dataset.run.histograms, tags, strict=True):
        np.testing.assert_array_equal(h.counts, counts)
    if points == 0:
        assert "empty good-bin support" in dataset.metadata["correction_error"]
    else:
        np.testing.assert_array_equal(reduce_run_to_dataset(dataset.run).time, dataset.time)


@pytest.mark.parametrize(
    "filename,first,last,length",
    [
        ("synthetic.bin", 16, 127, 128),
        ("musrroot-folder.root", 16, 127, 129),
        ("musrroot-directory.root", 16, 127, 129),
        ("legacy-lem.root", 4, 128, 129),
    ],
)
def test_serialized_root_defaults_are_distinct_from_container_length(filename, first, last, length):
    if filename.endswith(".root"):
        pytest.importorskip("uproot")
    dataset = load(FIXTURES / filename)
    run = dataset.run
    assert [h.n_bins for h in run.histograms] == [length] * 2
    assert [h.good_bin_end for h in run.histograms] == [last] * 2
    assert run.histograms[0].good_bin_start == first
    for i, h in enumerate(run.histograms):
        np.testing.assert_array_equal(h.counts[:128], (2 * i + 1) * (100 + np.arange(128)))
        np.testing.assert_array_equal(h.counts[128:], [] if length == 128 else [0])
    reduced = reduce_run_to_dataset(run)
    np.testing.assert_allclose(dataset.time, reduced.time, rtol=1e-14, atol=1e-16)
    np.testing.assert_array_equal(dataset.asymmetry, reduced.asymmetry)
    assert dataset.time.size == last - first - 3 + 1


@pytest.mark.parametrize("encoded", [False, True])
@pytest.mark.parametrize("exclude", [False, True])
def test_modern_root_per_detector_bounds_ignore_spectators(tmp_path, encoded, exclude):
    uproot = pytest.importorskip("uproot")

    from tests.io.test_root_loader import _write_header_strings

    path = tmp_path / "bounds.root"
    entries = []
    values = [("RunInfo", "Time Resolution", "10 ns", 3)]
    # Raw F 6..9 aligns with B 9..12. The spectator's unrelated
    # anchor/bounds must not move those four observations or their origin.
    for i, (anchor, first, last) in enumerate([(2.25, 4, 9), (5.125, 9, 14), (20, 22, 22)], 1):
        folder = f"DetectorInfo/Detector{i:03d}"
        values.extend(
            (folder, key, str(value), kind)
            for key, value, kind in [
                ("Name", ["Forw", "Back", "Spectator"][i - 1], 0),
                ("Histo Number", i, 1),
                ("Time Zero Bin", anchor, 2),
                ("First Good Bin", first, 1),
                ("Last Good Bin", last, 1),
            ]
        )
    with uproot.recreate(path) as root_file:
        for n, (folder, key, value, kind) in enumerate(values):
            if encoded:
                text = f"{n:03d} - {key}: {value} -@{kind}"
                entries.append((folder, text, text))
            else:
                root_file[f"RunHeader/{folder}/{key}"] = value
        if encoded:
            _write_header_strings(root_file, entries)
        tags = np.stack([100 + np.arange(32), 300 + 3 * np.arange(32), 900 + np.arange(32)])
        for i, counts in enumerate(tags, 1):
            root_file[f"histos/DecayAnaModule/hDecay{i:03d}"] = (counts, np.arange(33))
    dataset = load(path)
    run = dataset.run
    profile = profile_from_payload(
        dict(
            run.grouping,
            groups={1: [1, 3] if exclude else [1], 2: [2], 3: [3]},
            forward_group=1,
            backward_group=2,
            excluded_detectors=[3] if exclude else [],
        ),
        "Selected pair",
        ProfileFingerprint("", 3),
    )
    grouping = resolve_effective_grouping(profile, run)
    assert (grouping["first_good_bin"], grouping["last_good_bin"]) == (9, 12)
    reduced = reduce_run_to_dataset(replace(run, grouping=grouping))
    raw = np.arange(6, 10)
    f, b = tags[0, raw], tags[1, raw + 3]
    np.testing.assert_allclose(reduced.asymmetry, 100 * (f - b) / (f + b))
    np.testing.assert_allclose(reduced.time, (raw - 2.125) * 0.01, rtol=1e-14)
    align = resolve_alignment(run.histograms, [0], [1])
    assert align.first_bin == 3
    assert dict(align.detector_offsets) == {0: 0, 1: -3}
    assert dict(align.timing_residuals_us)[0] == pytest.approx(0.00125)
    for h, counts in zip(run.histograms, tags, strict=True):
        np.testing.assert_array_equal(h.counts, counts)


@pytest.fixture
def musrfit_export(tmp_path):
    """Run the versioned public CLI with isolated settings and no corrections."""
    executable = shutil.which("musrfit")
    if executable is None:
        pytest.skip("musrfit executable is not installed")
    version = subprocess.run(
        [executable, "--version"], capture_output=True, text=True, check=True, timeout=15
    ).stdout
    if "1.12.1" not in version or "708f777b" not in version:
        pytest.skip("reference requires musrfit 1.12.1, revision 708f777b; ROOT 6.40.04")

    def export(path, fmt, selection="", forward="1", asymmetry=False, backward="2"):
        (tmp_path / "musrfit_startup.xml").write_text(
            f'<?xml version="1.0"?><musrfit><data_path>{path.parent}</data_path></musrfit>'
        )
        settings = (
            f"fittype 2\nalpha 4\nbackgr.fix 0 0\nforward {forward}\nbackward {backward}"
            if asymmetry
            else f"fittype 0\nnorm 1\nbackgr.fit 2\nforward {forward}"
        )
        (tmp_path / "endpoint.msr").write_text(f"""Synthetic PSI interval comparison
###############################################################
FITPARAMETER
1 norm 1000 0 none
2 bg 0 0 none
3 asym 0 0 none
4 alpha 1 0 none
###############################################################
THEORY
asymmetry 3
###############################################################
RUN {path.stem} GPS PSI {fmt}
{settings}
{selection}
fit -1 10
packing 1
###############################################################
COMMANDS
SCALE_N0_BKG FALSE
MIGRAD
SAVE
###############################################################
PLOT {2 if asymmetry else 0}
runs 1
range -1 10
""")
        result = subprocess.run(
            [executable, "endpoint.msr", "-c", "--dump", "ascii"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=30,
        )
        (tmp_path / "native.log").write_text(result.stdout + result.stderr)
        assert result.returncode == 0, result.stdout + result.stderr
        return np.loadtxt(tmp_path / "endpoint_0.dat", comments="%", delimiter=",")

    return export


@pytest.mark.integration
@pytest.mark.parametrize(
    "kind,first,stop,anchor,shift",
    [
        ("BIN", 16, 58, 4, 3),
        ("M3", 35, 77, 7, 4),
        ("T4", 26, 58, 5, 3),
        ("T5", 26, 58, 5, 3),
    ],
)
def test_native_matched_marker_selection(
    boundary_file, musrfit_export, kind, first, stop, anchor, shift
):
    path, tags = boundary_file(kind)
    run = load(path).run
    fmt = "PSI-BIN" if kind == "BIN" else "PSI-MDU"
    exported = musrfit_export(path, fmt, f"data {first} {stop}", forward="1 2")
    raw = np.arange(first, stop)
    counts = tags[0, raw] + tags[1, raw + shift]
    assert exported.shape == (len(raw), 4)
    np.testing.assert_array_equal(exported[:, 1], counts)
    # ASCII has six significant digits; identify bins before comparing stamps.
    width = run.histograms[0].bin_width
    np.testing.assert_array_equal(np.rint(exported[:, 0] / width + anchor), raw)
    np.testing.assert_allclose(exported[:, 0], (raw - anchor) * width, rtol=5e-6)
    np.testing.assert_allclose(exported[:, 2], np.sqrt(counts), rtol=5e-6)
    assert load(path).time.size == len(raw)


@pytest.mark.integration
@pytest.mark.parametrize("last,stop", [(60, 60), (127, 127), (128, 128)])
def test_native_explicit_upper_is_exclusive(boundary_file, musrfit_export, last, stop):
    path, tags = boundary_file("BIN", last=last)
    exported = musrfit_export(path, "PSI-BIN", f"data 16 {last}")
    np.testing.assert_array_equal(exported[:, 1], tags[0, 16:stop])


@pytest.mark.integration
@pytest.mark.parametrize("explicit,first,stop", [(True, 16, 58), (False, 5, 125)])
def test_native_fb_defaults_and_matched_override(
    boundary_file, musrfit_export, explicit, first, stop
):
    path, tags = boundary_file("BIN")
    exported = musrfit_export(
        path, "PSI-BIN", "data 16 58 19 61" if explicit else "", asymmetry=True
    )
    raw = np.arange(first, stop)
    f, b = tags[0, raw].astype(float), tags[1, raw + 3].astype(float)
    assert exported.shape == (len(raw), 4)
    np.testing.assert_array_equal(np.rint(exported[:, 0] / 0.01 + 4), raw)
    np.testing.assert_allclose(exported[:, 1], (f - b) / (f + b), rtol=5e-6)
    np.testing.assert_allclose(exported[:, 2], 2 * np.sqrt(f * b / (f + b) ** 3), rtol=5e-6)
    assert load(path).time.size == 42


@pytest.mark.integration
@pytest.mark.parametrize("last", [0, 60, 127, 128])
def test_native_omitted_data_does_not_use_bin_good_fields(boundary_file, musrfit_export, last):
    path, tags = boundary_file("BIN", first=16, last=last)
    exported = musrfit_export(path, "PSI-BIN")
    # Omitted data uses the reference fallback t0+10 ns to histogram length,
    # even when its header good interval is empty or starts much later.
    np.testing.assert_array_equal(exported[:, 1], tags[0, 5:128])
    np.testing.assert_allclose(exported[:, 0], np.arange(1, 124) * 0.01, rtol=3e-8)


@pytest.mark.integration
@pytest.mark.parametrize("kind,first,anchor", [("M3", 23, 7), ("T4", 107, 5), ("T5", 107, 5)])
def test_native_mdu_omitted_data_uses_timing_fallback(
    boundary_file, musrfit_export, kind, first, anchor
):
    path, tags = boundary_file(kind)
    run = load(path).run
    exported = musrfit_export(path, "PSI-MDU")
    raw = np.arange(first, 128)
    np.testing.assert_array_equal(exported[:, 1], tags[0, raw])
    width = run.histograms[0].bin_width
    np.testing.assert_array_equal(np.rint(exported[:, 0] / width + anchor), raw)
    np.testing.assert_allclose(exported[:, 0], (raw - anchor) * width, rtol=5e-6)


@pytest.mark.integration
@pytest.mark.parametrize(
    "filename,fmt",
    [
        ("legacy-lem.root", "ROOT-NPP"),
        ("musrroot-folder.root", "MUSR-ROOT"),
    ],
)
def test_native_root_omitted_data_retains_terminal_storage(musrfit_export, filename, fmt):
    exported = musrfit_export(FIXTURES / filename, fmt)
    np.testing.assert_array_equal(exported[:, 1], np.append(100 + np.arange(5, 128), 0))
    np.testing.assert_allclose(exported[:, 0], np.arange(1, 125) * 0.01, rtol=3e-8)
