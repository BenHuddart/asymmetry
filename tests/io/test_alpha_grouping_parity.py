"""Cross-format parity: PSI-BIN and MusrRoot must reduce the same run identically.

Both raw-histogram loaders (:mod:`asymmetry.core.io.psi`,
:mod:`asymmetry.core.io.root`) attach the same default grouping to a run:
``alpha`` stays at the neutral 1.0 (calibration is an explicit, user-driven
step — an automatic integral-ratio estimate on load would absorb the baseline
asymmetry of ZF/LF data into alpha), and the PSI beam-referenced Forw/Back
labels are swapped to the muon-spin analysis pair the same way. This module
builds a synthetic PSI-BIN file and a synthetic MusrRoot file carrying the
*same* histogram counts, t0, and good-bin metadata, and asserts the two
loaders agree — so the same physical run reduces the same way regardless of
which format it was saved in.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

pytestmark = [pytest.mark.io]

from asymmetry.core.io.psi import PsiLoader
from asymmetry.core.io.root import RootLoader

uproot = pytest.importorskip("uproot")

#: Shared histogram counts for both fixtures: unequal forward/backward
#: integrals, so any future divergence in how the loaders derive alpha or
#: assign the analysis pair shows up as a real mismatch.
_BACK_COUNTS = np.array([0, 10, 20, 30, 40, 50], dtype=np.float64)
_FORW_COUNTS = np.array([0, 0, 0, 15, 25, 35], dtype=np.float64)
_T0_BACK = 1
_T0_FORW = 3
_FIRST_GOOD_BACK = 3
_FIRST_GOOD_FORW = 5
_LAST_GOOD_BACK = 5
_LAST_GOOD_FORW = 5


def _write_psi_bin_fixture(path) -> None:
    labels = [b"Back", b"Forw"]
    counts = np.stack([_BACK_COUNTS, _FORW_COUNTS]).astype("<i4")
    n_hist = len(labels)
    n_bins = int(counts.shape[1])
    header = bytearray(1024)
    header[0:2] = b"1N"
    struct.pack_into("<h", header, 2, 4)
    struct.pack_into("<h", header, 6, 4321)
    struct.pack_into("<h", header, 28, n_bins)
    struct.pack_into("<h", header, 30, n_hist)
    struct.pack_into("<h", header, 128, n_hist)
    struct.pack_into("<h", header, 130, n_bins)
    struct.pack_into("<h", header, 132, 1)
    struct.pack_into("<h", header, 134, 1)
    header[138:148] = b"Sample    "
    header[148:158] = b"50.0K     "
    header[158:168] = b"0.1T      "
    header[168:178] = b"TF        "
    header[218:227] = b"01-JAN-26"
    header[227:236] = b"01-JAN-26"
    header[236:244] = b"10:00:00"
    header[244:252] = b"11:00:00"
    header[860:922] = b"PSI/ROOT alpha-parity test".ljust(62, b" ")
    for i, label in enumerate(labels):
        header[948 + i * 4 : 952 + i * 4] = label[:4].ljust(4, b" ")
    struct.pack_into("<f", header, 1012, 0.01)
    t0_values = [_T0_BACK, _T0_FORW]
    first_good_values = [_FIRST_GOOD_BACK, _FIRST_GOOD_FORW]
    last_good_values = [_LAST_GOOD_BACK, _LAST_GOOD_FORW]
    for i, value in enumerate(t0_values):
        struct.pack_into("<h", header, 458 + i * 2, value)
    for i, value in enumerate(first_good_values):
        struct.pack_into("<h", header, 490 + i * 2, value)
    for i, value in enumerate(last_good_values):
        struct.pack_into("<h", header, 522 + i * 2, value)

    path.write_bytes(bytes(header) + counts.tobytes())


def _write_musrroot_fixture(path) -> None:
    edges = np.arange(-0.5, _BACK_COUNTS.size + 0.5, 1.0)
    with uproot.recreate(path) as root_file:
        root_file["RunHeader/RunInfo/Run Number"] = "4321"
        root_file["RunHeader/RunInfo/Run Title"] = "PSI/ROOT alpha-parity test"
        root_file["RunHeader/RunInfo/Laboratory"] = "PSI"
        root_file["RunHeader/RunInfo/Sample Name"] = "Sample"
        root_file["RunHeader/RunInfo/Sample Temperature"] = "50.0 K"
        root_file["RunHeader/RunInfo/Sample Magnetic Field"] = "0.1 T"
        root_file["RunHeader/RunInfo/No of Histos"] = "2"
        # Same bin width as the PSI-BIN fixture (0.01 microsecond).
        root_file["RunHeader/RunInfo/Time Resolution"] = "10 ns"
        root_file["RunHeader/DetectorInfo/Detector001/Name"] = "Back"
        root_file["RunHeader/DetectorInfo/Detector001/Histo Number"] = "1"
        root_file["RunHeader/DetectorInfo/Detector001/Time Zero Bin"] = str(_T0_BACK)
        root_file["RunHeader/DetectorInfo/Detector001/First Good Bin"] = str(_FIRST_GOOD_BACK)
        root_file["RunHeader/DetectorInfo/Detector001/Last Good Bin"] = str(_LAST_GOOD_BACK)
        root_file["RunHeader/DetectorInfo/Detector002/Name"] = "Forw"
        root_file["RunHeader/DetectorInfo/Detector002/Histo Number"] = "2"
        root_file["RunHeader/DetectorInfo/Detector002/Time Zero Bin"] = str(_T0_FORW)
        root_file["RunHeader/DetectorInfo/Detector002/First Good Bin"] = str(_FIRST_GOOD_FORW)
        root_file["RunHeader/DetectorInfo/Detector002/Last Good Bin"] = str(_LAST_GOOD_FORW)
        root_file["histos/DecayAnaModule/hDecay001"] = (_BACK_COUNTS, edges)
        root_file["histos/DecayAnaModule/hDecay002"] = (_FORW_COUNTS, edges)


def test_psi_bin_and_musrroot_agree_on_default_alpha_and_fb_semantics(tmp_path) -> None:
    bin_path = tmp_path / "deltat_pta_gpd_4321.bin"
    root_path = tmp_path / "lem_4321.root"
    _write_psi_bin_fixture(bin_path)
    _write_musrroot_fixture(root_path)

    bin_ds = PsiLoader().load(str(bin_path))
    root_ds = RootLoader().load(str(root_path))

    bin_grouping = bin_ds.run.grouping
    root_grouping = root_ds.run.grouping

    # Both loaders swap the PSI beam Forw/Back labels to the muon-spin
    # forward/backward analysis pair the same way.
    assert bin_grouping["group_names"][bin_grouping["forward_group"]] == "Back"
    assert bin_grouping["group_names"][bin_grouping["backward_group"]] == "Forw"
    assert root_grouping["group_names"][root_grouping["forward_group"]] == "Back"
    assert root_grouping["group_names"][root_grouping["backward_group"]] == "Forw"

    # Both loaders leave alpha at the neutral default; calibration is an
    # explicit user step (an automatic integral-ratio estimate would absorb
    # the baseline asymmetry of ZF/LF runs into alpha).
    assert bin_grouping["alpha"] == pytest.approx(1.0)
    assert root_grouping["alpha"] == pytest.approx(1.0)
