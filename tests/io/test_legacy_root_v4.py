"""Optional genuine v4 records and always-on normalisation boundary checks."""

import hashlib
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from asymmetry.core.io import load
from asymmetry.core.simulate import reduce_run_to_dataset
from tests.io import test_psi_good_intervals

pytestmark = pytest.mark.io
musrfit_export = test_psi_good_intervals.musrfit_export
uproot = pytest.importorskip("uproot")


@pytest.fixture(
    params=[
        ("lem08_his_2279.root", "e99524561544afe7be0b17c76f1a6621c78b340953b6f05a28b947b7ec65f292"),
        (
            "../BMWlibs/data/lem09_his_0916.root",
            "1e9597f0dd4c11ece5ed522916b87e322d5c04567c230765742622e8550afba9",
        ),
    ]
)
def v4_example(request):
    locator = os.environ.get("ASYMMETRY_MUSRFIT_DATA")
    if locator is None:
        pytest.skip("ASYMMETRY_MUSRFIT_DATA not set; genuine serialized v4 unavailable")
    relative, digest = request.param
    path = (Path(locator) / relative).resolve()
    if not path.is_file():
        pytest.skip(f"external musrfit example unavailable: {relative}")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    return path


def test_genuine_v4_calibration_and_all_stored_decay_arrays(v4_example):
    with uproot.open(v4_example) as root:
        record = root["RunInfo"].member("fFolders")[0]
        assert (record.classname, record.class_version) == ("TLemRunHeader", 4)
        assert record.member("fNHist") == 4
        assert record.member("fTimeResolution") == 0.1953125
        np.testing.assert_array_equal(record.member("fTimeZero")[:4], [3419] * 4)
        # Independently walk serialized folders, without the loader's lookup.
        folders = [root["histos"]]
        arrays = {}
        while folders:
            for obj in folders.pop().member("fFolders"):
                if obj.classname == "TFolder":
                    folders.append(obj)
                elif obj.classname.startswith("TH1") and str(obj.member("fName")).startswith(
                    "hDecay"
                ):
                    arrays[int(str(obj.member("fName"))[6:])] = obj.values(flow=False)
        run = load(v4_example).run
        assert run.run_number == record.member("fRunNumber")
        assert run.metadata["field"] == record.member("fSampleBField")
        assert run.metadata["temperature"] == record.member("fSampleTemperature")
    numbers = [0, 1, 2, 3, 20, 21, 22, 23]
    assert sorted(arrays) == numbers == run.metadata["root_histo_numbers"]
    assert run.metadata["root_format"] == "lem-root-v4"
    assert run.metadata["declared_units"]["time"]["RunInfo/Time Resolution"] == {
        "unit": "ns",
        "source": "format",
        "reference": "TLemRunHeader v4; musrfit 1.12.1 ROOT-NPP",
    }
    assert run.grouping["detector_t0_sources"] == ["file"] * 8
    for number, h in zip(numbers, run.histograms, strict=True):
        np.testing.assert_array_equal(h.counts, arrays[number])
        assert h.n_bins == 66601
        assert h.bin_width == 0.0001953125
        assert h.t0_bin == 3419
        assert h.t0_time_us == (3419.5 * h.bin_width)


@pytest.mark.integration
@pytest.mark.parametrize("fmt,pair", [("ROOT-NPP", (1, 3)), ("ROOT-PPC", (5, 7))])
@pytest.mark.parametrize("asymmetry", [False, True])
@pytest.mark.parametrize("file_t0", [True, False])
def test_native_v4_selected_observations(v4_example, musrfit_export, fmt, pair, asymmetry, file_t0):
    run = load(v4_example).run
    selected = [run.histograms[i - 1] for i in pair]
    grouping = dict(
        run.grouping,
        groups={1: [pair[0]], 2: [pair[1]]},
        forward_group=1,
        backward_group=2,
        alpha=1.0,
        beta=1.0,
        first_good_bin=3500,
        last_good_bin=3539,
        deadtime_correction=False,
        background_correction=False,
    )
    if not file_t0:
        selected = [replace(h, t0_bin=3429, t0_time_us=3429.5 * h.bin_width) for h in selected]
        histograms = list(run.histograms)
        for i, h in zip(pair, selected, strict=True):
            histograms[i - 1] = h
        run = replace(run, histograms=histograms)
    reduced = reduce_run_to_dataset(replace(run, grouping=grouping))
    selection = "data 3500 3540" + (" 3500 3540" if asymmetry else "")
    if not file_t0:
        selection += "\nt0 3429 3429"
    exported = musrfit_export(
        v4_example,
        fmt,
        selection,
        forward="1" if asymmetry else "1 3",
        backward="3",
        asymmetry=asymmetry,
    )
    f, b = [h.counts[3500:3540] for h in selected]
    assert np.all(f > 0) and np.all(b > 0)
    assert exported.shape == (40, 4)
    np.testing.assert_allclose(exported[:, 0], reduced.time, rtol=5e-6)
    if asymmetry:
        np.testing.assert_allclose(reduced.asymmetry, 100 * (f - b) / (f + b), rtol=1e-14)
        np.testing.assert_allclose(reduced.error, 200 * np.sqrt(f * b / (f + b) ** 3), rtol=1e-14)
        np.testing.assert_allclose(exported[:, 1], reduced.asymmetry / 100, rtol=5e-6, atol=5e-7)
        np.testing.assert_allclose(exported[:, 2], reduced.error / 100, rtol=5e-6)
    else:
        np.testing.assert_array_equal(exported[:, 1], f + b)
        np.testing.assert_allclose(exported[:, 2], np.sqrt(f + b), rtol=5e-6)


@pytest.mark.parametrize("version,banks", [(4, (0, 20)), (5, (0,))])
def test_lem_normalisation_boundary_sparse_identity(version, banks):
    """Own field values test normalisation, not serialized v4 decoding."""
    from types import SimpleNamespace

    from asymmetry.core.io.root import RootLoader, _RootHistogram
    from asymmetry.core.io.root_header import LEM_HEADER_UNITS, read_lem_header

    fields = dict(
        fRunNumber=123,
        fTimeResolution=25.0,
        fSampleTemperature=30.0,
        fSampleBField=100.0,
        fRunTitle="Title: Synthetic",
        fLemSetup="Setup: Test",
        fStartTimeString="Start: Test",
        fStopTimeString="Stop: Test",
        fNHist=4,
        fTimeZero=np.array([0.0, 7.25, -1.0, np.nan] + [-1.0] * 28),
    )
    record = SimpleNamespace(class_version=version, member=fields.__getitem__)
    header = read_lem_header(record)
    assert header["RunInfo/Header Class"] == f"TLemRunHeader v{version}"
    assert LEM_HEADER_UNITS[header["RunInfo/Header Class"]]["Time Resolution"].unit == "ns"
    loader = RootLoader()
    # Deliberately sparse and reordered: identity is neither ordinal nor length.
    numbers = [bank + slot for bank in reversed(banks) for slot in (3, 1, 0, 2)]
    histograms = [_RootHistogram(n, np.array([n + 1.0]), f"Synthetic {n}") for n in numbers]
    assert loader._select_histograms(header, histograms) == histograms
    for h in histograms:
        info = loader._detector_info(header, h)
        assert info["Histo Number"] == str(h.histo_number)
        anchor = fields["fTimeZero"][h.histo_number % 20]
        if np.isfinite(anchor) and anchor >= 0:
            assert float(info["Time Zero Bin"]) == anchor
        else:
            assert "Time Zero Bin" not in info
    if version == 4:
        for number in (4, 19, 24, 40):
            assert loader._detector_info(header, _RootHistogram(number, np.ones(2), "")) == {}


@pytest.mark.parametrize("anchors", [(0.0, 7.25, -1.0, np.nan), (np.inf, -np.inf, 19.5, 0.0)])
def test_genuine_v4_decoded_edge_payloads(monkeypatch, v4_example, anchors):
    """Decoded payload injections do not claim additional serialized fixtures."""
    with uproot.open(v4_example) as root:
        record = root["RunInfo"].member("fFolders")[0]
        record.members["fTimeZero"] = np.array(list(anchors) + [-1.0] * 28)
        record.members["fTimeResolution"] = 25.0
        monkeypatch.setattr(uproot, "open", lambda path: root)
        run = load(v4_example).run
    for i, h in enumerate(run.histograms):
        anchor = anchors[i % 4]
        available = np.isfinite(anchor) and anchor >= 0
        assert h.bin_width == 0.025
        assert run.grouping["detector_t0_sources"][i] == ("file" if available else "missing")
        assert h.t0_bin == (round(anchor) if available else 0)
        assert h.t0_time_us == (pytest.approx((anchor + 0.5) * 0.025) if available else None)
