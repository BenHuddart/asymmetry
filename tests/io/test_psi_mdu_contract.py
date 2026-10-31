"""MDU byte-format and raw-run contracts for the PSI loader."""

import struct
from dataclasses import replace

import numpy as np
import pytest

from asymmetry.core.io.psi import _FE_HEADER, PsiLoader, _MduHeader, _MduTag

pytestmark = pytest.mark.io


def mdu_header(fmt="T5", **changes):
    return replace(
        _MduHeader(
            fmt,
            "01-JAN-2026",
            "10:00:00",
            "01-JAN-2026",
            "11:00:00",
            2468,
            "Sample    25.0K     1.0G      TF",
            "MDU run",
            1,
            2,
            "0 1",
            "25.0",
            "0.1",
            0,
            _FE_HEADER.size,
            1972,
            60,
            8,
        ),
        **changes,
    )


def mdu_tag(label="F1", kind=b"P", low=0, high=5, t0=1, first=2, last=5):
    return _MduTag(label, kind, low, high, t0, first, last)


def tag_bytes(tag):
    return struct.pack(
        "<12sc3x11i",
        tag.label.encode(),
        tag.tag_type,
        0,
        0,
        0,
        0,
        0,
        0,
        tag.histominb,
        tag.histomaxb,
        tag.t0b,
        tag.tfb,
        tag.tlb,
    )


def mdu_bytes(fmt="T5", *, tags=None, payload_tags=None, code=None, **changes):
    count = 32 if fmt == "T5" else 16
    if tags is None:
        tags = [mdu_tag(), mdu_tag("B1")]
    tags = tags + [mdu_tag("", b"N", high=0)] * (count - len(tags))
    prefix = [0] * 13
    prefix[9] = (11 if fmt == "M3" else 100) if code is None else code
    settings = struct.pack("<13i", *prefix) + b"".join(map(tag_bytes, tags))
    header = mdu_header(fmt, num_bytes_settings=len(settings), **changes)
    data = (
        _FE_HEADER.pack(
            fmt[:1].encode(),
            fmt[1:].encode(),
            header.start_date.encode(),
            header.start_time.encode(),
            header.end_date.encode(),
            header.end_time.encode(),
            header.run_number,
            0,
            header.run_title.encode(),
            header.run_subtitle.encode(),
            b"",
            header.histo_resolution,
            0,
            0,
            header.number_of_detectors,
            header.detector_number_list.encode(),
            header.mean_temp.encode(),
            header.temp_dev.encode(),
            header.bin_size,
            header.num_bytes_header,
            header.num_bytes_settings,
            header.num_bytes_tag,
            header.num_bytes_statistics,
        )
        + settings
        + bytes(header.num_bytes_statistics)
    )
    for tag in tags if payload_tags is None else payload_tags:
        data += tag_bytes(tag)
        if tag.tag_type == b"P" and tag.n_bins > 1:
            data += np.arange(1, tag.n_bins + 1, dtype="<i4").tobytes()
    return data


@pytest.mark.parametrize(
    "fmt,width,length", [("M3", 0.000625, 5), ("T4", 0.00009765625, 6), ("T5", 0.00009765625, 6)]
)
def test_mdu_raw_run_formats(tmp_path, fmt, width, length):
    path = tmp_path / "tdc_hifi_2468.mdu"
    path.write_bytes(mdu_bytes(fmt))
    run = PsiLoader()._read_mdu(path)
    assert run.source_file == str(path)
    assert run.psi_format == "psi-mdu"
    assert run.run_number == 2468
    assert (run.title, run.comment, run.sample) == ("MDU run", "MDU run", "Sample")
    assert (run.temperature, run.field, run.orientation, run.setup) == (25.0, 1.0, "TF", "")
    assert (run.started, run.stopped) == ("01-JAN-2026 10:00:00", "01-JAN-2026 11:00:00")
    assert run.bin_width_us == width
    assert run.histogram_labels == ["F1", "B1"]
    assert run.t0_bins == [1, 1]
    assert run.first_good_bins == [2, 2]
    assert run.last_good_bins == [5, 5]
    assert run.temperature_logs is None
    assert run.sensor_temperatures == []
    for counts in run.counts:
        assert counts.dtype == np.float64
        np.testing.assert_array_equal(counts, np.arange(1, length + 1))


@pytest.mark.parametrize(
    "fmt,selected,labels",
    [("M3", "1", ["B1"]), ("T4", "", ["F1", "B1"]), ("T5", "31", ["F1", "B1"])],
)
def test_mdu_detector_selection(tmp_path, fmt, selected, labels):
    path = tmp_path / "run.mdu"
    path.write_bytes(mdu_bytes(fmt, detector_number_list=selected))
    assert PsiLoader()._read_mdu(path).histogram_labels == labels


@pytest.mark.parametrize("temperature,expected", [("", 25.0), ("0", 25.0), ("17.2", 17.2)])
def test_mdu_temperature_text_fallback(tmp_path, temperature, expected):
    path = tmp_path / "run.mdu"
    path.write_bytes(mdu_bytes(mean_temp=temperature))
    assert PsiLoader()._read_mdu(path).temperature == expected


def test_mdu_offsets_padding_crop_and_marker_scaling(tmp_path):
    path = tmp_path / "run.mdu"
    tags = [mdu_tag(low=2, high=5), mdu_tag("B1", low=-2, high=8)]
    data = bytearray(mdu_bytes(tags=tags, histo_resolution=3))
    start = _FE_HEADER.size + 1972 + 8 + 60
    struct.pack_into("<i", data, start, -17)
    path.write_bytes(data)
    run = PsiLoader()._read_mdu(path)
    np.testing.assert_array_equal(run.counts[0], [0, 0, -17, 2, 3, 4, 0, 0, 0])
    np.testing.assert_array_equal(run.counts[1], [1, 2, 3, 4, 5, 6, 7, 8, 9])
    assert run.t0_bins == [5, 5]
    assert run.first_good_bins == [8, 8]
    assert run.last_good_bins == [15, 15]


@pytest.mark.parametrize(
    "code,width",
    [
        (6, 0.00001953125),
        (7, 0.0000390625),
        (8, 0.000078125),
        (9, 0.00015625),
        (10, 0.0003125),
        (11, 0.000625),
    ],
)
@pytest.mark.parametrize("resolution", [-5, 1, 8])
def test_mdu_m3_resolution(code, width, resolution):
    settings = struct.pack("<13i", *([0] * 9 + [code, 0, 0, 0]))
    assert PsiLoader()._mdu_resolution(mdu_header("M3", histo_resolution=resolution), settings) == (
        width,
        2 ** max(0, resolution + 8 - code),
        16,
    )


@pytest.mark.parametrize("fmt,total", [("T4", 16), ("T5", 32)])
@pytest.mark.parametrize(
    "code,width",
    [(25, 0.0000244140625), (100, 0.00009765625), (200, 0.0001953125), (800, 0.00078125)],
)
@pytest.mark.parametrize("resolution", [-2, 0, 1, 4])
def test_mdu_tdc_resolution(fmt, total, code, width, resolution):
    settings = struct.pack("<13i", *([0] * 9 + [code, 0, 0, 0]))
    assert PsiLoader()._mdu_resolution(mdu_header(fmt, histo_resolution=resolution), settings) == (
        width,
        max(1, resolution),
        total,
    )


@pytest.mark.parametrize(
    "fmt,message",
    [
        ("M3", "Unsupported PSI-MDU pTA timespan: 0"),
        ("T4", "Unsupported PSI-MDU TDC resolution code: 0"),
        ("T5", "Unsupported PSI-MDU TDC resolution code: 0"),
    ],
)
def test_mdu_unsupported_resolution(fmt, message):
    with pytest.raises(ValueError, match=message):
        PsiLoader()._mdu_resolution(mdu_header(fmt), bytes(52))


def test_mdu_short_resolution_settings():
    with pytest.raises(struct.error):
        PsiLoader()._mdu_resolution(mdu_header(), bytes(51))


@pytest.mark.parametrize(
    "text,expected",
    [("", set()), ("0 1 1 31 32 -1 junk 2.0 +3 004", {0, 1, 3, 4, 31}), ("1,2\t7\n8", {7, 8})],
)
def test_mdu_detector_list_tokens(text, expected):
    assert PsiLoader()._selected_mdu_tags(text) == expected


@pytest.mark.parametrize(
    "fmt,selected,expected",
    [("M3", {1, 2}, (8, 1)), ("T4", set(), (9, 2)), ("T5", set(), (9, 2)), ("M3", set(), (0, 0))],
)
def test_mdu_shape_uses_largest_eligible_extent(fmt, selected, expected):
    tags = [mdu_tag(), mdu_tag("N", b"N"), mdu_tag("B1", low=2, high=8), mdu_tag("single", high=0)]
    assert PsiLoader()._mdu_histogram_shape(mdu_header(fmt), tags, selected, 4) == expected


def test_mdu_shape_total_limit_and_nonpositive_extent():
    loader = PsiLoader()
    assert loader._mdu_histogram_shape(mdu_header(), [mdu_tag(), mdu_tag()], set(), 1) == (6, 1)
    assert loader._mdu_histogram_shape(mdu_header(), [mdu_tag(low=-5, high=-2)], set(), 1) == (0, 1)


@pytest.mark.parametrize(
    "length,message",
    [
        (1, "too small to contain a header"),
        (_FE_HEADER.size + 3, "ended while reading settings"),
        (_FE_HEADER.size + 1972 + 3, "ended while reading statistics"),
        (_FE_HEADER.size + 1972 + 8 + 3, "ended while reading histogram tag records"),
        (_FE_HEADER.size + 1972 + 8 + 60 + 3, "ended while reading histogram data"),
    ],
)
def test_mdu_truncated_sections(tmp_path, length, message):
    path = tmp_path / "run.mdu"
    path.write_bytes(mdu_bytes()[:length])
    with pytest.raises(ValueError, match=message):
        PsiLoader()._read_mdu(path)


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"num_bytes_header": 1}, "Unsupported PSI-MDU header size: 1"),
        ({"num_bytes_tag": 59}, "Unsupported PSI-MDU tag size: 59"),
    ],
)
def test_mdu_invalid_record_sizes(tmp_path, changes, message):
    path = tmp_path / "run.mdu"
    path.write_bytes(mdu_bytes(**changes))
    with pytest.raises(ValueError, match=message):
        PsiLoader()._read_mdu(path)


def test_mdu_no_usable_histograms(tmp_path):
    path = tmp_path / "run.mdu"
    path.write_bytes(mdu_bytes(tags=[]))
    with pytest.raises(ValueError, match="does not contain usable positron histograms"):
        PsiLoader()._read_mdu(path)


def test_mdu_missing_payload_histogram_rejects_inconsistent_eligibility(tmp_path):
    path = tmp_path / "run.mdu"
    tags = [mdu_tag(), mdu_tag("B1")]
    payload = [mdu_tag(), mdu_tag("", b"N", high=0)] + [mdu_tag("", b"N", high=0)] * 30
    path.write_bytes(mdu_bytes(tags=tags, payload_tags=payload))
    with pytest.raises(ValueError, match="settings and histogram eligibility disagree for tag 1"):
        PsiLoader()._read_mdu(path)


def test_mdu_zero_extent_does_not_determine_array_length():
    tags = [mdu_tag(low=-3, high=-1), mdu_tag(high=7)]
    assert PsiLoader()._mdu_histogram_shape(mdu_header(), tags, set(), 2) == (8, 2)


@pytest.mark.parametrize("fmt", ["T4", "T5"])
@pytest.mark.parametrize("reverse", [False, True])
def test_mdu_unequal_extents_preserve_all_counts(tmp_path, fmt, reverse):
    path = tmp_path / "run.mdu"
    tags = [mdu_tag(high=2, t0=11, first=12, last=13), mdu_tag("B1", low=2, high=8)]
    expected = [[1, 2, 3, 0, 0, 0, 0, 0, 0], [0, 0, 1, 2, 3, 4, 5, 6, 7]]
    if reverse:
        tags.reverse()
        expected.reverse()
    path.write_bytes(mdu_bytes(fmt, tags=tags))

    run = PsiLoader()._read_mdu(path)

    assert run.histogram_labels == [tag.label for tag in tags]
    for counts, values in zip(run.counts, expected, strict=True):
        np.testing.assert_array_equal(counts, values)
    assert run.t0_bins == [tag.t0b for tag in tags]
    assert run.first_good_bins == [tag.tfb for tag in tags]
    assert run.last_good_bins == [tag.tlb for tag in tags]


@pytest.mark.parametrize("settings_high", [8, 2**30])
def test_mdu_rejects_inconsistent_ranges_before_any_count_allocation(
    tmp_path, monkeypatch, settings_high
):
    path = tmp_path / "run.mdu"
    settings_tags = [mdu_tag(), mdu_tag("B1", high=settings_high)]
    payload_tags = [mdu_tag(), mdu_tag("B1")] + [mdu_tag("", b"N", high=0)] * 30
    path.write_bytes(mdu_bytes(tags=settings_tags, payload_tags=payload_tags))

    def reject_allocation(*args, **kwargs):
        pytest.fail("Count arrays must be allocated only after all record ranges are validated")

    monkeypatch.setattr(np, "zeros", reject_allocation)
    with pytest.raises(
        ValueError, match="PSI-MDU settings and histogram bin ranges disagree for tag 1"
    ):
        PsiLoader()._read_mdu(path)


def test_mdu_record_labels_and_markers_may_differ_from_settings(tmp_path):
    path = tmp_path / "run.mdu"
    payload_tags = [mdu_tag("actual", t0=17, first=18, last=29)]
    payload_tags += [mdu_tag("", b"N", high=0)] * 31
    path.write_bytes(mdu_bytes(tags=[mdu_tag()], payload_tags=payload_tags))

    run = PsiLoader()._read_mdu(path)

    assert run.histogram_labels == ["actual"]
    assert run.t0_bins == [17]
    assert run.first_good_bins == [18]
    assert run.last_good_bins == [29]
    np.testing.assert_array_equal(run.counts[0], [1, 2, 3, 4, 5, 6])


@pytest.mark.parametrize("truncated", [False, True])
def test_mdu_closes_source_and_returns_independent_writable_counts(
    tmp_path, monkeypatch, truncated
):
    path = tmp_path / "run.mdu"
    data = mdu_bytes()
    path.write_bytes(data[: _FE_HEADER.size + 1972 + 8 + 60 + 3] if truncated else data)
    original_open = type(path).open
    opened = []

    def track_open(source, *args, **kwargs):
        stream = original_open(source, *args, **kwargs)
        opened.append(stream)
        return stream

    monkeypatch.setattr(type(path), "open", track_open)
    if truncated:
        with pytest.raises(ValueError, match="ended while reading histogram data"):
            PsiLoader()._read_mdu(path)
    else:
        run = PsiLoader()._read_mdu(path)
        path.unlink()
        assert all(counts.flags.owndata and counts.flags.writeable for counts in run.counts)
        run.counts[0][0] = -123
        assert run.counts[1][0] == 1
        np.testing.assert_array_equal(run.counts[0], [-123, 2, 3, 4, 5, 6])
    assert opened and all(stream.closed for stream in opened)


@pytest.mark.parametrize("fmt", ["M3", "T4", "T5"])
def test_mdu_rejects_relocated_eligibility_before_allocation(tmp_path, monkeypatch, fmt):
    path = tmp_path / "run.mdu"
    settings_tags = [mdu_tag(high=2**30), mdu_tag("B1", kind=b"N")]
    payload_tags = [mdu_tag(kind=b"N", high=0), mdu_tag("B1")]
    payload_tags += [mdu_tag("", b"N", high=0)] * ((32 if fmt == "T5" else 16) - 2)
    path.write_bytes(mdu_bytes(fmt, tags=settings_tags, payload_tags=payload_tags))

    def reject_allocation(*args, **kwargs):
        pytest.fail("Relocated histograms must be rejected before count allocation")

    monkeypatch.setattr(np, "zeros", reject_allocation)
    with pytest.raises(
        ValueError, match="PSI-MDU settings and histogram eligibility disagree for tag 0"
    ):
        PsiLoader()._read_mdu(path)


@pytest.mark.parametrize("trailing_bytes", [0, 16 * 1024**2])
def test_mdu_peak_memory_is_close_to_returned_counts(tmp_path, trailing_bytes):
    import gc
    import tracemalloc

    path = tmp_path / "run.mdu"
    path.write_bytes(mdu_bytes(tags=[mdu_tag(str(index), high=65535) for index in range(32)]))
    with path.open("ab") as stream:
        stream.truncate(path.stat().st_size + trailing_bytes)
    gc.collect()
    tracemalloc.start()
    try:
        run = PsiLoader()._read_mdu(path)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    output_bytes = sum(counts.nbytes for counts in run.counts)
    assert peak < output_bytes * 1.5
