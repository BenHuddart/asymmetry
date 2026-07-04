"""Tests for the ISIS ICP ``.log`` sidecar field-reading parser.

The parser is deliberately narrow: it recovers the applied-field *magnitude*
from whichever ``Field_*`` channel matches the currently-selected magnet
(``a_selected_magnet``), and only ever claims a "Zero field" *direction* when
the log explicitly says a ZF magnet is selected. It never infers TF/LF from a
magnet name or from field magnitude, and never raises on malformed input.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from asymmetry.core.io.icp_log import (
    IcpFieldReading,
    parse_icp_log_file,
    parse_icp_log_text,
    sibling_icp_log_path,
)

pytestmark = [pytest.mark.io]

# A trimmed but realistic excerpt: every Field_* channel logs continuously
# regardless of which magnet is selected (Field_Hifi keeps reporting ~15000 G
# on a run where ZF is the active magnet) — the parser must not be fooled by
# that into reporting a huge field.
_ZF_RUN_LOG = """\
2012-03-10T09:14:33\tField_ZF_Magnitude\t0.00102202618171322
2012-03-10T09:14:33\tField_Danfysik\t0
2012-03-10T09:14:33\tField_Hifi\t15006.4
2012-03-10T09:15:46\tField_ZF_Magnitude\t0.000144465902061504
2012-03-10T09:19:37\ta_selected_magnet\tActive ZF
2012-03-10T09:19:37\tField_ZF_Mode\t1
2012-03-10T09:19:41\tField_ZF_Magnitude\t0.000200071079235183
2012-03-10T09:19:45\tField_ZF_Magnitude\t0.00027827089616567
"""

_DANFYSIK_RUN_LOG = """\
2012-03-10T09:14:33\tField_Danfysik\t99.8
2012-03-10T09:14:33\tField_ZF_Magnitude\t0.0001
2012-03-10T09:19:37\ta_selected_magnet\tDanfysik
2012-03-10T09:19:40\tField_Danfysik\t100.0
"""


def test_zf_selected_magnet_reads_zero_field_direction_and_zf_magnitude() -> None:
    reading = parse_icp_log_text(_ZF_RUN_LOG)
    assert reading is not None
    assert reading.field_direction == "Zero field"
    assert reading.field_gauss == pytest.approx(0.00027827089616567)
    assert reading.selected_magnet == "Active ZF"


def test_zf_run_ignores_unrelated_field_channels() -> None:
    """Field_Hifi ~15000 G is logged on the ZF run but must not be picked up."""
    reading = parse_icp_log_text(_ZF_RUN_LOG)
    assert reading is not None
    assert reading.field_gauss < 1.0


def test_named_magnet_reads_magnitude_with_no_direction_claim() -> None:
    reading = parse_icp_log_text(_DANFYSIK_RUN_LOG)
    assert reading is not None
    assert reading.field_gauss == pytest.approx(100.0)
    # A magnet name alone never asserts TF vs LF.
    assert reading.field_direction == ""
    assert reading.selected_magnet == "Danfysik"


def test_last_selected_magnet_wins_when_magnet_changes_mid_run() -> None:
    text = (
        "2012-01-01T00:00:00\ta_selected_magnet\tDanfysik\n"
        "2012-01-01T00:00:00\tField_Danfysik\t50.0\n"
        "2012-01-01T00:05:00\ta_selected_magnet\tActive ZF\n"
        "2012-01-01T00:05:01\tField_ZF_Magnitude\t0.0002\n"
    )
    reading = parse_icp_log_text(text)
    assert reading is not None
    assert reading.field_direction == "Zero field"
    assert reading.field_gauss == pytest.approx(0.0002)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "\n\n\n",
        "not tab separated at all",
        "2012-01-01T00:00:00\tsome_channel\n",  # missing value field
        "garbage\tgarbage\tgarbage\n",  # unknown channel, no a_selected_magnet
    ],
)
def test_malformed_or_uninformative_text_returns_none(text: str) -> None:
    assert parse_icp_log_text(text) is None


def test_unrecognised_magnet_name_returns_none() -> None:
    text = "2012-01-01T00:00:00\ta_selected_magnet\tSomeNewMagnetType\n"
    assert parse_icp_log_text(text) is None


def test_selected_magnet_with_no_matching_channel_reading_returns_none() -> None:
    # Magnet is named but its channel was never actually logged.
    text = "2012-01-01T00:00:00\ta_selected_magnet\tDanfysik\n"
    assert parse_icp_log_text(text) is None


def test_non_numeric_field_value_is_skipped_not_raised() -> None:
    text = (
        "2012-01-01T00:00:00\ta_selected_magnet\tDanfysik\n"
        "2012-01-01T00:00:01\tField_Danfysik\tNOT_A_NUMBER\n"
        "2012-01-01T00:00:02\tField_Danfysik\t42.0\n"
    )
    reading = parse_icp_log_text(text)
    assert reading is not None
    assert reading.field_gauss == pytest.approx(42.0)


def test_parse_icp_log_file_missing_file_returns_none(tmp_path) -> None:
    assert parse_icp_log_file(tmp_path / "does_not_exist.log") is None


def test_parse_icp_log_file_reads_real_content(tmp_path) -> None:
    path = tmp_path / "MUSR00099999.log"
    path.write_text(_ZF_RUN_LOG)
    reading = parse_icp_log_file(path)
    assert reading is not None
    assert reading.field_direction == "Zero field"


def test_sibling_icp_log_path_matches_stem() -> None:
    assert sibling_icp_log_path("/data/MUSR00038241.nxs") == Path("/data/MUSR00038241.log")
    assert sibling_icp_log_path("/data/MUSR00038241.nxs_v2") == Path("/data/MUSR00038241.log")


def test_reading_is_frozen_dataclass() -> None:
    reading = IcpFieldReading(field_gauss=1.0, field_direction="", selected_magnet="Danfysik")
    with pytest.raises(AttributeError):
        reading.field_gauss = 2.0  # type: ignore[misc]
