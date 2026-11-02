"""Keep raw binning parsing behind the BinningSpec authority."""

import pytest

from tests._harness import _load_harness


def test_current_binning_reads_match_the_frozen_count():
    assert _load_harness().find_binning_payload_violations() == []


@pytest.mark.parametrize(
    "owner,expression,violates",
    [
        (False, 'grouping.get("bunching_factor", 1)', True),
        (False, 'payload.get("binning_mode", "fixed")', True),
        (False, 'grouping.get("alpha", 1)', False),
        (True, 'grouping.get("bunching_factor", 1)', False),
    ],
)
def test_only_the_parser_can_add_raw_binning_reads(tmp_path, owner, expression, violates):
    folder = tmp_path / "src/asymmetry/core/transform"
    folder.mkdir(parents=True)
    (folder / ("rebin.py" if owner else "consumer.py")).write_text(expression + "\n")
    assert bool(_load_harness().find_binning_payload_violations(tmp_path)) is violates
