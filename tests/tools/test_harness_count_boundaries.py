"""Structural ratchets keep array truncation and local F/B window parsing out."""

from pathlib import Path

import pytest

from tests._harness import _load_harness


def test_repository_uses_the_count_boundary() -> None:
    assert _load_harness().find_count_boundary_violations() == []


@pytest.mark.parametrize(
    ("source", "violates"),
    [
        ("n = min(f.size, b.size)", True),
        ("n = min(len(f), len(b))", True),
        ("n = min(len(hist.counts) + offset for hist, offset in selected)", False),
        ("n = min(window.last, f.size - 1)", False),
        ('window = grouping.get("first_good_bin"); binned_fb_asymmetry(counts)', True),
        ('window = grouping.get("last_good_bin"); slice_to_good_window(a, e, window)', True),
        ('window = grouping.get("first_good_bin")', False),
        ("binned_fb_asymmetry(counts, grouping=grouping, alpha=alpha)", False),
    ],
)
def test_ratchet_distinguishes_reduction_from_alignment_and_metadata(
    tmp_path: Path, source: str, violates: bool
) -> None:
    folder = tmp_path / "src/asymmetry/core/transform"
    folder.mkdir(parents=True)
    (folder / "reduction.py").write_text("def reduce():\n    " + source + "\n")
    assert bool(_load_harness().find_count_boundary_violations(tmp_path)) is violates
