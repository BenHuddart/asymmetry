"""Keep the comparison's local destinations and public labels usable."""

import re
from pathlib import Path

from docutils.nodes import make_id

DOCS = Path(__file__).resolve().parents[2] / "docs"
COMPARISON = DOCS / "explanation" / "comparison.rst"
PUBLIC_LABELS = {
    "reviewed-count-reduction",
    "comparison-asymmetry-errors",
    "comparison-binning",
    "comparison-membership",
    "comparison-calibration-policies",
    "comparison-count-calibration",
    "comparison-timing",
    "comparison-file-manual-windows",
    "comparison-nexus-good-times",
    "comparison-root",
    "comparison-psi-intervals",
    "comparison-deadtime-units",
    "comparison-deadtime-availability",
    "comparison-deadtime-scope",
    "comparison-deadtime-calibration",
    "comparison-reference-timing",
    "comparison-reference-errors",
    "comparison-physical-backgrounds",
    "comparison-spectral-conventions",
    "comparison-selected-timing",
    "comparison-count-backgrounds",
}


def test_comparison_retains_public_labels() -> None:
    labels = set(re.findall(r"^\.\. _([^:]+):$", COMPARISON.read_text(), re.MULTILINE))
    assert PUBLIC_LABELS <= labels


def test_comparison_local_references_resolve() -> None:
    labels = {
        label
        for path in DOCS.rglob("*.rst")
        if "_build" not in path.parts
        for label in re.findall(r"^\.\. _([^:]+):$", path.read_text(), re.MULTILINE)
    }
    text = COMPARISON.read_text()
    for role, content in re.findall(r":(doc|ref):`([^`]+)`", text):
        target = content.rsplit("<", 1)[-1].removesuffix(">")
        if role == "ref":
            assert target in labels, target
        else:
            parent = DOCS if target.startswith("/") else COMPARISON.parent
            assert (parent / target.lstrip("/")).with_suffix(".rst").is_file(), target


def test_comparison_avoids_repository_internal_destinations() -> None:
    text = COMPARISON.read_text()
    targets = re.findall(r"<([^>]+)>`_", text)
    targets += re.findall(r"``([^`]+)``", text)
    for target in targets:
        assert not re.search(
            r"(?:^|/)(?:docs/(?:plans|porting|studies|audit|investigations)|docs/PLANS\.md)(?:/|$)",
            target,
        ), target
        assert not re.search(r"github\.com/[^/]+/[^/]+/(?:blob|tree)/", target), target


def test_comparison_heading_ids_remain_unique() -> None:
    headings = re.findall(r"^([^\n]+)\n[-=~+]{3,}\s*$", COMPARISON.read_text(), re.MULTILINE)
    identifiers = [make_id(heading) for heading in headings]
    assert len(identifiers) == len(set(identifiers))
