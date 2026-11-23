"""The displayed authoring example uses the live profile/resolver contract."""

from pathlib import Path
from textwrap import dedent


def test_displayed_instrument_authoring_example_executes_against_bundled_profile():
    page = Path(__file__).parents[2] / "docs/contributing.rst"
    marker = ".. code-block:: python\n   :name: instrument-authoring-example\n"
    source = page.read_text().split(marker, 1)[1].split("\nA numbering slot", 1)[0]
    namespace = {}
    exec(compile(dedent(source), str(page), "exec"), namespace)
    slots = namespace["slots"]
    assert {slot.ordinal for slot in slots} == set(range(1, 16))
    assert len({slot.id for slot in slots}) == 15
    assert {segment.detector_id for segment in namespace["individual"].active_segments} == {
        1,
        2,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
    }
    assert next(slot for slot in slots if slot.channel == "M").physical == ()
