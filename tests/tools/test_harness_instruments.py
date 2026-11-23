"""Instrument-name runtime decisions stay with the profile authority."""

import pytest

from tools.harness_checks import (
    INSTRUMENT_DISPATCH_BASELINE,
    find_instrument_dispatch_violations,
)


def write(root, relative, source):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


@pytest.fixture
def profiles(tmp_path):
    write(
        tmp_path,
        "src/asymmetry/core/instruments/example.py",
        'raise RuntimeError("Profiles must not be imported")\n'
        'identity = Identity("isis.emu", "ISIS", "EMU", "1")\n'
        'recognition = Recognition(("GPS",), filename_tokens=("NewAlias",))\n',
    )
    return tmp_path


def test_existing_instrument_dispatch_matches_exact_expression_budget():
    assert sum(sum(counts.values()) for counts in INSTRUMENT_DISPATCH_BASELINE.values()) == 5
    assert len(INSTRUMENT_DISPATCH_BASELINE) == 3
    assert find_instrument_dispatch_violations() == []


@pytest.mark.parametrize("folder", ["core/io", "core/transform", "gui"])
@pytest.mark.parametrize(
    "condition",
    [
        'instrument.strip().upper() == "EMU"',
        'metadata["facility"] in {"ISIS", "PSI"}',
        '"gps" == run.beamline.casefold()',
        'profile_key not in ("isis.emu",)',
        'layout_key in ["NewAlias"]',
    ],
)
def test_new_loader_or_consumer_dispatch_is_rejected(profiles, folder, condition):
    path = write(profiles, f"src/asymmetry/{folder}/example.py", f"if {condition}:\n    pass\n")
    failures = find_instrument_dispatch_violations(profiles, baseline={})
    assert len(failures) == 1
    assert failures[0].path == path
    assert failures[0].line == 1
    assert "core/instruments" in failures[0].message


@pytest.mark.parametrize(
    "source",
    [
        'result = instrument == "EMU"\n',
        'def supported(facility):\n    return facility == "ISIS"\n',
    ],
)
def test_assigned_and_returned_boolean_decisions_are_rejected(profiles, source):
    write(profiles, "src/asymmetry/core/io/example.py", source)
    assert len(find_instrument_dispatch_violations(profiles, baseline={})) == 1


def test_metadata_labels_and_versioned_migrations_do_not_dispatch_runtime_profiles(profiles):
    source = 'if facility == "ISIS":\n    pass\n'
    for relative in (
        "src/asymmetry/core/instruments/profile_logic.py",
        "src/asymmetry/core/project/instrument_migration.py",
        "src/asymmetry/core/project/schema.py",
        "tests/core/example.py",
        "docs/example.py",
    ):
        write(profiles, relative, source)
    write(
        profiles,
        "src/asymmetry/core/io/example.py",
        (
            'metadata["instrument"] = "EMU"\nlabel = "GPS"\n'
            'if label == "GPS":\n    pass\n'
            'if facility == "unknown":\n    pass\n'
            "if profile.source.pulsed:\n    pass\n"
        ),
    )
    assert find_instrument_dispatch_violations(profiles, baseline={}) == []


def test_only_versioned_migration_paths_are_exempt(profiles):
    write(profiles, "src/asymmetry/core/io/schema.py", 'if instrument == "EMU":\n    pass\n')
    assert len(find_instrument_dispatch_violations(profiles, baseline={})) == 1


@pytest.mark.parametrize("change", ["add", "replace", "remove", "delete"])
def test_expression_baseline_cannot_grow_swap_or_remain_stale(profiles, change):
    relative = "src/asymmetry/core/io/example.py"
    original = 'if instrument == "EMU":\n    pass\n'
    path = write(profiles, relative, original)
    baseline = {relative: {"instrument == 'EMU'": 1}}
    assert find_instrument_dispatch_violations(profiles, baseline=baseline) == []
    if change == "delete":
        path.unlink()
    else:
        path.write_text(
            {
                "add": original * 2,
                "replace": 'if instrument == "GPS":\n    pass\n',
                "remove": "pass\n",
            }[change]
        )
    assert len(find_instrument_dispatch_violations(profiles, baseline=baseline)) == 1
    if change in {"remove", "delete"}:
        assert find_instrument_dispatch_violations(profiles, baseline={}) == []
