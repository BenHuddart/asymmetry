"""Unit tests for :class:`AxisLimitPolicy`, the per-axis Auto/Hold resolver.

The policy is the single decision point for "does this axis follow the data".
These tests pin the contract the plot, frequency and ALC panels build on in
later phases of ``docs/plans/axis-limit-policy.md``: Auto follows the bounds,
Held ignores them, an axis with no held value frames itself (there is no
first-paint latch), a quantity change refits a held axis exactly once, and a
resolved value is always the new held value so the fields, the axes and the
policy can never disagree.

Pure Python — no Qt widgets and no ``QApplication`` are involved.
"""

from __future__ import annotations

from asymmetry.gui.widgets.axis_limits import AxisLimitPolicy


def test_new_axis_is_auto_and_holds_nothing() -> None:
    policy = AxisLimitPolicy()

    assert policy.is_auto("x") is True
    assert policy.held("x") is None


def test_first_resolve_frames_from_bounds_and_becomes_held() -> None:
    policy = AxisLimitPolicy()

    assert policy.resolve({"x": (0.0, 10.0)}) == {"x": (0.0, 10.0)}
    assert policy.held("x") == (0.0, 10.0)


def test_auto_axis_follows_changed_bounds() -> None:
    policy = AxisLimitPolicy()
    policy.resolve({"x": (0.0, 10.0)})

    assert policy.resolve({"x": (0.0, 32.0)}) == {"x": (0.0, 32.0)}
    assert policy.held("x") == (0.0, 32.0)


def test_held_axis_ignores_changed_bounds() -> None:
    policy = AxisLimitPolicy()
    policy.set_manual("x", 1.0, 4.0)

    assert policy.resolve({"x": (0.0, 32.0)}) == {"x": (1.0, 4.0)}
    assert policy.held("x") == (1.0, 4.0)


def test_set_manual_sorts_an_inverted_pair_and_clears_auto() -> None:
    policy = AxisLimitPolicy()

    policy.set_manual("y", 30.0, -30.0)

    assert policy.held("y") == (-30.0, 30.0)
    assert policy.is_auto("y") is False


def test_set_auto_turns_following_back_on_without_forgetting_the_held_value() -> None:
    policy = AxisLimitPolicy()
    policy.set_manual("x", 1.0, 4.0)

    policy.set_auto("x", True)

    assert policy.held("x") == (1.0, 4.0)
    assert policy.resolve({"x": (0.0, 32.0)}) == {"x": (0.0, 32.0)}


def test_record_gesture_holds_only_the_axes_that_moved() -> None:
    policy = AxisLimitPolicy()
    policy.resolve({"x": (0.0, 10.0), "y": (-30.0, 30.0)})

    policy.record_gesture(
        {"x": (0.0, 10.0), "y": (-30.0, 30.0)},
        {"x": (2.0, 4.0), "y": (-30.0, 30.0)},
    )

    assert policy.is_auto("x") is False
    assert policy.held("x") == (2.0, 4.0)
    assert policy.is_auto("y") is True
    # The untouched Auto y still follows the data on the next render.
    assert policy.resolve({"y": (-5.0, 5.0)}) == {"y": (-5.0, 5.0)}


def test_record_gesture_holds_an_axis_absent_from_the_before_snapshot() -> None:
    policy = AxisLimitPolicy()

    policy.record_gesture({}, {"y:P_x": (-1.0, 1.0)})

    assert policy.is_auto("y:P_x") is False
    assert policy.held("y:P_x") == (-1.0, 1.0)


def test_quantity_change_refits_a_held_axis_exactly_once() -> None:
    policy = AxisLimitPolicy()
    policy.resolve({"y": (-30.0, 30.0)}, {"y": "asymmetry"})
    policy.set_manual("y", -10.0, 10.0)

    # A new view mode is a new quantity: the held value no longer means
    # anything on this axis, so it refits from the bounds.
    assert policy.resolve({"y": (0.0, 4000.0)}, {"y": "counts"}) == {"y": (0.0, 4000.0)}
    # ...and then holds again, because the quantity is now stamped.
    assert policy.resolve({"y": (0.0, 9000.0)}, {"y": "counts"}) == {"y": (0.0, 4000.0)}
    assert policy.is_auto("y") is False


def test_quantity_change_is_invisible_on_an_auto_axis() -> None:
    policy = AxisLimitPolicy()
    policy.resolve({"y": (-30.0, 30.0)}, {"y": "asymmetry"})

    assert policy.resolve({"y": (0.0, 4000.0)}, {"y": "counts"}) == {"y": (0.0, 4000.0)}
    assert policy.resolve({"y": (0.0, 9000.0)}, {"y": "counts"}) == {"y": (0.0, 9000.0)}
    assert policy.is_auto("y") is True


def test_first_quantity_stamp_does_not_count_as_a_change() -> None:
    policy = AxisLimitPolicy()
    policy.set_manual("x", 1.0, 4.0)

    assert policy.resolve({"x": (0.0, 32.0)}, {"x": "MHz"}) == {"x": (1.0, 4.0)}
    assert policy.axis("x").quantity == "MHz"


def test_convert_maps_the_held_value_and_stamps_the_new_quantity() -> None:
    policy = AxisLimitPolicy()
    policy.set_manual("x", 10.0, 20.0)
    policy.resolve({"x": (0.0, 100.0)}, {"x": "MHz"})

    policy.convert("x", lambda v: v / 13.55, "field")

    assert policy.held("x") == (10.0 / 13.55, 20.0 / 13.55)
    assert policy.is_auto("x") is False
    # The quantity is already stamped, so the next render holds rather than refits.
    assert policy.resolve({"x": (0.0, 7.4)}, {"x": "field"}) == (
        {"x": (10.0 / 13.55, 20.0 / 13.55)}
    )


def test_convert_sorts_the_pair_when_the_function_reverses_it() -> None:
    policy = AxisLimitPolicy()
    policy.set_manual("x", 1.0, 4.0)

    policy.convert("x", lambda v: -v, "flipped")

    assert policy.held("x") == (-4.0, -1.0)


def test_convert_on_an_unframed_axis_only_stamps_the_quantity() -> None:
    policy = AxisLimitPolicy()

    policy.convert("x", lambda v: v * 2.0, "field")

    assert policy.held("x") is None
    assert policy.axis("x").quantity == "field"


def test_none_bounds_leave_a_held_axis_where_it_is() -> None:
    policy = AxisLimitPolicy()
    policy.set_manual("x", 1.0, 4.0)

    assert policy.resolve({"x": None}) == {"x": (1.0, 4.0)}
    assert policy.held("x") == (1.0, 4.0)


def test_none_bounds_leave_an_auto_axis_at_its_last_frame() -> None:
    policy = AxisLimitPolicy()
    policy.resolve({"x": (0.0, 10.0)})

    assert policy.resolve({"x": None}) == {"x": (0.0, 10.0)}


def test_none_bounds_on_a_never_framed_axis_resolve_to_nothing() -> None:
    policy = AxisLimitPolicy()

    assert policy.resolve({"x": None}) == {}
    assert policy.held("x") is None


def test_axes_absent_from_bounds_are_untouched_and_absent_from_the_result() -> None:
    policy = AxisLimitPolicy()
    policy.set_manual("y", -1.0, 1.0)

    resolved = policy.resolve({"x": (0.0, 10.0)})

    assert resolved == {"x": (0.0, 10.0)}
    assert policy.held("y") == (-1.0, 1.0)


def test_unknown_axis_ids_in_resolve_are_created_auto() -> None:
    policy = AxisLimitPolicy()

    assert policy.resolve({"y:group:2": (0.0, 1.0)}) == {"y:group:2": (0.0, 1.0)}
    assert policy.is_auto("y:group:2") is True
    assert policy.resolve({"y:group:2": (0.0, 8.0)}) == {"y:group:2": (0.0, 8.0)}


def test_resolve_sorts_inverted_bounds() -> None:
    policy = AxisLimitPolicy()

    assert policy.resolve({"y": (30.0, -30.0)}) == {"y": (-30.0, 30.0)}


def test_reset_reframes_on_the_next_resolve_but_keeps_the_toggles() -> None:
    policy = AxisLimitPolicy()
    policy.set_manual("x", 1.0, 4.0)
    policy.resolve({"x": (0.0, 10.0)}, {"x": "time"})

    policy.reset()

    assert policy.held("x") is None
    assert policy.is_auto("x") is False
    assert policy.resolve({"x": (0.0, 32.0)}, {"x": "time"}) == {"x": (0.0, 32.0)}


def test_state_round_trips_through_restore() -> None:
    policy = AxisLimitPolicy()
    policy.set_manual("x", 1.0, 4.0)
    policy.resolve({"x": (0.0, 10.0), "y": (-30.0, 30.0)}, {"x": "time", "y": "asymmetry"})
    snapshot = policy.state()

    assert snapshot == {
        "axes": {
            "x": {"auto": False, "held": [1.0, 4.0], "quantity": "time"},
            "y": {"auto": True, "held": [-30.0, 30.0], "quantity": "asymmetry"},
        }
    }

    restored = AxisLimitPolicy()
    restored.restore(snapshot)

    assert restored.held("x") == (1.0, 4.0)
    assert restored.is_auto("x") is False
    assert restored.axis("x").quantity == "time"
    assert restored.held("y") == (-30.0, 30.0)
    assert restored.is_auto("y") is True
    assert restored.resolve({"x": (0.0, 32.0)}, {"x": "time"}) == {"x": (1.0, 4.0)}


def test_state_round_trips_an_axis_that_holds_nothing() -> None:
    policy = AxisLimitPolicy()
    policy.set_auto("y", False)
    snapshot = policy.state()

    assert snapshot["axes"]["y"] == {"auto": False, "held": None, "quantity": None}

    restored = AxisLimitPolicy()
    restored.restore(snapshot)

    assert restored.held("y") is None
    assert restored.is_auto("y") is False
    # Nothing to hold, so the axis still frames itself on the next render.
    assert restored.resolve({"y": (-5.0, 5.0)}) == {"y": (-5.0, 5.0)}


def test_restore_replaces_the_previous_axes() -> None:
    policy = AxisLimitPolicy()
    policy.set_manual("x", 1.0, 4.0)
    policy.set_manual("y:P_x", -1.0, 1.0)

    policy.restore({"axes": {"x": {"auto": True, "held": [0.0, 8.0], "quantity": None}}})

    assert policy.state()["axes"] == {"x": {"auto": True, "held": [0.0, 8.0], "quantity": None}}
    # The dropped axis comes back as a fresh Auto axis, not its old held value.
    assert policy.is_auto("y:P_x") is True
    assert policy.held("y:P_x") is None


def test_restore_ignores_unknown_keys() -> None:
    policy = AxisLimitPolicy()

    policy.restore(
        {
            "version": 2,
            "axes": {"x": {"auto": False, "held": [1.0, 4.0], "quantity": "time", "lock": True}},
        }
    )

    assert policy.held("x") == (1.0, 4.0)
    assert policy.is_auto("x") is False
