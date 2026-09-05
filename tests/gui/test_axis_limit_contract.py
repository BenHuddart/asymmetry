"""Contract: the per-axis Auto/Hold policy behaves the same on every plot
representation (docs/plans/axis-limit-policy.md).

Each representation is wrapped in a small adapter exposing the same surface
(render/buttons/fields/type_x/type_y/gesture/blank/teardown); one scenario
table is run against every representation the scenario applies to. Two
scenarios do not generalise across representations (a time-view-mode change;
a frequency unit switch) and are tested standalone at the bottom of the file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pyside6 = pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # type: ignore  # noqa: E402

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run  # noqa: E402
from asymmetry.gui.panels.alc_panel import ALCScanView  # noqa: E402
from asymmetry.gui.panels.plot_panel import PlotPanel  # noqa: E402


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


# ── synthetic data ──────────────────────────────────────────────────────────


def _decay_curve(seed: int, amplitude: float, *, phase: float = 0.0) -> tuple[np.ndarray, ...]:
    """A damped oscillation whose amplitude and span both scale with *amplitude*.

    "Double amplitude and double span" therefore falls out of one knob: the
    envelope peaks at the start of the range (the "large early amplitude" the
    scenarios narrow away from) and both curve height and domain length track
    *amplitude* together.
    """
    rng = np.random.default_rng(seed)
    span = 10.0 * amplitude
    x = np.linspace(0.0, span, 400)
    y = amplitude * np.exp(-0.3 * x) * np.cos(0.8 * x + phase) + rng.normal(0.0, 1e-4, x.size)
    err = np.full_like(x, 0.01)
    return x, y, err


def _decay_dataset(
    seed: int, amplitude: float, run_number: int, *, phase: float = 0.0
) -> MuonDataset:
    x, y, err = _decay_curve(seed, amplitude, phase=phase)
    return MuonDataset(time=x, asymmetry=y, error=err, metadata={"run_number": run_number})


def _frequency_dataset(seed: int, amplitude: float, run_number: int) -> MuonDataset:
    """A spectrum with a DC peak (ignored by framing) and one real line.

    The line's height and the span both scale with *amplitude*; the line sits
    near the low-frequency end, so it plays the same "large early amplitude"
    role the time-domain envelope does for the shared scenarios.
    """
    rng = np.random.default_rng(seed)
    span = 500.0 * amplitude
    freqs = np.linspace(0.0, span, 2048)
    values = np.full_like(freqs, 1.0) + rng.normal(0.0, 0.01, freqs.size)
    values[:3] = 3.0e5
    values[int(len(freqs) * 0.05)] = 5.0e3 * amplitude
    return MuonDataset(
        time=freqs,
        asymmetry=values,
        error=np.full_like(freqs, 0.5),
        metadata={
            "run_number": run_number,
            "plot_domain": "frequency",
            "x_label": "Frequency (MHz)",
        },
    )


# ── adapters ─────────────────────────────────────────────────────────────────


@dataclass
class LimitPair:
    """One axis's limits, read from both the fields and the live axes."""

    field: tuple[float, float]
    axis: tuple[float, float]


def _assert_consistent(pair: LimitPair) -> None:
    """The fields and the live axes must never disagree (within field rounding)."""
    assert pair.field == pytest.approx(pair.axis, abs=2e-3, rel=1e-4)


def _set_checked(button, value: bool) -> None:
    """Drive *button* to *value* through its real click path (never setChecked)."""
    if button.isChecked() != value:
        button.click()


class _PlotPanelAdapter:
    """Shared plumbing for every ``PlotPanel``-based representation."""

    def __init__(self, panel: PlotPanel) -> None:
        self.panel = panel

    @property
    def auto_x_btn(self):
        return self.panel._auto_x_btn

    @property
    def auto_y_btn(self):
        return self.panel._auto_y_btn

    def render(self, seed: int, amplitude: float) -> None:
        raise NotImplementedError

    def _active_y_axes_obj(self):
        axes = self.panel._displayed_y_axes()
        return axes.get(self.panel._active_y_axis(), self.panel._ax)

    def x_limits(self) -> LimitPair:
        return LimitPair(
            field=(self.panel._x_min.value(), self.panel._x_max.value()),
            axis=tuple(self.panel._ax.get_xlim()),
        )

    def active_y_limits(self) -> LimitPair:
        return LimitPair(
            field=(self.panel._y_min.value(), self.panel._y_max.value()),
            axis=tuple(self._active_y_axes_obj().get_ylim()),
        )

    def type_x(self, lo: float, hi: float) -> None:
        self.panel._x_min.setValue(lo)
        self.panel._x_max.setValue(hi)
        self.panel._on_x_limit_field_edited()

    def type_y(self, lo: float, hi: float) -> None:
        self.panel._y_min.setValue(lo)
        self.panel._y_max.setValue(hi)
        self.panel._on_y_limit_field_edited()

    def gesture(
        self, *, x: tuple[float, float] | None = None, y: tuple[float, float] | None = None
    ) -> None:
        ax = self.panel._ax
        self.panel._set_navigation_mode("zoom")
        self.panel._on_canvas_button_press(SimpleNamespace(button=1))
        if x is not None:
            ax.set_xlim(*x)
        if y is not None:
            ax.set_ylim(*y)
        self.panel._on_canvas_button_release(SimpleNamespace(button=1))
        self.panel._set_navigation_mode("none")

    def blank(self) -> None:
        self.panel.clear()

    def teardown(self) -> None:
        self.panel.reset_view_limits()


class _TimeSingleAdapter(_PlotPanelAdapter):
    def render(self, seed: int, amplitude: float) -> None:
        self.panel.plot_dataset(_decay_dataset(seed, amplitude, run_number=seed))

    @classmethod
    def fresh(cls) -> _TimeSingleAdapter:
        return cls(PlotPanel())


class _TimeOverlayAdapter(_PlotPanelAdapter):
    def render(self, seed: int, amplitude: float) -> None:
        self.panel.plot_datasets(
            [
                _decay_dataset(seed, amplitude, run_number=seed),
                _decay_dataset(seed, amplitude, run_number=seed + 1, phase=1.2),
            ]
        )

    @classmethod
    def fresh(cls) -> _TimeOverlayAdapter:
        return cls(PlotPanel())


class _StackedAdapter(_PlotPanelAdapter):
    def render(self, seed: int, amplitude: float) -> None:
        self.panel._current_polarization_axis = "ALL"
        self.panel.plot_vector_subplots(
            {
                "P_x": [_decay_dataset(seed, amplitude, run_number=seed)],
                "P_y": [_decay_dataset(seed, amplitude, run_number=seed, phase=1.2)],
            }
        )

    @classmethod
    def fresh(cls) -> _StackedAdapter:
        return cls(PlotPanel())


class _GroupedAdapter(_PlotPanelAdapter):
    def render(self, seed: int, amplitude: float) -> None:
        datasets = [
            _decay_dataset(seed, amplitude, run_number=-1),
            _decay_dataset(seed, amplitude, run_number=-2, phase=1.2),
        ]
        self.panel.plot_grouped_time_domain_subplots(datasets)
        self.panel.set_fit_target_projection(self.panel._grouped_subplot_axis_key(datasets[0]))

    @classmethod
    def fresh(cls) -> _GroupedAdapter:
        return cls(PlotPanel())


class _FrequencyAdapter(_PlotPanelAdapter):
    def render(self, seed: int, amplitude: float) -> None:
        self.panel.plot_dataset(_frequency_dataset(seed, amplitude, run_number=seed))

    @classmethod
    def fresh(cls) -> _FrequencyAdapter:
        return cls(PlotPanel(domain="frequency"))


class _AlcAdapter:
    """ALC has no pan/zoom toolbar (no ``gesture``) and its own render/reset."""

    def __init__(self, view: ALCScanView) -> None:
        self.view = view

    @property
    def auto_x_btn(self):
        return self.view._auto_x_btn

    @property
    def auto_y_btn(self):
        return self.view._auto_y_btn

    def render(self, seed: int, amplitude: float) -> None:
        rng = np.random.default_rng(seed)
        span = 100.0 * amplitude
        x = np.linspace(0.0, span, 60)
        y = amplitude * 20.0 * np.exp(-0.05 * x) + rng.normal(0.0, 0.05, x.size)
        err = np.full_like(x, 0.1)
        self.view.show_scan(x, y, err, list(range(1, x.size + 1)), x_label="B (G)", y_label="A (%)")

    def x_limits(self) -> LimitPair:
        return LimitPair(
            field=(self.view._x_min.value(), self.view._x_max.value()),
            axis=tuple(self.view._ax.get_xlim()),
        )

    def active_y_limits(self) -> LimitPair:
        return LimitPair(
            field=(self.view._y_min.value(), self.view._y_max.value()),
            axis=tuple(self.view._ax.get_ylim()),
        )

    def type_x(self, lo: float, hi: float) -> None:
        self.view._x_min.setValue(lo)
        self.view._x_max.setValue(hi)
        self.view._on_x_limit_edited()

    def type_y(self, lo: float, hi: float) -> None:
        self.view._y_min.setValue(lo)
        self.view._y_max.setValue(hi)
        self.view._on_y_limit_edited()

    def blank(self) -> None:
        self.view.clear()

    def teardown(self) -> None:
        self.view.reset()

    @classmethod
    def fresh(cls) -> _AlcAdapter:
        return cls(ALCScanView())


_REPRESENTATIONS = {
    "time_single": _TimeSingleAdapter,
    "time_overlay": _TimeOverlayAdapter,
    "stacked": _StackedAdapter,
    "grouped": _GroupedAdapter,
    "frequency": _FrequencyAdapter,
    "alc": _AlcAdapter,
}
_PLOT_PANEL_REPS = ["time_single", "time_overlay", "stacked", "grouped", "frequency"]
_ALL_REPS = [*_PLOT_PANEL_REPS, "alc"]


def _make(rep: str):
    return _REPRESENTATIONS[rep].fresh()


# ── scenario table ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("rep", _PLOT_PANEL_REPS)
def test_browse_holds(qapp: QApplication, rep: str) -> None:
    """Render A, then B with double amplitude and span: the window stays put."""
    adapter = _make(rep)
    adapter.render(1, 1.0)
    x_before, y_before = adapter.x_limits(), adapter.active_y_limits()
    _assert_consistent(x_before)

    adapter.render(2, 2.0)
    x_after, y_after = adapter.x_limits(), adapter.active_y_limits()
    assert x_after.field == pytest.approx(x_before.field, abs=2e-3)
    assert y_after.field == pytest.approx(y_before.field, abs=2e-3)

    _set_checked(adapter.auto_x_btn, True)
    x_final = adapter.x_limits()
    _assert_consistent(x_final)
    assert x_final.field[1] > x_before.field[1] * 1.5


@pytest.mark.parametrize("rep", _ALL_REPS)
def test_auto_y_follows_typed_x(qapp: QApplication, rep: str) -> None:
    """Auto Y on; narrowing x past the large early amplitude shrinks y max.

    ALC is the one exception: its Auto Y frames the whole scan by design
    (``_auto_data_bounds`` deliberately ignores the x window, so a dragged
    region/peak handle outside the visible span stays reachable), so a typed
    x window leaves y exactly where it was rather than shrinking it.
    """
    adapter = _make(rep)
    adapter.render(1, 1.0)
    _set_checked(adapter.auto_y_btn, True)
    y_before = adapter.active_y_limits()

    x_before = adapter.x_limits()
    span = x_before.field[1] - x_before.field[0]
    adapter.type_x(x_before.field[0] + span * 0.6, x_before.field[1])
    y_after = adapter.active_y_limits()

    _assert_consistent(y_after)
    if rep == "alc":
        assert y_after.field == pytest.approx(y_before.field, abs=2e-3)
    else:
        assert y_after.field[1] < y_before.field[1]


@pytest.mark.parametrize("rep", _PLOT_PANEL_REPS)
def test_horizontal_gesture_keeps_auto_y(qapp: QApplication, rep: str) -> None:
    """A gesture moving only x leaves Auto Y on and holds x at the new window."""
    adapter = _make(rep)
    adapter.render(1, 1.0)
    _set_checked(adapter.auto_y_btn, True)

    x_before = adapter.x_limits()
    span = x_before.field[1] - x_before.field[0]
    window = (x_before.field[0] + span * 0.2, x_before.field[1] - span * 0.2)
    adapter.gesture(x=window)

    assert adapter.auto_y_btn.isChecked() is True
    assert adapter.auto_x_btn.isChecked() is False
    x_after = adapter.x_limits()
    _assert_consistent(x_after)
    assert x_after.field == pytest.approx(window, abs=2e-3)


@pytest.mark.parametrize("rep", _PLOT_PANEL_REPS)
def test_auto_x_leaves_typed_y_alone_across_switch(qapp: QApplication, rep: str) -> None:
    """Auto X refits on a run switch; a typed, held y is untouched."""
    adapter = _make(rep)
    adapter.render(1, 1.0)
    x_before = adapter.x_limits()

    adapter.type_y(-5.0, 5.0)
    _set_checked(adapter.auto_x_btn, True)
    y_before = adapter.active_y_limits()

    adapter.render(2, 2.0)

    y_after, x_after = adapter.active_y_limits(), adapter.x_limits()
    assert y_after.field == pytest.approx(y_before.field, abs=2e-3)
    _assert_consistent(y_after)
    _assert_consistent(x_after)
    assert x_after.field[1] > x_before.field[1] * 1.5


@pytest.mark.parametrize("rep", _PLOT_PANEL_REPS)
def test_blank_canvas_keeps_holds(qapp: QApplication, rep: str) -> None:
    """A blank canvas never touches held values; the next render sees them."""
    adapter = _make(rep)
    adapter.render(1, 1.0)
    adapter.type_x(1.0, 3.0)
    adapter.type_y(-2.0, 2.0)
    x_before, y_before = adapter.x_limits(), adapter.active_y_limits()

    adapter.blank()
    adapter.render(1, 1.0)

    x_after, y_after = adapter.x_limits(), adapter.active_y_limits()
    assert x_after.field == pytest.approx(x_before.field, abs=2e-3)
    assert y_after.field == pytest.approx(y_before.field, abs=2e-3)


@pytest.mark.parametrize("rep", _ALL_REPS)
def test_teardown_forgets_holds(qapp: QApplication, rep: str) -> None:
    """Only teardown forgets a hold; the next render frames afresh."""
    adapter = _make(rep)
    adapter.render(1, 1.0)
    adapter.type_x(1.0, 3.0)
    adapter.type_y(-2.0, 2.0)

    adapter.teardown()
    adapter.render(2, 2.0)

    x_after, y_after = adapter.x_limits(), adapter.active_y_limits()
    _assert_consistent(x_after)
    _assert_consistent(y_after)
    assert x_after.field != pytest.approx((1.0, 3.0))
    assert y_after.field != pytest.approx((-2.0, 2.0))


@pytest.mark.parametrize("rep", _PLOT_PANEL_REPS)
def test_state_round_trip(qapp: QApplication, rep: str) -> None:
    """A typed x and Auto Y round-trip through get_state/restore_state."""
    adapter = _make(rep)
    adapter.type_x(2.0, 6.0)
    _set_checked(adapter.auto_y_btn, True)

    state = adapter.panel.get_state()
    fresh = _make(rep)
    fresh.panel.restore_state(state, dataset=None)

    assert fresh.panel._limits.held("x") == pytest.approx((2.0, 6.0))
    assert fresh.panel._auto_x_btn.isChecked() is False
    assert fresh.panel._auto_y_btn.isChecked() is True


# ── representation-specific scenarios ────────────────────────────────────────


def test_time_view_mode_change_refits_held_y_once(qapp: QApplication) -> None:
    """A view-mode change is a new y quantity: it refits a held y exactly once."""
    counts = np.full(200, 900.0)
    hist = Histogram(counts=counts, bin_width=0.05, t0_bin=0)
    run = Run(run_number=9100, histograms=[hist])
    t = hist.time_axis.copy()
    ds = MuonDataset(
        time=t,
        asymmetry=20.0 * np.exp(-0.4 * t),
        error=np.full_like(t, 0.2),
        metadata={"run_number": 9100},
        run=run,
    )
    panel = PlotPanel()
    panel.set_time_view_modes(["fb_asymmetry", "raw_counts"], "fb_asymmetry")
    panel.plot_dataset(ds)
    panel._y_min.setValue(-100.0)
    panel._y_max.setValue(100.0)
    panel._on_y_limit_field_edited()
    panel.plot_dataset(ds)
    assert panel._limits.held("y") == pytest.approx((-100.0, 100.0))

    panel.set_current_time_view_mode("raw_counts")
    panel.plot_dataset(ds)
    refit = panel._limits.held("y")
    assert refit is not None
    assert refit != pytest.approx((-100.0, 100.0))

    panel.plot_dataset(ds)
    assert panel._limits.held("y") == pytest.approx(refit)


def test_frequency_unit_switch_converts_held_x(qapp: QApplication) -> None:
    """Switching x unit converts a held window rather than refitting it."""
    panel = PlotPanel(domain="frequency")
    panel.plot_dataset(_frequency_dataset(11, 1.0, run_number=42))
    panel._x_min.setValue(50.0)
    panel._x_max.setValue(150.0)
    panel._on_x_limit_field_edited()
    assert panel._auto_x_btn.isChecked() is False

    old_unit = panel._current_frequency_x_unit
    mode = panel._frequency_axis_mode

    def _expected(value: float) -> float:
        canonical = panel._convert_display_limit_to_canonical_mhz(value, unit=old_unit, mode=mode)
        return panel._convert_canonical_mhz_to_display_limit(
            canonical, unit="field_gauss", mode=mode
        )

    expected = sorted((_expected(50.0), _expected(150.0)))

    panel._switch_frequency_axis_display(unit="field_gauss")

    assert panel._auto_x_btn.isChecked() is False
    got = sorted((panel._x_min.value(), panel._x_max.value()))
    assert got == pytest.approx(expected, rel=1e-4)
    assert sorted(panel._ax.get_xlim()) == pytest.approx(got, abs=2e-3)
