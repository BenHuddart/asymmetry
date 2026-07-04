"""Side-by-side comparison of two global parameter fit studies.

The comparison dialog answers a model-selection question — "is the LCR term
justified?", "does making D_2D local actually pay for itself?" — by putting two
:class:`~asymmetry.core.representation.global_fit_study.GlobalFitStudy` objects
of the *same* parameter next to each other:

- a grid of per-group panels overlaying both studies' total model curves on the
  shared data points (study A solid, study B dashed), matched by ``group_id``;
- a side-by-side statistics block (χ², χ²ᵣ, n, k, AIC, AICc, BIC via
  :func:`~asymmetry.gui.windows.global_fit_window_helpers.information_criteria`)
  with a ``Δ (B − A)`` sub-block that bolds the better (lower) side; and
- a two-column global-parameter table (union of names) that highlights any
  parameter whose two values disagree by more than ``2σ``.

It is deliberately lean and read-only: no annotations, no GLE export, no
decorations — those live on the main results window. The per-group curve
sampling logic mirrors
``GlobalParameterFitWindow._sample_group_fit_curve`` /
``_model_kwargs`` but is copied here (rather than importing the window's
internals) so the dialog stays self-contained.
"""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.fitting.parameters import get_param_info
from asymmetry.gui.widgets.mpl_canvas import create_canvas
from asymmetry.gui.windows.global_fit_window_helpers import (
    format_value_with_error,
    information_criteria,
)

__all__ = ["GlobalFitCompareDialog"]

#: Curve sample count per panel (mirrors the results window constant).
_CURVE_SAMPLE_COUNT = 400

#: Study-A curve is solid C0; study-B curve is dashed C3.
_CURVE_COLOR_A = "#1f77b4"  # matplotlib C0
_CURVE_COLOR_B = "#d62728"  # matplotlib C3

#: Light warning background for a >2σ parameter discrepancy cell.
_DISCREPANCY_BG = QColor(255, 236, 179)


def _local_group_x_key(x_key: str) -> str:
    """Return the per-panel abscissa key (mirrors the results window).

    A field-series study plots each group's data against temperature and vice
    versa; anything else is a plain run axis. Only ``"field"`` selects the
    geomspace sampling convention below.
    """
    if x_key == "field":
        return "temperature"
    if x_key == "temperature":
        return "field"
    return "run"


def _model_kwargs(result, model, group_id: str) -> dict[str, float]:
    """Build a group's model kwargs (global + fixed + local, defaults-filled).

    Copied from ``GlobalParameterFitWindow._model_kwargs`` so the dialog does
    not reach into the window's internals.
    """
    kwargs = {p.name: p.value for p in result.global_parameters}
    for p in result.fixed_parameters:
        kwargs[p.name] = p.value
    local = result.local_parameters.get(group_id)
    if local is not None:
        for p in local:
            kwargs[p.name] = p.value
    missing = [name for name in getattr(model, "param_names", []) if name not in kwargs]
    defaults = getattr(model, "param_defaults", {})
    for name in missing:
        if isinstance(defaults, dict) and name in defaults:
            kwargs[name] = float(defaults[name])
    return kwargs


def _sample_group_curve(study, group_id: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Sample a study's total model curve over one group's x-range.

    Mirrors ``GlobalParameterFitWindow._sample_group_fit_curve``: sort the
    finite x, clip to the study's ``fit_x_min/max`` window when set, then sample
    ``_CURVE_SAMPLE_COUNT`` points geometrically for a field abscissa or
    linearly otherwise. Returns ``None`` when the group is absent, the model is
    missing, or fewer than two finite x remain.
    """
    if study.model is None or study.result is None:
        return None
    group = next((g for g in study.groups if g.group_id == group_id), None)
    if group is None:
        return None
    kwargs = _model_kwargs(study.result, study.model, group_id)
    if not kwargs:
        return None

    xx = np.asarray(group.x, dtype=float)
    finite_x = xx[np.isfinite(xx)]
    if finite_x.size < 2:
        return None
    xx = np.sort(finite_x)
    if (
        np.isfinite(study.fit_x_min)
        and np.isfinite(study.fit_x_max)
        and study.fit_x_max > study.fit_x_min
    ):
        mask = (xx >= study.fit_x_min) & (xx <= study.fit_x_max)
        xx = xx[mask]
    if xx.size < 2:
        return None

    x_min = float(np.nanmin(xx))
    x_max = float(np.nanmax(xx))
    if _local_group_x_key(study.x_key) == "field" and x_min > 0.0 and x_max > 0.0:
        xs = np.geomspace(x_min, x_max, _CURVE_SAMPLE_COUNT)
    else:
        xs = np.linspace(x_min, x_max, _CURVE_SAMPLE_COUNT)
    try:
        ys = np.asarray(study.model.function(xs, **kwargs), dtype=float)
    except KeyError:
        return None
    finite = np.isfinite(xs) & np.isfinite(ys)
    if not np.any(finite):
        return None
    return xs[finite], ys[finite]


def _ordered_group_ids(study_a, study_b) -> list[str]:
    """Union of both studies' group ids: A's order first, then B-only ones."""
    ids: list[str] = []
    seen: set[str] = set()
    for group in study_a.groups:
        if group.group_id not in seen:
            ids.append(group.group_id)
            seen.add(group.group_id)
    for group in study_b.groups:
        if group.group_id not in seen:
            ids.append(group.group_id)
            seen.add(group.group_id)
    return ids


def _group_by_id(study, group_id: str):
    return next((g for g in study.groups if g.group_id == group_id), None)


def _global_value_and_err(result, name: str) -> tuple[float | None, float | None]:
    """Return ``(value, err)`` for a global parameter *name* in *result*.

    ``(None, None)`` when the parameter is not a fitted global of this study.
    """
    if result is None:
        return None, None
    for p in result.global_parameters:
        if p.name == name:
            err = result.global_uncertainties.get(name)
            if err is not None:
                try:
                    err = float(err)
                    if not math.isfinite(err) or err < 0:
                        err = None
                except (TypeError, ValueError):
                    err = None
            return float(p.value), err
    return None, None


class GlobalFitCompareDialog(QDialog):
    """A read-only side-by-side comparison of two global parameter fit studies."""

    def __init__(self, study_a, study_b, parent=None) -> None:
        super().__init__(parent)
        self._study_a = study_a
        self._study_b = study_b
        self.setWindowTitle(f"Compare: {study_a.name} vs {study_b.name}")
        self.resize(1000, 760)

        layout = QVBoxLayout(self)

        # ── Comparability caveats ───────────────────────────────────────────
        mode_a = str(getattr(study_a.result, "error_mode", "") or "")
        mode_b = str(getattr(study_b.result, "error_mode", "") or "")
        n_a = int(getattr(study_a.result, "n_points", 0) or 0)
        n_b = int(getattr(study_b.result, "n_points", 0) or 0)
        same_mode_and_n = (mode_a == mode_b) and (n_a == n_b)
        same_snapshot = str(study_a.input_digest) == str(study_b.input_digest)
        #: Criteria differences are only meaningful with matching data + error
        #: mode. "Matching data" includes the input digest: two studies fit to
        #: different data snapshots (e.g. one refit after a trend edit) must not
        #: have their Δ column bolded even when the error mode and point count
        #: coincide. When any of these differ we show a caveat and suppress the
        #: Δ bolding.
        self._criteria_comparable = same_mode_and_n and same_snapshot

        if not same_mode_and_n:
            caveat = QLabel("Criteria not comparable: different data/error mode")
            caveat.setStyleSheet(
                "QLabel { background-color: #ffd9b3; color: #7a3b00; "
                "font-weight: bold; padding: 4px 8px; }"
            )
            caveat.setWordWrap(True)
            layout.addWidget(caveat)

        if not same_snapshot:
            snap_caveat = QLabel("Fitted to different data snapshots")
            snap_caveat.setStyleSheet(
                "QLabel { background-color: #fff3cd; color: #66512c; padding: 4px 8px; }"
            )
            snap_caveat.setWordWrap(True)
            layout.addWidget(snap_caveat)

        # ── Log X / Log Y toggles + legend ──────────────────────────────────
        from PySide6.QtWidgets import QCheckBox

        controls = QHBoxLayout()
        self._log_x_check = QCheckBox("Log X")
        self._log_y_check = QCheckBox("Log Y")
        self._log_x_check.toggled.connect(self._refresh_plot)
        self._log_y_check.toggled.connect(self._refresh_plot)
        controls.addWidget(self._log_x_check)
        controls.addWidget(self._log_y_check)
        controls.addSpacing(16)
        legend = QLabel(
            f'<span style="color:{_CURVE_COLOR_A};">──</span> {study_a.name} &nbsp; '
            f'<span style="color:{_CURVE_COLOR_B};">– –</span> {study_b.name}'
        )
        controls.addWidget(legend)
        controls.addStretch(1)
        layout.addLayout(controls)

        # ── Per-group grid of overlay panels ────────────────────────────────
        self._figure = None
        self._canvas = None
        self._group_ids = _ordered_group_ids(study_a, study_b)
        try:
            self._figure, self._canvas = create_canvas(layout="tight")
            layout.addWidget(self._canvas, 3)
        except ImportError:
            layout.addWidget(QLabel("matplotlib not installed — plots unavailable"), 3)

        # ── Bottom: stats + parameter table (scrollable) ────────────────────
        bottom = QWidget()
        bottom_layout = QHBoxLayout(bottom)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        self._stats_table = self._build_stats_table()
        bottom_layout.addWidget(self._stats_table, 1)
        self._param_table = self._build_param_table()
        bottom_layout.addWidget(self._param_table, 2)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(bottom)
        layout.addWidget(scroll, 2)

        if self._figure is not None:
            self._refresh_plot()

    # ── plot ────────────────────────────────────────────────────────────────

    def _refresh_plot(self) -> None:
        if self._figure is None or self._canvas is None:
            return
        self._figure.clear()
        group_ids = self._group_ids
        n = len(group_ids)
        if n == 0:
            ax = self._figure.add_subplot(111)
            ax.set_title("No groups")
            self._canvas.draw()
            return
        ncols = max(1, min(int(np.ceil(np.sqrt(n))), 4))
        nrows = (n + ncols - 1) // ncols

        log_x = self._log_x_check.isChecked()
        log_y = self._log_y_check.isChecked()

        for idx, group_id in enumerate(group_ids):
            ax = self._figure.add_subplot(nrows, ncols, idx + 1)
            group_a = _group_by_id(self._study_a, group_id)
            group_b = _group_by_id(self._study_b, group_id)
            # Data points: prefer A's group, else B's.
            data_group = group_a if group_a is not None else group_b
            if data_group is not None:
                ax.errorbar(
                    data_group.x,
                    data_group.y,
                    yerr=data_group.yerr,
                    fmt="o",
                    linestyle="none",
                    color="black",
                    capsize=2,
                    markersize=3,
                )

            curve_a = _sample_group_curve(self._study_a, group_id)
            curve_b = _sample_group_curve(self._study_b, group_id)
            if curve_a is not None:
                ax.plot(curve_a[0], curve_a[1], color=_CURVE_COLOR_A, linewidth=1.5)
            if curve_b is not None:
                ax.plot(
                    curve_b[0],
                    curve_b[1],
                    color=_CURVE_COLOR_B,
                    linewidth=1.5,
                    linestyle="--",
                )

            title = group_a.group_name if group_a is not None else None
            if title is None and group_b is not None:
                title = group_b.group_name
            ax.set_title(title or group_id, pad=8, fontsize=9)

            # Corner note for a single-study group.
            if group_a is None and group_b is not None:
                note = "B only"
            elif group_b is None and group_a is not None:
                note = "A only"
            else:
                note = ""
            if note:
                ax.text(
                    0.97,
                    0.95,
                    note,
                    transform=ax.transAxes,
                    fontsize=8,
                    ha="right",
                    va="top",
                    color="#7a3b00",
                )

            if log_x:
                try:
                    ax.set_xscale("log")
                except Exception:
                    pass
            if log_y:
                try:
                    ax.set_yscale("log")
                except Exception:
                    pass
            ax.grid(True, alpha=0.3)

        self._figure.tight_layout(h_pad=1.4)
        self._canvas.draw()

    # ── stats table ──────────────────────────────────────────────────────────

    def _build_stats_table(self) -> QTableWidget:
        """A (metric × [A, B, Δ]) table of χ²/χ²ᵣ/n/k/AIC/AICc/BIC."""
        ra = self._study_a.result
        rb = self._study_b.result
        ic_a = information_criteria(ra) or {}
        ic_b = information_criteria(rb) or {}

        def _g(mapping, key):
            v = mapping.get(key)
            return None if v is None else float(v)

        metrics: list[tuple[str, float | None, float | None, bool]] = [
            ("χ²", float(ra.chi_squared), float(rb.chi_squared), False),
            ("χ²ᵣ", float(ra.reduced_chi_squared), float(rb.reduced_chi_squared), True),
            ("n", float(ra.n_points), float(rb.n_points), False),
            ("k", _g(ic_a, "k"), _g(ic_b, "k"), False),
            ("AIC", _g(ic_a, "aic"), _g(ic_b, "aic"), True),
            ("AICc", _g(ic_a, "aicc"), _g(ic_b, "aicc"), True),
            ("BIC", _g(ic_a, "bic"), _g(ic_b, "bic"), True),
        ]
        # ``delta_metric`` flags rows carrying a Δ (B − A) with better-side bolding.

        table = QTableWidget(len(metrics), 4, self)
        table.setHorizontalHeaderLabels(["", self._study_a.name, self._study_b.name, "Δ (B − A)"])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        def _fmt(v):
            return "—" if v is None else f"{v:.6g}"

        for row, (label, va, vb, is_delta) in enumerate(metrics):
            table.setItem(row, 0, QTableWidgetItem(label))
            item_a = QTableWidgetItem(_fmt(va))
            item_b = QTableWidgetItem(_fmt(vb))
            table.setItem(row, 1, item_a)
            table.setItem(row, 2, item_b)
            if is_delta and va is not None and vb is not None:
                delta = vb - va
                delta_item = QTableWidgetItem(f"{delta:+.6g}")
                table.setItem(row, 3, delta_item)
                # Bold the better (lower) side — only when criteria comparable.
                if self._criteria_comparable and math.isfinite(delta) and delta != 0.0:
                    better = item_b if delta < 0 else item_a
                    font = better.font()
                    font.setBold(True)
                    better.setFont(font)
            else:
                table.setItem(row, 3, QTableWidgetItem("—"))

        table.resizeColumnsToContents()
        return table

    # ── global parameter union table ─────────────────────────────────────────

    def _build_param_table(self) -> QTableWidget:
        """Union global-parameter table: rows = names, cols = [param, A, B].

        A cell pair is highlighted (both value cells) when the two studies'
        values disagree by more than ``2·sqrt(err_A² + err_B²)`` — skipped when
        either uncertainty is missing.
        """
        ra = self._study_a.result
        rb = self._study_b.result
        names: list[str] = []
        seen: set[str] = set()
        for result in (ra, rb):
            if result is None:
                continue
            for p in result.global_parameters:
                if p.name not in seen:
                    names.append(p.name)
                    seen.add(p.name)

        table = QTableWidget(len(names), 3, self)
        table.setHorizontalHeaderLabels(["Parameter", self._study_a.name, self._study_b.name])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        for row, name in enumerate(names):
            info = get_param_info(name)
            symbol = info.unicode_label(include_unit=True)
            table.setItem(row, 0, QTableWidgetItem(symbol))

            va, ea = _global_value_and_err(ra, name)
            vb, eb = _global_value_and_err(rb, name)
            text_a = "—" if va is None else format_value_with_error(va, ea)
            text_b = "—" if vb is None else format_value_with_error(vb, eb)
            item_a = QTableWidgetItem(text_a)
            item_b = QTableWidgetItem(text_b)
            table.setItem(row, 1, item_a)
            table.setItem(row, 2, item_b)

            if va is not None and vb is not None and ea is not None and eb is not None:
                sigma = math.sqrt(ea * ea + eb * eb)
                if sigma > 0 and abs(va - vb) > 2.0 * sigma:
                    item_a.setBackground(_DISCREPANCY_BG)
                    item_b.setBackground(_DISCREPANCY_BG)
                    tip = (
                        f"|Δ| = {abs(va - vb):.4g} > 2σ = {2.0 * sigma:.4g} "
                        "— the two studies disagree on this parameter"
                    )
                    item_a.setToolTip(tip)
                    item_b.setToolTip(tip)

        table.resizeColumnsToContents()
        return table
