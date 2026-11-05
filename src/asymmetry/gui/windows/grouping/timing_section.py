"""Timing controls consume shared core resolution and cached raw worker results."""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from typing import Any

from PySide6.QtCore import QSignalBlocker

from asymmetry.core.data.dataset import Run
from asymmetry.core.io.periods import (
    acquired_period_histograms,
    held_period_histograms,
    histogram_grouping_facts,
    select_period_grouping,
    selected_period_index,
)
from asymmetry.core.transform.grouping import InvalidGroupingError, common_t0_for_groups
from asymmetry.core.transform.t0 import (
    RunT0Search,
    T0Assessment,
    assess_t0,
    find_t0_for_run,
    resolve_selected_timing,
    selected_t0_diagnostics,
    source_is_pulsed,
)
from asymmetry.gui.styles import tokens
from asymmetry.gui.tasks import check_task_cancelled
from asymmetry.gui.windows.grouping.channel_entries import shift_channel_entries

_T0_STRATEGY_LABELS = {"prompt_peak": "prompt peak", "pulse_edge": "pulse-edge midpoint"}


class TimingControls:
    """Grouping dialog timing presentation; acquisition and selection resolve in core."""

    def _t0_preview_run(self, period_mode: str | None = None) -> Run:
        """Read selected acquired timing without copying count arrays."""
        run = self._run
        index = selected_period_index(run.grouping, period_mode or self._current_period_mode())
        if index is None:
            return (
                replace(run, metadata=dict(run.metadata, raw_period_histograms_available=False))
                if "period_reduced" in run.grouping and "period_histograms" not in run.grouping
                else run
            )
        histograms = held_period_histograms(run.histograms, run.grouping, index)
        return replace(
            run,
            histograms=histograms,
            metadata=dict(
                run.metadata,
                raw_period_histograms_available=acquired_period_histograms(run.grouping, index)
                is not None,
            ),
            grouping=histogram_grouping_facts(
                histograms, select_period_grouping(run.grouping, index)
            ),
        )

    def _common_t0_for_preview_run(
        self, detector_t0_bins: list[int] | None, *, period_mode: str | None = None
    ) -> int:
        """Resolve the selected period's anchor over live, non-excluded groups.

        With unavailable membership, the acquired maximum remains the editing
        base while the requested reduction stays unavailable."""
        if self._run is None or not self._run.histograms:
            return 0
        run = self._t0_preview_run(period_mode)
        try:
            forward_idx = self._filtered_group_indices(int(self._forward_combo.currentData() or 1))
            backward_idx = self._filtered_group_indices(
                int(self._backward_combo.currentData() or 2)
            )
        except InvalidGroupingError:
            return common_t0_for_groups(
                run.histograms,
                list(range(len(run.histograms))),
                detector_t0_bins=detector_t0_bins,
            )
        return common_t0_for_groups(
            run.histograms, forward_idx, backward_idx, detector_t0_bins=detector_t0_bins
        )

    def _detected_common_t0_for_preview_run(self, search: RunT0Search) -> int:
        """Align detected per-detector anchors over the live selected groups."""
        run = self._t0_preview_run()
        grouping = self._t0_assessment_grouping()
        resolve_selected_timing(
            run.histograms, grouping, run.metadata, mode="auto_detect", search=search
        )
        return int(grouping["t0_bin"])

    def _seed_t0_spin_from_preview(self) -> None:
        """Show the acquired selected anchor while retaining unresolved raw data."""
        self._set_t0_spin_bin(self._timing_base_for_preview_run())

    def _timing_base_for_preview_run(self, period_mode: str | None = None) -> int:
        """Resolve the file/fallback base independently of an authored manual offset."""
        run = self._t0_preview_run(period_mode)
        grouping = (
            dict(
                run.grouping,
                groups={
                    gid: shift_channel_entries(values, 1) for gid, values in self._groups.items()
                },
                forward_group=int(self._forward_combo.currentData() or 1),
                backward_group=int(self._backward_combo.currentData() or 2),
            )
            | self._exclusion_payload()
        )
        try:
            resolve_selected_timing(
                run.histograms,
                grouping,
                run.metadata,
                search_provider=partial(
                    self._cached_t0_search,
                    acquisition_key=self._raw_timing_key(run.histograms, run.metadata),
                ),
            )
        except ValueError:
            return self._common_t0_for_preview_run(None, period_mode=period_mode)
        return grouping["t0_bin"]

    def _cached_t0_search(
        self, histograms, metadata, *, acquisition_key=None, **options
    ) -> RunT0Search:
        """Only completed raw scans seed controls or previews on the GUI thread."""
        search = self._t0_search_cache.get(acquisition_key or self._t0_cache_key())
        if search is None:
            raise ValueError("Automatic t0 pending raw-channel search.")
        return search

    def _seed_t0_spin_from_detection(self) -> None:
        """Show selected own estimates only after the raw worker completes."""
        cached = self._t0_search_cache.get(self._t0_cache_key())
        if cached is None:
            self._schedule_t0_detection()
            return
        try:
            common = self._detected_common_t0_for_preview_run(cached)
        except ValueError:
            return
        self._set_t0_spin_bin(common)

    def _set_t0_spin_bin(self, consensus_t0: int) -> None:
        """Display a zero-based t0 in the file's bin-number convention."""
        blocked = self._t0_spin.blockSignals(True)
        try:
            self._t0_spin.setValue(consensus_t0 + self._bin_index_base())
        finally:
            self._t0_spin.blockSignals(blocked)
        with QSignalBlocker(self._last_good_spin), QSignalBlocker(self._t_good_offset_spin):
            self._on_t0_changed()

    # -- the always-on file / detected / Δ line (D11) --------------------

    def _raw_timing_key(self, histograms, metadata) -> tuple:
        """Acquired arrays are unchanged by editing; source facts and anchors may change."""
        return (
            id(self._run),
            source_is_pulsed(metadata),
            metadata.get("raw_period_histograms_available", True),
            tuple((id(hist.counts), hist.t0_bin, hist.bin_width) for hist in histograms),
        )

    def _t0_cache_key(self, period_mode: str | None = None) -> tuple:
        """Cache identity of the selected acquired period and strategy facts."""
        run = self._t0_preview_run(period_mode)
        return self._raw_timing_key(run.histograms, run.metadata)

    def _schedule_t0_detection(self) -> None:
        """Queue the preview run's detection unless it is cached or in flight."""
        if any(
            self._t0_cache_key(mode) not in self._t0_search_cache
            and self._t0_cache_key(mode) != self._t0_detection_key
            for mode in self._timing_period_modes()
        ):
            self._t0_detect_timer.start()

    def _timing_period_modes(self) -> tuple[str, ...]:
        """The acquisitions required by the current single or combined request."""
        mode = self._current_period_mode()
        return ("red", "green") if mode in ("green_minus_red", "green_plus_red") else (mode,)

    def _start_t0_detection(self) -> None:
        """Scan each required raw acquisition once through the existing task owner."""
        if self._t0_detection_key is not None:
            return
        pending = [
            mode
            for mode in self._timing_period_modes()
            if self._t0_cache_key(mode) not in self._t0_search_cache
        ]
        if not pending:
            return
        run = self._t0_preview_run(pending[0])
        self._t0_detection_key = self._t0_cache_key(pending[0])
        histograms, metadata = list(run.histograms), dict(run.metadata)
        policy, displayed_t0 = self._current_t0_policy(), self._t0_spin.value()
        self._tasks.start(
            lambda worker: find_t0_for_run(
                histograms, metadata, cancel_check=partial(check_task_cancelled, worker)
            ),
            on_finished=partial(
                self._on_t0_detection_finished, policy=policy, displayed_t0=displayed_t0
            ),
            on_error=partial(self._on_t0_detection_error, policy=policy, displayed_t0=displayed_t0),
        )

    def _on_t0_detection_error(self, message: str, *, policy=None, displayed_t0=None) -> None:
        """Retire failed scans through the same current-acquisition barrier."""
        self._on_t0_detection_finished(
            RunT0Search([], 0, 0, "unknown", False, message),
            policy=policy,
            displayed_t0=displayed_t0,
        )

    def _on_t0_detection_finished(self, result: object, *, policy=None, displayed_t0=None) -> None:
        """Store a finished detection and repaint the line (GUI thread)."""
        key, self._t0_detection_key = self._t0_detection_key, None
        assert isinstance(result, RunT0Search)
        assert key is not None
        unchanged_manual = (
            policy is not None
            and policy.mode == "manual"
            and self._current_t0_mode() == "manual"
            and self._t0_spin.value() == displayed_t0
            and self._current_t0_policy().to_dict() == policy.to_dict()
        )
        self._t0_search_cache[key] = result
        if key in {self._t0_cache_key(mode) for mode in self._timing_period_modes()}:
            if self._current_t0_mode() == "auto_detect":
                self._seed_t0_spin_from_detection()
            elif self._current_t0_mode() == "from_file":
                self._seed_t0_spin_from_preview()
            elif unchanged_manual:
                self._set_t0_spin_bin(
                    self._timing_base_for_preview_run() + int(policy.offset_bins or 0)
                )
            self._refresh_preview()
        else:
            self._refresh_t0_line()

    def _t0_assessment_grouping(self) -> dict[str, Any]:
        """Compare live groups with the selected period's acquired t0.

        Effective policy overrides cannot move the file baseline. Display-base
        bin numbers and live good-window controls keep the verdict in the
        coordinates of the edited request."""
        run_grouping = self._t0_preview_run().grouping
        grouping: dict[str, Any] = {
            "groups": {
                gid: shift_channel_entries(values, 1) for gid, values in self._groups.items()
            },
            "forward_group": int(self._forward_combo.currentData() or 1),
            "backward_group": int(self._backward_combo.currentData() or 2),
            "bin_index_base": self._bin_index_base(),
            "first_good_bin": self._resolve_good_bin_limits_from_controls()[2],
        } | self._exclusion_payload()
        if run_grouping.get("t0_source") is not None:
            grouping["t0_source"] = run_grouping["t0_source"]
        return grouping

    def _current_t0_verdict(self) -> T0Assessment:
        """The live t0 verdict for the preview run (D8), from the cached detection."""
        return assess_t0(
            self._t0_preview_run().histograms,
            self._t0_assessment_grouping(),
            self._t0_search_cache.get(self._t0_cache_key()),
        )

    def _refresh_t0_line(self) -> None:
        """Repaint the file / detected / Δ line and its verdict button (D8, D11).

        Cheap by construction — a cache read plus :func:`assess_t0`, which is
        O(detectors) — so it is safe on the per-edit refresh seam. The scan it
        displays is the debounced worker's, never this call's.

        The detected bin is the file bin plus the **median per-detector shift**,
        and the spread is the range of those shifts, not of the raw estimates:
        on per-detector-t0 data (PSI) the raw median and the group's header
        maximum are different quantities, so quoting them side by side reported
        a divergence that was only the detectors' real stagger.
        """
        if self._run is None or not self._run.histograms:
            self._t0_detected_label.setText("")
            self._t0_verdict_button.hide()
            return
        self._schedule_t0_detection()
        search = self._t0_search_cache.get(self._t0_cache_key())
        self._find_t0_btn.setEnabled(self._current_t0_mode() == "manual" and search is not None)
        verdict = self._current_t0_verdict()

        base = self._bin_index_base()
        file_common = self._file_common_t0_for_preview_run()
        source = self._t0_preview_run().grouping.get("t0_source")
        if source in ("detected", "missing"):
            file_part = "File: none"
        else:
            file_part = f"File: bin {file_common + base}"
        if search is None:
            detected_part = "Detected: …"
        elif not search.ok:
            detected_part = "Detected: unavailable"
        else:
            try:
                detected = self._detected_common_t0_for_preview_run(search)
            except ValueError as exc:
                detected_part = f"Detected: unavailable ({exc})"
            else:
                strategy = _T0_STRATEGY_LABELS[search.strategy]
                spread = selected_t0_diagnostics(
                    self._t0_preview_run().histograms, self._t0_assessment_grouping(), search
                )[1]
                detected_part = f"Detected: bin {detected + base} ({strategy}, spread {spread})"
        parts = [file_part, detected_part]
        if verdict.delta_bins is not None:
            parts.append(f"Δ {verdict.delta_bins:+d}")
        text = " · ".join(parts)
        # The label is an ElidedLabel: it elides itself to whatever width the
        # row is actually given (down to its own readable-prefix floor), so a
        # run with unusually long bin numbers no longer needs a manual elide
        # here. set_hover_text (not the default elision-only tooltip) always
        # shows the untruncated line on hover, even when the current width
        # happens to fit it — matching the line's previous, always-on tooltip.
        self._t0_detected_label.setText(text)
        self._t0_detected_label.set_hover_text(text)
        color = {"error": tokens.ERROR, "warn": tokens.WARN}.get(verdict.level, tokens.TEXT_MUTED)
        self._t0_detected_label.set_pen_color(color)
        self._refresh_t0_verdict_button(verdict)

    def _refresh_t0_verdict_button(self, verdict: T0Assessment) -> None:
        """Show (or hide) the ⚠ button carrying the verdict messages.

        Hidden at level ``ok`` — a clean run says nothing. Otherwise the button
        is tinted by severity and holds every message twice over: in the tooltip,
        for a hover, and as *disabled* menu entries, so a click parks them on
        screen to be read while nothing about them is clickable.
        """
        self._t0_verdict_menu.clear()
        if verdict.level == "ok":
            self._t0_verdict_button.hide()
            return
        tint = tokens.ERROR if verdict.level == "error" else tokens.WARN
        self._t0_verdict_button.setStyleSheet(f"color: {tint};")
        self._t0_verdict_button.setToolTip("\n".join(verdict.messages))
        for message in verdict.messages:
            self._t0_verdict_menu.addAction(message).setEnabled(False)
        self._t0_verdict_button.show()
