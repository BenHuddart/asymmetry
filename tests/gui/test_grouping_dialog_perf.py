"""Regression tests for GroupingDialog resolve_effective_grouping call counts.

Guards two performance fixes (audit tasks B1/B2):

* B1 — ``_reload_controls_from_seed`` used to call ``self._seed_source()`` five
  separate times, each re-resolving the draft from scratch via
  ``resolve_effective_grouping`` (a t0 auto-detect scan plus a per-run alpha
  estimate can run there). It now resolves once per reseed and passes the
  result through.
* B2 — ``_populate_group_table`` used to populate the group ``QTableWidget``
  without blocking signals while ``itemChanged`` was wired to both
  ``_mark_dirty`` and ``_refresh_preview`` (a synchronous
  ``resolve_effective_grouping``), so every table refresh fired up to
  ``4 * N_groups`` redundant resolves. Population now blocks ``itemChanged``
  and callers trigger the dirty/preview side effects explicitly, once.

``resolve_effective_grouping`` is monkeypatched with a call-counting wrapper at
both places it is bound: ``asymmetry.core.project.profiles`` (the source
module, re-imported fresh on every call inside
``GroupingDialog._preview_effective_grouping``) and
``asymmetry.gui.windows.grouping.profile_bridge`` (which imports the name once
at module load time for ``payload_from_profile_for_preview``, so patching the
source module alone would not intercept that path).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

import asymmetry.core.project.profiles as profiles_module
import asymmetry.gui.windows.grouping.dialog as grouping_dialog_dialog_module
import asymmetry.gui.windows.grouping.profile_bridge as profile_bridge_module
from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.gui.windows.grouping_dialog import GroupingDialog


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class _ResolveCounter:
    """Call-counting wrapper around the real ``resolve_effective_grouping``.

    Calls through to the original implementation so dialog behavior (groups,
    alpha, t0) stays real; only the call count is instrumented.
    """

    def __init__(self) -> None:
        self.count = 0
        self._original = profiles_module.resolve_effective_grouping

    def __call__(self, profile, run):
        self.count += 1
        return self._original(profile, run)


@pytest.fixture
def resolve_counter(monkeypatch: pytest.MonkeyPatch) -> _ResolveCounter:
    counter = _ResolveCounter()
    monkeypatch.setattr(profiles_module, "resolve_effective_grouping", counter, raising=True)
    monkeypatch.setattr(profile_bridge_module, "resolve_effective_grouping", counter, raising=True)
    return counter


def _dataset_with_histograms(run_number: int = 4001) -> MuonDataset:
    h1 = Histogram(counts=np.array([100.0, 100.0, 100.0, 100.0]), bin_width=0.01)
    h2 = Histogram(counts=np.array([50.0, 50.0, 50.0, 50.0]), bin_width=0.01)
    run = Run(
        run_number=run_number,
        histograms=[h1, h2],
        metadata={"run_number": run_number, "title": "Grouping Perf Test"},
        grouping={
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "first_good_bin": 0,
            "last_good_bin": 3,
        },
    )
    t = np.array([0.0, 0.01, 0.02, 0.03])
    return MuonDataset(
        time=t,
        asymmetry=np.zeros_like(t),
        error=np.full_like(t, 0.01),
        metadata={"run_number": run_number},
        run=run,
    )


def _dataset_with_eight_groups(run_number: int = 4002) -> MuonDataset:
    """A reference run whose grouping has 8 detector groups, 1 detector each."""
    histograms = [
        Histogram(counts=np.array([100.0, 100.0, 100.0, 100.0]), bin_width=0.01) for _ in range(8)
    ]
    run = Run(
        run_number=run_number,
        histograms=histograms,
        metadata={"run_number": run_number, "title": "Eight Group Perf Test"},
        grouping={
            "groups": {i: [i - 1] for i in range(1, 9)},
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "first_good_bin": 0,
            "last_good_bin": 3,
        },
    )
    t = np.array([0.0, 0.01, 0.02, 0.03])
    return MuonDataset(
        time=t,
        asymmetry=np.zeros_like(t),
        error=np.full_like(t, 0.01),
        metadata={"run_number": run_number},
        run=run,
    )


def test_opening_dialog_resolves_bounded_number_of_times(
    qapp: QApplication, resolve_counter: _ResolveCounter
) -> None:
    """Opening the dialog performs exactly one GUI-thread resolve.

    That one resolve seeds the form controls (``_seed_source`` in ``__init__``,
    now shared by every field read from it instead of re-resolved per field).
    The initial live preview no longer resolves synchronously at all: since B3,
    ``_refresh_preview`` hands the unresolved draft to the preview pane, whose
    worker thread resolves + reduces (covered by
    ``test_grouping_preview_pane.py``). Before the B1–B3 fixes this
    construction path performed several more GUI-thread resolves (five from
    ``_seed_source`` fan-out alone, one from the synchronous preview, plus one
    per group-table cell from the unblocked ``itemChanged`` storm).
    """
    GroupingDialog([_dataset_with_histograms()])
    assert resolve_counter.count == 1


def test_reload_controls_from_seed_resolves_once(
    qapp: QApplication, resolve_counter: _ResolveCounter
) -> None:
    dialog = GroupingDialog([_dataset_with_histograms()])
    resolve_counter.count = 0

    dialog._reload_controls_from_seed()

    # Exactly one GUI-thread resolve, from the single _seed_source() call. The
    # explicit _refresh_preview() at the end of the reseed no longer resolves
    # here — since B3 it passes the unresolved draft to the preview pane's
    # worker thread.
    assert resolve_counter.count == 1


def test_refresh_preview_never_resolves_on_the_gui_thread(
    qapp: QApplication, resolve_counter: _ResolveCounter
) -> None:
    """The per-keystroke preview slot performs zero synchronous resolves (B3).

    ``_refresh_preview`` fires from nearly every form control; under an
    ``auto_detect`` t0 policy a synchronous resolve scans every detector's full
    histogram (~276 ms at 128 detectors x 1M bins), so it must only build the
    cheap draft profile here and leave resolution to the preview pane's worker.
    """
    dialog = GroupingDialog([_dataset_with_histograms()])
    resolve_counter.count = 0

    for _ in range(5):  # a burst, as from typing in the exclude field
        dialog._refresh_preview()

    assert resolve_counter.count == 0


def test_populate_group_table_does_not_resolve_or_refresh_preview(
    qapp: QApplication, resolve_counter: _ResolveCounter, monkeypatch: pytest.MonkeyPatch
) -> None:
    dialog = GroupingDialog([_dataset_with_eight_groups()])
    assert len(dialog._groups) == 8

    refresh_calls = []
    monkeypatch.setattr(dialog, "_refresh_preview", lambda *a, **k: refresh_calls.append((a, k)))
    resolve_counter.count = 0

    dialog._populate_group_table()

    assert dialog._group_table.rowCount() == 8
    assert resolve_counter.count == 0
    assert refresh_calls == []


def test_preset_apply_refreshes_preview_exactly_once_after_repopulation(
    qapp: QApplication, resolve_counter: _ResolveCounter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A caller that repopulates the table must trigger one preview refresh.

    ``_on_preset_combo_activated`` repopulates the group table (itemChanged
    blocked, per B2) and then must call ``_refresh_preview`` explicitly once —
    it can no longer rely on the itemChanged storm to do it implicitly. Drives
    the real ``_on_preset_combo_activated`` path (rather than replaying its
    body in the test) so a regression that drops the explicit call is caught.
    """
    dialog = GroupingDialog([_dataset_with_eight_groups()])

    payload = {
        "groups": {1: [1, 2], 2: [3, 4]},
        "group_names": {1: "Forward", 2: "Backward"},
        "forward_group": 1,
        "backward_group": 2,
    }
    # Preset resolution itself (instrument layout lookup, preset table) is
    # exercised elsewhere; stub it here so the test isolates the
    # populate-then-refresh contract under B2.
    monkeypatch.setattr(dialog, "_current_instrument_layout", lambda: object())
    monkeypatch.setattr(
        grouping_dialog_dialog_module, "preset_payload", lambda layout, name: payload
    )
    dialog._preset_combo.addItem("Test Preset", "test-preset")
    index = dialog._preset_combo.count() - 1

    refresh_calls = []
    monkeypatch.setattr(dialog, "_refresh_preview", lambda *a, **k: refresh_calls.append((a, k)))

    dialog._on_preset_combo_activated(index)

    assert dialog._group_table.rowCount() == 2
    assert len(refresh_calls) == 1


# --------------------------------------------------------------------------- #
# Auto-detect t0: the read-only display reuses the resolve's consensus instead
# of running a second full-detector find_t0_for_run scan on the GUI thread.
# --------------------------------------------------------------------------- #


def _dataset_with_prompt_peak(run_number: int = 4003, n_det: int = 4) -> MuonDataset:
    """A run whose histograms have a real prompt peak, so ``find_t0_for_run``
    succeeds (and the resolve writes its consensus/strategy provenance)."""
    n_bins = 256
    t0 = 13
    idx = np.arange(n_bins)
    decay = np.where(idx >= t0, 1000.0 * np.exp(-(idx - t0) / 40.0), 0.0)
    counts = decay + 5.0  # flat baseline under the decay
    histograms = [Histogram(counts=counts.copy(), bin_width=0.016, t0_bin=t0) for _ in range(n_det)]
    half = n_det // 2
    run = Run(
        run_number=run_number,
        histograms=histograms,
        metadata={"run_number": run_number, "instrument": "EMU", "title": "Prompt Peak Perf"},
        grouping={
            "groups": {1: list(range(half)), 2: list(range(half, n_det))},
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "first_good_bin": t0 + 2,
            "last_good_bin": n_bins - 1,
            "t0_bin": t0,
            "instrument": "EMU",
        },
    )
    t = idx * 0.016
    return MuonDataset(
        time=t,
        asymmetry=np.zeros_like(t),
        error=np.full_like(t, 0.01),
        metadata={"run_number": run_number, "instrument": "EMU"},
        run=run,
    )


class _T0ScanCounter:
    """Call-counting wrapper around the real ``find_t0_for_run`` (the expensive
    per-detector scan), calling through so consensus/strategy stay real."""

    def __init__(self) -> None:
        self.count = 0
        self._original = profiles_module.find_t0_for_run

    def __call__(self, histograms, metadata=None, *, pulsed=None):
        self.count += 1
        return self._original(histograms, metadata, pulsed=pulsed)


@pytest.fixture
def t0_scan_counter(monkeypatch: pytest.MonkeyPatch) -> _T0ScanCounter:
    counter = _T0ScanCounter()
    # find_t0_for_run is bound in profiles (the resolve path) and in the dialog
    # module (the read-only display path); patch both so either call is counted.
    monkeypatch.setattr(profiles_module, "find_t0_for_run", counter, raising=True)
    monkeypatch.setattr(grouping_dialog_dialog_module, "find_t0_for_run", counter, raising=True)
    return counter


def test_auto_detect_reseed_scans_each_detector_once(
    qapp: QApplication, t0_scan_counter: _T0ScanCounter
) -> None:
    """Under auto-detect, a reseed runs exactly one full-detector t0 scan.

    The resolve scans once and writes the consensus into the payload; the
    read-only t0 spinbox now reads that back instead of re-scanning. Before this
    fix the display ran a second scan (``find_t0_for_run`` twice per reseed —
    the residual GUI-thread stall that decimation, PR #229, did not touch).
    """
    dialog = GroupingDialog([_dataset_with_prompt_peak()])
    dialog._draft.t0_policy = profiles_module.T0Policy(mode="auto_detect")
    t0_scan_counter.count = 0

    dialog._reload_controls_from_seed()

    assert t0_scan_counter.count == 1


def test_from_file_reseed_never_scans(qapp: QApplication, t0_scan_counter: _T0ScanCounter) -> None:
    """The default from_file policy performs no GUI-thread t0 scan at all."""
    dialog = GroupingDialog([_dataset_with_prompt_peak()])
    dialog._draft.t0_policy = profiles_module.T0Policy(mode="from_file")
    t0_scan_counter.count = 0

    dialog._reload_controls_from_seed()

    assert t0_scan_counter.count == 0


def test_auto_detect_display_matches_resolved_t0(qapp: QApplication) -> None:
    """The spinbox shows the exact consensus the resolve computed.

    Guards the latent divergence: the old display scan merged the
    reference-dataset metadata that core's resolve does not, so ``source_is_pulsed``
    could pick a different strategy and show a t0 the reduction never used.
    Reading the value from the resolved payload makes them equal by construction.
    """
    dialog = GroupingDialog([_dataset_with_prompt_peak()])
    dialog._draft.t0_policy = profiles_module.T0Policy(mode="auto_detect")

    dialog._reload_controls_from_seed()

    resolved = dialog._last_resolved_seed
    assert resolved is not None
    assert resolved.get("t0_search_strategy")  # the resolve took the auto path
    base = dialog._bin_index_base()
    assert dialog._t0_spin.value() == int(resolved["t0_bin"]) + base


def test_auto_detect_display_falls_back_when_payload_lacks_t0_bin(
    qapp: QApplication, t0_scan_counter: _T0ScanCounter
) -> None:
    """Provenance without a scalar ``t0_bin`` must not display t0 = 0.

    ``_apply_t0_policy`` writes ``t0_search_strategy`` before its ``delta == 0``
    early return but leaves ``t0_bin`` unwritten there, so a run with
    per-detector-only t0 can carry provenance without a consensus. The display
    must fall through to a scan and show the real consensus, not a ``t0_bin``
    default of 0.
    """
    dialog = GroupingDialog([_dataset_with_prompt_peak()])
    dialog._draft.t0_policy = profiles_module.T0Policy(mode="auto_detect")
    # Strategy present but no scalar t0_bin — the delta==0 corner.
    dialog._last_resolved_seed = {"t0_search_strategy": "pulse_edge"}
    t0_scan_counter.count = 0

    dialog._seed_t0_spin_from_detection()

    assert t0_scan_counter.count == 1  # fast path skipped; scanned instead
    base = dialog._bin_index_base()
    expected = t0_scan_counter._original(dialog._run.histograms, dialog._run.metadata or {})
    assert dialog._t0_spin.value() == int(expected.consensus_t0_bin) + base


def test_toggle_to_auto_detect_scans_once_with_run_metadata(
    qapp: QApplication, t0_scan_counter: _T0ScanCounter
) -> None:
    """An explicit switch to auto-detect (no fresh resolve) scans exactly once,
    with core's ``run.metadata`` only, so the display matches the reduction."""
    dialog = GroupingDialog([_dataset_with_prompt_peak()])
    dialog._draft.t0_policy = profiles_module.T0Policy(mode="from_file")
    dialog._reload_controls_from_seed()  # from_file: no auto provenance cached
    t0_scan_counter.count = 0

    dialog._set_t0_mode_combo("auto_detect")
    dialog._apply_t0_mode_to_controls()

    assert t0_scan_counter.count == 1
    base = dialog._bin_index_base()
    expected = t0_scan_counter._original(dialog._run.histograms, dialog._run.metadata or {})
    assert dialog._t0_spin.value() == int(expected.consensus_t0_bin) + base
