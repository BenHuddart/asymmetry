"""Tests for fit-panel helpers and worker glue."""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from asymmetry.gui.panels.fit_panel import GlobalFitWorker, _format_param_label


def test_format_param_label_known_and_unknown() -> None:
    assert _format_param_label("A0") == "A₀ (%)"
    assert _format_param_label("Lambda") == "λ (μs⁻¹)"
    assert _format_param_label("custom") == "custom"


def test_global_fit_worker_emits_finished() -> None:
    class _Engine:
        def global_fit(self, *_args, **_kwargs):
            return {1: "ok"}, {"A0": 0.2}

    worker = GlobalFitWorker(_Engine(), ["d"], lambda *_a, **_k: None, ["A0"], ["Lambda"], {})

    out = {}
    worker.finished.connect(lambda results, glob: out.update({"results": results, "global": glob}))

    worker.run()

    assert out["results"] == {1: "ok"}
    assert out["global"] == {"A0": 0.2}


def test_global_fit_worker_emits_error() -> None:
    class _Engine:
        def global_fit(self, *_args, **_kwargs):
            raise RuntimeError("boom")

    worker = GlobalFitWorker(_Engine(), ["d"], lambda *_a, **_k: None, ["A0"], ["Lambda"], {})

    errors = []
    worker.error.connect(lambda msg: errors.append(msg))

    worker.run()

    assert errors and "boom" in errors[0]
