"""Data Browser route from a memory-only acquisition to verified file backing."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING

from PySide6.QtWidgets import QFileDialog, QMessageBox

from asymmetry.core.io.persist_run import persist_histogram_run
from asymmetry.core.transform.preparation import histogram_observation, prepare_histogram_asymmetry
from asymmetry.gui.tasks import check_task_cancelled
from asymmetry.gui.utils.scientific_input import ScientificInputDialog

if TYPE_CHECKING:
    from asymmetry.gui.panels.data_browser import DataBrowserPanel


def save_histogram_run(browser: DataBrowserPanel, run_number: int) -> None:
    """Keep the current scientific owner and its history while capturing its file."""
    dataset = browser.get_dataset(run_number)
    if dataset is None or dataset.run is None:
        return
    run = dataset.run
    model = browser._project_model
    source_file, source_binding = run.source_file, run.source_binding
    observation = dataset.histogram_observation
    recipe = dataset.effective_recipe
    path, _ = QFileDialog.getSaveFileName(
        browser, "Save as NeXus", f"SIM{run_number}.nxs", "NeXus files (*.nxs)"
    )
    if not path:
        return
    if not path.lower().endswith(".nxs"):
        path += ".nxs"
    if run.grouping.get("period_histograms") or run.grouping.get("period_inputs"):
        QMessageBox.warning(
            browser,
            "Could Not Save NeXus",
            "Save as NeXus requires a single-period histogram run",
        )
        return
    if any(hist.count_variance is not None for hist in run.histograms):
        QMessageBox.warning(
            browser,
            "Could Not Save NeXus",
            "Save as NeXus requires raw counts without supplied counting variance",
        )
        return
    # Counts are acquired facts: retain their buffers while the worker copies
    # and writes them; settings and histogram clocks are detached here.
    acquired_counts = tuple(hist.counts for hist in run.histograms)
    histograms = [replace(hist, counts=hist.counts.view()) for hist in run.histograms]
    for hist in histograms:
        hist.counts.setflags(write=False)
    snapshot = replace(
        run,
        histograms=histograms,
        metadata=deepcopy(run.metadata),
        grouping=deepcopy(run.grouping),
    )

    def capture(worker):
        checkpoint = partial(check_task_cancelled, worker)
        saved = persist_histogram_run(snapshot, path, checkpoint=checkpoint)
        current_observation, current_recipe = observation, recipe
        if observation is not None:
            source = replace(
                observation.source,
                source_file=saved.source_file,
                source_binding=saved.source_binding,
            )
            current_observation = histogram_observation(
                source,
                facility=observation.facility,
                reference=observation.reference,
                binning=observation.binning,
                cancel_check=checkpoint,
            )
            current_recipe = prepare_histogram_asymmetry(
                source,
                merge=1,
                facility=observation.facility,
                reference=observation.reference,
                binning=observation.binning,
                cancel_check=checkpoint,
            ).effective_recipe
            if recipe is not None and "curve_operations" in recipe.to_dict():
                current_recipe = current_recipe.with_facts(
                    curve_operations=recipe.to_dict()["curve_operations"]
                )
        checkpoint()
        return saved, current_observation, current_recipe

    def is_current():
        return (
            browser._project_model is model
            and browser.get_dataset(run_number) is dataset
            and dataset.run is run
            and run.source_file == source_file
            and run.source_binding == source_binding
            and len(run.histograms) == len(snapshot.histograms)
            and all(
                current.counts is acquired
                and current.bin_width == captured.bin_width
                and current.t0_bin == captured.t0_bin
                and current.good_bin_start == captured.good_bin_start
                and current.good_bin_end == captured.good_bin_end
                and current.t0_time_us == captured.t0_time_us
                for current, captured, acquired in zip(
                    run.histograms, snapshot.histograms, acquired_counts, strict=True
                )
            )
            and dataset.histogram_observation is observation
            and dataset.effective_recipe is recipe
        )

    dialog = ScientificInputDialog(browser, "Saving and verifying NeXus…", "Save as NeXus")
    try:
        result = dialog.run(capture, is_current=is_current)
    except ValueError as exc:
        QMessageBox.warning(browser, "Could Not Save NeXus", str(exc))
        return
    finally:
        dialog.deleteLater()
    if result is None:
        return
    saved, dataset.histogram_observation, dataset.effective_recipe = result
    run.source_file = saved.source_file
    run.source_binding = saved.source_binding
    dataset.metadata["grouping_overrides"] = True
    browser.datasets_changed.emit()
