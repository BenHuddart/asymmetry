"""Get Info provenance from a serialized synthetic legacy LEM ROOT acquisition.

The same production loader used by the GUI supplies numeric timing, scalar
metadata, two histogram identities, and retained terminal bins. This capture
exercises the calibration boundary without asserting physical t0 accuracy.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEventLoop, Qt, QTimer
from PySide6.QtWidgets import QApplication

from asymmetry.core.io import load

from ._base import CaptureContext, Scenario, _optimize_png, register


class RunInfoProvenanceScenario(Scenario):
    name = "run_info_provenance"
    description = (
        "Run Info dialog showing the preserved provenance of a loaded legacy LEM ROOT run."
    )
    size = (780, 660)

    def capture(self, ctx: CaptureContext) -> Path:  # noqa: D401
        from asymmetry.gui.windows.run_info_dialog import RunInfoDialog

        fixture = Path(__file__).parents[3] / "tests/fixtures/psi_root/legacy-lem.root"
        dataset = load(str(fixture))
        assert dataset.run.histograms[0].bin_width == 0.01
        assert dataset.run.grouping["detector_t0_bins"] == [4, 7]
        # Logical synthetic provenance keeps workstation paths out of the image.
        dataset.metadata["source_file"] = fixture.name
        dataset.run.source_file = fixture.name

        dialog = RunInfoDialog(dataset)
        dialog.resize(*self.size)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        dialog.show()
        # The total-counts cells are backfilled from a QTimer.singleShot(0, ...)
        # deferred sum (see RunInfoDialog._fill_total_counts), so pump the event
        # loop until they land before grabbing — otherwise the capture shows the
        # transient "computing…" placeholder.
        _pump_events(200)

        pix = dialog.grab()
        out_path = ctx.output_dir / f"{self.name}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not pix.save(str(out_path), "PNG"):
            raise RuntimeError(f"Failed to save screenshot to {out_path}")
        _optimize_png(out_path)

        dialog.close()
        dialog.deleteLater()
        _pump_events(40)
        return out_path


def _pump_events(milliseconds: int) -> None:
    loop = QEventLoop()
    QTimer.singleShot(int(milliseconds), loop.quit)
    loop.exec()
    QApplication.processEvents()


register(RunInfoProvenanceScenario())


class RunInfoUnitsScenario(Scenario):
    name = "run_info_units"
    description = "NeXus declarations beside interpreted physical quantities."
    size = (1280, 680)

    def capture(self, ctx: CaptureContext) -> Path:
        import shutil
        from tempfile import TemporaryDirectory

        import h5py

        from asymmetry.gui.windows.run_info_dialog import RunInfoDialog

        fixture = (
            Path(__file__).parents[3] / "tests/fixtures/reference_cases/manual-phase/synthetic.nxs"
        )
        with TemporaryDirectory() as folder:
            source = Path(folder) / "synthetic.nxs"
            shutil.copyfile(fixture, source)
            with h5py.File(source, "r+") as file:
                sample = file["raw_data_1/sample"]
                sample["temperature"][...] = 300
                sample["temperature"].attrs["units"] = "mK"
                sample["magnetic_field"][...] = 100
                sample["magnetic_field"].attrs["units"] = "K"
                resolution = file["raw_data_1/instrument/detector_1/resolution"]
                resolution[...] = 10
                del resolution.attrs["units"]
            dataset = load(str(source))
        dataset.metadata["source_file"] = "synthetic.nxs"
        dataset.run.source_file = "synthetic.nxs"
        dialog = RunInfoDialog(dataset)
        dialog.resize(*self.size)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        dialog.show()
        _pump_events(200)
        out = ctx.output_dir / f"{self.name}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        if not dialog.grab().save(str(out), "PNG"):
            raise RuntimeError("Could not save units screenshot")
        _optimize_png(out)
        dialog.close()
        dialog.deleteLater()
        _pump_events(40)
        return out


register(RunInfoUnitsScenario())


class LegacyNexusGroupingLogScenario(Scenario):
    """Capture import warnings using synthetic V1 acquisitions."""

    name = "legacy_nexus_grouping_log"
    description = "The desktop log reports legacy EMU grouping and a conflicting run number."
    size = (1080, 720)

    def capture(self, ctx: CaptureContext) -> Path:
        from contextlib import chdir
        from shutil import copyfile
        from tempfile import TemporaryDirectory

        import h5py
        import numpy as np

        from asymmetry.gui.mainwindow import MainWindow

        window = MainWindow()
        try:
            window.resize(*self.size)
            window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
            window.show()
            window._dock_log.show()
            window.resizeDocks([window._dock_log], [150], Qt.Orientation.Vertical)
            window._log_panel.clear()
            with TemporaryDirectory() as directory:
                source = Path(directory) / "synthetic-emu.nxs"
                with h5py.File(source, "w") as handle:
                    entry = handle.create_group("run")
                    entry.create_dataset("analysis", data=b"muonTD")
                    entry.create_dataset("number", data=42)
                    entry.create_dataset("switching_states", data=1)
                    entry.create_dataset("instrument/beam/period_type", data=[1])
                    entry.create_dataset("instrument/name", data=b"EMU")
                    entry.create_dataset("instrument/detector/orientation", data=b"l")
                    entry.create_dataset("instrument/detector/number", data=96)
                    detector = entry.create_group("histogram_data_1")
                    counts = np.tile(np.arange(24, 0, -1), (96, 1))
                    detector.create_dataset("counts", data=counts).attrs.update(
                        t0_bin=3, first_good_bin=5, last_good_bin=20
                    )
                    detector.create_dataset("resolution", data=20).attrs["units"] = "ns"
                    detector.create_dataset("grouping", data=np.zeros(96, dtype=int)).attrs[
                        "available"
                    ] = 1
                conflicting_path = Path("another") / source.name
                with chdir(directory):
                    conflicting_path.parent.mkdir()
                    copyfile(source, conflicting_path)
                    window._load_files([source.name, str(conflicting_path)])
            log = window._log_panel.to_plain_text()
            if "Applied the standard EMU" not in log or "Loaded 1 file(s), 1 failed" not in log:
                raise RuntimeError("Legacy EMU import did not report both warnings")
            _pump_events(200)
            output = ctx.output_dir / f"{self.name}.png"
            output.parent.mkdir(parents=True, exist_ok=True)
            if not window._dock_log.grab().save(str(output), "PNG"):
                raise RuntimeError("Could not save legacy NeXus grouping screenshot")
            _optimize_png(output)
            return output
        finally:
            self.teardown(window)


register(LegacyNexusGroupingLogScenario())
