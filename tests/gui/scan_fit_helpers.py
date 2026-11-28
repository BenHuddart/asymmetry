"""Semantic terminal waits for the modeless scalar-scan fit actions."""

from PySide6.QtCore import QElapsedTimer
from PySide6.QtTest import QTest


def load_scan_selection(window, *datasets):
    """Load the scan inputs and select their Data Browser rows."""
    browser = window._data_browser
    with browser.batch_updates():
        for dataset in datasets:
            browser.add_dataset(dataset)
    browser.select_runs([dataset.run_number for dataset in datasets])


def wait_for_fit(window):
    """Await owned work and queued publication before inspecting fit consumers."""
    timer = QElapsedTimer()
    timer.start()
    while not window._scan_fits.idle and timer.elapsed() < 15000:
        QTest.qWait(5)
    assert window._scan_fits.idle, "Scan fit did not deliver its terminal"
