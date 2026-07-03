from __future__ import annotations

from pathlib import Path

from asymmetry.gui.export_paths import resolve_gle_export_paths


def test_resolve_gle_export_paths_defaults_to_foldered_bundle(tmp_path: Path) -> None:
    selected_path = tmp_path / "example_plot.gle"

    output_path, export_dir = resolve_gle_export_paths(selected_path)

    assert export_dir == tmp_path / "example_plot.gleplot"
    assert output_path == export_dir / "example_plot.gle"


def test_resolve_gle_export_paths_can_leave_files_in_place(tmp_path: Path) -> None:
    selected_path = tmp_path / "example_plot.gle"

    output_path, export_dir = resolve_gle_export_paths(selected_path, folder=False)

    assert export_dir == tmp_path
    assert output_path == selected_path


def test_resolve_gle_export_paths_accepts_gleplot_bundle_names(tmp_path: Path) -> None:
    selected_path = tmp_path / "example_plot.gleplot"

    output_path, export_dir = resolve_gle_export_paths(selected_path)

    assert export_dir == selected_path
    assert output_path == selected_path / "example_plot.gle"
