"""Save, load, and migrate Asymmetry project-state files."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from asymmetry.core.project.schema import CURRENT_SCHEMA_VERSION, load_project, migrate_to_current, save_project


def main() -> None:
    state = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "created_with_app_version": "0.1.0",
        "datasets": [{"run_number": 3301, "source_file": "run3301.nxs", "metadata_overrides": {}}],
        "browser_state": {"filters": {}, "selected_run_numbers": [3301], "selected_group_ids": [], "data_groups": [], "extra_columns": []},
        "plot_state": {"current_run_number": 3301, "bunch_factor": 1},
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "example.asymp"
        save_project(state, path)
        loaded = load_project(path)
        print("loaded schema:", loaded["schema_version"])

        old = {
            "schema_version": 2,
            "created_with_app_version": "0.0.9",
            "datasets": [{"run_number": 1200, "source_file": "legacy.nxs", "metadata_overrides": {}}],
            "browser_state": {"filters": {}, "selected_run_numbers": [1200], "selected_group_ids": [], "data_groups": []},
        }
        migrated = migrate_to_current(old)
        print("migrated schema:", migrated["schema_version"])
        print("has extra_columns:", "extra_columns" in migrated.get("browser_state", {}))

        raw_text = path.read_text(encoding="utf-8")
        print("saved bytes:", len(raw_text.encode("utf-8")))
        json.loads(raw_text)


if __name__ == "__main__":
    main()
