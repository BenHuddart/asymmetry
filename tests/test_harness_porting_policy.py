from __future__ import annotations

import runpy
from pathlib import Path

HARNESS = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools" / "harness.py"))
find_porting_policy_violations = HARNESS["find_porting_policy_violations"]


def _write_porting_root(root: Path) -> Path:
    porting_root = root / "docs" / "porting"
    porting_root.mkdir(parents=True)
    (porting_root / "README.md").write_text("# Porting Workflow\n", encoding="utf-8")
    return porting_root


def test_porting_policy_accepts_empty_study_index(tmp_path: Path) -> None:
    porting_root = _write_porting_root(tmp_path)
    (porting_root / "index.json").write_text(
        '{"version": 1, "studies": []}\n',
        encoding="utf-8",
    )

    failures = find_porting_policy_violations(tmp_path)

    assert failures == []


def test_porting_policy_requires_full_study_artifacts(tmp_path: Path) -> None:
    porting_root = _write_porting_root(tmp_path)
    study_dir = porting_root / "background-correction"
    study_dir.mkdir()
    (study_dir / "README.md").write_text("# Background correction\n", encoding="utf-8")
    (study_dir / "comparison.md").write_text("# Comparison\n", encoding="utf-8")
    (study_dir / "implementation-options.md").write_text("# Options\n", encoding="utf-8")
    (study_dir / "test-data.md").write_text("# Test data\n", encoding="utf-8")
    (porting_root / "index.json").write_text(
        """
        {
          "version": 1,
          "studies": [
            {
              "slug": "background-correction",
              "feature_name": "Background correction",
              "status": "study",
              "path": "docs/porting/background-correction",
              "references": ["WiMDA", "musrfit", "Mantid"],
              "docs": {
                "readme": "docs/porting/background-correction/README.md",
                "comparison": "docs/porting/background-correction/comparison.md",
                "implementation_options": "docs/porting/background-correction/implementation-options.md",
                "test_data": "docs/porting/background-correction/test-data.md",
                "verification_plan": "docs/porting/background-correction/verification-plan.md"
              }
            }
          ]
        }
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    failures = find_porting_policy_violations(tmp_path)

    assert len(failures) == 1
    assert failures[0].path == study_dir / "verification-plan.md"


def test_porting_policy_requires_index_entry_for_study_dir(tmp_path: Path) -> None:
    porting_root = _write_porting_root(tmp_path)
    study_dir = porting_root / "background-correction"
    study_dir.mkdir()
    for filename in (
        "README.md",
        "comparison.md",
        "implementation-options.md",
        "test-data.md",
        "verification-plan.md",
    ):
        (study_dir / filename).write_text(f"# {filename}\n", encoding="utf-8")
    (porting_root / "index.json").write_text(
        '{"version": 1, "studies": []}\n',
        encoding="utf-8",
    )

    failures = find_porting_policy_violations(tmp_path)

    assert len(failures) == 1
    assert failures[0].path == study_dir
