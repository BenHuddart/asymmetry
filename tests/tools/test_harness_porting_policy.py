from __future__ import annotations

import runpy
from pathlib import Path

HARNESS = runpy.run_path(str(Path(__file__).resolve().parents[2] / "tools" / "harness.py"))
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


def _write_candidate_dir(candidates_root: Path, slug: str) -> Path:
    candidate_dir = candidates_root / slug
    candidate_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("README.md", "comparison.md", "scoring.md"):
        (candidate_dir / filename).write_text(f"# {filename}\n", encoding="utf-8")
    return candidate_dir


def _candidate_index_entry(slug: str) -> str:
    return f"""{{
      "slug": "{slug}",
      "feature_name": "Feature {slug}",
      "status": "candidate",
      "tier": "now",
      "score": 10,
      "path": "docs/porting/candidates/{slug}",
      "references": ["WiMDA"],
      "docs": {{
        "readme": "docs/porting/candidates/{slug}/README.md",
        "comparison": "docs/porting/candidates/{slug}/comparison.md",
        "scoring": "docs/porting/candidates/{slug}/scoring.md"
      }},
      "updated": "2026-01-01"
    }}"""


def test_porting_policy_accepts_valid_candidate(tmp_path: Path) -> None:
    porting_root = _write_porting_root(tmp_path)
    _write_candidate_dir(porting_root / "candidates", "my-feature")
    (porting_root / "index.json").write_text(
        '{"version": 1, "studies": [' + _candidate_index_entry("my-feature") + "]}\n",
        encoding="utf-8",
    )

    failures = find_porting_policy_violations(tmp_path)

    assert failures == []


def test_porting_policy_flags_candidate_missing_scoring(tmp_path: Path) -> None:
    porting_root = _write_porting_root(tmp_path)
    candidate_dir = _write_candidate_dir(porting_root / "candidates", "my-feature")
    (candidate_dir / "scoring.md").unlink()
    (porting_root / "index.json").write_text(
        '{"version": 1, "studies": [' + _candidate_index_entry("my-feature") + "]}\n",
        encoding="utf-8",
    )

    failures = find_porting_policy_violations(tmp_path)

    assert len(failures) == 1
    assert failures[0].path == candidate_dir / "scoring.md"


def test_porting_policy_flags_candidate_dir_not_in_index(tmp_path: Path) -> None:
    porting_root = _write_porting_root(tmp_path)
    _write_candidate_dir(porting_root / "candidates", "my-feature")
    (porting_root / "index.json").write_text(
        '{"version": 1, "studies": []}\n',
        encoding="utf-8",
    )

    failures = find_porting_policy_violations(tmp_path)

    assert len(failures) == 1
    assert failures[0].path == porting_root / "candidates" / "my-feature"


def test_porting_policy_flags_indexed_candidate_missing_directory(tmp_path: Path) -> None:
    porting_root = _write_porting_root(tmp_path)
    (porting_root / "index.json").write_text(
        '{"version": 1, "studies": [' + _candidate_index_entry("my-feature") + "]}\n",
        encoding="utf-8",
    )

    failures = find_porting_policy_violations(tmp_path)

    assert len(failures) == 1
    assert "my-feature" in failures[0].message


def test_porting_policy_ignores_non_study_dirs(tmp_path: Path) -> None:
    porting_root = _write_porting_root(tmp_path)
    (porting_root / "practical-workflows").mkdir()
    (porting_root / "reference").mkdir()
    (porting_root / "index.json").write_text(
        '{"version": 1, "studies": []}\n',
        encoding="utf-8",
    )

    failures = find_porting_policy_violations(tmp_path)

    assert failures == []
