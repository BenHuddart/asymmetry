"""Project-source commands are read-only unless selected owners and output are explicit."""

import json

import pytest

from asymmetry import cli
from asymmetry.core.project import load_prepared_project, save_project, verify_project_sources
from tests.io.test_acquisition_content import acquisition
from tests.project.test_source_decisions import owned_project

pytestmark = pytest.mark.io


def test_cli_verification_replacement_and_atomic_output(tmp_path, capsys):
    path, state = owned_project(tmp_path)
    project = tmp_path / "owned.asymp"
    save_project(state, project)
    before = project.read_bytes()
    acquisition(tmp_path, 5)
    cli.main(["project-sources", str(project), "--json"])
    report = json.loads(capsys.readouterr().out)
    assert report["sources"][0]["status"] == "changed"
    assert not report["written"]
    assert str(tmp_path) not in str(report)
    assert project.read_bytes() == before
    cli.main(
        ["project-sources", str(project), "--replace", "0", "--output", str(project), "--json"]
    )
    assert json.loads(capsys.readouterr().out)["written"]
    assert project.with_suffix(".asymp.bak").read_bytes() == before
    facts = verify_project_sources(load_prepared_project(project), project).facts
    assert facts[0].status == "matched"
    assert all(f.status == "changed" and not f.source.owner.active for f in facts[1:])


@pytest.mark.parametrize(
    "arguments",
    [
        ["--replace", "0"],
        ["--output", "new.asymp"],
        ["--replace", "1", "--output", "new.asymp"],
        ["--candidate", "999=missing"],
        ["--candidate", "invalid"],
    ],
)
def test_cli_rejects_implicit_or_historical_actions(tmp_path, capsys, arguments):
    path, state = owned_project(tmp_path)
    project = tmp_path / "owned.asymp"
    save_project(state, project)
    before = project.read_bytes()
    with pytest.raises(SystemExit) as exc:
        cli.main(["project-sources", str(project), *arguments])
    assert exc.value.code == 1
    assert project.read_bytes() == before
    assert "Traceback" not in capsys.readouterr().err
    assert not (tmp_path / "new.asymp").exists()


def test_cli_candidate_relocation_is_a_comparison(tmp_path, capsys):
    path, state = owned_project(tmp_path)
    project = tmp_path / "owned.asymp"
    save_project(state, project)
    moved = tmp_path / "relocated.nxs"
    path.rename(moved)
    cli.main(["project-sources", str(project), "--candidate", f"0={moved}", "--json"])
    report = json.loads(capsys.readouterr().out)
    assert report["sources"][0]["status"] == "matched"
    assert not report["written"]
    assert load_prepared_project(project).state["datasets"][0]["source_file"] == str(path)
