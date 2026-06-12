from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_inno_installer_uses_selected_windows_defaults() -> None:
    iss_text = _read("packaging/windows/asymmetry.iss")

    assert 'AppId      "io.github.benhuddart.asymmetry"' in iss_text
    assert "PrivilegesRequired=lowest" in iss_text
    assert "PrivilegesRequiredOverridesAllowed=dialog" in iss_text
    assert "ArchitecturesAllowed=x64compatible" in iss_text
    assert "ArchitecturesInstallIn64BitMode=x64compatible" in iss_text
    assert "CloseApplications=yes" in iss_text
    assert "CloseApplicationsFilter={#AppExeName}" in iss_text
    assert "RestartApplications=yes" in iss_text
    assert 'Name: "desktopicon"' in iss_text
    assert 'Tasks: desktopicon' in iss_text
    assert 'Description: "{cm:LaunchProgram,{#AppName}}"' in iss_text
    assert 'SetupIconFile={#IconFile}' in iss_text
    assert 'UninstallDisplayIcon={app}\\{#IconName}' in iss_text
    assert 'DestName: "{#IconName}"' in iss_text
    assert 'Type: filesandordirs; Name: "{app}"' in iss_text


def test_windows_workflows_use_inno_setup_installer() -> None:
    release_text = _read(".github/workflows/release.yml")
    preview_text = _read(".github/workflows/preview-windows.yml")

    for workflow_text in (release_text, preview_text):
        assert "choco install innosetup -y" in workflow_text
        assert "packaging\\windows\\asymmetry.iss" in workflow_text
        assert ".exe" in workflow_text
        assert "installer.wxs" not in workflow_text
        assert "wix build" not in workflow_text
