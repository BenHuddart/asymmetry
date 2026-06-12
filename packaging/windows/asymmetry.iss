; Asymmetry Windows Installer (Inno Setup 6)
;
; Build variables (passed via /D on the iscc command line):
;   AppVersion   - version string shown by the installer
;   AppDir       - absolute path to the PyInstaller onedir output (dist\Asymmetry)
;   IconFile     - absolute path to .ico
;   OutputDir    - directory to place the finished installer
;   OutputName   - (optional) installer base filename without extension;
;                  defaults to Asymmetry-{AppVersion}-windows-x64-setup

#define AppName    "Asymmetry"
#define AppExeName "Asymmetry.exe"
#define Publisher  "Asymmetry Contributors"
#define AppId      "io.github.benhuddart.asymmetry"
#define IconName   "asymmetry.ico"
#ifndef OutputName
  #define OutputName "Asymmetry-" + AppVersion + "-windows-x64-setup"
#endif

[Setup]
AppId={#AppId}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#Publisher}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
OutputDir={#OutputDir}
OutputBaseFilename={#OutputName}
SetupIconFile={#IconFile}
UninstallDisplayIcon={app}\{#IconName}
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
MinVersion=10.0
CloseApplications=yes
CloseApplicationsFilter={#AppExeName}
RestartApplications=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "{#AppDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#IconFile}"; DestDir: "{app}"; DestName: "{#IconName}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"; IconFilename: "{app}\{#IconName}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; IconFilename: "{app}\{#IconName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}"
