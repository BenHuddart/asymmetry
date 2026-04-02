Unicode true
ManifestDPIAware true

!include "MUI2.nsh"
!include "FileFunc.nsh"
!include "x64.nsh"

!define APP_NAME "Asymmetry"
!define APP_PUBLISHER "Asymmetry Contributors"

!ifndef VERSION
  !define VERSION "0.0.0-dev"
!endif

!ifndef DIST_DIR
  !error "DIST_DIR define is required (path to PyInstaller onedir output)."
!endif

!ifndef ICON_FILE
  !define ICON_FILE "build\\icons\\Asymmetry.ico"
!endif

!ifndef OUTPUT_FILE
  !define OUTPUT_FILE "Asymmetry-${VERSION}-windows-x64-setup.exe"
!endif

Name "${APP_NAME}"
OutFile "${OUTPUT_FILE}"
BrandingText "${APP_PUBLISHER}"
RequestExecutionLevel admin
InstallDir "$PROGRAMFILES64\\${APP_NAME}"
InstallDirRegKey HKLM "Software\\${APP_NAME}" "Install_Dir"

Icon "${ICON_FILE}"
UninstallIcon "${ICON_FILE}"

!define MUI_ABORTWARNING
!define MUI_ICON "${ICON_FILE}"
!define MUI_UNICON "${ICON_FILE}"

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_UNPAGE_FINISH

!insertmacro MUI_LANGUAGE "English"

Section "Install" SEC_INSTALL
  SetOutPath "$INSTDIR"
  File /r "${DIST_DIR}\\*.*"

  WriteUninstaller "$INSTDIR\\uninstall.exe"

  CreateDirectory "$SMPROGRAMS\\${APP_NAME}"
  CreateShortCut "$SMPROGRAMS\\${APP_NAME}\\${APP_NAME}.lnk" "$INSTDIR\\Asymmetry.exe" "" "$INSTDIR\\Asymmetry.exe" 0
  CreateShortCut "$SMPROGRAMS\\${APP_NAME}\\Uninstall ${APP_NAME}.lnk" "$INSTDIR\\uninstall.exe"
  CreateShortCut "$DESKTOP\\${APP_NAME}.lnk" "$INSTDIR\\Asymmetry.exe" "" "$INSTDIR\\Asymmetry.exe" 0

  WriteRegStr HKLM "Software\\${APP_NAME}" "Install_Dir" "$INSTDIR"

  WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APP_NAME}" "DisplayName" "${APP_NAME}"
  WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APP_NAME}" "DisplayVersion" "${VERSION}"
  WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APP_NAME}" "Publisher" "${APP_PUBLISHER}"
  WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APP_NAME}" "InstallLocation" "$INSTDIR"
  WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APP_NAME}" "DisplayIcon" "$INSTDIR\\Asymmetry.exe"
  WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APP_NAME}" "UninstallString" '"$INSTDIR\\uninstall.exe"'

  ${GetSize} "$INSTDIR" "/S=0K" $0 $1 $2
  IntFmt $0 "0x%08X" $0
  WriteRegDWORD HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APP_NAME}" "EstimatedSize" "$0"
  WriteRegDWORD HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APP_NAME}" "NoModify" 1
  WriteRegDWORD HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APP_NAME}" "NoRepair" 1
SectionEnd

Section "Uninstall"
  Delete "$DESKTOP\\${APP_NAME}.lnk"
  Delete "$SMPROGRAMS\\${APP_NAME}\\${APP_NAME}.lnk"
  Delete "$SMPROGRAMS\\${APP_NAME}\\Uninstall ${APP_NAME}.lnk"
  RMDir "$SMPROGRAMS\\${APP_NAME}"

  DeleteRegKey HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APP_NAME}"
  DeleteRegKey HKLM "Software\\${APP_NAME}"

  RMDir /r "$INSTDIR"
SectionEnd