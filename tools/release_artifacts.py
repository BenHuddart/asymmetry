"""Bind native bundle inventories and final installer bytes to reviewed inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import plistlib
import re
import shutil
import sys
import tarfile
import tempfile
import time
import tomllib
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath

# Script execution with -I deliberately does not borrow the source import path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import release_environment as release  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BUILD_TOOLS = (
    "pip",
    "setuptools",
    "setuptools-scm",
    "wheel",
    "packaging",
    "pyinstaller",
    "pyinstaller-hooks-contrib",
    "pillow",
    "dmgbuild",
)
NATIVE_FIELDS = (
    "ImageOS",
    "ImageVersion",
    "RUNNER_OS",
    "RUNNER_ARCH",
    "os_version",
    "xcode",
    "sdk",
)
PYTHON_CONTAINERS = ("PYZ", "CArchive_modules", "CArchive_scripts")
SYSTEM_COMPONENTS = [
    "Operating-system loader, APIs, drivers, and system frameworks remain external; this inventory covers the collected application tree."
]


def write_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def portable(name: str) -> str:
    if (
        not name
        or "\\" in name
        or PurePosixPath(name).is_absolute()
        or PureWindowsPath(name).drive
        or any(part in ("", ".", "..") for part in name.split("/"))
        or any(ord(char) < 32 for char in name)
    ):
        raise ValueError("public bundle identity must be a portable relative path")
    return name


def safe_text(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9 ._+()\n-]+", value):
        raise ValueError("public observation contains an unsafe value")
    return value


def file_record(path: Path) -> dict:
    return {
        "filename": portable(path.name),
        "size": path.stat().st_size,
        "sha256": release.digest(path),
    }


def expected_identity(target: str, commit: str) -> dict:
    actual = release.verify_source(ROOT)
    if not re.fullmatch(r"[a-f0-9]{40}", commit) or actual != commit:
        raise ValueError("expected source identity differs from triggering checkout")
    inputs = ROOT / "packaging/release"
    tools = read_json(inputs / "tools.json")
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    return {
        "source_commit": commit,
        "version": project["version"],
        "target": target,
        "python": {
            "implementation": tools["implementation"],
            "version": tools["python"],
            **tools["targets"][target],
        },
        "input_hashes": {
            name: release.digest(inputs / name)
            for name in (f"{target}.txt", "tools.json", "windows-archives.json")
        },
    }


def public_digest(value: str) -> str:
    if not re.fullmatch(r"[a-f0-9]{64}", value):
        raise ValueError("invalid public SHA256 identity")
    return value


def public_wheels(wheels: dict) -> dict:
    result = {}
    for name, item in sorted(wheels.items()):
        filename = portable(item["filename"])
        if "/" in filename:
            raise ValueError("invalid public wheel identity")
        result[safe_text(name)] = {"filename": filename, "sha256": public_digest(item["sha256"])}
    return result


def prepare(replay: Path, target: str, commit: str) -> None:
    release.verify_build(replay)
    identity = expected_identity(target, commit)
    observed = read_json(replay / "replay-1/observed.json")
    if any(observed[key] != identity[key] for key in ("source_commit", "target", "input_hashes")):
        raise ValueError("build observation identity differs from checkout")
    reviewed = release.reviewed_wheels(ROOT / f"packaging/release/{target}.txt")
    index_wheels = public_wheels(observed["downloaded_wheels"])
    if {name: item["sha256"] for name, item in index_wheels.items()} != {
        name: item["sha256"] for name, item in reviewed.items()
    }:
        raise ValueError("index wheel observations differ from reviewed inputs")
    if index_wheels != release.downloaded_wheels(replay / "replay-1"):
        raise ValueError("index wheel observations differ from allowlisted download reports")
    if observed["native"] != release.native_identity():
        raise ValueError("native tool observations changed after replay")
    wheels = list((replay / "replay-1/app-wheel").glob("*.whl"))
    if len(wheels) != 1 or {
        key: observed["wheels"]["asymmetry"][key] for key in ("filename", "sha256")
    } != {
        "filename": wheels[0].name,
        "sha256": release.digest(wheels[0]),
    }:
        raise ValueError("source wheel observation differs from replay bytes: asymmetry")
    environment = {
        "identity": identity,
        "distributions": observed["environment"]["distributions"],
        "source_wheels": observed["wheels"],
        "index_wheels": index_wheels,
        "native_tools": observed["native"],
    }
    write_json(ROOT / "build/build-identity.json", identity)
    write_json(ROOT / "build/build-environment.json", public_environment(environment, identity))


def public_environment(record: dict, identity: dict) -> dict:
    if record.get("identity") != identity:
        raise ValueError("public build identity differs from expected checkout")
    index_wheels = public_wheels(record["index_wheels"])
    reviewed = release.reviewed_wheels(ROOT / f"packaging/release/{identity['target']}.txt")
    if {name: item["sha256"] for name, item in index_wheels.items()} != {
        name: item["sha256"] for name, item in reviewed.items()
    }:
        raise ValueError("public index wheel identity differs from reviewed inputs")
    source_wheels = public_wheels(record["source_wheels"])
    if set(source_wheels) != {"asymmetry"}:
        raise ValueError("unreviewed source wheel identity")
    distributions = {
        safe_text(name): safe_text(version)
        for name, version in sorted(record["distributions"].items())
    }
    return {
        "identity": identity,
        "distributions": distributions,
        "build_tool_distributions": [name for name in BUILD_TOOLS if name in distributions],
        "source_wheels": source_wheels,
        "index_wheels": index_wheels,
        "native_tools": {
            key: safe_text(value)
            for key, value in record["native_tools"].items()
            if key in NATIVE_FIELDS
        },
    }


def public_collection_names(record: dict) -> dict:
    return {
        key: sorted({portable(name) for name in record[key]})
        for key in ("native_libraries", "data")
    }


def public_analysis(record: dict) -> dict:
    return {
        **public_collection_names(record),
        "python_modules": sorted({safe_text(name) for name in record["python_modules"]}),
    }


def public_collected_analysis(record: dict) -> dict:
    return {
        **public_collection_names(record),
        "python_archives": {
            kind: sorted({safe_text(name) for name in record["python_archives"][kind]})
            for kind in PYTHON_CONTAINERS
        },
    }


def public_components(record: dict, identity: dict) -> dict:
    if record["schema"] != 1:
        raise ValueError("unsupported public component record")
    pins = (
        read_json(ROOT / "packaging/release/windows-archives.json")
        if identity["target"] == "windows-x64"
        else []
    )
    archives = record["external_archives"]
    if [
        {key: archive[key] for key in ("filename", "url", "sha256")} for archive in archives
    ] != pins:
        raise ValueError("public archive identities differ from reviewed inputs")
    return {
        "schema": 1,
        **public_environment(record, identity),
        "analysis": public_collected_analysis(record["analysis"]),
        "components": record["components"],
        "external_archives": [
            {
                **pin,
                "dlls": [
                    {
                        "member": portable(dll["member"]),
                        "path": portable(dll["path"]),
                        "sha256": public_digest(dll["sha256"]),
                    }
                    for dll in archive["dlls"]
                ],
            }
            for pin, archive in zip(pins, archives, strict=True)
        ],
        "external_system_components": SYSTEM_COMPONENTS,
    }


def write_analysis(pure: list, binaries: list, datas: list) -> None:
    """Export documented Analysis identities, never the host origin column."""
    record = {
        "python_modules": sorted({safe_text(name) for name, _origin, _type in pure}),
        "native_libraries": sorted(
            {portable(name.replace("\\", "/")) for name, _origin, _type in binaries}
        ),
        "data": sorted({portable(name.replace("\\", "/")) for name, _origin, _type in datas}),
    }
    write_json(ROOT / "build/bundle-analysis.json", public_analysis(record))


def inspect_python_archive(executable: Path, modules: list[str]) -> dict:
    output = release.run(["pyi-archive_viewer", "-l", "-r", str(executable)], ROOT, display=False)
    observed = {kind: set() for kind in PYTHON_CONTAINERS}
    section = ""
    for line in output.splitlines():
        if line.startswith("Contents of "):
            section = (
                "PYZ"
                if line.endswith("(PYZ):")
                else "CArchive"
                if line.endswith("(PKG/CArchive):")
                else ""
            )
        elif section == "PYZ":
            match = re.fullmatch(r"\s*\d+,\s*\d+,\s*\d+, '([^']+)'", line)
            if match:
                observed["PYZ"].add(safe_text(match[1]))
        elif section == "CArchive":
            match = re.fullmatch(r"\s*\d+,\s*\d+,\s*\d+,\s*[01], '([mMs])', '([^']+)'", line)
            if match:
                kind = "CArchive_scripts" if match[1] == "s" else "CArchive_modules"
                observed[kind].add(safe_text(match[2]))
    missing = set(modules) - observed["PYZ"] - observed["CArchive_modules"]
    if missing:
        raise ValueError(
            f"collected Python archives are missing Analysis modules: {sorted(missing)}"
        )
    return {kind: sorted(names) for kind, names in observed.items()}


def bundle_inventory(bundle: Path) -> list[dict]:
    result = []
    for path in sorted(bundle.rglob("*")):
        name = portable(path.relative_to(bundle).as_posix())
        if path.is_symlink():
            link = os.readlink(path)
            if (
                PurePosixPath(link).is_absolute()
                or PureWindowsPath(link).drive
                or "\\" in link
                or not path.resolve(strict=True).is_relative_to(bundle.resolve())
            ):
                raise ValueError(f"bundle symlink escapes collected tree: {name}")
            result.append({"path": name, "kind": "symlink", "link": link})
        elif path.is_file():
            with path.open("rb") as stream:
                magic = stream.read(4)
            lower = path.name.lower()
            kind = "resource"
            if lower == "third_party_licenses.md":
                kind = "notice"
            elif lower in ("asymmetry", "asymmetry.exe"):
                kind = "executable"
            elif (
                lower.endswith((".dll", ".dylib", ".so", ".pyd"))
                or ".so." in lower
                or magic
                in (
                    b"\xcf\xfa\xed\xfe",
                    b"\xfe\xed\xfa\xcf",
                    b"\xca\xfe\xba\xbe",
                    b"\xbe\xba\xfe\xca",
                )
            ):
                kind = "native_library"
            elif lower.endswith((".zip", ".pyz")):
                kind = "python_archive"
            elif lower.endswith((".json", ".plist")) or ".dist-info/" in name:
                kind = "metadata"
            item = {
                "path": name,
                "kind": kind,
                "size": path.stat().st_size,
                "sha256": release.digest(path),
            }
            if lower == "base_library.zip":
                with zipfile.ZipFile(path) as archive:
                    item["members"] = [
                        {
                            "path": portable(member),
                            "sha256": hashlib.sha256(archive.read(member)).hexdigest(),
                        }
                        for member in sorted(archive.namelist())
                        if not member.endswith("/")
                    ]
            result.append(item)
        elif not path.is_dir():
            raise ValueError(f"unsupported collected bundle member: {name}")
    return result


def verified_staged_archives(directory: Path) -> list:
    archives = read_json(directory / "archives.json")
    reviewed = read_json(ROOT / "packaging/release/windows-archives.json")
    if [{key: item[key] for key in ("filename", "url", "sha256")} for item in archives] != reviewed:
        raise ValueError("external archive identities differ from reviewed inputs")
    for pin in reviewed:
        archive = directory / "_download" / pin["filename"]
        if release.digest(archive) != pin["sha256"]:
            raise ValueError(f"verified archive bytes changed before collection: {archive.name}")
    result = []
    for item, pin in zip(archives, reviewed, strict=True):
        archive = directory / "_download" / pin["filename"]
        dlls = []
        with tarfile.open(archive, "r:bz2") as source:
            for member in source.getmembers():
                if not member.name.startswith("Library/bin/") or not member.name.lower().endswith(
                    ".dll"
                ):
                    continue
                if not member.isfile() or len(PurePosixPath(member.name).parts) != 3:
                    raise ValueError("verified DLL must be a regular direct archive member")
                name = portable(PurePosixPath(member.name).name)
                sha256 = hashlib.sha256(source.extractfile(member).read()).hexdigest()
                if release.digest(directory / name) != sha256:
                    raise ValueError(f"staged DLL identity differs from verified archive: {name}")
                dlls.append({"member": member.name, "name": name, "sha256": sha256})
        dlls.sort(key=lambda dll: dll["name"])
        if item["dlls"] != dlls:
            raise ValueError("staged DLL record identity differs from verified archive members")
        result.append({**pin, "dlls": dlls})
    return result


def staged_binaries(directory: Path) -> list[tuple[str, str]]:
    """Select intended DLLs without sweeping unrelated local files into a bundle."""
    return [
        (str(directory / dll["name"]), ".")
        for archive in verified_staged_archives(directory)
        for dll in archive["dlls"]
    ]


def external_archives(bundle: Path) -> list:
    records = verified_staged_archives(ROOT / "build/hdf4-dlls")
    for archive in records:
        for dll in archive["dlls"]:
            included = bundle / "_internal" / dll.pop("name")
            if release.digest(included) != dll["sha256"]:
                raise ValueError(
                    f"bundled DLL identity differs from verified archive: {included.name}"
                )
            dll["path"] = included.relative_to(bundle).as_posix()
    return records


def collect(bundle: Path, target: str, commit: str) -> None:
    identity = expected_identity(target, commit)
    environment = public_environment(read_json(ROOT / "build/build-environment.json"), identity)
    inventory = bundle_inventory(bundle)
    embedded = [
        item
        for item in inventory
        if item["path"].endswith("/build-identity.json") and item["kind"] != "symlink"
    ]
    notices = [item for item in inventory if item["kind"] == "notice"]
    if len(embedded) != 1 or read_json(bundle / embedded[0]["path"]) != identity:
        raise ValueError("packaged build identity is missing or stale")
    if not notices or any(
        item["sha256"] != release.digest(ROOT / "THIRD_PARTY_LICENSES.md") for item in notices
    ):
        raise ValueError("bundled third-party notices are missing or changed")
    analysis = public_analysis(read_json(ROOT / "build/bundle-analysis.json"))
    executable = bundle / (
        "Asymmetry.exe" if target == "windows-x64" else "Contents/MacOS/Asymmetry"
    )
    containers = inspect_python_archive(executable, analysis["python_modules"])
    analysis = {**public_collection_names(analysis), "python_archives": containers}
    record = {
        "schema": 1,
        **environment,
        "analysis": analysis,
        "components": inventory,
        "external_archives": external_archives(bundle) if target == "windows-x64" else [],
        "external_system_components": SYSTEM_COMPONENTS,
    }
    write_json(ROOT / f"dist/{target}-build-components.json", public_components(record, identity))


def verify_bundle(bundle: Path, record: dict) -> None:
    if bundle_inventory(bundle) != record["components"]:
        raise ValueError("collected bundle changed after component inventory")


def native_stage(name, command, work, timeout, events):
    """Retain measured native command outcomes separately from the fixed public manifest."""
    started = time.monotonic()
    event = {"stage": name, "timeout_seconds": timeout, "success": False}
    try:
        output = release.run(command, work, display=False, timeout=timeout)
        event["success"] = True
        return output
    except Exception as exc:
        event["failure_type"] = type(exc).__name__
        event["cleanup"] = list(exc.__notes__) if hasattr(exc, "__notes__") else []
        raise
    finally:
        event["elapsed_seconds"] = time.monotonic() - started
        events.append(event)


def attached_image_device(binary, work, events):
    """Identify the whole device from the actual image, including unmounted attachments."""
    output = native_stage("image attachments", ["hdiutil", "info", "-plist"], work, 30, events)
    images = plistlib.loads(output.encode())["images"]
    matches = [image for image in images if Path(image["image-path"]).resolve() == binary.resolve()]
    if not matches:
        events.append({"stage": "image ownership", "success": True, "attached": False})
        return None
    if len(matches) != 1:
        raise ValueError("owned image has ambiguous attached devices")
    # hdiutil lists the backing device before its partitions or synthesized volumes.
    device = matches[0]["system-entities"][0]["dev-entry"]
    if not re.fullmatch(r"/dev/disk[0-9]+", device):
        raise ValueError("owned image has no identifiable backing device")
    events.append(
        {"stage": "image ownership", "success": True, "attached": True, "device": Path(device).name}
    )
    return device


def detach_image(binary, mount, work, events):
    """Retain query failures while still attempting cleanup of the owned mount."""
    errors = []
    device = None
    try:
        device = attached_image_device(binary, work, events)
    except Exception as exc:
        errors.append(exc)
    try:
        if device is not None or mount.is_mount():
            native_stage("detach", ["hdiutil", "detach", device or str(mount)], work, 60, events)
    except Exception as exc:
        errors.append(exc)
    try:
        if attached_image_device(binary, work, events) is not None or mount.is_mount():
            raise ValueError("temporary DMG remains attached or mounted after detach")
    except Exception as exc:
        errors.append(exc)
    if len(errors) == 1:
        raise errors[0]
    if errors:
        raise ExceptionGroup("owned image cleanup failures", errors)


def inspect_payload(binary, expected, target, required_resources, visitor=None, *, events=None):
    """Inspect and visit the final installed/mounted app within one checked lifetime."""
    if target not in ("windows-x64", "macos-arm64"):
        raise ValueError("unsupported native payload target")
    events = [] if events is None else events
    if target == "macos-arm64" and attached_image_device(binary, binary.parent, events) is not None:
        raise ValueError("refusing to reuse a pre-existing image attachment")
    work = Path(tempfile.mkdtemp(prefix="asymmetry-payload-"))
    application = work / ("application" if target == "windows-x64" else "volume/Asymmetry.app")
    mount = work / "volume"
    uninstaller = application / "unins000.exe"
    errors = []
    try:
        if target == "windows-x64":
            native_stage(
                "install",
                [
                    str(binary.resolve()),
                    "/SP-",
                    "/VERYSILENT",
                    "/SUPPRESSMSGBOXES",
                    "/NORESTART",
                    "/NOICONS",
                    "/TASKS=",
                    "/NOCLOSEAPPLICATIONS",
                    "/NORESTARTAPPLICATIONS",
                    f"/DIR={application}",
                ],
                work,
                120,
                events,
            )
            if not uninstaller.is_file():
                raise ValueError("final installer payload has no cleanup executable")
            inventory = bundle_inventory(application)
            generated = {"unins000.exe", "unins000.dat", "unins000.msg"}
            payload = [
                item for item in inventory if item["path"] not in generated | {"asymmetry.ico"}
            ]
            if payload != expected:
                raise ValueError("final installer payload differs from inventoried bundle")
            icon = application / "asymmetry.ico"
            result = {
                "method": "silent Inno payload inspection",
                "additional_resources": [
                    {
                        "path": "asymmetry.ico",
                        "sha256": release.digest(icon),
                        "size": icon.stat().st_size,
                    }
                ],
                "generated_installer_files": sorted(
                    item["path"] for item in inventory if item["path"] in generated
                ),
            }
        else:
            mount.mkdir()
            native_stage(
                "attach",
                [
                    "hdiutil",
                    "attach",
                    "-readonly",
                    "-nobrowse",
                    "-mountpoint",
                    str(mount),
                    str(binary.resolve()),
                ],
                work,
                60,
                events,
            )
            if bundle_inventory(application) != expected:
                raise ValueError("final DMG payload differs from inventoried bundle")
            applications = mount / "Applications"
            if not applications.is_symlink() or os.readlink(applications) != "/Applications":
                raise ValueError("final DMG Applications system link differs")
            extras = []
            for path in sorted(mount.rglob("*")):
                relative = path.relative_to(mount)
                if relative.parts[0] == "Asymmetry.app" or path == applications:
                    continue
                if path.is_symlink():
                    raise ValueError("unsupported final DMG assembly symlink")
                if path.is_dir():
                    continue
                if not path.is_file():
                    raise ValueError("unsupported final DMG assembly resource")
                extras.append(
                    {
                        "path": portable(relative.as_posix()),
                        "sha256": release.digest(path),
                        "size": path.stat().st_size,
                    }
                )
            result = {
                "method": "read-only mounted DMG payload inspection",
                "additional_resources": extras,
                "external_system_link": {
                    "path": "Applications",
                    "system_location": "Applications directory",
                },
            }
        for digest in required_resources:
            if sum(item["sha256"] == digest for item in result["additional_resources"]) != 1:
                raise ValueError("final assembly resource differs from required input")
        if visitor is not None:
            visitor(application, work)
    except Exception as exc:
        errors.append(exc)
    finally:
        try:
            if target == "windows-x64" and uninstaller.is_file():
                native_stage(
                    "uninstall",
                    [str(uninstaller), "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART"],
                    work,
                    120,
                    events,
                )
                deadline = time.monotonic() + 30
                while application.exists() and time.monotonic() < deadline:
                    time.sleep(0.1)
                if application.exists():
                    raise ValueError("temporary installed payload remains after uninstall")
            elif target == "windows-x64" and application.exists():
                raise ValueError("partial installation has no cleanup executable")
            elif target == "macos-arm64":
                detach_image(binary, mount, work, events)
            events.append({"stage": "native cleanup", "success": True})
        except Exception as exc:
            events.append(
                {"stage": "native cleanup", "success": False, "failure_type": type(exc).__name__}
            )
            errors.append(exc)
        try:
            if target == "macos-arm64" and mount.is_mount():
                raise OSError("temporary cleanup refused while owned DMG remains mounted")
            shutil.rmtree(work)
            events.append({"stage": "temporary cleanup", "success": True})
        except OSError as exc:
            events.append(
                {"stage": "temporary cleanup", "success": False, "failure_type": type(exc).__name__}
            )
            errors.append(exc)
    if len(errors) == 1:
        raise errors[0]
    if errors:
        raise ExceptionGroup("native payload and cleanup failures", errors)
    return result


def inspect_assembled_artifact(binary: Path, bundle: Path, target: str) -> dict:
    resource = ROOT / (
        "build/icons/Asymmetry.ico" if target == "windows-x64" else "build/dmg-bg.png"
    )
    return inspect_payload(binary, bundle_inventory(bundle), target, {release.digest(resource)})


def finalize(binary: Path, bundle: Path, target: str, commit: str, label: str) -> None:
    identity = expected_identity(target, commit)
    component = binary.parent / f"{target}-build-components.json"
    record = read_json(component)
    if record["identity"] != identity:
        raise ValueError("component identity differs from expected checkout")
    verify_bundle(bundle, record)
    record = public_components(record, identity)
    if target == "windows-x64":
        compiler = read_json(ROOT / "build/inno-observed.json")
        tools = read_json(ROOT / "packaging/release/tools.json")
        if compiler["version"] != tools["inno"]:
            raise ValueError("installer compiler identity differs from reviewed inputs")
        record["installer"] = {
            "version": safe_text(compiler["version"]),
            "components": {
                name: public_digest(compiler["components"][name])
                for name in ("ISCC.exe", "ISCmplr.dll")
            },
        }
    expected_name = artifact_name(target, label)
    if binary.name != expected_name:
        raise ValueError("final artifact name differs from target and label")
    artifact = file_record(binary)
    record["assembly"] = inspect_assembled_artifact(binary, bundle, target)
    if file_record(binary) != artifact:
        raise ValueError("final artifact changed during payload inspection")
    print(
        f"Verified final {target} payload against {len(record['components'])} collected components."
    )
    write_json(component, record)
    final = {
        "schema": 1,
        "identity": identity,
        "artifact_label": label,
        "artifact": artifact,
        "build_components": file_record(component),
    }
    write_json(binary.parent / f"{target}-artifact.json", final)
    (binary.parent / f"{target}-SHA256SUMS").write_text(
        checksum_text(final), encoding="utf-8", newline="\n"
    )


def artifact_name(target: str, label: str) -> str:
    safe_text(label)
    return f"Asymmetry-{label}-" + (
        "windows-x64-setup.exe" if target == "windows-x64" else "macos-arm64.dmg"
    )


def checksum_text(final: dict) -> str:
    return "".join(
        f"{final[key]['sha256']}  {final[key]['filename']}\n"
        for key in ("artifact", "build_components")
    )


def verify_download(directory: Path, target: str, commit: str) -> list[Path]:
    identity = expected_identity(target, commit)
    return verify_downloaded(directory, identity, release.digest(ROOT / "THIRD_PARTY_LICENSES.md"))


def verify_downloaded(directory: Path, identity: dict, notice_digest: str) -> list[Path]:
    """Verify the same downloaded-byte contract using an already source-bound expectation."""
    target, commit = identity["target"], identity["source_commit"]
    manifest = directory / f"{target}-artifact.json"
    final = read_json(manifest)
    if (
        final.get("schema") != 1
        or final.get("identity") != identity
        or final["artifact_label"] != identity["version"]
    ):
        raise ValueError("downloaded artifact identity differs from triggering checkout")
    expected_names = {
        "artifact": artifact_name(target, identity["version"]),
        "build_components": f"{target}-build-components.json",
    }
    selected = [manifest]
    for key, name in expected_names.items():
        path = directory / name
        if final[key] != file_record(path):
            raise ValueError(f"downloaded {key} bytes or identity differ")
        selected.append(path)
    components = read_json(directory / expected_names["build_components"])
    if components.get("schema") != 1 or components.get("identity") != identity:
        raise ValueError("downloaded component identity differs from triggering checkout")
    method = (
        "silent Inno payload inspection"
        if target == "windows-x64"
        else "read-only mounted DMG payload inspection"
    )
    if components.get("assembly", {}).get("method") != method:
        raise ValueError("downloaded final payload inspection record is missing or swapped")
    notices = [item for item in components["components"] if item["kind"] == "notice"]
    if not notices or any(item["sha256"] != notice_digest for item in notices):
        raise ValueError("downloaded notice identity differs from checkout")
    checksum = directory / f"{target}-SHA256SUMS"
    if checksum.read_text(encoding="utf-8") != checksum_text(final):
        raise ValueError("downloaded checksum text differs from authoritative record")
    selected.append(checksum)
    print(f"Verified downloaded {target} artifact, component record, and checksum at {commit}.")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "collect", "finalize", "verify"))
    parser.add_argument("--target", required=True, choices=("windows-x64", "macos-arm64"))
    parser.add_argument("--commit", required=True)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--label")
    parser.add_argument("--directory", type=Path)
    args = parser.parse_args()
    if args.operation == "prepare":
        prepare(args.replay, args.target, args.commit)
    elif args.operation == "collect":
        collect(args.bundle, args.target, args.commit)
    elif args.operation == "finalize":
        finalize(args.artifact, args.bundle, args.target, args.commit, args.label)
    else:
        verify_download(args.directory, args.target, args.commit)


if __name__ == "__main__":
    main()
