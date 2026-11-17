"""Bind native bundle inventories and final installer bytes to reviewed inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from urllib.parse import urlsplit

# Script execution with -I deliberately does not borrow the source import path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import release_environment as release  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


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
    tools = read_json(ROOT / "packaging/release/tools.json")
    gle = tools["gle"]
    url = urlsplit(gle["url"])
    if (
        url.scheme != "https"
        or url.hostname != "github.com"
        or url.username
        or url.password
        or url.query
        or url.fragment
        or not re.fullmatch(r"/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", url.path)
    ):
        raise ValueError("unsafe required immutable VCS identity")
    if observed["gle_commit"] != gle["commit"]:
        raise ValueError("build GLE identity differs from reviewed input")
    distributions = observed["environment"]["distributions"]
    environment = {
        "identity": identity,
        "distributions": {
            safe_text(name): safe_text(version)
            for name, version in sorted(observed["environment"]["distributions"].items())
        },
        "build_tool_versions": {
            name: safe_text(distributions[name])
            for name in (
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
            if name in distributions
        },
        "source_wheels": public_wheels(observed["wheels"]),
        "index_wheels": public_wheels(observed["downloaded_wheels"]),
        "direct_sources": {
            "gleplot": {key: gle[key] for key in ("url", "requested_revision", "commit", "version")}
        },
        "native_tools": {
            key: safe_text(value)
            for key, value in observed["native"].items()
            if key
            in ("ImageOS", "ImageVersion", "RUNNER_OS", "RUNNER_ARCH", "os_version", "xcode", "sdk")
        },
    }
    write_json(ROOT / "build/build-identity.json", identity)
    write_json(ROOT / "build/build-environment.json", environment)


def write_analysis(pure: list, binaries: list, datas: list) -> None:
    """Export documented Analysis identities, never the host origin column."""
    record = {
        "python_modules": sorted({safe_text(name) for name, _origin, _type in pure}),
        "native_libraries": sorted(
            {portable(name.replace("\\", "/")) for name, _origin, _type in binaries}
        ),
        "data": sorted({portable(name.replace("\\", "/")) for name, _origin, _type in datas}),
    }
    write_json(ROOT / "build/bundle-analysis.json", record)


def inspect_python_archive(executable: Path, modules: list[str]) -> None:
    # The documented recursive archive viewer observes the PYZ inside the actual EXE.
    output = release.run(["pyi-archive_viewer", "-l", "-r", str(executable)], ROOT, display=False)
    missing = [name for name in modules if repr(name) not in output]
    if missing:
        raise ValueError(f"collected Python archive is missing Analysis modules: {missing}")


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


def external_archives(bundle: Path) -> list:
    if not (ROOT / "build/hdf4-dlls/archives.json").exists():
        raise ValueError("verified external runtime record is missing")
    archives = read_json(ROOT / "build/hdf4-dlls/archives.json")
    reviewed = read_json(ROOT / "packaging/release/windows-archives.json")
    if [{key: item[key] for key in ("filename", "url", "sha256")} for item in archives] != reviewed:
        raise ValueError("external archive identities differ from reviewed inputs")
    result = []
    for item in archives:
        archive = ROOT / "build/hdf4-dlls/_download" / item["filename"]
        if release.digest(archive) != item["sha256"]:
            raise ValueError(f"verified archive bytes changed before collection: {archive.name}")
        with tarfile.open(archive, "r:bz2") as source:
            members = {
                member.name: hashlib.sha256(source.extractfile(member).read()).hexdigest()
                for member in source.getmembers()
                if member.isfile()
                and member.name.startswith("Library/bin/")
                and member.name.lower().endswith(".dll")
            }
        dlls = []
        if {dll["member"] for dll in item["dlls"]} != set(members):
            raise ValueError("staged DLL record differs from verified archive members")
        for dll in item["dlls"]:
            name = portable(dll["name"])
            staged = ROOT / "build/hdf4-dlls" / name
            included = bundle / "_internal" / name
            if (
                members[dll["member"]] != dll["sha256"]
                or release.digest(staged) != dll["sha256"]
                or release.digest(included) != dll["sha256"]
            ):
                raise ValueError(f"staged/bundled DLL identity differs: {name}")
            dlls.append(
                {
                    "member": portable(dll["member"]),
                    "path": included.relative_to(bundle).as_posix(),
                    "sha256": dll["sha256"],
                }
            )
        result.append({**{key: item[key] for key in ("filename", "url", "sha256")}, "dlls": dlls})
    return result


def collect(bundle: Path, target: str, commit: str) -> None:
    identity = expected_identity(target, commit)
    environment = read_json(ROOT / "build/build-environment.json")
    if environment["identity"] != identity:
        raise ValueError("prepared build identity is stale")
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
    analysis = read_json(ROOT / "build/bundle-analysis.json")
    executable = bundle / (
        "Asymmetry.exe" if target == "windows-x64" else "Contents/MacOS/Asymmetry"
    )
    inspect_python_archive(executable, analysis["python_modules"])
    record = {
        "schema": 1,
        **environment,
        "analysis": analysis,
        "components": inventory,
        "external_archives": external_archives(bundle) if target == "windows-x64" else [],
        "external_system_components": [
            "Operating-system loader, APIs, drivers, and system frameworks remain external; this inventory covers the collected application tree."
        ],
    }
    write_json(ROOT / f"dist/{target}-build-components.json", record)


def verify_bundle(bundle: Path, record: dict) -> None:
    if bundle_inventory(bundle) != record["components"]:
        raise ValueError("collected bundle changed after component inventory")


def finalize(binary: Path, bundle: Path, target: str, commit: str, label: str) -> None:
    identity = expected_identity(target, commit)
    component = binary.parent / f"{target}-build-components.json"
    record = read_json(component)
    if record["identity"] != identity:
        raise ValueError("component identity differs from expected checkout")
    verify_bundle(bundle, record)
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
        write_json(component, record)
    expected_name = artifact_name(target, label)
    if binary.name != expected_name:
        raise ValueError("final artifact name differs from target and label")
    final = {
        "schema": 1,
        "identity": identity,
        "artifact_label": label,
        "artifact": file_record(binary),
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
    notices = [item for item in components["components"] if item["kind"] == "notice"]
    if not notices or any(
        item["sha256"] != release.digest(ROOT / "THIRD_PARTY_LICENSES.md") for item in notices
    ):
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
