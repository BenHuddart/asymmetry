"""Replay reviewed desktop build inputs in two fresh, dedicated environments."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import tomllib
import venv
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "packaging/release"
EXTRAS = ("gui", "gle", "hdf5", "hdf4", "root")


def normalized(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def reviewed_requirements(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^ ]+) --hash=sha256:([a-f0-9]{64})", line)
        if match is None:
            raise ValueError("release requirements must be exact hashed wheel pins")
        name = normalized(match[1])
        if name in result:
            raise ValueError(f"duplicate release distribution: {name}")
        result[name] = match[2]
    if "pip" not in result:
        raise ValueError("release bootstrap pip is missing")
    return result


def observe() -> dict:
    distributions = {}
    origins = {}
    for dist in importlib.metadata.distributions():
        name = normalized(dist.metadata["Name"])
        if name in distributions:
            raise ValueError(f"duplicate installed distribution: {name}")
        distributions[name] = dist.version
        direct = dist.read_text("direct_url.json")
        if direct is not None:
            origins[name] = json.loads(direct)
    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "platform": sys.platform,
            "machine": platform.machine(),
        },
        "distributions": dict(sorted(distributions.items())),
        "origins": origins,
    }


def verify_environment(actual: dict, python: dict, expected: dict, wheel_hashes: dict) -> dict:
    if actual["python"] != python:
        raise ValueError("release Python identity differs from reviewed inputs")
    if actual["distributions"] != expected:
        raise ValueError(
            f"release distribution drift: expected {expected}, observed {actual['distributions']}"
        )
    if set(actual["origins"]) != set(wheel_hashes):
        raise ValueError("unexpected release direct origins")
    for name, sha256 in wheel_hashes.items():
        origin = actual["origins"][name]
        if origin.get("archive_info", {}).get("hashes", {}).get("sha256") != sha256:
            raise ValueError(f"release wheel identity differs: {name}")
    return {"python": python, "distributions": actual["distributions"]}


def verify_replays(first: dict, second: dict) -> None:
    if first != second:
        raise ValueError("release replay versions or VCS identity differ")


def verify_inno(output: str, version: str) -> None:
    if (
        re.search(
            r"^Compiler engine version: Inno Setup " + re.escape(version) + r"\s*$",
            output,
            re.MULTILINE,
        )
        is None
    ):
        raise ValueError("Inno Setup compiler differs from reviewed input")


def verify_selected_extras(gle: dict) -> None:
    from packaging.requirements import Requirement
    from packaging.version import Version

    installed = observe()["distributions"]
    for text in importlib.metadata.requires("asymmetry"):
        requirement = Requirement(text)
        if requirement.marker is not None and not any(
            requirement.marker.evaluate({"extra": extra}) for extra in ("", *EXTRAS)
        ):
            continue
        name = normalized(requirement.name)
        if name not in installed or Version(installed[name]) not in requirement.specifier:
            raise ValueError(f"selected desktop requirement is unsatisfied: {name}")
        if requirement.url is not None and (
            name != "gleplot" or requirement.url != f"git+{gle['url']}@{gle['requested_revision']}"
        ):
            raise ValueError(f"unreviewed desktop direct requirement: {name}")


def clean_environment() -> dict[str, str]:
    env = {
        key: value for key, value in os.environ.items() if not key.startswith(("PYTHON", "PIP_"))
    }
    env.update(
        PYTHONNOUSERSITE="1",
        PIP_CONFIG_FILE=os.devnull,
        PIP_DISABLE_PIP_VERSION_CHECK="1",
        PIP_NO_INPUT="1",
    )
    return env


def run(command: list[str], cwd: Path, *, offline: bool = False, display: bool = True) -> str:
    env = clean_environment()
    if offline:
        env["PIP_NO_INDEX"] = "1"
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if display or result.returncode:
        print(result.stdout, flush=True)
    result.check_returncode()
    return result.stdout


def environment_python(directory: Path) -> Path:
    return directory / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def build_wheel(python: Path, source: Path, output: Path) -> Path:
    output.mkdir()
    run(
        [
            str(python),
            "-I",
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--no-cache-dir",
            "--wheel-dir",
            str(output),
            str(source),
        ],
        output,
        offline=True,
    )
    wheels = list(output.glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError("source build must produce exactly one wheel")
    return wheels[0]


def verify_source(root: Path) -> str:
    if run(["git", "status", "--porcelain", "--untracked-files=all"], root).strip():
        raise ValueError("release source must be a clean committed candidate")
    tracked = set(run(["git", "ls-files", "-z", "--", "src"], root, display=False).split("\0"))
    for path in (root / "src").rglob("*"):
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts or relative.parts[1] == "asymmetry.egg-info":
            continue
        if path.is_symlink() or (path.is_file() and relative.as_posix() not in tracked):
            raise ValueError("unreviewed file in release source: " + relative.as_posix())
    return run(["git", "rev-parse", "HEAD"], root).strip()


def replay(target: str, output: Path) -> None:
    tools = json.loads((INPUTS / "tools.json").read_text(encoding="utf-8"))
    identity = {
        "implementation": tools["implementation"],
        "version": tools["python"],
        **tools["targets"][target],
    }
    if observe()["python"] != identity:
        raise ValueError("release Python identity differs from reviewed inputs")
    output = output.resolve()
    if output.is_relative_to(ROOT) or output.exists():
        raise ValueError("release replay output must be new and outside source")
    commit = verify_source(ROOT)
    output.mkdir(parents=True)
    requirements = INPUTS / f"{target}.txt"
    expected = reviewed_requirements(requirements)
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    expected.update(asymmetry=project["version"], gleplot=tools["gle"]["version"])
    replay_identities = []
    for index in (1, 2):
        work = output / f"replay-{index}"
        work.mkdir()
        envdir = work / "environment"
        venv.EnvBuilder(with_pip=True).create(envdir)
        python = environment_python(envdir)
        bootstrap = work / "bootstrap.txt"
        bootstrap.write_text(
            next(
                line
                for line in requirements.read_text(encoding="utf-8").splitlines()
                if line.startswith("pip==")
            )
            + "\n",
            encoding="utf-8",
        )
        run(
            [
                str(python),
                "-I",
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--only-binary=:all:",
                "--require-hashes",
                "--force-reinstall",
                "-r",
                str(bootstrap),
                "--report",
                str(work / "bootstrap-report.json"),
            ],
            work,
        )
        run(
            [
                str(python),
                "-I",
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--only-binary=:all:",
                "--require-hashes",
                "-r",
                str(requirements),
                "--report",
                str(work / "install-report.json"),
            ],
            work,
        )
        gle_source = work / "gle-source"
        run(["git", "init", str(gle_source)], work)
        run(
            [
                "git",
                "-C",
                str(gle_source),
                "fetch",
                "--depth=1",
                tools["gle"]["url"],
                tools["gle"]["commit"],
            ],
            work,
        )
        run(["git", "-C", str(gle_source), "checkout", "--detach", "FETCH_HEAD"], work)
        gle_commit = run(["git", "-C", str(gle_source), "rev-parse", "HEAD"], work).strip()
        if gle_commit != tools["gle"]["commit"]:
            raise ValueError("GLE source commit differs from reviewed input")
        wheels = {
            "gleplot": build_wheel(python, gle_source, work / "gle-wheel"),
            "asymmetry": build_wheel(python, ROOT, work / "app-wheel"),
        }
        hashes = {name: digest(wheel) for name, wheel in wheels.items()}
        run(
            [str(python), "-I", "-m", "pip", "install", "--no-deps", *map(str, wheels.values())],
            work,
            offline=True,
        )
        run([str(python), "-I", "-m", "pip", "check"], work, offline=True)
        actual = json.loads(
            run([str(python), "-I", str(Path(__file__).resolve()), "--observe"], work, offline=True)
        )
        observed = verify_environment(actual, identity, expected, hashes)
        run([str(python), "-I", str(Path(__file__).resolve()), "--extras"], work, offline=True)
        stable = {"environment": observed, "gle_commit": gle_commit}
        replay_identities.append(stable)
        record = {
            **stable,
            "target": target,
            "source_commit": commit,
            "wheels": {
                name: {"filename": wheel.name, "sha256": hashes[name]}
                for name, wheel in wheels.items()
            },
            "input_hashes": {
                path.name: digest(path) for path in (requirements, INPUTS / "tools.json")
            },
            "downloaded_wheels": downloaded_wheels(work),
            "native": native_identity(),
        }
        (work / "observed.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    verify_replays(*replay_identities)
    if verify_source(ROOT) != commit:
        raise ValueError("release source commit changed during replay")
    scripts = environment_python(output / "replay-1/environment").parent
    if "GITHUB_PATH" in os.environ:
        with Path(os.environ["GITHUB_PATH"]).open("a", encoding="utf-8") as stream:
            stream.write(str(scripts) + "\n")


def downloaded_wheels(work: Path) -> dict:
    wheels = {}
    for filename in ("bootstrap-report.json", "install-report.json"):
        report = json.loads((work / filename).read_text(encoding="utf-8"))
        for item in report["install"]:
            info = item["download_info"]
            origin = urlsplit(info["url"])
            if (
                origin.scheme != "https"
                or origin.hostname != "files.pythonhosted.org"
                or origin.username
                or origin.password
                or origin.query
                or origin.fragment
            ):
                raise ValueError("unreviewed package index origin")
            filename = origin.path.rsplit("/", 1)[-1]
            sha256 = info["archive_info"]["hashes"]["sha256"]
            if not re.fullmatch(r"[A-Za-z0-9_.+-]+\.whl", filename) or not re.fullmatch(
                r"[a-f0-9]{64}", sha256
            ):
                raise ValueError("invalid downloaded wheel identity")
            wheels[normalized(item["metadata"]["name"])] = {"filename": filename, "sha256": sha256}
    return dict(sorted(wheels.items()))


def native_identity() -> dict:
    result = {
        key: os.environ[key]
        for key in ("ImageOS", "ImageVersion", "RUNNER_OS", "RUNNER_ARCH")
        if key in os.environ
    }
    if sys.platform == "darwin":
        result.update(
            os_version=platform.mac_ver()[0],
            xcode=run(["xcodebuild", "-version"], ROOT).strip(),
            sdk=run(["xcrun", "--show-sdk-version"], ROOT).strip(),
        )
    elif sys.platform == "win32":
        result["os_version"] = platform.win32_ver()[1]
    return result


def verify_build(output: Path) -> None:
    tools = json.loads((INPUTS / "tools.json").read_text(encoding="utf-8"))
    record = json.loads((output / "replay-1/observed.json").read_text(encoding="utf-8"))
    if verify_source(ROOT) != record["source_commit"]:
        raise ValueError("release source commit changed after replay")
    if record["target"] not in tools["targets"]:
        raise ValueError("unreviewed release target")
    requirements = INPUTS / f"{record['target']}.txt"
    if set(record["input_hashes"]) != {requirements.name, "tools.json"}:
        raise ValueError("release input record is incomplete")
    for name, sha256 in record["input_hashes"].items():
        if digest(INPUTS / name) != sha256:
            raise ValueError(f"release inputs changed after replay: {name}")
    if record["gle_commit"] != tools["gle"]["commit"]:
        raise ValueError("release GLE commit changed after replay")
    identity = {
        "implementation": tools["implementation"],
        "version": tools["python"],
        **tools["targets"][record["target"]],
    }
    expected = reviewed_requirements(requirements)
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    expected.update(asymmetry=project["version"], gleplot=tools["gle"]["version"])
    if record["environment"] != {"python": identity, "distributions": expected}:
        raise ValueError("release observation differs from reviewed inputs")
    verify_environment(
        observe(),
        identity,
        expected,
        {name: wheel["sha256"] for name, wheel in record["wheels"].items()},
    )
    verify_selected_extras(tools["gle"])


def inno_compile(arguments: list[str]) -> None:
    tools = json.loads((INPUTS / "tools.json").read_text(encoding="utf-8"))
    compiler = Path(os.environ["ProgramFiles(x86)"]) / "Inno Setup 6/ISCC.exe"
    with tempfile.TemporaryDirectory(prefix="asymmetry-inno-") as directory:
        probe = Path(directory) / "compiler.iss"
        probe.write_text(
            "[Setup]\nAppName=Compiler probe\nAppVersion=0\nCreateAppDir=no\nUninstallable=no\nOutput=no\n",
            encoding="utf-8",
        )
        verify_inno(run([str(compiler), str(probe)], Path(directory)), tools["inno"])
    run([str(compiler), *arguments], ROOT)
    record = {
        "version": tools["inno"],
        "components": {
            name: digest(compiler.parent / name) for name in ("ISCC.exe", "ISCmplr.dll")
        },
    }
    (ROOT / "build/inno-observed.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("windows-x64", "macos-arm64"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--observe", action="store_true")
    parser.add_argument("--extras", action="store_true")
    parser.add_argument("--inno", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.verify:
        verify_build(args.verify)
    elif args.observe:
        print(json.dumps(observe()))
    elif args.extras:
        verify_selected_extras(
            json.loads((INPUTS / "tools.json").read_text(encoding="utf-8"))["gle"]
        )
    elif args.inno is not None:
        inno_compile(args.inno)
    elif args.target and args.output:
        replay(args.target, args.output)
    else:
        parser.error("select --target and --output, --observe, --extras, or --inno")


if __name__ == "__main__":
    main()
