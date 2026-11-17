"""Verify conda-forge HDF4 runtime archives and stage their regular DLLs.

Usage: python packaging/windows/fetch_hdf4_dlls.py <target_dir>
Local development, previews, and release builds share these reviewed inputs.
Cached archives are rechecked before any extraction or runtime mutation.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath

ARCHIVE_INPUT = Path(__file__).resolve().parents[1] / "release/windows-archives.json"


def main(target: str) -> int:
    target_dir = Path(target).resolve()
    work = target_dir / "_download"
    work.mkdir(parents=True, exist_ok=True)
    packages = json.loads(ARCHIVE_INPUT.read_text(encoding="utf-8"))
    for package in packages:
        archive = work / package["filename"]
        if not archive.exists():
            print(f"downloading {package['url']}")
            urllib.request.urlretrieve(package["url"], archive)  # noqa: S310 - reviewed input
        with archive.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != package["sha256"]:
            raise ValueError(
                f"{archive.name}: expected SHA256 {package['sha256']}, actual SHA256 {actual}"
            )
        print(f"verified {archive.name}: {actual}")

    # All archive hashes and members must pass before replacing any runtime DLL.
    destinations = set()
    records = []
    with tempfile.TemporaryDirectory(prefix="asymmetry-hdf4-") as directory:
        staged = Path(directory)
        for package in packages:
            dlls = []
            with tarfile.open(work / package["filename"], "r:bz2") as archive:
                for member in archive.getmembers():
                    name = member.name.replace("\\", "/")
                    if not name.startswith("Library/bin/") or not name.lower().endswith(".dll"):
                        continue
                    parts = PurePosixPath(name).parts
                    if len(parts) != 3 or not member.isfile():
                        raise ValueError(
                            f"{package['filename']}: DLL must be a regular direct member: {name}"
                        )
                    destination = parts[-1]
                    if destination.casefold() in destinations:
                        raise ValueError(f"ambiguous DLL destination: {destination}")
                    destinations.add(destination.casefold())
                    content = archive.extractfile(member).read()
                    (staged / destination).write_bytes(content)
                    dlls.append(
                        {
                            "name": destination,
                            "member": name,
                            "sha256": hashlib.sha256(content).hexdigest(),
                        }
                    )
            records.append({**package, "dlls": sorted(dlls, key=lambda item: item["name"])})
        for path in staged.iterdir():
            destination = target_dir / path.name
            if destination.is_symlink() or (destination.exists() and not destination.is_file()):
                raise ValueError(f"DLL destination must be a regular file: {path.name}")
        for path in staged.iterdir():
            shutil.copyfile(path, target_dir / path.name)
    (target_dir / "archives.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"staged {len(destinations)} verified DLLs")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
