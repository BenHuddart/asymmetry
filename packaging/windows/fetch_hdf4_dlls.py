"""Fetch the HDF4 C runtime DLLs (conda-forge ``hdf4``) for Windows.

pyhdf's Windows PyPI wheel ships only the ``_hdfext`` extension module; it
links against external ``hdf.dll`` / ``mfhdf.dll`` (the HDF4 C library), which
the wheel does not bundle. This mirrors how Mantid sources HDF4 on Windows
(conda-forge ``hdf4`` + its dependency DLLs). The script downloads the
conda-forge ``hdf4`` package (and the transitive runtime deps it links), and
extracts every ``Library/bin/*.dll`` into a target directory so that
``os.add_dll_directory(target)`` makes ``import pyhdf`` work.

Usage:
    python packaging/windows/fetch_hdf4_dlls.py <target_dir>

Used both for local development/test runs and (in CI) to stage the DLLs the
PyInstaller Windows build bundles into the frozen app.
"""

from __future__ import annotations

import sys
import tarfile
import urllib.request
from pathlib import Path

CHANNEL = "https://conda.anaconda.org/conda-forge/win-64"

# hdf4 + the runtime libraries its DLLs link against. Pinned to known-good
# conda-forge builds; bz2 form so Python's stdlib tarfile can extract them
# (the newer .conda form needs zstd, absent from the 3.12 stdlib).
PACKAGES = [
    "hdf4-4.2.15-h1b1b6ef_5.tar.bz2",
    "libjpeg-turbo-2.1.4-hcfcfb64_0.tar.bz2",
    "zlib-1.2.13-hcfcfb64_4.tar.bz2",
]


def main(target: str) -> int:
    target_dir = Path(target).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    work = target_dir / "_download"
    work.mkdir(exist_ok=True)

    extracted: list[str] = []
    for pkg in PACKAGES:
        url = f"{CHANNEL}/{pkg}"
        archive = work / pkg
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, archive)  # noqa: S310 - trusted channel
        with tarfile.open(archive, "r:bz2") as tar:
            for member in tar.getmembers():
                name = member.name.replace("\\", "/")
                if name.startswith("Library/bin/") and name.lower().endswith(".dll"):
                    member.name = Path(name).name
                    tar.extract(member, target_dir)
                    extracted.append(member.name)

    print(f"\nextracted {len(extracted)} DLLs into {target_dir}:")
    for dll in sorted(extracted):
        print(f"  {dll}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
