"""The WiMDA muon-school corpus, for tests that read real instrument data.

The corpus is never committed; ``ASYMMETRY_CORPUS_ROOT`` names the extracted
release (see AGENTS.md § "Test corpus"). Tests that need it skip when the
variable is unset, which is always the case on CI. Once it is set, a missing
file is a configuration error (the wrong release, a renamed run), not a skip.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

CORPUS_ROOT = (
    Path(os.environ["ASYMMETRY_CORPUS_ROOT"]) if "ASYMMETRY_CORPUS_ROOT" in os.environ else None
)


def corpus_path(relative: str) -> Path | None:
    """*relative* under the corpus root, or None when no corpus is configured."""
    if CORPUS_ROOT is None:
        return None
    path = CORPUS_ROOT / relative
    if not path.exists():
        raise FileNotFoundError(
            f"{relative!r} is not in the corpus at ASYMMETRY_CORPUS_ROOT={CORPUS_ROOT}"
        )
    return path


def requires_corpus(path: Path | None) -> pytest.MarkDecorator:
    """Skip unless a corpus is configured (*path* is from :func:`corpus_path`)."""
    return pytest.mark.skipif(
        path is None, reason="WiMDA corpus not configured (set ASYMMETRY_CORPUS_ROOT)"
    )


def authored_grouping_copy(source: Path, destination: Path, memberships: list[int]) -> Path:
    """Author only grouping in a private copy; retain all acquired source facts.

    Membership is a test choice, not a claim about exported detector wiring.
    The new container receives its own content binding through ordinary loading.
    """
    import shutil
    from contextlib import ExitStack

    import h5py
    import numpy as np

    from asymmetry.core.io.hdf4 import _require_pyhdf, is_hdf4
    from asymmetry.core.io.icp_log import sibling_icp_log_path

    group_count = sum(group > 0 for group in set(memberships))
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / source.name
    shutil.copyfile(source, target)
    sidecar = sibling_icp_log_path(source)
    if sidecar.is_file():
        shutil.copyfile(sidecar, sibling_icp_log_path(target))
    if is_hdf4(str(target)):
        _, _, sd_type, sd_constants = _require_pyhdf()

        with ExitStack() as resources:
            handle = sd_type(str(target), sd_constants.WRITE)
            resources.callback(handle.end)
            node = handle.select("grouping")
            resources.callback(node.endaccess)
            assert node[:].shape == (len(memberships),)
            node[:] = np.asarray(memberships, dtype=np.int32)
            for name in ("available", "number_groups"):
                if name in node.attributes():
                    node.attr(name).set(sd_constants.INT32, group_count)
    else:
        with h5py.File(target, "r+") as handle:
            entry = handle["run"] if "run" in handle else handle["raw_data_1"]
            node = (
                entry["histogram_data_1/grouping"]
                if "histogram_data_1" in entry
                else entry["instrument/detector_1/grouping"]
            )
            assert node.shape == (len(memberships),)
            node[:] = memberships
            for name in ("available", "number_groups"):
                if name in node.attrs:
                    node.attrs[name] = group_count
    return target
