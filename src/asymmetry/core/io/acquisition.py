"""Bind new acquisition facts to stable decoding at built-in loader boundaries."""

from dataclasses import replace
from functools import wraps
from pathlib import Path

from asymmetry.core.data.acquisition import ContentIdentity, SourceBinding
from asymmetry.core.utils.content import ContentReadError, file_stamp


def capture_acquisition(decode):
    """Capture once per physical load, including direct supported loader calls.

    Two complete hashes bracket decoding. Matching digests and an unchanged
    inode/size/mtime/ctime across the interval establish an ordinary stable read;
    concurrent writers must not defeat filesystem change reporting. A detected
    change or hash I/O failure leaves usable decoded counts explicitly unknown.
    No raw snapshot is retained and no digest is attached to cached old counts.
    """

    @wraps(decode)
    def load(self, filepath, *, checkpoint=lambda: None, progress=lambda done, total: None):
        path = Path(filepath)
        checkpoint()
        try:
            before = file_stamp(path.stat())
        except OSError:
            before = None
        try:
            identity = ContentIdentity.read(path, checkpoint=checkpoint, progress=progress)
        except ContentReadError:
            identity = None
        result = decode(self, filepath)
        checkpoint()
        if identity is not None:
            try:
                after = ContentIdentity.read(path, checkpoint=checkpoint, progress=progress)
            except ContentReadError:
                after = None
            try:
                stable = identity == after and before == file_stamp(path.stat())
            except OSError:
                stable = False
            if not stable:
                identity = None
        binding = SourceBinding(identity, identity)

        def bind(run):
            grouping = run.grouping
            if "period_inputs" in grouping:
                grouping = dict(
                    grouping, period_inputs=tuple(bind(r) for r in grouping["period_inputs"])
                )
            return replace(run, source_binding=binding, grouping=grouping)

        datasets = result if isinstance(result, list) else [result]
        captured = [replace(ds, run=bind(ds.run)) if ds.run is not None else ds for ds in datasets]
        checkpoint()
        return captured if isinstance(result, list) else captured[0]

    return load


def restore_source_binding(dataset, saved):
    """Retain a saved expectation, including legacy unknown, alongside fresh counts."""
    expected = SourceBinding.from_dict(saved).expected

    def restore(run):
        grouping = run.grouping
        if "period_inputs" in grouping:
            grouping = dict(
                grouping, period_inputs=tuple(restore(r) for r in grouping["period_inputs"])
            )
        return replace(
            run,
            source_binding=SourceBinding(expected, run.source_binding.observed),
            grouping=grouping,
        )

    return replace(dataset, run=restore(dataset.run)) if dataset.run is not None else dataset
