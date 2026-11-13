"""Chunked whole-file SHA-256 shared by acquisition reads and workdir caches."""

import hashlib
import os
from collections.abc import Callable
from pathlib import Path

FILE_HASH_CHUNK = 1024 * 1024


class ContentReadError(OSError):
    """Acquisition I/O failed; callback exceptions are never converted to this."""


def file_stamp(status: os.stat_result) -> tuple[int, ...]:
    """Change detector spanning a read; this stamp alone never proves byte equality."""
    return (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns, status.st_ctime_ns)


def hash_file(
    path: str | Path,
    *,
    checkpoint: Callable[[], None] = lambda: None,
    progress: Callable[[int, int], None] = lambda done, total: None,
) -> str:
    """Hash all bytes, checking cancellation between chunks; isolate I/O errors.

    Concurrent writes or path replacement make this read unavailable. Timestamps
    are race detectors only; equality always requires the entire content digest.
    """
    path = Path(path)
    checkpoint()
    digest = hashlib.sha256()
    try:
        stream = path.open("rb")
    except OSError as exc:
        raise ContentReadError(str(exc)) from exc
    with stream:
        try:
            before = os.fstat(stream.fileno())
        except OSError as exc:
            raise ContentReadError(str(exc)) from exc
        done = 0
        progress(done, before.st_size)
        while True:
            checkpoint()
            try:
                chunk = stream.read(FILE_HASH_CHUNK)
            except OSError as exc:
                raise ContentReadError(str(exc)) from exc
            if not chunk:
                break
            digest.update(chunk)
            done += len(chunk)
            progress(done, before.st_size)
        checkpoint()
        try:
            changed = file_stamp(before) != file_stamp(os.fstat(stream.fileno())) or file_stamp(
                before
            ) != file_stamp(path.stat())
        except OSError as exc:
            raise ContentReadError(str(exc)) from exc
        if changed:
            raise ContentReadError("Source changed during content read")
    return digest.hexdigest()
