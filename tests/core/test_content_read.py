"""Whole-file hashing retains descriptor and current-path race protection."""

import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from asymmetry.core.utils import content


@pytest.mark.parametrize("field", ["st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"])
def test_descriptor_change_in_each_original_stamp_field_is_rejected(tmp_path, monkeypatch, field):
    path = tmp_path / "bytes"
    path.write_bytes(b"original")
    real = os.fstat
    calls = []

    def changed(fd):
        status = real(fd)
        calls.append(fd)
        if len(calls) == 2:
            values = {
                name: getattr(status, name)
                for name in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
            }
            values[field] += 1
            return SimpleNamespace(**values)
        return status

    monkeypatch.setattr(content.os, "fstat", changed)
    with pytest.raises(content.ContentReadError, match="Source changed"):
        content.hash_file(path)


def test_in_place_content_mutation_remains_unavailable(tmp_path):
    path = tmp_path / "bytes"
    path.write_bytes(b"a" * (2 * content.FILE_HASH_CHUNK))
    initial = path.stat()

    def mutate(done, total):
        if done == content.FILE_HASH_CHUNK:
            with path.open("r+b") as target:
                target.seek(-1, 2)
                target.write(b"b")
            os.utime(path, ns=(initial.st_atime_ns, initial.st_mtime_ns + 1_000_000_000))

    with pytest.raises(content.ContentReadError, match="Source changed"):
        content.hash_file(path, progress=mutate)


@pytest.mark.skipif(os.name == "nt", reason="Windows prevents replacement of this open file")
def test_current_path_replacement_is_rejected_even_with_unchanged_read_descriptor(
    tmp_path, monkeypatch
):
    path, replacement = tmp_path / "bytes", tmp_path / "replacement"
    path.write_bytes(b"old")
    replacement.write_bytes(b"new")
    real = os.fstat
    original = []

    def stable_original(fd):
        if not original:
            original.append((fd, real(fd)))
        return original[0][1] if fd == original[0][0] else real(fd)

    def replace_path(done, total):
        if done == total:
            os.replace(replacement, path)

    monkeypatch.setattr(content.os, "fstat", stable_original)
    with pytest.raises(content.ContentReadError, match="Source changed"):
        content.hash_file(path, progress=replace_path)
    assert path.read_bytes() == b"new"


@pytest.mark.parametrize("failure", ["open", "stat"])
def test_current_path_observation_failure_closes_all_owned_handles(tmp_path, monkeypatch, failure):
    path = tmp_path / "bytes"
    path.write_bytes(b"original")
    opened = []
    real_open, real_stat = Path.open, os.fstat

    def observe(self, *args, **kwargs):
        if opened and failure == "open":
            raise FileNotFoundError("current path unavailable")
        result = real_open(self, *args, **kwargs)
        opened.append(result)
        return result

    def status(fd):
        if len(opened) == 2 and failure == "stat":
            raise OSError("current path metadata unavailable")
        return real_stat(fd)

    monkeypatch.setattr(content.Path, "open", observe)
    monkeypatch.setattr(content.os, "fstat", status)
    with pytest.raises(content.ContentReadError, match="unavailable"):
        content.hash_file(path)
    assert all(handle.closed for handle in opened)


@pytest.mark.parametrize("callback", ["checkpoint", "progress"])
@pytest.mark.parametrize("error", [OSError, InterruptedError, RuntimeError])
def test_callback_failures_are_not_converted_to_hash_io_failure(tmp_path, callback, error):
    path = tmp_path / "bytes"
    path.write_bytes(b"original")
    failure = error("caller checkpoint failed")

    def abort(*args):
        raise failure

    with pytest.raises(error) as caught:
        content.hash_file(path, **{callback: abort})
    assert caught.value is failure


def test_stable_hash_is_the_complete_byte_digest(tmp_path):
    path = tmp_path / "bytes"
    payload = b"a" * content.FILE_HASH_CHUNK + b"tail"
    path.write_bytes(payload)
    assert content.hash_file(path) == hashlib.sha256(payload).hexdigest()
