"""Exercise the production fetcher without network access."""

import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def fetcher(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location(
        "fetch_hdf4", ROOT / "packaging/windows/fetch_hdf4_dlls.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    payloads = {}
    records = []
    for index in range(3):
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w:bz2") as archive:
            content = f"runtime {index}".encode()
            member = tarfile.TarInfo(f"Library/bin/runtime{index}.dll")
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
            member = tarfile.TarInfo("Library/bin/ignored.txt")
            archive.addfile(member, io.BytesIO())
        name = f"archive{index}.tar.bz2"
        payloads[name] = stream.getvalue()
        records.append(
            {
                "filename": name,
                "url": f"https://example.org/{name}",
                "sha256": hashlib.sha256(payloads[name]).hexdigest(),
            }
        )
    inputs = tmp_path / "archives.json"
    inputs.write_text(json.dumps(records))
    monkeypatch.setattr(module, "ARCHIVE_INPUT", inputs, raising=False)
    monkeypatch.setattr(module, "PACKAGES", list(payloads), raising=False)

    def download(url, destination):
        Path(destination).write_bytes(payloads[url.rsplit("/", 1)[-1]])

    monkeypatch.setattr(module.urllib.request, "urlretrieve", download)
    return module, payloads, records


def test_fetcher_records_verified_regular_dlls(fetcher, tmp_path):
    module, _, records = fetcher
    target = tmp_path / "runtime"
    assert module.main(str(target)) == 0
    assert sorted(path.name for path in target.glob("*.dll")) == [
        "runtime0.dll",
        "runtime1.dll",
        "runtime2.dll",
    ]
    observed = json.loads((target / "archives.json").read_text())
    assert [item["sha256"] for item in observed] == [item["sha256"] for item in records]
    assert observed[0]["dlls"][0]["sha256"] == hashlib.sha256(b"runtime 0").hexdigest()
    assert not (target / "ignored.txt").exists()


@pytest.mark.parametrize("index", [0, 2])
def test_all_hashes_checked_before_extraction_or_target_mutation(
    fetcher, tmp_path, monkeypatch, index
):
    module, payloads, records = fetcher
    payloads[records[index]["filename"]] += b"tampered"
    target = tmp_path / "runtime"
    target.mkdir()
    (target / "old.dll").write_bytes(b"existing runtime")

    def forbidden(*args, **kwargs):
        pytest.fail("opened archive before all hashes verified")

    monkeypatch.setattr(module.tarfile, "open", forbidden)
    with pytest.raises(ValueError, match=records[index]["filename"] + ".*expected.*actual"):
        module.main(str(target))
    assert {p.name: p.read_bytes() for p in target.glob("*.dll")} == {
        "old.dll": b"existing runtime"
    }


def test_cached_archive_tamper_is_rejected(fetcher, tmp_path, monkeypatch):
    module, _, records = fetcher
    target = tmp_path / "runtime"
    module.main(str(target))
    before = {p.name: p.read_bytes() for p in target.glob("*.dll")}
    (target / "_download" / records[1]["filename"]).write_bytes(b"cache tamper")

    def forbidden(*args):
        pytest.fail("cached archive must be verified, not replaced")

    monkeypatch.setattr(module.urllib.request, "urlretrieve", forbidden)
    with pytest.raises(ValueError, match="archive1.*expected.*actual"):
        module.main(str(target))
    assert {p.name: p.read_bytes() for p in target.glob("*.dll")} == before


@pytest.mark.parametrize("kind", ["symlink", "duplicate", "nested"])
def test_fetcher_rejects_ambiguous_or_nonregular_dlls_before_mutation(fetcher, tmp_path, kind):
    module, payloads, records = fetcher
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:bz2") as archive:
        name = "Library/bin/sub/nested.dll" if kind == "nested" else "Library/bin/runtime0.dll"
        member = tarfile.TarInfo(name)
        if kind == "symlink":
            member.type = tarfile.SYMTYPE
            member.linkname = "elsewhere"
        archive.addfile(member, io.BytesIO())
    payloads[records[2]["filename"]] = stream.getvalue()
    records[2]["sha256"] = hashlib.sha256(stream.getvalue()).hexdigest()
    module.ARCHIVE_INPUT.write_text(json.dumps(records))
    target = tmp_path / "runtime"
    target.mkdir()
    (target / "old.dll").write_bytes(b"unchanged")
    with pytest.raises(ValueError, match="regular direct member|ambiguous DLL destination"):
        module.main(str(target))
    assert list(target.glob("*.dll")) == [target / "old.dll"]
    assert (target / "old.dll").read_bytes() == b"unchanged"


def test_verified_archive_without_dlls_is_recorded_honestly(fetcher, tmp_path):
    module, payloads, records = fetcher
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:bz2"):
        pass
    payloads[records[2]["filename"]] = stream.getvalue()
    records[2]["sha256"] = hashlib.sha256(stream.getvalue()).hexdigest()
    module.ARCHIVE_INPUT.write_text(json.dumps(records))
    target = tmp_path / "runtime"
    module.main(str(target))
    assert json.loads((target / "archives.json").read_text())[2]["dlls"] == []
    assert len(list(target.glob("*.dll"))) == 2


@pytest.mark.parametrize("tamper", [None, "staged", "bundled", "archive", "self_blessed_record"])
def test_actual_archive_to_bundle_lineage(fetcher, tmp_path, monkeypatch, tamper):
    import shutil

    from tools import release_artifacts as artifacts

    module, _, _ = fetcher
    monkeypatch.setattr(artifacts, "ROOT", tmp_path)
    reviewed = tmp_path / "packaging/release/windows-archives.json"
    reviewed.parent.mkdir(parents=True)
    reviewed.write_bytes(module.ARCHIVE_INPUT.read_bytes())
    staged = tmp_path / "build/hdf4-dlls"
    module.main(str(staged))
    bundle = tmp_path / "bundle"
    (bundle / "_internal").mkdir(parents=True)
    for dll in staged.glob("*.dll"):
        shutil.copyfile(dll, bundle / "_internal" / dll.name)
    if tamper == "staged":
        (staged / "runtime0.dll").write_bytes(b"changed")
    elif tamper == "bundled":
        (bundle / "_internal/runtime0.dll").write_bytes(b"changed")
    elif tamper == "archive":
        (staged / "_download/archive0.tar.bz2").write_bytes(b"changed")
    elif tamper == "self_blessed_record":
        record = json.loads((staged / "archives.json").read_text())
        record[0]["dlls"][0]["sha256"] = hashlib.sha256(b"changed").hexdigest()
        (staged / "archives.json").write_text(json.dumps(record))
        (staged / "runtime0.dll").write_bytes(b"changed")
        (bundle / "_internal/runtime0.dll").write_bytes(b"changed")
    if tamper:
        with pytest.raises(ValueError, match="identity differs|archive bytes changed"):
            artifacts.external_archives(bundle)
    else:
        records = artifacts.external_archives(bundle)
        assert records[0]["dlls"][0]["path"] == "_internal/runtime0.dll"
        assert records[0]["dlls"][0]["sha256"] == hashlib.sha256(b"runtime 0").hexdigest()


def test_packaging_selects_only_verified_dlls_and_preserves_local_files(
    fetcher, tmp_path, monkeypatch
):
    from tools import release_artifacts as artifacts

    module, _, _ = fetcher
    monkeypatch.setattr(artifacts, "ROOT", tmp_path)
    reviewed = tmp_path / "packaging/release/windows-archives.json"
    reviewed.parent.mkdir(parents=True)
    reviewed.write_bytes(module.ARCHIVE_INPUT.read_bytes())
    staged = tmp_path / "runtime"
    module.main(str(staged))
    (staged / "old.dll").write_bytes(b"unrelated local runtime")
    selected = artifacts.staged_binaries(staged)
    assert sorted(Path(source).name for source, _ in selected) == [
        "runtime0.dll",
        "runtime1.dll",
        "runtime2.dll",
    ]
    assert (staged / "old.dll").read_bytes() == b"unrelated local runtime"


def test_packaging_trace_verifies_all_archives_before_member_reads(fetcher, tmp_path, monkeypatch):
    from tools import release_artifacts as artifacts

    module, _, records = fetcher
    monkeypatch.setattr(artifacts, "ROOT", tmp_path)
    reviewed = tmp_path / "packaging/release/windows-archives.json"
    reviewed.parent.mkdir(parents=True)
    reviewed.write_bytes(module.ARCHIVE_INPUT.read_bytes())
    staged = tmp_path / "runtime"
    module.main(str(staged))
    (staged / "_download" / records[2]["filename"]).write_bytes(b"later cache tamper")

    def forbidden(*args, **kwargs):
        pytest.fail("opened an archive before checking all hashes")

    monkeypatch.setattr(artifacts.tarfile, "open", forbidden)
    with pytest.raises(ValueError, match="archive bytes changed"):
        artifacts.staged_binaries(staged)
