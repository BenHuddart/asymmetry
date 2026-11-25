"""Bind Python release files and clean-install evidence to one source commit."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import tarfile
import zipfile
from email.parser import BytesParser
from email.policy import default
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from distribution_smoke import INSTALL_PROFILES, artifact_contract
from release_environment import verify_source

PYTHONS = ("3.11", "3.12", "3.13")
RESOLUTIONS = ("constrained", "unconstrained")


def verify_publisher_environment(environment: dict, policies: dict) -> None:
    """Require a maintainer review and release-tag-only deployment authority."""
    if environment["name"] != "pypi" or environment["deployment_branch_policy"] != {
        "protected_branches": False,
        "custom_branch_policies": True,
    }:
        raise ValueError("pypi must use custom release-tag deployment policies")
    reviewers = [r for r in environment["protection_rules"] if r["type"] == "required_reviewers"]
    if len(reviewers) != 1 or [
        (r["type"], r["reviewer"]["login"]) for r in reviewers[0]["reviewers"]
    ] != [("User", "BenHuddart")]:
        raise ValueError("pypi must require BenHuddart approval")
    if policies["total_count"] != 1 or [
        (p["name"], p["type"]) for p in policies["branch_policies"]
    ] != [("v*", "tag")]:
        raise ValueError("pypi may deploy only v* tags")


def candidate_identity(source: Path, packet: Path, commit: str) -> dict:
    """Check source, distribution contract, and both UTF-8 long descriptions."""
    if verify_source(source) != commit:
        raise ValueError("candidate source commit differs from triggering commit")
    dist = packet / "dist"
    (sdist,) = dist.glob("*.tar.gz")
    (wheel,) = dist.glob("*.whl")
    if set(dist.iterdir()) != {sdist, wheel} or any(p.is_symlink() for p in (sdist, wheel)):
        raise ValueError("candidate must contain exactly one regular sdist and wheel")
    contract = artifact_contract(source, sdist, wheel)
    readme = (source / "README.md").read_bytes()
    with tarfile.open(sdist) as archive:
        (name,) = (n for n in archive.getnames() if n.count("/") == 1 and n.endswith("/PKG-INFO"))
        member = archive.extractfile(name)
        assert member is not None
        source_metadata = member.read()
    with zipfile.ZipFile(wheel) as archive:
        (name,) = (n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))
        wheel_metadata = archive.read(name)
    for raw in (source_metadata, wheel_metadata):
        metadata = BytesParser(policy=default).parsebytes(raw)
        if metadata["Description-Content-Type"] != "text/markdown":
            raise ValueError("candidate long description must be Markdown")
        if metadata.get_payload(decode=True) != readme:
            raise ValueError("candidate long description differs from source README")
    return {
        "schema": 1,
        "source_commit": commit,
        "name": contract["metadata"]["Name"][0],
        "version": contract["version"],
        "readme_sha256": hashlib.sha256(readme).hexdigest(),
        "artifacts": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (sdist, wheel)},
    }


def record_candidate(source: Path, packet: Path, commit: str) -> None:
    """Retain portable identities, checksums, and a rendered review preview."""
    from readme_renderer.markdown import render

    identity = candidate_identity(source, packet, commit)
    preview = render((source / "README.md").read_text(encoding="utf-8"))
    if preview is None:
        raise ValueError("candidate README did not render")
    (packet / "README.html").write_text(preview, encoding="utf-8")
    (packet / "python-release.json").write_text(json.dumps(identity, indent=2) + "\n")
    (packet / "SHA256SUMS").write_text(
        "".join(f"{digest}  dist/{name}\n" for name, digest in identity["artifacts"].items())
    )


def verify_candidate(source: Path, packet: Path, commit: str, reports: Path | None = None) -> dict:
    """Reject changed files or evidence for another candidate or incomplete matrix."""
    identity = candidate_identity(source, packet, commit)
    if json.loads((packet / "python-release.json").read_text()) != identity:
        raise ValueError("Python release record differs from candidate")
    expected = "".join(f"{digest}  dist/{name}\n" for name, digest in identity["artifacts"].items())
    if (packet / "SHA256SUMS").read_text() != expected:
        raise ValueError("Python release checksums differ from candidate")
    if reports is not None:
        for python, resolution in itertools.product(PYTHONS, RESOLUTIONS):
            manifest = json.loads(
                (reports / f"python-installs-{python}-{resolution}" / "manifest.json").read_text()
            )
            if (
                manifest["source_head"] != commit
                or manifest["artifacts"] != identity["artifacts"]
                or manifest["resolution"] != resolution
                or manifest["contract"]["version"] != identity["version"]
            ):
                raise ValueError("installation evidence differs from candidate")
            targets = manifest["targets"]
            if len(targets) != len(INSTALL_PROFILES) or {t["profile"] for t in targets} != set(
                INSTALL_PROFILES
            ):
                raise ValueError("installation evidence lacks required profiles")
            for target in targets:
                if not target["version"].startswith(python + "."):
                    raise ValueError("installation evidence uses another Python version")
                if tuple(target["required_modules"]) != INSTALL_PROFILES[target["profile"]][1]:
                    raise ValueError("installation evidence uses another feature profile")
    return identity


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("record", "verify", "environment"))
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--commit")
    parser.add_argument("--reports", type=Path)
    args = parser.parse_args()
    if args.command == "environment":
        verify_publisher_environment(
            json.loads((args.packet / "environment.json").read_text()),
            json.loads((args.packet / "policies.json").read_text()),
        )
    elif not args.commit:
        parser.error("record and verify require --commit")
    elif args.command == "record":
        record_candidate(args.source, args.packet, args.commit)
    else:
        verify_candidate(args.source, args.packet, args.commit, args.reports)
    print(f"Python {args.command} check passed")
