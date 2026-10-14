#!/usr/bin/env python3
"""Download pinned official DGM4 or CLIP assets and verify SHA-256 checksums."""
import argparse
import hashlib
import json
import stat
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_asset(path, asset):
    if path.exists():
        if sha256_file(path) == asset["sha256"]:
            print(f"Verified existing {path}", flush=True)
            return
        raise FileExistsError(f"Existing file has a different checksum: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    digest = hashlib.sha256()
    with urllib.request.urlopen(asset["url"], timeout=120) as response, partial.open("wb") as stream:
        for block in iter(lambda: response.read(1024 * 1024), b""):
            digest.update(block)
            stream.write(block)
    if partial.stat().st_size != asset["bytes"] or digest.hexdigest() != asset["sha256"]:
        raise RuntimeError(f"Download checksum or size mismatch: {partial}; original files are unchanged")
    partial.replace(path)
    print(f"Downloaded and verified {path}", flush=True)


def extract_archive(path):
    """Extract official archives next to their ZIP after checking member paths."""
    destination = path.parent.resolve()
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            name = PurePosixPath(member.filename)
            mode = member.external_attr >> 16
            if name.is_absolute() or ".." in name.parts or stat.S_ISLNK(mode):
                raise ValueError(f"Unsafe archive member: {member.filename}")
            target = destination / member.filename
            if not target.resolve().is_relative_to(destination):
                raise ValueError(f"Archive member escapes destination: {member.filename}")
        archive.extractall(destination)
    print(f"Extracted {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", required=True, choices=("dgm4", "clip"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata-only", action="store_true", help="Download only DGM4 split metadata; not enough for model inference")
    parser.add_argument("--extract", action="store_true", help="Extract DGM4 image ZIP files after verification")
    args = parser.parse_args()
    if args.asset == "clip" and (args.metadata_only or args.extract):
        parser.error("--metadata-only and --extract apply only to DGM4")
    name = "dgm4_assets.json" if args.asset == "dgm4" else "pretrained_assets.json"
    manifest = json.loads(Path(__file__).with_name(name).read_text(encoding="utf-8"))
    for filename, asset in manifest["files"].items():
        if args.metadata_only and not filename.startswith("metadata/"):
            continue
        path = args.output / filename
        download_asset(path, asset)
        if args.extract and path.suffix == ".zip":
            extract_archive(path)


if __name__ == "__main__":
    main()
