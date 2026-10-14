#!/usr/bin/env python3
"""Verify the original DGM4 splits and derive source subsets without resplitting.

Only fingerprints and aggregate counts belong in the public repository. Optional
record/ID manifests are generated into the user's local, ignored data directory.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path, PurePosixPath

SOURCES = ("bbc", "guardian", "usa_today", "washington_post")
SPLITS = ("train", "val", "test")
STATES = ("Real", "Text-only", "Image-only", "Joint")
DGM4_REVISION = "9efd3f1ca14cb1dac1692cdce0beea6844b968a6"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        if stream.read(1) == "[":
            stream.seek(0)
            rows = json.load(stream)
        else:
            stream.seek(0)
            text = stream.read()
            rows = json.loads(text) if text.lstrip().startswith("[") else [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(rows, list):
        raise ValueError(f"Expected a record array or JSONL: {path}")
    return rows


def infer_labels(item):
    """Match the submitted implementation, including annotation/name fallbacks."""
    fake_cls = str(item.get("fake_cls", "")).lower().strip()
    fake_text_pos = item.get("fake_text_pos", []) or []
    fake_image_box = item.get("fake_image_box", []) or []
    text_manip = int(bool(fake_text_pos) or "text_" in fake_cls or "text" in fake_cls)
    image_manip = int((len(fake_image_box) == 4 and fake_image_box[2] > fake_image_box[0] and fake_image_box[3] > fake_image_box[1]) or "face_" in fake_cls or "face" in fake_cls or "image" in fake_cls)
    real_names = {"orig", "original", "real", "none", "pristine", "clean"}
    fake = int(not ((fake_cls in real_names or fake_cls == "") and not text_manip and not image_manip))
    state = 3 if text_manip and image_manip else 1 if text_manip else 2 if image_manip else 0
    return fake, text_manip, image_manip, state


def resolve_image_path(data_root, relative):
    """Use the exact candidate order of the submitted data loader."""
    relative = str(relative).replace("\\", "/")
    candidates = [data_root / relative, data_root.parent / relative]
    if relative.startswith("DGM4/"):
        candidates.append(data_root / relative[len("DGM4/"):])
    return next((path for path in candidates if path.is_file()), candidates[0])


def digest_records(rows):
    """Hash ordered canonical JSON records; retain duplicates and all fields."""
    digest = hashlib.sha256()
    for row in rows:
        digest.update((json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))
    return digest.hexdigest()


def digest_ids(rows):
    return hashlib.sha256("".join(str(row["id"]) + "\n" for row in rows).encode("utf-8")).hexdigest()


def build_manifest(data_root, source_metadata=None, output=None, check_images=False, write_subsets=False, write_id_manifests=False):
    """Read original assets; never change their contents or split membership."""
    data_root = Path(data_root)
    metadata_root = data_root / "metadata"
    source_by_id = {}
    if source_metadata is not None:
        source_metadata = Path(source_metadata)
        source_rows = load_rows(source_metadata)
        for row in source_rows:
            key, source = str(row["id"]), row["source"]
            if key in source_by_id and source_by_id[key] != source:
                raise ValueError(f"Conflicting VisualNews source labels for ID {key}")
            source_by_id[key] = source
        del source_rows
    else:
        # Every used ID has an original-image record in the fixed DGM4 release.
        # This uses source metadata only; no labels or predictions select samples.
        for split in SPLITS:
            for row in load_rows(metadata_root / f"{split}.json"):
                parts = str(row["image"]).replace("\\", "/").split("/")
                if len(parts) > 3 and parts[:2] == ["DGM4", "origin"]:
                    key, source = str(row["id"]), parts[2]
                    if key in source_by_id and source_by_id[key] != source:
                        raise ValueError(f"Conflicting original-image sources for ID {key}")
                    source_by_id[key] = source
    manifest = {
        "schema_version": 1,
        "dataset": "rshaojimmy/DGM4",
        "official_revision": DGM4_REVISION,
        "split_policy": "Original metadata partitions; stable filter by VisualNews id-to-source; no resplit or deduplication",
        "record_digest_encoding": "UTF-8 canonical JSON per record, sorted keys, compact separators, newline terminator, original order",
        "id_digest_encoding": "UTF-8 str(id) followed by newline, original order, duplicates retained",
        "state_order": list(STATES),
        "binary_labels": {"Real": 0, "Manipulated": 1},
        "inputs": {},
        "source_metadata_mode": "provided_id_to_source" if source_metadata is not None else "derived_from_original_image_paths",
        "sources": {source: {} for source in SOURCES},
    }
    if source_metadata is not None:
        manifest["inputs"]["VisualNews/data.json"] = {"sha256": sha256_file(source_metadata)}
    all_ids = {}
    for split in SPLITS:
        path = metadata_root / f"{split}.json"
        rows = load_rows(path)
        manifest["inputs"][f"DGM4/metadata/{split}.json"] = {"sha256": sha256_file(path), "count": len(rows)}
        groups = {source: [] for source in SOURCES}
        all_ids[split] = set()
        for row in rows:
            key = str(row["id"])
            source = source_by_id.get(key)
            if source not in groups:
                raise ValueError(f"Missing or unsupported VisualNews source for ID {key}: {source}")
            image = str(row["image"]).replace("\\", "/")
            pure = PurePosixPath(image)
            if pure.is_absolute() or ".." in pure.parts or ":" in pure.parts[0]:
                raise ValueError(f"Image path must be portable and relative: {image}")
            if check_images and not resolve_image_path(data_root, image).is_file():
                raise FileNotFoundError(f"Missing image for {split} ID {key}: {image}")
            groups[source].append(row)
            all_ids[split].add(key)
        for source, subset in groups.items():
            state_counts = Counter(infer_labels(row)[3] for row in subset)
            labels = Counter(infer_labels(row)[0] for row in subset)
            manifest["sources"][source][split] = {
                "count": len(subset),
                "unique_ids": len({str(row["id"]) for row in subset}),
                "state_counts": {name: state_counts[index] for index, name in enumerate(STATES)},
                "binary_counts": {"Real": labels[0], "Manipulated": labels[1]},
                "ordered_ids_sha256": digest_ids(subset),
                "ordered_records_sha256": digest_records(subset),
            }
            if output is not None and (write_subsets or write_id_manifests):
                directory = Path(output) / source
                directory.mkdir(parents=True, exist_ok=True)
                if write_subsets:
                    with (directory / f"{split}.jsonl").open("w", encoding="utf-8") as stream:
                        for row in subset:
                            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                if write_id_manifests:
                    (directory / f"{split}.ids.txt").write_text("".join(str(row["id"]) + "\n" for row in subset), encoding="utf-8")
    used_ids = sorted(set().union(*all_ids.values()))
    mapping_rows = [{"id": key, "source": source_by_id[key]} for key in used_ids]
    manifest["used_source_mapping"] = {"unique_ids": len(used_ids), "sha256": digest_records(mapping_rows)}
    if output is not None:
        Path(output).mkdir(parents=True, exist_ok=True)
        with (Path(output) / "source_by_id.jsonl").open("w", encoding="utf-8") as stream:
            for row in mapping_rows:
                stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    manifest["cross_split_id_overlap"] = {
        f"{left}/{right}": len(all_ids[left] & all_ids[right])
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    if any(manifest["cross_split_id_overlap"].values()):
        raise ValueError("Original news IDs overlap across data splits; inspect the dataset version")
    return manifest


def compare_expected(actual, expected):
    """Compare byte fingerprints and content/order, not filenames or totals alone."""
    errors = []
    for name, fingerprint in expected["inputs"].items():
        if name.startswith("DGM4/") and actual["inputs"].get(name) != fingerprint:
            errors.append(name)
    for section in ("sources", "used_source_mapping", "cross_split_id_overlap"):
        if actual[section] != expected[section]:
            errors.append(section)
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Extracted DGM4 directory containing metadata/, origin/, manipulation/")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--source-metadata", type=Path, help="Original VisualNews data.json or a verified minimal id/source JSONL")
    source_group.add_argument("--derive-source-metadata", action="store_true", help="Reconstruct id/source from original-image paths and verify equality against the audited VisualNews mapping")
    parser.add_argument("--output", type=Path, default=Path("data/prepared"), help="Local output directory, excluded from Git")
    parser.add_argument("--expected", type=Path, default=Path(__file__).with_name("expected_splits.json"))
    parser.add_argument("--check-images", action="store_true", help="Check that every referenced image exists; does not decode all images")
    parser.add_argument("--write-subsets", action="store_true", help="Write local source-filtered records containing dataset text; never commit these files")
    parser.add_argument("--write-id-manifests", action="store_true", help="Write local ordered ID-only manifests; these are not included in the public release")
    args = parser.parse_args()
    # Verify first; do not emit alternative data records after a failed audit.
    actual = build_manifest(args.data_root, args.source_metadata, output=args.output, check_images=args.check_images)
    expected = json.loads(args.expected.read_text(encoding="utf-8"))
    differences = compare_expected(actual, expected)
    actual["verification"] = {"status": "FAIL" if differences else "PASS", "differing_sections": differences, "image_paths_checked": args.check_images}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "split_audit.json").write_text(json.dumps(actual, indent=2) + "\n", encoding="utf-8")
    if differences:
        raise SystemExit("Dataset differs from the reported protocol in: " + ", ".join(differences) + ". No subsets written; inspect split_audit.json.")
    if args.write_subsets or args.write_id_manifests:
        build_manifest(args.data_root, args.source_metadata, output=args.output, write_subsets=args.write_subsets, write_id_manifests=args.write_id_manifests)
    print(json.dumps({"status": "PASS", "counts": {source: {split: actual["sources"][source][split]["count"] for split in SPLITS} for source in SOURCES}, "image_paths_checked": args.check_images}, indent=2))


if __name__ == "__main__":
    main()
