"""Check staged input bytes without loading models, caches or research tables."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


MANIFEST = Path(__file__).with_name("input-manifest.json")


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def input_path(root, relative):
    """Resolve manifest paths inside the supplied root, including HF blob links."""
    relative = PurePosixPath(relative)
    if relative.is_absolute() or ".." in relative.parts or any(
            ":" in part or "\\" in part for part in relative.parts):
        raise ValueError(f"Unsafe input path: {relative}")
    root = Path(root).resolve()
    path = root.joinpath(*relative.parts).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Input resolves outside its root: {relative}")
    return path


def check_files(entries, root):
    """Report every missing/mismatched input; never repair or acquire it."""
    failures = []
    checked = 0
    for entry in entries:
        name = entry["path"]
        path = input_path(root, name)
        if not path.is_file():
            failures.append(f"Missing: {name}")
        elif path.stat().st_size != entry["bytes"]:
            failures.append(f"Size differs: {name}")
        elif sha256(path) != entry["sha256"]:
            failures.append(f"SHA-256 differs: {name}")
        else:
            checked += 1
    return checked, failures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("references", "regions"), required=True)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--region-data-root", type=Path)
    parser.add_argument("--hf-hub-root", type=Path)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("schema") != "fusion-studies.inputs.v1":
        raise ValueError("Unsupported input manifest")
    if args.scope == "references":
        if args.reference_root is None:
            parser.error("references requires --reference-root")
        groups = [("Reference result JSONs and best weights", manifest["references"], args.reference_root)]
    else:
        if args.region_data_root is None and args.hf_hub_root is None:
            parser.error("regions requires --region-data-root and/or --hf-hub-root")
        groups = []
        if args.region_data_root is not None:
            groups.append(("A.6 source tables and homology map", manifest["region_data"], args.region_data_root))
        if args.hf_hub_root is not None:
            groups.append(("Pinned pretrained snapshots and references", manifest["encoder_files"], args.hf_hub_root))
    failures = []
    for label, entries, root in groups:
        if not entries:
            raise ValueError(f"Empty required inventory: {label}")
        checked, missing = check_files(entries, root)
        print(f"{label}: {checked}/{len(entries)} files verified.")
        failures.extend(missing)
    if failures:
        raise SystemExit("Input verification failed:\n" + "\n".join(failures))
    print("Byte verification only; no model loading, cohort preparation, extraction or training.")


if __name__ == "__main__":
    main()
