"""
Verify a prepared RPI2241 workspace or train one split and seed on Newton.
Each training job runs the six declared arms through the shared pair-study code.
"""

import argparse
from contextlib import ExitStack, contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
from tempfile import NamedTemporaryFile


WORKSPACE = Path(__file__).resolve().parent
SPLITS = ("left", "right", "both")
SEEDS = (0, 1, 2)
EXPECTED_GPU = "NVIDIA H100 PCIe"
OUTPUTS = "outputs/rpi2241-newton"


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def workspace_path(root, name):
    parts = PurePosixPath(name)
    if parts.is_absolute() or not parts.parts or ".." in parts.parts or "\\" in name or ":" in name:
        raise ValueError(f"Invalid workspace path: {name}")
    path = root.joinpath(*parts.parts)
    if path.resolve() != path or path.is_symlink():
        raise ValueError(f"Linked workspace path: {name}")
    return path


def verify_package(root):
    """Check the prepared sources before importing any experiment code."""
    contents = workspace_path(root, "source-snapshot.json").read_bytes()
    snapshot = json.loads(contents)
    if snapshot.get("schema") != "partner-tasks.package.v1":
        raise ValueError("Prepare the partner_tasks recipe before running this launcher.")

    names = set()
    for entry in snapshot["files"]:
        name = entry["destination"]
        if name.casefold() in names:
            raise ValueError(f"Duplicate prepared source: {name}")
        names.add(name.casefold())
        path = workspace_path(root, name)
        if not path.is_file() or sha256(path) != entry["initial_copy_sha256"]:
            raise ValueError(f"Missing or changed prepared source: {name}")
    required = {
        "run_partner_tasks.py", "input-identity.json", "batches.tsv",
        "code/shared_code/fusion/pair_study.py",
    }
    if not required <= names:
        raise ValueError(f"Missing package entries: {sorted(required - names)}")
    print(f"Verified {len(names)} prepared source files.")
    return hashlib.sha256(contents).hexdigest()


def check_gpu(torch, expected_gpu):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Training requires an allocated Slurm job.")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose one allocated CUDA GPU to the RPI2241 job.")
    name = torch.cuda.get_device_name(0)
    if not expected_gpu.strip() or expected_gpu.casefold() not in name.casefold():
        raise RuntimeError(f"Unexpected GPU: {name} (expected {expected_gpu}).")
    return name


def read_inputs(root, reference, stack):
    """Match the built table, recorded splits and frozen caches before training."""
    from shared_code.fusion import entity_embeddings, pair_sources, pair_study, pair_tasks

    table_dir = root / "inputs/rpi2241"
    frame = pair_sources.read_pair_dataset(table_dir)
    if pair_study.table_record(frame) != reference["table"]:
        raise ValueError("Staged RPI2241 table differs from Julian's recorded table.")
    splits = {name: pair_tasks.disjoint_split(frame, held_out=name, seed=reference["split_seed"])
              for name in SPLITS}
    split_records = {name: pair_study.split_record(split) for name, split in splits.items()}
    for name, expected in reference["recorded_splits"].items():
        if split_records[name] != expected:
            raise ValueError(f"Staged groups or split differ for {name} held out.")

    dataset_metadata = json.loads((table_dir / "dataset.json").read_text(encoding="utf-8"))
    input_files = [table_dir / name for name in ("dataset.json", "pairs.csv", "rna.fasta", "protein.fasta")]
    groups = table_dir / "groups.json"
    if dataset_metadata["schema"] == "pair-dataset.v2" or groups.exists():
        input_files.append(groups)

    embeddings, summaries = {}, {}
    for side in pair_tasks.SIDES:
        expected = reference["caches"][side]
        encoder = entity_embeddings.encoder_for(frame.modality(side), expected["encoder"])
        directory = entity_embeddings.cache_dir_for(root / "inputs/pair_entities", frame, side, encoder.key)
        cache = entity_embeddings.load(directory)
        stack.callback(cache.close)
        names, texts = entity_embeddings.inputs_for(encoder, frame.sequences(side))
        fingerprint = entity_embeddings.identity_fingerprint(names, texts)
        if cache.names != names or cache.meta["input_sha256"] != fingerprint:
            raise ValueError(f"{side} cache does not match the staged entity IDs and sequences.")
        embeddings[side] = cache
        summaries[side] = entity_embeddings.summarize(directory)
        input_files.append(directory / "entities.json")
        input_files.extend(directory / f"shard-{index:05d}.npz"
                           for index in range(int(cache.shard_of[-1]) + 1))
    if pair_study.cache_record(summaries) != reference["caches"]:
        raise ValueError("Staged encoder caches differ from Julian's recorded input metadata.")

    hashes = {path.relative_to(root).as_posix(): sha256(path) for path in input_files}
    print(f"Verified RPI2241 table, three splits and both frozen entity caches ({len(hashes)} files).")
    return frame, splits, split_records, embeddings, summaries, hashes


@contextmanager
def output_lock(path):
    """Coordinate cluster jobs that could write the same output."""
    import fcntl

    with path.open("a", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another job is writing {path.name}; retry after it finishes.") from error
        yield


def pin_run(root, output, evidence):
    """Bind the nine jobs to one set of sources, inputs and runtime versions."""
    import fcntl

    record = workspace_path(root, f"{OUTPUTS}/run-inputs.json")
    lock = workspace_path(root, f"{OUTPUTS}/.run-inputs.lock")
    with lock.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        if record.exists():
            if json.loads(record.read_bytes()) != evidence:
                raise ValueError("Existing Newton run uses a different snapshot, input or runtime.")
            return
        if any(output.glob("rpi2241/held_out_*/*.json")):
            raise ValueError("The output folder already contains fits without a Newton run record.")
        pending = NamedTemporaryFile("w", encoding="utf-8", dir=output, suffix=".partial", delete=False)
        try:
            with pending:
                json.dump(evidence, pending, indent=2)
                pending.write("\n")
            os.replace(pending.name, record)
        finally:
            Path(pending.name).unlink(missing_ok=True)


def train_job(root, args, snapshot_sha256, gpu, torch, reference, inputs):
    from shared_code.fusion import pair_study, pair_training, run_identity

    frame, splits, split_records, embeddings, summaries, hashes = inputs
    split = splits[args.split]
    config = pair_training.PairConfig(**{**reference["config"], "seed": args.seed})
    keys = pair_study.panel_keys()
    if len(keys) != 6:
        raise ValueError("The RPI2241 recipe requires the six core arms.")

    output = workspace_path(root, OUTPUTS)
    output.mkdir(parents=True, exist_ok=True)
    producer = run_identity.producer_record(
        device="cuda", repository=root, packages=("numpy", "pandas", "scipy", "scikit-learn", "torch")
    )
    evidence = {"schema": "rpi2241-newton.run.v1", "source_snapshot_sha256": snapshot_sha256,
                "input_sha256": hashes, "splits": split_records,
                "gpu": gpu, "cuda": torch.version.cuda, "python": producer["python"],
                "packages": producer["packages"]}
    pin_run(root, output, evidence)
    evidence_sha256 = sha256(output / "run-inputs.json")
    producer["partner_execution"] = {"run_inputs_sha256": evidence_sha256,
                                      "slurm_job_id": os.environ["SLURM_JOB_ID"], "gpu": gpu}

    with output_lock(workspace_path(root, f"{OUTPUTS}/.{args.split}-seed{args.seed}.lock")):
        for key in keys:
            path = pair_study.result_path(output, frame, key, config, held_out=args.split)
            workspace_path(root, path.relative_to(root).as_posix())
            if path.exists():
                identity = pair_study.study_identity(frame, split, key, config, summaries)
                payload, issue = pair_study.read_result(path, identity)
                provenance = (payload or {}).get("producer", {}).get("partner_execution", {})
                if issue is not None or provenance.get("run_inputs_sha256") != evidence_sha256:
                    raise ValueError(f"Existing output is outside this Newton run: {path}")

        print(f"RPI2241 / {args.split} held out / training seed {args.seed} / {gpu}")
        panel = pair_study.run_panel(frame, split, embeddings, config, summaries, output,
                                     device="cuda:0", keys=keys, producer=producer)
        if set(panel["results"]) != set(keys):
            raise RuntimeError("The job ended without all six declared fit records.")
        print(f"6/6 fit records complete ({len(panel['reused'])} resumed from this Newton run).")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify-only", action="store_true", help="verify package sources")
    mode.add_argument("--verify-inputs", action="store_true", help="verify staged table and caches")
    parser.add_argument("--split", choices=SPLITS)
    parser.add_argument("--seed", choices=SEEDS, type=int)
    parser.add_argument("--expected-gpu", default=EXPECTED_GPU)
    args = parser.parse_args(argv)
    verify = args.verify_only or args.verify_inputs
    if verify and (args.split is not None or args.seed is not None):
        parser.error("Verification takes no split or training seed.")
    if not verify and (args.split is None or args.seed is None):
        parser.error("Training requires both --split and --seed.")

    root = WORKSPACE.resolve()
    snapshot = verify_package(root)
    if args.verify_only:
        return
    if not verify and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Training requires an allocated Slurm job.")

    sys.path.insert(0, str(root / "Code"))
    import torch

    gpu = check_gpu(torch, args.expected_gpu) if not verify else None
    reference = json.loads((root / "input-identity.json").read_bytes())
    with ExitStack() as stack:
        inputs = read_inputs(root, reference, stack)
        if verify:
            print("Input verification complete. No fits started.")
        else:
            train_job(root, args, snapshot, gpu, torch, reference, inputs)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        raise SystemExit(f"RPI2241 stopped: {error}") from error
