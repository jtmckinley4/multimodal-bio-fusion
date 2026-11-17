"""Execute one prepared Newton benchmark/seed, or report the finished cohort.
Importing this module only defines helpers; it never starts execution.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import uuid

from nbconvert.preprocessors import ExecutePreprocessor

BENCHMARKS = ("cov_vac_degradation", "fungal_expression", "ecoli_proteins", "mrna_stability")
WORKSPACE = Path(__file__).resolve().parent


class LiveOutputPreprocessor(ExecutePreprocessor):
    """Mirror notebook text streams to Slurm without changing saved cell outputs."""

    def process_message(self, msg, cell, cell_index):
        output = super().process_message(msg, cell, cell_index)
        if msg["msg_type"] == "stream":
            content = msg["content"]
            stream = sys.stderr if content["name"] == "stderr" else sys.stdout
            # Forward chunks immediately, including partial lines and carriage returns.
            stream.write(content["text"])
            stream.flush()
        return output


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def execution_settings():
    """Default to reading fresh outputs; computation requires the explicit launcher."""
    mode = os.environ.get("ARCC_MODE", "read")
    if mode not in {"read", "train", "report"}:
        raise ValueError("ARCC_MODE must be read, train or report.")
    benchmark = os.environ.get("ARCC_BENCHMARK", "all")
    if benchmark not in (*BENCHMARKS, "all"):
        raise ValueError("Unknown ARCC_BENCHMARK.")
    seed = int(os.environ.get("ARCC_SEED", "0"))
    if seed not in (0, 1, 2):
        raise ValueError("Training seeds must be 0, 1 or 2.")
    if mode == "train" and benchmark == "all":
        raise ValueError("Train one benchmark/seed per invocation.")
    if mode != "read" and not os.environ.get("ARCC_LAUNCH_RECORD"):
        raise RuntimeError("Use run_encoder_swaps.py for training or report publication.")
    return {
        "mode": mode, "seed": seed,
        "benchmarks": BENCHMARKS if benchmark == "all" else (benchmark,),
        "cache_root": Path(os.environ.get("ARCC_CACHE_ROOT",
            str(WORKSPACE / "Code/stage3/embeddings"))).expanduser().resolve(),
        "output_root": WORKSPACE / "outputs",
        "historical_root": WORKSPACE / "historical_results/Code",
        "expected_gpu": os.environ.get("ARCC_EXPECTED_GPU", ""),
        "launch_record": os.environ.get("ARCC_LAUNCH_RECORD"),
    }


def verify_preserved_inputs():
    """Check CSV/selection sources against the original preparation snapshot."""
    snapshot = json.loads((WORKSPACE / "source-snapshot.json").read_text(encoding="utf-8"))
    checked = {}
    for entry in snapshot["files"]:
        relative = entry["destination"]
        if not relative.startswith(("datasets/", "historical_results/")):
            continue
        path = (WORKSPACE / relative).resolve()
        if not path.is_relative_to(WORKSPACE) or not path.is_file():
            raise RuntimeError(f"Missing preserved input: {relative}")
        actual = sha256(path)
        if actual != entry["initial_copy_sha256"]:
            raise RuntimeError(f"Preserved input changed: {relative}; inspect before continuing.")
        checked[relative] = actual
    if not checked:
        raise RuntimeError("The source snapshot contains no preserved inputs.")
    return snapshot, checked


def acquire_locks(stack, directory, benchmarks, seeds):
    """Prevent two cooperative launchers writing the same fit/checkpoint."""
    import fcntl  # Newton/Linux.
    directory.mkdir(parents=True, exist_ok=True)
    for benchmark in sorted(benchmarks):
        for seed in sorted(seeds):
            handle = stack.enter_context((directory / f"{benchmark}-seed{seed}.lock").open("a+"))
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"Another job holds {benchmark}, seed {seed}.") from exc



def require_arcc_producer(data):
    """Reject desktop results/checkpoints copied into the fresh output cohort."""
    context = data.get("producer", {}).get("arcc_execution", {})
    if context.get("source_snapshot_sha256") != sha256(WORKSPACE / "source-snapshot.json"):
        raise ValueError("Fresh results require this ARCC cohort's producer context; preserve the file.")
    if not context.get("launch_record") or not context.get("selection_source_sha256"):
        raise ValueError("ARCC launch/selection evidence is missing; preserve the file.")


def check_fresh_results(benchmarks):
    for benchmark in benchmarks:
        paths = list((WORKSPACE / "outputs/phaseA" / benchmark).glob("*.json"))
        paths += list((WORKSPACE / "outputs/phaseB" / benchmark).glob("*/*.json"))
        for path in paths:
            try:
                require_arcc_producer(json.loads(path.read_text(encoding="utf-8")))
            except (ValueError, TypeError, AttributeError) as exc:
                raise ValueError(f"Unsupported fresh result {path}: {exc}") from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "report"), required=True)
    parser.add_argument("--benchmark", choices=(*BENCHMARKS, "all"), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2))
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--cache-root", type=Path, default=Path(os.environ.get(
        "ARCC_CACHE_ROOT", str(WORKSPACE / "Code/stage3/embeddings"))))
    parser.add_argument("--expected-gpu", default=os.environ.get("ARCC_EXPECTED_GPU", ""),
                        help="GPU-name substring established during preflight")
    args = parser.parse_args()
    if args.mode == "train":
        if args.benchmark == "all" or args.seed is None:
            parser.error("Training requires one --benchmark and one --seed.")
        if not os.environ.get("SLURM_JOB_ID"):
            parser.error("Training requires an allocated Slurm compute job.")
        if not args.expected_gpu.strip():
            parser.error("Set --expected-gpu from the preflight GPU name.")
    elif args.seed is not None:
        parser.error("Reporting reads seeds 0, 1 and 2 together; omit --seed.")

    snapshot, inputs = verify_preserved_inputs()
    benchmarks = BENCHMARKS if args.benchmark == "all" else (args.benchmark,)
    seed = args.seed if args.seed is not None else 0
    lock_seeds = (seed,) if args.mode == "train" else (0, 1, 2)
    with ExitStack() as stack:
        acquire_locks(stack, WORKSPACE / "outputs/.locks", benchmarks, lock_seeds)
        import nbformat

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        name = f"{args.mode}-{args.benchmark}-seed{seed}-{stamp}-{uuid.uuid4().hex[:8]}"
        log_dir, notebook_dir = WORKSPACE / "outputs/logs", WORKSPACE / "outputs/notebooks"
        log_dir.mkdir(parents=True, exist_ok=True)
        notebook_dir.mkdir(parents=True, exist_ok=True)
        record_path = log_dir / f"{name}.json"
        executed_path = notebook_dir / f"{name}.ipynb"
        source_path = WORKSPACE / "Code/stage4/encoder_swaps.ipynb"
        source_files = [source_path, Path(__file__).resolve(),
                        *sorted((WORKSPACE / "Code/shared_code").rglob("*.py"))]
        record = {
            "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
            "mode": args.mode, "benchmarks": list(benchmarks),
            "training_seed": args.seed, "comparison_seeds": [0, 1, 2],
            "selection": "preserved historical seed-0 validation choices",
            "historical_source_commit": snapshot["source_commit"],
            "snapshot_sha256": sha256(WORKSPACE / "source-snapshot.json"),
            "preserved_input_sha256": inputs,
            "execution_source_sha256": {str(p.relative_to(WORKSPACE)): sha256(p) for p in source_files},
            "host": platform.node(), "launcher_python": sys.version,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "kernel": args.kernel, "expected_gpu": args.expected_gpu,
            "cache_root": str(args.cache_root.expanduser().resolve()),
            "executed_notebook": str(executed_path.relative_to(WORKSPACE)),
        }
        record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        os.environ.update({
            "ARCC_MODE": args.mode, "ARCC_BENCHMARK": args.benchmark, "ARCC_SEED": str(seed),
            "ARCC_CACHE_ROOT": record["cache_root"], "ARCC_EXPECTED_GPU": args.expected_gpu,
            "ARCC_LAUNCH_RECORD": str(record_path), "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
        })
        notebook = nbformat.read(source_path, as_version=4)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                cell.outputs = []
                cell.execution_count = None
                cell.metadata.pop("execution", None)
        processor = LiveOutputPreprocessor(
            timeout=None, kernel_name=args.kernel, allow_errors=False, record_timing=True,
        )
        print(f"Executing {args.mode}: {benchmarks}, seed {args.seed}; {executed_path}", flush=True)
        primary_error = None
        try:
            processor.preprocess(notebook, {"metadata": {"path": str(WORKSPACE)}})
        except BaseException as exc:
            primary_error = exc
            record["status"] = "failed_or_interrupted"
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
        else:
            record["status"] = "completed"
        finally:
            # A forced kill may prevent this; fit JSONs/checkpoints persist separately.
            record["ended_utc"] = datetime.now(timezone.utc).isoformat()
            save_errors = []
            try:
                nbformat.write(notebook, executed_path)
            except Exception as exc:
                save_errors.append(exc)
            try:
                record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            except Exception as exc:
                save_errors.append(exc)
            if save_errors:
                if primary_error is None:
                    raise RuntimeError("Could not save execution evidence") from save_errors[0]
                if hasattr(primary_error, "add_note"):
                    primary_error.add_note(f"Saving execution evidence also failed: {save_errors}")



if __name__ == "__main__":
    main()
