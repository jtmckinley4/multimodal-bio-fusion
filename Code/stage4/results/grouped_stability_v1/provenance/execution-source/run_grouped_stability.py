"""Verify or execute one seed of the declared 12-fit grouped-stability panel.

Run only inside a separately prepared workspace. Verification starts no kernel;
training requires an explicit seed, kernel, external cache root and Slurm GPU.
"""
from contextlib import ExitStack
from datetime import datetime, timezone
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import uuid


def _load_support():
    here = Path(__file__).resolve().parent
    path = here / "arcc_execution_support.py"
    if not path.is_file():
        path = here.parent / "encoder_swaps/run_encoder_swaps.py"
    spec = importlib.util.spec_from_file_location("arcc_execution_support", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


support = _load_support()
WORKSPACE = Path(__file__).resolve().parent
NOTEBOOK = "Code/stage4/grouped_stability.ipynb"
SCHEMA = "biological-fusion.grouped-stability-package"


def verify_kernel(kernel):
    """Check the registered interpreter without starting a notebook kernel."""
    from jupyter_client.kernelspec import KernelSpecManager

    specification = KernelSpecManager().get_kernel_spec(kernel)
    if not specification.argv:
        raise ValueError(f"Kernel {kernel!r} has no interpreter command.")
    command = specification.argv[0]
    executable = Path(shutil.which(command) or command).expanduser().resolve()
    expected = Path(sys.executable).resolve()
    if executable != expected:
        raise ValueError(
            f"Kernel {kernel!r} uses {executable}; the activated environment uses {expected}. "
            "Register the intended environment's ipykernel before submission."
        )
    return {"kernel": kernel, "registered_python": str(executable)}


def verify_package(workspace=None):
    root = Path(workspace or WORKSPACE).resolve()
    snapshot = json.loads((root / "source-snapshot.json").read_text(encoding="utf-8"))
    if snapshot.get("schema") != SCHEMA or snapshot.get("schema_version") != 1:
        raise ValueError("Use a newly prepared grouped-stability package; preserve other snapshots.")
    if snapshot.get("training_seeds") != [0, 1, 2] or snapshot.get("fits_per_seed") != 12:
        raise ValueError("The package does not identify the agreed 36-fit panel.")
    files = {}
    for entry in snapshot.get("files", []):
        name = entry["destination"]
        path = support.package_path(root, name)
        if name.casefold() in {item.casefold() for item in files}:
            raise ValueError(f"Duplicate package file: {name}")
        if not path.is_file():
            raise FileNotFoundError(f"Missing prepared file: {name}")
        digest = support.sha256(path)
        if digest != entry["initial_copy_sha256"]:
            raise ValueError(f"Prepared file changed: {name}; prepare a new snapshot.")
        files[name] = digest
    required = {NOTEBOOK, "run_grouped_stability.py", "arcc_execution_support.py",
                "Code/shared_code/fusion/grouped_stability.py",
                "Code/shared_code/fusion/run_identity.py", "datasets/mRNA_Stability.csv"}
    if not required <= files.keys():
        raise ValueError(f"Incomplete execution inventory: {sorted(required - files.keys())}")
    return snapshot, files


class NotebookExecution:
    """Execution evidence and reuse guards, separate from scientific configuration."""

    def __init__(self, config):
        self.config = config
        self.root = Path(config["workspace"]).resolve()
        self.output = Path(config["output_root"]).resolve()
        self.snapshot, self.files = verify_package(self.root)
        support.validate_destinations(self.root, self.output, config["cache_root"],
                                      self.files, "newton")
        self.snapshot_sha = support.sha256(self.root / "source-snapshot.json")

    def observe_kernel(self, torch):
        if not os.environ.get("SLURM_JOB_ID"):
            raise RuntimeError("Newton training requires an allocated Slurm job.")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; refusing CPU fallback.")
        actual = torch.cuda.get_device_name(0)
        expected = self.config["expected_gpu"]
        if not expected.strip() or expected.casefold() not in actual.casefold():
            raise RuntimeError(f"Observed GPU {actual!r} differs from requested {expected!r}.")
        import importlib.metadata as metadata
        names = ("numpy", "pandas", "scipy", "scikit-learn", "torch", "nbconvert", "nbclient")
        self.environment = {
            "host": platform.node(), "python": sys.version, "platform": platform.platform(),
            "device": "cuda", "gpu": actual, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "packages": {name: metadata.version(name) for name in names},
        }
        path = Path(self.config["launch_record"])
        record = json.loads(path.read_text(encoding="utf-8"))
        record["kernel_environment"] = self.environment
        support.save_record(path, record)
        print("Kernel environment:", json.dumps(self.environment), flush=True)

    def producer(self, *unused):
        from shared_code.fusion import run_identity
        record = run_identity.producer_record(device="cuda", repository=self.root)
        record["grouped_stability_execution"] = {
            "source_snapshot_sha256": self.snapshot_sha,
            "output_root": str(self.output), "launch_record": Path(self.config["launch_record"]).name,
            "training_seed": self.config["seed"], "kernel_environment": self.environment,
        }
        return record

    def require_producer(self, record):
        producer = record.get("producer") if isinstance(record, dict) else None
        context = producer.get("grouped_stability_execution") if isinstance(producer, dict) else None
        if not isinstance(context, dict):
            raise ValueError("Result/checkpoint lacks grouped-stability execution evidence.")
        expected = {"source_snapshot_sha256": self.snapshot_sha,
                    "output_root": str(self.output), "training_seed": self.config["seed"]}
        if any(context.get(key) != value for key, value in expected.items()):
            raise ValueError("Result/checkpoint belongs to another execution cohort; preserve it.")
        name = context.get("launch_record")
        if not isinstance(name, str) or Path(name).name != name or "/" in name or "\\" in name:
            raise ValueError("Launch record must be a basename inside this output's logs.")
        launch = json.loads((self.output / "logs" / name).read_text(encoding="utf-8"))
        if not isinstance(launch, dict):
            raise ValueError("Referenced launch must be a structured execution record.")
        settings = launch.get("configuration", {})
        environment = context.get("kernel_environment")
        if (launch.get("snapshot_sha256") != self.snapshot_sha
                or launch.get("execution_source_sha256") != self.files
                or not isinstance(settings, dict)
                or settings.get("workspace") != str(self.root)
                or settings.get("output_root") != str(self.output)
                or settings.get("seed") != self.config["seed"]
                or launch.get("status") not in {"running", "completed", "failed_or_interrupted"}
                or not isinstance(environment, dict)
                or environment != launch.get("kernel_environment")):
            raise ValueError("Result/checkpoint launch or kernel evidence does not match.")
        gpu = environment.get("gpu")
        if (environment.get("device") != "cuda" or not isinstance(gpu, str)
                or self.config["expected_gpu"].casefold() not in gpu.casefold()):
            raise ValueError("Result/checkpoint lacks the matching observed GPU.")
        seed = record.get("config", {}).get("seed")
        if seed is not None and seed != self.config["seed"]:
            raise ValueError("Fit seed differs from its execution evidence.")

    def checkpoint_preflight(self, path, requested):
        import torch
        from shared_code.fusion import train
        # Only explicitly retained, trusted local recovery files enter this route.
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.require_producer(payload)
        del payload
        train.checkpoint_preflight(path, requested)


def prepare_notebook(notebook, config):
    import nbformat
    notebook = copy.deepcopy(notebook)
    settings = [i for i, cell in enumerate(notebook.cells)
                if "grouped-stability-settings" in cell.get("metadata", {}).get("tags", [])]
    if len(settings) != 1:
        raise ValueError("Expected one maintained grouped-stability-settings tag.")
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
            cell.metadata.pop("execution", None)
    source = "\n".join([
        "# Execution paths and provenance; protocol settings stay in the source notebook.",
        f"_execution_config = {config!r}",
        "if SPLIT_SEED != 42 or tuple(TRAINING_SEEDS) != (0, 1, 2) or SELECTION_SEED != 0:",
        "    raise ValueError('The prepared launcher requires the agreed split and seed policy.')",
        "if len(PANEL) != 12 or len({spec.key for spec in PANEL}) != 12:",
        "    raise ValueError('The prepared launcher requires twelve unique fits per seed.')",
        "sys.path.insert(0, _execution_config['workspace'])",
        "from run_grouped_stability import NotebookExecution",
        "EXECUTION = NotebookExecution(_execution_config)",
        "EXECUTION.observe_kernel(torch)",
        "READ_SAVED_RESULTS = False", "RUN_TRAINING = True", "RUN_COMPARISONS = False",
        "SEED = _execution_config['seed']", "DEVICE = 'cuda'",
        "CACHE_ROOT = Path(_execution_config['cache_root'])",
        "RESULTS_ROOT = Path(_execution_config['output_root']) / 'study'",
        "DATA_DIR = Path(_execution_config['workspace']) / 'datasets'",
        "FIT_PRODUCER = EXECUTION.producer",
        "RESULT_VALIDATOR = EXECUTION.require_producer",
        "CHECKPOINT_VALIDATOR = EXECUTION.checkpoint_preflight",
        "print('Grouped stability:', 'seed', SEED, '12 fits;', RESULTS_ROOT, flush=True)",
    ])
    notebook.cells.insert(settings[0] + 1, nbformat.v4.new_code_cell(source, id="arcc-grouped-execution"))
    return notebook


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--seed", type=int, choices=(0, 1, 2))
    parser.add_argument("--kernel")
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--expected-gpu", default=os.environ.get("ARCC_EXPECTED_GPU", ""))
    args = parser.parse_args(argv)
    snapshot, files = verify_package()
    if args.verify_only:
        if args.kernel:
            print(json.dumps(verify_kernel(args.kernel)))
        print(f"Verified {len(files)} files; no kernel, split generation or cache reads started.")
        return
    if args.seed is None or not args.kernel or args.cache_root is None:
        parser.error("Training requires --seed, --kernel and --cache-root.")
    if not os.environ.get("SLURM_JOB_ID") or not args.expected_gpu.strip():
        parser.error("Training requires an allocated Slurm job and an explicit --expected-gpu.")
    output = (args.output_root or WORKSPACE / "outputs").expanduser().resolve()
    cache = args.cache_root.expanduser().resolve()
    support.validate_destinations(WORKSPACE, output, cache, files, "newton")
    config = {"workspace": str(WORKSPACE), "output_root": str(output), "cache_root": str(cache),
              "seed": args.seed, "expected_gpu": args.expected_gpu}
    import nbformat
    with ExitStack() as stack:
        support.acquire_locks(stack, output / ".locks", ("grouped_stability",), (args.seed,))
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        name = f"grouped-stability-seed{args.seed}-{stamp}-{uuid.uuid4().hex[:8]}"
        logs, notebooks = output / "logs", output / "notebooks"
        logs.mkdir(parents=True, exist_ok=True)
        notebooks.mkdir(parents=True, exist_ok=True)
        record_path, executed_path = logs / f"{name}.json", notebooks / f"{name}.ipynb"
        config["launch_record"] = str(record_path)
        record = {
            "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
            "configuration": config, "snapshot_sha256": support.sha256(WORKSPACE / "source-snapshot.json"),
            "source_commit": snapshot.get("source_commit"), "execution_source_sha256": files,
            "host": platform.node(), "launcher_python": sys.version,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "executed_notebook": str(executed_path.relative_to(WORKSPACE)),
        }
        notebook = prepare_notebook(nbformat.read(WORKSPACE / NOTEBOOK, as_version=4), config)
        support.save_record(record_path, record)
        for key, value in {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                           "TOKENIZERS_PARALLELISM": "false"}.items():
            previous = os.environ.get(key)
            stack.callback(lambda k=key, p=previous: os.environ.pop(k, None) if p is None
                           else os.environ.__setitem__(k, p))
            os.environ[key] = value
        error = None
        try:
            processor = support.LiveOutputPreprocessor(timeout=None, kernel_name=args.kernel,
                                                       allow_errors=False, record_timing=True)
            processor.preprocess(notebook, {"metadata": {"path": str(WORKSPACE)}})
        except BaseException as caught:
            error = caught
            raise
        finally:
            failures = []
            try:
                nbformat.write(notebook, executed_path)
            except Exception as caught:
                failures.append(caught)
            try:
                record = json.loads(record_path.read_text(encoding="utf-8"))
                record.update(status="failed_or_interrupted" if error else "completed",
                              ended_utc=datetime.now(timezone.utc).isoformat())
                if error:
                    record["error"] = f"{type(error).__name__}: {error}"
                support.save_record(record_path, record)
            except Exception as caught:
                failures.append(caught)
            if failures:
                if error is None:
                    raise RuntimeError("Could not save execution evidence") from failures[0]
                if hasattr(error, "add_note"):
                    error.add_note(f"Saving execution evidence also failed: {failures}")


if __name__ == "__main__":
    main()
