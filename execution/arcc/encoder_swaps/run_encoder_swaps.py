"""Run a declared swap workload from a verified, separately prepared workspace.

--verify-only reads the package inventory without starting a kernel. Scientific
cells come from the maintained notebook; only execution settings are injected.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import sys
import uuid

from nbconvert.preprocessors import ExecutePreprocessor

WORKSPACE = Path(__file__).resolve().parent
BENCHMARKS = ("cov_vac_degradation", "fungal_expression", "ecoli_proteins", "mrna_stability")
NOTEBOOK = "Code/stage4/encoder_swaps.ipynb"
EXPERIMENTS = ("full", "ecoli_ernie")


class LiveOutputPreprocessor(ExecutePreprocessor):
    """Forward text immediately while preserving nbconvert's saved outputs."""

    def process_message(self, msg, cell, cell_index):
        output = super().process_message(msg, cell, cell_index)
        if msg["msg_type"] == "stream":
            content = msg["content"]
            stream = sys.stderr if content["name"] == "stderr" else sys.stdout
            stream.write(content["text"])
            stream.flush()
        return output


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_path(root, relative):
    """Manifest names use portable relative paths, never external file references."""
    if not isinstance(relative, str) or "\\" in relative or ":" in relative:
        raise ValueError(f"Invalid package path: {relative!r}")
    name = PurePosixPath(relative)
    if name.is_absolute() or ".." in name.parts or not name.parts:
        raise ValueError(f"Invalid package path: {relative!r}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Package path escapes workspace: {relative}")
    return path


def verify_preserved_inputs(workspace=None):
    """Verify copied code and preserved inputs; never repair or update a snapshot."""
    root = Path(workspace or WORKSPACE).resolve()
    snapshot = json.loads((root / "source-snapshot.json").read_text(encoding="utf-8"))
    if snapshot.get("schema_version") != 2:
        raise ValueError("Use a newly prepared v2 workspace; preserve older ARCC snapshots unchanged.")
    entries = snapshot.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("The package has no file inventory.")
    checked = {}
    for entry in entries:
        relative = entry["destination"]
        if relative in checked:
            raise ValueError(f"Duplicate inventory entry: {relative}")
        path = package_path(root, relative)
        if not path.is_file():
            raise FileNotFoundError(f"Missing prepared file: {relative}")
        actual = sha256(path)
        if actual != entry["initial_copy_sha256"]:
            raise ValueError(f"Prepared file changed: {relative}; prepare a new snapshot.")
        checked[relative] = actual
    required = {NOTEBOOK, "run_encoder_swaps.py", "Code/shared_code/fusion/swap_plan.py",
                "Code/shared_code/fusion/result_sets.py"}
    if not required <= checked.keys():
        raise ValueError(f"Missing execution inventory: {sorted(required - checked.keys())}")
    if not any(p.startswith("datasets/") for p in checked):
        raise ValueError("No preserved dataset inputs in the inventory.")
    if not any(p.startswith("historical_results/Code/") for p in checked):
        raise ValueError("No fixed historical selection inputs in the inventory.")
    return snapshot, checked


def overlaps(first, second):
    first, second = Path(first).resolve(), Path(second).resolve()
    return first.is_relative_to(second) or second.is_relative_to(first)


def validate_destinations(workspace, output_root, cache_root, files, purpose):
    """Keep publication inside its workspace and away from copied inputs/caches."""
    workspace, output_root = Path(workspace).resolve(), Path(output_root).resolve()
    if output_root == workspace or not output_root.is_relative_to(workspace):
        raise ValueError("Output root must be a separate directory inside the prepared workspace.")
    if overlaps(output_root, cache_root):
        raise ValueError("Outputs and embedding caches must not overlap.")
    protected = ("Code", "datasets", "historical_results", "evidence")
    if any(overlaps(output_root, workspace / p) for p in protected):
        raise ValueError("Outputs overlap a protected input/source directory.")
    if any(overlaps(output_root, package_path(workspace, p)) for p in files):
        raise ValueError("Outputs overlap a prepared file.")
    if purpose == "local_pilot" and overlaps(output_root, workspace / "outputs"):
        raise ValueError("Local pilots must use a separate destination, such as pilot_outputs.")
    if purpose == "newton" and overlaps(output_root, workspace / "pilot_outputs"):
        raise ValueError("Newton results must stay separate from local pilot outputs.")


def validate_extension(workspace, output_root, cache_root, new_cache_root, files):
    """An extension may write new RNA caches, never the previously used cache tree."""
    workspace = Path(workspace).resolve()
    output_root, new_cache_root = Path(output_root).resolve(), Path(new_cache_root).resolve()
    if not new_cache_root.is_relative_to(workspace) or new_cache_root == workspace:
        raise ValueError("New RNA caches must live inside the separate prepared workspace.")
    if any(overlaps(new_cache_root, root) for root in (output_root, cache_root)):
        raise ValueError("New RNA caches must be separate from outputs and existing caches.")
    if any(overlaps(new_cache_root, workspace / name)
           for name in ("Code", "datasets", "historical_results", "evidence")):
        raise ValueError("New RNA caches overlap a protected source directory.")
    if any(overlaps(new_cache_root, package_path(workspace, name)) for name in files):
        raise ValueError("New RNA caches overlap a prepared file.")
    if output_root in {workspace / "outputs", workspace / "pilot_outputs"}:
        raise ValueError("The E. coli extension needs its own output subdirectory.")


def acquire_locks(stack, directory, benchmarks, seeds):
    """OS locks release after failure; duplicate cooperative writers fail promptly."""
    directory.mkdir(parents=True, exist_ok=True)
    for benchmark in sorted(benchmarks):
        for seed in sorted(seeds):
            handle = stack.enter_context((directory / f"{benchmark}-seed{seed}.lock").open("a+b"))
            try:
                if os.name == "nt":
                    import msvcrt
                    if handle.seek(0, 2) == 0:
                        handle.write(b"\0")
                        handle.flush()
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise RuntimeError(f"Another writer holds {benchmark}, seed {seed}.") from error


def save_record(path, record):
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(record, indent=2) + "\n")
    os.replace(temporary, path)


class NotebookExecution:
    """Execution paths and producer checks; no scientific model configuration."""

    def __init__(self, config):
        self.config = config
        self.root = Path(config["workspace"]).resolve()
        self.output = Path(config["output_root"]).resolve()
        self.snapshot, self.files = verify_preserved_inputs(self.root)
        validate_destinations(self.root, self.output, config["cache_root"],
                              self.files, config["purpose"])
        self.experiment = config.get("experiment", "full")
        if self.experiment != self.snapshot.get("experiment", "full"):
            raise ValueError("Requested experiment differs from the prepared snapshot.")
        if self.experiment == "ecoli_ernie":
            if config["benchmarks"] != ["ecoli_proteins"]:
                raise ValueError("The E. coli extension cannot run another benchmark.")
            validate_extension(self.root, self.output, config["cache_root"],
                               config["new_cache_root"], self.files)
            if any(overlaps(config["reference_root"], path)
                   for path in (self.root, self.output, config["new_cache_root"])):
                raise ValueError("The completed reference workspace must remain separate and read-only.")
        self.snapshot_sha = sha256(self.root / "source-snapshot.json")
        self.device = config["device"]

    def observe_kernel(self, torch):
        """Check the executing kernel, not merely the submission shell's request."""
        if self.config["mode"] in {"train", "extract"}:
            if self.config["purpose"] == "newton" and not os.environ.get("SLURM_JOB_ID"):
                raise RuntimeError("Newton training requires an allocated Slurm job.")
            if self.device == "cuda":
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA unavailable; refusing CPU fallback.")
                actual = torch.cuda.get_device_name(0)
                expected = self.config["expected_gpu"]
                if not expected or expected.casefold() not in actual.casefold():
                    raise RuntimeError(f"Observed GPU {actual!r} does not match {expected!r}.")
            elif self.config["purpose"] == "newton":
                raise RuntimeError("Newton production fits require CUDA.")
        import importlib.metadata as metadata
        packages = {}
        for name in ("numpy", "pandas", "scipy", "scikit-learn", "torch", "nbconvert", "nbclient"):
            try:
                packages[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                packages[name] = None
        self.environment = {
            "host": platform.node(), "python": sys.version, "platform": platform.platform(),
            "device": self.device, "gpu": torch.cuda.get_device_name(0) if self.device == "cuda" else None,
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "packages": packages,
        }
        launch = Path(self.config["launch_record"])
        record = json.loads(launch.read_text(encoding="utf-8"))
        record["kernel_environment"] = self.environment
        save_record(launch, record)
        print("Kernel environment:", json.dumps(self.environment), flush=True)

    def result_paths(self):
        from shared_code.fusion.result_sets import SwapResultSet, historical_originals
        historical = self.root / "historical_results/Code"
        analysis = self.output / "phaseB/analyses/paired-report-v1"
        return SwapResultSet(
            name=self.config["purpose"], root=self.root,
            originals={key: [self.output / "phaseA" / key] for key in BENCHMARKS},
            swaps=self.output / "phaseB", selection_originals=historical_originals(historical),
            selection_swaps=historical / "stage4/results/phaseB",
            saved_report_dirs=(analysis,), analysis_dir=analysis,
            fixed_historical_selection=True, imported=False,
        )

    def require_producer(self, record):
        if not isinstance(record, dict) or not isinstance(record.get("producer"), dict):
            raise ValueError("Result/checkpoint has no producer record.")
        context = record["producer"].get("arcc_execution")
        if not isinstance(context, dict):
            raise ValueError("Result/checkpoint has no execution cohort record.")
        expected = {"source_snapshot_sha256": self.snapshot_sha, "purpose": self.config["purpose"],
                    "output_root": str(self.output)}
        if any(context.get(key) != value for key, value in expected.items()):
            raise ValueError("Result/checkpoint belongs to another execution cohort; preserve it.")
        if context.get("experiment", "full") != self.experiment:
            raise ValueError("Result/checkpoint belongs to another experiment.")
        if not context.get("launch_record") or not context.get("selection_source_sha256"):
            raise ValueError("Result/checkpoint lacks launch or fixed-selection evidence.")
        if context.get("selection_seed") != 0:
            raise ValueError("The historical selection seed must remain zero.")
        sources = context["selection_source_sha256"]
        if not isinstance(sources, dict) or any(
            not isinstance(name, str) or not name.startswith("historical_results/Code/")
            or self.files.get(name) != digest
            for name, digest in sources.items()
        ):
            raise ValueError("Historical selection hashes do not match the prepared inputs.")
        try:
            if any(sha256(package_path(self.root, name)) != digest for name, digest in sources.items()):
                raise ValueError("Historical selection evidence changed after package verification.")
        except OSError as error:
            raise ValueError("Historical selection evidence is no longer readable.") from error
        name = context["launch_record"]
        if not isinstance(name, str) or Path(name).name != name or "/" in name or "\\" in name:
            raise ValueError("Launch evidence must name a file inside this output's logs.")
        try:
            launch = json.loads((self.output / "logs" / name).read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError("Referenced launch evidence is missing or invalid.") from error
        if not isinstance(launch, dict):
            raise ValueError("Referenced launch evidence must be an object.")
        settings = launch.get("configuration", {})
        if not isinstance(settings, dict):
            raise ValueError("Referenced launch has no configuration.")
        if (launch.get("snapshot_sha256") != self.snapshot_sha
                or settings.get("purpose") != self.config["purpose"]
                or settings.get("output_root") != str(self.output)
                or settings.get("workspace") != str(self.root)
                or settings.get("experiment", "full") != self.experiment
                or settings.get("mode") != "train"
                or launch.get("status") not in {"running", "completed", "failed_or_interrupted"}):
            raise ValueError("Launch evidence does not describe this training cohort.")
        environment = context.get("kernel_environment")
        if not isinstance(environment, dict) or environment != launch.get("kernel_environment"):
            raise ValueError("Observed kernel environment is missing or differs from its launch.")
        if self.config["purpose"] == "newton":
            gpu = environment.get("gpu")
            expected_gpu = settings.get("expected_gpu")
            if (environment.get("device") != "cuda" or not isinstance(gpu, str)
                    or not isinstance(expected_gpu, str) or not expected_gpu
                    or expected_gpu.casefold() not in gpu.casefold()):
                raise ValueError("Newton fit lacks matching observed CUDA/GPU evidence.")
            if (self.config["mode"] == "train" and self.config["expected_gpu"].casefold()
                    not in gpu.casefold()):
                raise ValueError("Completed fit used a different GPU from the requested cohort.")
        if record.get("benchmark") is not None and record["benchmark"] not in settings.get("benchmarks", []):
            raise ValueError("Fit benchmark differs from its launch.")
        seed = record.get("config", {}).get("seed")
        if seed is not None and seed != settings.get("seed"):
            raise ValueError("Fit seed differs from its launch.")

    def check_results(self):
        for key in self.config["benchmarks"]:
            paths = list((self.output / "phaseA" / key).glob("*.json"))
            paths += list((self.output / "phaseB" / key).glob("*/*.json"))
            for path in paths:
                try:
                    self.require_producer(json.loads(path.read_text(encoding="utf-8")))
                except (ValueError, TypeError, AttributeError) as error:
                    raise ValueError(f"Unsupported existing result {path}: {error}") from error

    def checkpoint_preflight(self, path, requested):
        import torch
        from shared_code.fusion import train
        # These are trusted local recovery files; torch.load is not an untrusted-file reader.
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.require_producer(payload)
        del payload
        train.checkpoint_preflight(path, requested)

    def producer(self, plan):
        from shared_code.fusion import run_identity
        producer = run_identity.producer_record(device=self.device, repository=self.root)
        producer["arcc_execution"] = {
            "purpose": self.config["purpose"], "output_root": str(self.output),
            "experiment": self.experiment,
            "launch_record": Path(self.config["launch_record"]).name,
            "source_snapshot_sha256": self.snapshot_sha, "selection_seed": 0,
            "selected_single_variants": plan.selected_singles,
            "selection_source_sha256": {
                Path(path).relative_to(self.root).as_posix(): sha256(path)
                for path in plan.selection_sources
            },
            "kernel_environment": self.environment,
        }
        return producer


def prepare_notebook(notebook, config):
    """Inject paths/mode after Setup settings; never replace a scientific cell."""
    import copy
    import nbformat
    notebook = copy.deepcopy(notebook)
    indices = [i for i, cell in enumerate(notebook.cells) if cell.get("id") == "swaps-settings"]
    if len(indices) != 1:
        raise ValueError("Expected one maintained swaps-settings cell.")
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
            cell.metadata.pop("execution", None)
    source = "\n".join([
        "# Launcher-supplied execution boundary; scientific settings remain above.",
        "import sys", "from pathlib import Path",
        f"_execution_config = {config!r}",
        "sys.path.insert(0, _execution_config['workspace'])",
        "from run_encoder_swaps import NotebookExecution",
        "EXECUTION = NotebookExecution(_execution_config)",
        "EXECUTION.observe_kernel(torch)", "EXECUTION.check_results()",
        "EXPERIMENT = _execution_config.get('experiment', 'full')",
        "READ_SAVED_RESULTS = _execution_config['mode'] not in ('train', 'extract')",
        "RUN_TRAINING = _execution_config['mode'] == 'train'",
        "RUN_SAVED_COMPARISONS = _execution_config['mode'] == 'report'",
        "RUN_BOOTSTRAP = False", "CACHE_REUSE_ONLY = True",
        "if _execution_config['mode'] == 'extract': CACHE_REUSE_ONLY = False",
        "SEED = _execution_config['seed']", "BENCHMARKS = tuple(_execution_config['benchmarks'])",
        "DEVICE = EXECUTION.device", "CACHE_ROOT = Path(_execution_config['cache_root'])",
        "NEW_CACHE_ROOT = (Path(_execution_config['new_cache_root']) if EXPERIMENT == 'ecoli_ernie' else None)",
        "if EXPERIMENT == 'ecoli_ernie':",
        "    SWAPS = [('rna', 'rnaernie'), ('rna', 'ernierna')]",
        "    STABILITY_SWAPS = []",
        "    REFERENCE_SWAP_ROOT = Path(_execution_config['reference_root']) / 'outputs/phaseB'",
        "RESULT_PATHS = EXECUTION.result_paths()", "RESULTS_ROOT = RESULT_PATHS.swaps",
        "ORIGINAL_RESULTS = RESULT_PATHS.originals", "COMPARISON_DIR = RESULT_PATHS.analysis_dir",
        "CHECKPOINT_VALIDATOR = EXECUTION.checkpoint_preflight",
        "FIT_PRODUCER = lambda benchmark: EXECUTION.producer(FIT_PLANS[benchmark])",
        "print('Execution:', _execution_config['purpose'], BENCHMARKS, 'seed', SEED, 'outputs:', EXECUTION.output)",
    ])
    notebook.cells.insert(indices[0] + 1, nbformat.v4.new_code_cell(source, id="execution-settings"))
    return notebook


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--mode", choices=("train", "report", "extract"))
    parser.add_argument("--experiment", choices=EXPERIMENTS,
                        help="Must match the separately prepared snapshot; defaults to its experiment")
    parser.add_argument("--benchmark", choices=(*BENCHMARKS, "all"))
    parser.add_argument("--seed", type=int, choices=(0, 1, 2))
    parser.add_argument("--kernel")
    parser.add_argument("--purpose", choices=("newton", "local_pilot"), default="newton")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--cache-root", type=Path, default=WORKSPACE / "Code/stage3/embeddings")
    parser.add_argument("--new-cache-root", type=Path,
                        help="New RNA caches inside the extension workspace; existing caches stay read-only")
    parser.add_argument("--reference-root", type=Path,
                        help="Completed Newton swap workspace, read-only reference for the extension")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--expected-gpu", default=os.environ.get("ARCC_EXPECTED_GPU", ""))
    args = parser.parse_args(argv)
    snapshot, files = verify_preserved_inputs()
    if args.verify_only:
        print(f"Verified {len(files)} prepared files; no kernel, cache arrays or computation started.")
        return
    experiment = args.experiment or snapshot.get("experiment", "full")
    if experiment not in EXPERIMENTS or experiment != snapshot.get("experiment", "full"):
        parser.error("Experiment must match the prepared snapshot.")
    if experiment == "ecoli_ernie" and args.benchmark is None:
        args.benchmark = "ecoli_proteins"
    if experiment == "ecoli_ernie" and args.benchmark != "ecoli_proteins":
        parser.error("The E. coli extension only supports ecoli_proteins.")
    if experiment == "ecoli_ernie" and args.mode == "report":
        parser.error("The extension has no selected-single/three-seed report; read its saved fit table.")
    if args.mode == "extract" and experiment != "ecoli_ernie":
        parser.error("Extraction is limited to the explicitly prepared E. coli extension.")
    if args.mode == "extract" and args.seed is not None:
        parser.error("RNA extraction is shared across training seeds; omit --seed.")
    if experiment == "ecoli_ernie" and args.mode == "train" and args.seed is None:
        args.seed = 0
    if experiment == "ecoli_ernie" and args.reference_root is None:
        parser.error("The extension requires --reference-root for completed Newton comparison evidence.")
    if not args.mode or not args.benchmark or not args.kernel:
        parser.error("Execution requires --mode, --benchmark and --kernel.")
    if args.mode in {"train", "extract"}:
        if args.benchmark == "all" or (args.mode == "train" and args.seed is None):
            parser.error("Train one explicit benchmark and seed per invocation.")
        if args.purpose == "newton" and not os.environ.get("SLURM_JOB_ID"):
            parser.error("Newton training requires an allocated Slurm job.")
        if args.purpose == "local_pilot" and args.device is None:
            parser.error("A local pilot requires an explicit --device and separate output destination.")
        device = args.device or "cuda"
        if args.purpose == "newton" and device != "cuda":
            parser.error("Newton production fits require CUDA.")
        if device == "cuda" and not args.expected_gpu.strip():
            parser.error("Supply --expected-gpu from an observed preflight.")
    else:
        if args.seed is not None or args.device == "cuda":
            parser.error("Reporting uses CPU and all three seeds; omit --seed and --device cuda.")
        device = "cpu"
    default_output = WORKSPACE / ("outputs" if args.purpose == "newton" else "pilot_outputs")
    if experiment == "ecoli_ernie":
        default_output /= "ecoli_ernie"
    output = (args.output_root or default_output).expanduser().resolve()
    cache = args.cache_root.expanduser().resolve()
    validate_destinations(WORKSPACE, output, cache, files, args.purpose)
    new_cache = (args.new_cache_root or WORKSPACE / "extension_embeddings").expanduser().resolve()
    if experiment == "ecoli_ernie":
        validate_extension(WORKSPACE, output, cache, new_cache, files)
        reference = args.reference_root.expanduser().resolve()
        if not (reference / "outputs/phaseB/ecoli_proteins/rinalmo").is_dir():
            parser.error("--reference-root lacks the completed Newton E. coli/RiNALMo directory.")
        if overlaps(reference, WORKSPACE):
            parser.error("Use a separate extension workspace; preserve the completed Newton reference workspace.")
    elif args.new_cache_root is not None:
        parser.error("--new-cache-root belongs only to the E. coli extension.")
    else:
        reference = None
    benchmarks = BENCHMARKS if args.benchmark == "all" else (args.benchmark,)
    config = {"workspace": str(WORKSPACE), "output_root": str(output), "cache_root": str(cache),
              "mode": args.mode, "purpose": args.purpose, "device": device,
              "experiment": experiment, "new_cache_root": str(new_cache) if experiment == "ecoli_ernie" else None,
              "reference_root": str(reference) if reference is not None else None,
              "benchmarks": list(benchmarks), "seed": args.seed or 0, "expected_gpu": args.expected_gpu}
    import nbformat
    with ExitStack() as stack:
        seeds = (args.seed,) if args.mode == "train" else (0, 1, 2)
        acquire_locks(stack, output / ".locks", benchmarks, seeds)
        if args.mode == "extract":
            acquire_locks(stack, new_cache / ".locks", benchmarks, (0,))
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        name = f"{args.mode}-{args.benchmark}-seed{args.seed}-{stamp}-{uuid.uuid4().hex[:8]}"
        logs, notebooks = output / "logs", output / "notebooks"
        logs.mkdir(parents=True, exist_ok=True)
        notebooks.mkdir(parents=True, exist_ok=True)
        record_path, executed_path = logs / f"{name}.json", notebooks / f"{name}.ipynb"
        config["launch_record"] = str(record_path)
        record = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
                  "configuration": config, "snapshot_sha256": sha256(WORKSPACE / "source-snapshot.json"),
                  "source_commit": snapshot.get("source_commit"), "execution_source_sha256": files,
                  "selection": ("historical seed-0 MIL penalty only; new single baseline not selected"
                                if experiment == "ecoli_ernie" else "preserved historical seed-0 validation choices"),
                  "host": platform.node(), "launcher_python": sys.version,
                  "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                  "executed_notebook": str(executed_path.relative_to(WORKSPACE))}
        notebook = prepare_notebook(nbformat.read(WORKSPACE / NOTEBOOK, as_version=4), config)
        save_record(record_path, record)
        # Even explicit extraction uses staged pinned weights; no network fallback.
        for key, value in {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                           "TOKENIZERS_PARALLELISM": "false"}.items():
            previous = os.environ.get(key)
            stack.callback(lambda k=key, p=previous: os.environ.pop(k, None) if p is None
                           else os.environ.__setitem__(k, p))
            os.environ[key] = value
        print(f"Starting {args.mode}: {benchmarks}, seed {args.seed}; {executed_path}", flush=True)
        error = None
        try:
            processor = LiveOutputPreprocessor(timeout=None, kernel_name=args.kernel,
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
                # The kernel adds observed environment evidence to this same launch record.
                record = json.loads(record_path.read_text(encoding="utf-8"))
                record.update(status="failed_or_interrupted" if error else "completed",
                              ended_utc=datetime.now(timezone.utc).isoformat())
                if error:
                    record["error"] = f"{type(error).__name__}: {error}"
                save_record(record_path, record)
            except Exception as caught:
                failures.append(caught)
            if failures:
                if error is None:
                    raise RuntimeError("Could not save execution evidence") from failures[0]
                if hasattr(error, "add_note"):
                    error.add_note(f"Saving execution evidence also failed: {failures}")


if __name__ == "__main__":
    main()
