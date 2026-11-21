"""Execute one explicit study/benchmark/seed in a verified, separate Newton snapshot."""

from contextlib import ExitStack
from datetime import datetime, timezone
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import uuid


WORKSPACE = Path(__file__).resolve().parent
STUDIES = {"correction": "anchor_correction", "gates": "gate_diagnostics",
           "attention": "anchor_cross_attention", "regions": "region_aware_fusion"}
PANELS = {
    "correction": ("cov_vac_degradation", "fungal_expression", "mrna_stability"),
    "gates": ("cov_vac_degradation", "ecoli_proteins"),
    "attention": ("cov_vac_degradation", "ecoli_proteins"),
    "regions": ("rnahl-human", "translation-efficiency-human"),
}


def support_module():
    path = WORKSPACE / "arcc_execution_support.py"
    if not path.exists():
        path = WORKSPACE.parent / "encoder_swaps/run_encoder_swaps.py"
    spec = importlib.util.spec_from_file_location("arcc_execution_support", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


support = support_module()


def verify_package(root=WORKSPACE):
    root = Path(root).resolve()
    snapshot = json.loads((root / "source-snapshot.json").read_text(encoding="utf-8"))
    if snapshot.get("schema") != "fusion-studies.package.v1":
        raise ValueError("Use a new fusion-studies package, not an old execution snapshot.")
    files = {}
    for item in snapshot.get("files", []):
        name = item["destination"]
        if name.casefold() in {p.casefold() for p in files}:
            raise ValueError("Duplicate package destination.")
        path = support.package_path(root, name)
        if not path.is_file() or support.sha256(path) != item["sha256"]:
            raise ValueError(f"Missing or changed prepared source: {name}")
        files[name] = item["sha256"]
    required = {f"Code/stage4/{name}.ipynb" for name in STUDIES.values()}
    required.update(("run_fusion_study.py", "arcc_execution_support.py",
                     "Code/shared_code/fusion/train.py"))
    if not required <= files.keys():
        raise ValueError(f"Incomplete preparation: {sorted(required - files.keys())}")
    return snapshot, files


def observe_kernel(torch, configuration):
    """Observe the actual kernel/GPU before any study can open inputs or fit."""
    preparation = configuration["action"] == "prepare"
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("An allocated Slurm job is required for this execution recipe.")
    if not preparation and not torch.cuda.is_available():
        raise RuntimeError("An allocated Slurm CUDA GPU is required; no CPU fallback.")
    gpu = None if preparation else torch.cuda.get_device_name(0)
    if not preparation and configuration["expected_gpu"].casefold() not in gpu.casefold():
        raise RuntimeError(f"Unexpected GPU: {gpu}")
    from shared_code.fusion import run_identity
    producer = run_identity.producer_record(device="cpu" if preparation else "cuda", repository=configuration["workspace"])
    producer["fusion_study_execution"] = {
        "snapshot_sha256": configuration["snapshot_sha256"],
        "study": configuration["study"], "benchmark": configuration["benchmark"],
        "seed": configuration["seed"], "action": configuration["action"],
        "gpu": gpu, "cuda": torch.version.cuda, "host": platform.node(),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "launch_record": configuration["launch_record"],
    }
    if configuration.get("slice_size") is not None:
        producer["fusion_study_execution"]["slice_size"] = configuration["slice_size"]
    path = Path(configuration["launch_record"])
    record = json.loads(path.read_text(encoding="utf-8"))
    record["kernel_producer"] = producer
    support.save_record(path, record)
    print(f"Observed GPU: {gpu}; {configuration['study']} {configuration['action']}", flush=True)
    return producer


def prepare_notebook(notebook, configuration):
    import nbformat
    result = copy.deepcopy(notebook)
    settings = [i for i, cell in enumerate(result.cells)
                if "fusion-study-settings" in cell.get("metadata", {}).get("tags", [])]
    if len(settings) != 1:
        raise ValueError("Expected one fusion-study-settings cell.")
    for cell in result.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    lines = [f"_execution = {configuration!r}",
             "sys.path.insert(0, _execution['workspace'])",
             "from run_fusion_study import observe_kernel",
             "PRODUCER = observe_kernel(torch, _execution)",
             "DEVICE = 'cpu' if _execution['action'] == 'prepare' else 'cuda'",
             "READ_SAVED_RESULTS = _execution['action'] == 'diagnostics'",
             "RUN_PREPARATION = _execution['action'] == 'prepare'",
             "RUN_TRAINING = _execution['action'] == 'train'",
             "RUN_EXTRACTION = _execution['action'] == 'extract'",
             "RUN_DIAGNOSTICS = _execution['action'] == 'diagnostics'",
             "RUN_ANALYSIS = False", "SEED = _execution['seed']",
             "CONFIG = replace(BASE_CONFIG, seed=SEED)",
             "BENCHMARK = _execution['benchmark']", "DATASET_KEY = BENCHMARK",
             "RESULTS_ROOT = Path(_execution['output_root']) / _execution['study']",
             "DATA_ROOT = Path(_execution['data_root'])",
             "CACHE_ROOT = Path(_execution['cache_root'])",
             "REFERENCE_ROOT = Path(_execution['reference_root'])",
             "GROUPED_SPLIT_PATH = Path(_execution['grouped_split'])",
             "GROUPED_REFERENCE_ROOT = GROUPED_SPLIT_PATH.parent",
             "DATA_CACHE_ROOT = Path(_execution['region_data_root'])",
             "TOKEN_CACHE_ROOT = Path(_execution['region_cache_root'])",
             "if _execution.get('fit_keys') is not None:",
             "    FIT_KEYS = _execution['fit_keys']",
             "if _execution['study'] == 'regions':",
             "    BATCH_SLICE_SIZE = _execution.get('slice_size')",
             "print('Explicit job:', BENCHMARK, 'seed', SEED, _execution['action'], flush=True)"]
    result.cells.insert(settings[0] + 1, nbformat.v4.new_code_cell(
        "\n".join(lines), id="newton-study-execution"))
    return result


def preflight_notebook(configuration):
    """Check the selected kernel and imports without opening research inputs."""
    import nbformat
    setup = [
        "import importlib", "import importlib.metadata as metadata", "import json",
        "from pathlib import Path", "import sys",
        f"_execution = {configuration!r}",
        "sys.path.insert(0, str(Path(_execution['workspace']) / 'Code'))",
        "sys.path.insert(0, _execution['workspace'])",
        "import numpy as np", "import torch",
        "from run_fusion_study import observe_kernel",
        "PRODUCER = observe_kernel(torch, _execution)",
    ]
    checks = [
        "packages = {'numpy': 'numpy', 'pandas': 'pandas', 'scipy': 'scipy',",
        "    'sklearn': 'scikit-learn', 'matplotlib': 'matplotlib', 'Bio': 'biopython',",
        "    'torch': 'torch', 'transformers': 'transformers', 'multimolecule': 'multimolecule',",
        "    'einops': 'einops', 'pyarrow': 'pyarrow', 'ipykernel': 'ipykernel',",
        "    'nbconvert': 'nbconvert', 'nbformat': 'nbformat', 'nbclient': 'nbclient'}",
        "for module in packages:",
        "    importlib.import_module(module)",
        "from shared_code.fusion import (correction_experiment, controlled_experiment,",
        "    region_experiment, region_training)",
        "versions = {distribution: metadata.version(distribution) for distribution in packages.values()}",
        "print('Kernel packages:', json.dumps(versions, indent=2), flush=True)",
        "x = torch.ones((32, 32), device='cuda')",
        "answer = x @ x",
        "torch.cuda.synchronize()",
        "if answer.sum().item() != 32768:",
        "    raise RuntimeError('The synthetic CUDA arithmetic check failed.')",
        "del answer, x",
        "print('Preflight passed: imports and synthetic GPU arithmetic only; '",
        "      'no research data, caches, encoder weights or training were opened.', flush=True)",
    ]
    return nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_code_cell("\n".join(setup), id="preflight-environment"),
        nbformat.v4.new_code_cell("\n".join(checks), id="preflight-imports-and-arithmetic"),
    ])


def validate_configuration(configuration, files):
    study, benchmark, action = (configuration[name] for name in ("study", "benchmark", "action"))
    if study not in PANELS or benchmark not in PANELS[study]:
        raise ValueError("This benchmark is outside the declared study panel.")
    if action not in ("prepare", "train", "extract", "diagnostics"):
        raise ValueError("Unknown execution action.")
    if action in ("prepare", "extract") and study != "regions":
        raise ValueError("Existing-benchmark studies reuse frozen caches; extraction is A6 only.")
    if type(configuration["seed"]) is not int or configuration["seed"] not in (0, 1, 2):
        raise ValueError("Select one training seed from 0, 1, 2.")
    if study == "correction" and action != "train":
        raise ValueError("Correction reports are separate local analyses, not diagnostic jobs.")
    if action in ("prepare", "extract") and configuration["seed"] != 0:
        raise ValueError("Frozen A6 cache extraction is seed-independent; prepare it once with seed 0.")
    keys = configuration.get("fit_keys")
    if keys is not None and (study != "regions" or action not in ("train", "diagnostics")
                            or not keys or len(set(keys)) != len(keys)):
        raise ValueError("An explicit unique fit subset is supported only for A6 fits/diagnostics.")
    slice_size = configuration.get("slice_size")
    if slice_size is not None and (
        type(slice_size) is not int or slice_size != 4
        or (study, benchmark, action, configuration["seed"]) != ("regions", "rnahl-human", "train", 0)
        or keys != ["rnafm__native__attention_cds__true"]
    ):
        raise ValueError("--slice-size 4 requires the single native RNA-FM attention fit for human half-life seed 0.")
    if action != "prepare" and not configuration["expected_gpu"].strip():
        raise ValueError("Declare the GPU name confirmed by preflight.")
    output = configuration["output_root"]
    support.validate_destinations(configuration["workspace"], output,
                                  configuration["cache_root"], files, "newton")
    if slice_size is not None:
        original = Path(configuration["workspace"]) / "outputs"
        existing_paths = (*STUDIES, "logs", "notebooks", ".locks")
        if any(support.overlaps(output, original / name) for name in existing_paths):
            raise ValueError("Use a separate --output-root for the sliced fit to preserve existing evidence.")
    for name in ("data_root", "reference_root", "grouped_split", "region_data_root", "region_cache_root"):
        if support.overlaps(output, configuration[name]):
            raise ValueError(f"Outputs overlap {name}; preserve inputs and completed results.")
    if support.overlaps(configuration["region_cache_root"], configuration["cache_root"]):
        raise ValueError("A6 token caches must be separate from existing BLF caches.")
    if study == "regions":
        cache = configuration["region_cache_root"]
        for name in ("data_root", "reference_root", "grouped_split", "region_data_root"):
            if support.overlaps(cache, configuration[name]):
                raise ValueError(f"A6 cache destination overlaps protected {name}.")
        source_root = Path(configuration["workspace"]) / "Code"
        if support.overlaps(cache, source_root) or any(
                support.overlaps(cache, support.package_path(Path(configuration["workspace"]), name))
                for name in files):
            raise ValueError("A6 cache destination overlaps prepared scientific source.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inspection = parser.add_mutually_exclusive_group()
    inspection.add_argument("--verify-only", action="store_true")
    inspection.add_argument("--preflight", action="store_true",
                            help="Check imports and a tiny CUDA calculation in the selected kernel; no research inputs.")
    parser.add_argument("--study", choices=STUDIES)
    parser.add_argument("--benchmark")
    parser.add_argument("--seed", type=int, choices=(0, 1, 2))
    parser.add_argument("--action", choices=("prepare", "train", "extract", "diagnostics"), default="train")
    parser.add_argument("--kernel")
    parser.add_argument("--fit-key", action="append", dest="fit_keys",
                        help="One declared A6 core fit key; repeat for a bounded subset.")
    parser.add_argument("--slice-size", type=int, choices=(4,),
                        help="GPU slice size for the native RNA-FM attention retry at half-life seed 0.")
    parser.add_argument("--expected-gpu", default=os.environ.get("ARCC_EXPECTED_GPU", ""))
    for name in ("data-root", "cache-root", "reference-root", "grouped-split", "region-data-root", "region-cache-root"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--output-root", type=Path, default=WORKSPACE / "outputs")
    args = parser.parse_args(argv)
    if args.verify_only and args.slice_size is not None:
        parser.error("--slice-size requires a training job.")
    snapshot, files = verify_package()
    if args.verify_only:
        print(f"Verified {len(files)} source files; no kernel, data, cache or model was opened.")
        return
    if args.preflight:
        scientific_options = (args.study, args.benchmark, args.seed, args.fit_keys, args.slice_size,
                              args.data_root, args.cache_root, args.reference_root,
                              args.grouped_split, args.region_data_root, args.region_cache_root)
        if any(value is not None for value in scientific_options) or args.action != "train":
            parser.error("Preflight is separate from study, seed, action and research-input arguments.")
        if not args.kernel or not args.expected_gpu.strip():
            parser.error("Preflight requires --kernel and --expected-gpu explicitly.")
    elif not args.study or not args.benchmark or args.seed is None or not args.kernel:
        parser.error("Select --study, --benchmark, --seed and --kernel explicitly.")
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Research execution requires an allocated Slurm job.")
    configuration = vars(args).copy()
    configuration.pop("verify_only")
    configuration.pop("preflight")
    if args.preflight:
        configuration.update(study="preflight", benchmark="environment", action="preflight")
    configuration["workspace"] = str(WORKSPACE)
    needed = (set() if args.preflight else
              {"region_data_root", "region_cache_root"} if args.study == "regions" else
              {"data_root", "cache_root"})
    if args.study in ("gates", "attention") or (args.study == "correction" and args.benchmark != "mrna_stability"):
        needed.add("reference_root")
    if args.benchmark == "mrna_stability":
        needed.add("grouped_split")
    for name in ("data_root", "cache_root", "reference_root", "grouped_split", "region_data_root", "region_cache_root", "output_root"):
        if configuration[name] is None:
            if name in needed:
                parser.error(f"Declare --{name.replace('_', '-')} explicitly.")
            configuration[name] = WORKSPACE / "unused" / name
        configuration[name] = str(configuration[name].expanduser().resolve())
    if args.preflight:
        support.validate_destinations(WORKSPACE, configuration["output_root"],
                                      configuration["cache_root"], files, "newton")
    else:
        validate_configuration(configuration, files)
    output = Path(configuration["output_root"])
    with ExitStack() as stack:
        support.acquire_locks(stack, output / ".locks",
            ("preflight",) if args.preflight else (f"{args.study}-{args.benchmark}",),
            (0,) if args.preflight else (args.seed,))
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        prefix = "preflight" if args.preflight else f"{args.study}-{args.benchmark}-seed{args.seed}"
        name = f"{prefix}-{stamp}-{uuid.uuid4().hex[:8]}"
        logs, notebooks = output / "logs", output / "notebooks"
        logs.mkdir(parents=True, exist_ok=True)
        notebooks.mkdir(parents=True, exist_ok=True)
        record_path, executed_path = logs / f"{name}.json", notebooks / f"{name}.ipynb"
        configuration["launch_record"] = str(record_path)
        configuration["snapshot_sha256"] = support.sha256(WORKSPACE / "source-snapshot.json")
        record = {"status": "running", "configuration": configuration,
                  "started_utc": datetime.now(timezone.utc).isoformat(),
                  "execution_source_sha256": files, "slurm_job_id": os.environ["SLURM_JOB_ID"]}
        support.save_record(record_path, record)
        notebook = None
        error = None
        try:
            import nbformat
            if args.preflight:
                notebook = preflight_notebook(configuration)
            else:
                notebook = prepare_notebook(nbformat.read(
                    WORKSPACE / "Code/stage4" / f"{STUDIES[args.study]}.ipynb", as_version=4), configuration)
            # Training always stays offline. A6 extraction may read already staged
            # encoder weights; acquisition is a separate reviewed preparation step.
            os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
            processor = support.LiveOutputPreprocessor(timeout=None, kernel_name=args.kernel,
                allow_errors=False, record_timing=True)
            processor.preprocess(notebook, {"metadata": {"path": str(WORKSPACE)}})
        except BaseException as caught:
            error = caught
            raise
        finally:
            try:
                if notebook is not None:
                    nbformat.write(notebook, executed_path)
                record = json.loads(record_path.read_text(encoding="utf-8"))
                record.update(status="failed_or_interrupted" if error else "completed",
                              ended_utc=datetime.now(timezone.utc).isoformat())
                if error:
                    record["error"] = f"{type(error).__name__}: {error}"
                support.save_record(record_path, record)
            except Exception as saving_error:
                if error is None:
                    raise
                error.add_note(f"Saving execution evidence also failed: {saving_error}")


if __name__ == "__main__":
    main()
