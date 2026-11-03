"""Plan missing encoder-swap fits from saved JSON metadata, without computing them.

Selection stays at seed 0 while the caller chooses one training seed. The plan
includes the same-seed original references needed to compare every replacement.
Existing files are never modified. Only an explicitly supplied checkpoint
validator may deserialize a checkpoint. This is a preflight snapshot, not a lock
or proof of unrecorded input/producer identity.
"""

from dataclasses import dataclass, field
import math
from pathlib import Path

from . import swap_results


ORIGINAL_VARIANTS = (
    "unimodal:dna", "unimodal:rna", "unimodal:prot", "unimodal-grid:rna",
    "concat", "mil", "cross",
)
CONFIG_FIELDS = {
    "lr", "weight_decay", "batch_size", "max_epochs", "early_stopping_patience",
    "plateau_patience", "entropy_lambda", "channels", "tconv", "seed",
}


@dataclass
class RunSpec:
    benchmark: str
    role: str
    encoder_key: str | None
    modality: str | None
    variant: str
    config: dict
    encoders: dict
    destination: Path
    status: str
    reason: str
    source_path: Path | None = None
    legacy_identity: bool = False

    @property
    def result_path(self):
        """Read the actual compatible historical file, or publish at the new destination."""
        return self.source_path if self.source_path is not None else self.destination

    @property
    def path(self):
        """The path to read for reuse, or the canonical result tag to train/resume."""
        return self.result_path


@dataclass
class ExecutionPlan:
    benchmark: str
    training_seed: int
    selection_seed: int
    selection_lambda: float | None
    runs: list[RunSpec] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    @property
    def ready(self):
        return not self.issues and bool(self.runs) and all(
            run.status in {"reuse", "new", "resume"} for run in self.runs
        )

    @property
    def lambda_(self):
        return self.selection_lambda

    def require_ready(self):
        """Fail before caches/models are opened; do not execute a partially blocked plan."""
        problems = [*self.issues, *(f"{run.variant}: {run.reason}"
                                  for run in self.runs if run.status not in {"reuse", "new", "resume"})]
        if not self.ready:
            raise ValueError(f"{self.benchmark}: execution preflight blocked: "
                             + "; ".join(problems or ["no planned fits"]))
        return self.runs

    def rows(self):
        """Small display records; paths identify reuse and publication separately."""
        return [{
            "benchmark": run.benchmark, "role": run.role, "encoder": run.encoder_key or "original",
            "variant": run.variant, "seed": run.config["seed"],
            "lambda": run.config["entropy_lambda"], "status": run.status,
            "reason": run.reason, "source": str(run.source_path or ""),
            "destination": str(run.destination), "legacy identity": run.legacy_identity,
        } for run in self.runs]


def result_path(directory, benchmark, variant, config):
    """The shared train.run filename convention, including one explicit training seed."""
    tag = (f"{benchmark}__{variant.replace(':', '-')}__lambda{config['entropy_lambda']}"
           f"__ch{config['channels']}__{config['tconv']}__seed{config['seed']}")
    return Path(directory) / (tag + ".json")


def _validate_request(benchmark, config, selection_seed, original_encoders, swaps, penalties):
    if not isinstance(benchmark, str) or not benchmark or any(c in benchmark for c in "/\\"):
        raise ValueError("Benchmark must be one nonempty path component")
    if set(config) != CONFIG_FIELDS:
        raise ValueError("Pass the full TrainConfig dictionary, without missing or additional fields")
    if type(config["seed"]) is not int or config["seed"] not in (0, 1, 2):
        raise ValueError("Choose one training seed: 0, 1, or 2")
    if type(selection_seed) is not int or selection_seed != 0:
        raise ValueError("The retained MIL selection seed must remain 0")
    if config["tconv"] != "equation" or config["entropy_lambda"] != 0.0:
        raise ValueError("The swap base recipe requires equation convolution and zero base penalty")
    for name in ("batch_size", "max_epochs", "early_stopping_patience", "plateau_patience", "channels"):
        if type(config[name]) is not int or config[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("lr", "weight_decay"):
        value = config[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if config["lr"] == 0:
        raise ValueError("lr must be positive")
    if set(original_encoders) != set(swap_results.MODALITIES):
        raise ValueError("Original encoders must name exactly the DNA, RNA, and protein slots")
    for key in original_encoders.values():
        if not isinstance(key, str) or not key or any(c in key for c in "/\\"):
            raise ValueError("Encoder keys must be nonempty path components")
    if (not penalties or len(set(penalties)) != len(penalties)
            or any(type(p) not in (int, float) or not math.isfinite(p) or p <= 0 for p in penalties)):
        raise ValueError("Positive MIL penalties must be distinct finite positive numbers")
    seen = set()
    for modality, encoder, grid in swaps:
        if modality not in original_encoders or type(grid) is not bool:
            raise ValueError("Each swap needs a known modality and an explicit boolean grid requirement")
        if not isinstance(encoder, str) or not encoder or any(c in encoder for c in "/\\"):
            raise ValueError("Replacement encoder must be one nonempty path component")
        if encoder == original_encoders[modality] or encoder in seen:
            raise ValueError("Replacement encoder keys must be distinct and actually replace their slot")
        seen.add(encoder)


def _plan_fit(*, benchmark, role, encoder_key, modality, variant, config, encoders,
              folders, destination_directory, native_fallback, checkpoint_validator):
    destination = result_path(destination_directory, benchmark, variant, config).resolve()
    run = RunSpec(benchmark, role, encoder_key, modality, variant, dict(config), dict(encoders),
                  destination, "blocked", "Inspection did not complete")
    try:
        report = swap_results.read_results(
            folders, benchmark=benchmark, config=config, encoders=encoders,
            native_fallback=native_fallback, mil_lambda=config["entropy_lambda"],
            variants=(variant,),
        )
        if report.duplicates:
            run.reason = "Ambiguous compatible results; preserve all copies: " + "; ".join(report.duplicates)
            return run
        entry = report.records[0] if report.records else None
        # A malformed/mismatched canonical target cannot be silently replaced or ignored
        # in favor of a different historical filename.
        if destination.exists() and (entry is None or entry["path"].resolve() != destination):
            detail = next((i["reason"] for i in report.issues
                           if Path(i["source"]).resolve() == destination), "not the unique compatible result")
            run.reason = "Canonical destination already exists; preserve it: " + detail
            return run
        if entry is not None:
            run.status = "reuse"
            run.source_path = entry["path"]
            run.legacy_identity = entry["legacy"]
            run.reason = ("Compatible saved score record; reuse this actual path without a new fit. "
                          + ("Historical encoder/input identity is incomplete."
                             if entry["legacy"] else "Recorded encoder mapping matches."))
            return run
        for suffix, explanation in (
            (".weights.pt", "Final weights exist without a reusable completed result"),
            (".json.partial", "A partial result publication requires separate recovery review"),
        ):
            artifact = destination.with_suffix(suffix)
            if artifact.exists():
                run.reason = explanation + "; preserve " + str(artifact)
                return run
        checkpoint = destination.with_suffix(".checkpoint")
        if checkpoint.exists():
            if checkpoint_validator is None:
                run.status = "checkpoint_requires_review"
                run.reason = "Checkpoint not deserialized in metadata mode; compatibility preflight required."
            else:
                requested = {"config": dict(config), "variant": variant, "benchmark": benchmark}
                if checkpoint_validator(checkpoint, requested) is not None:
                    raise ValueError("Checkpoint validator must return None on success")
                run.status = "resume"
                run.reason = "Checkpoint passed the supplied schema/recorded-recipe preflight; resume this fit."
            return run
        # Do not create directories while planning, including to probe writability.
        ancestor = destination.parent
        while not ancestor.exists() and ancestor != ancestor.parent:
            ancestor = ancestor.parent
        if not ancestor.is_dir():
            run.reason = "Destination has a file where a directory is required: " + str(ancestor)
            return run
        run.status = "new"
        run.reason = "No compatible completed result or recovery artifact; one new fit is needed."
    except (OSError, ValueError, TypeError, KeyError) as error:
        run.reason = "Metadata inspection failed; preserve existing evidence: " + str(error)
    return run


def _check_selection_singles(plan, *, config, original_encoders, original_folders, swaps, swap_root):
    """Repeats hold the seed-0 single-encoder selection fixed for every triple."""
    selection_config = {**config, "seed": plan.selection_seed}
    required_originals = ORIGINAL_VARIANTS[:4]
    try:
        report = swap_results.read_results(
            original_folders, benchmark=plan.benchmark, config=selection_config,
            encoders=original_encoders, native_fallback=True, variants=required_originals,
        )
        originals = {entry["data"]["variant"] for entry in report.require_unique()}
        missing = set(required_originals) - originals
        if missing:
            raise ValueError("missing " + ", ".join(sorted(missing)))
    except (OSError, ValueError, TypeError, KeyError) as error:
        plan.issues.append("Seed-0 original single-encoder selection unavailable: " + str(error))
    for modality, encoder_key, grid in swaps:
        required = (f"unimodal:{modality}", *((f"unimodal-grid:{modality}",) if grid else ()))
        try:
            report = swap_results.read_results(
                (swap_root / encoder_key,), benchmark=plan.benchmark, config=selection_config,
                encoders={**original_encoders, modality: encoder_key}, variants=required,
            )
            available = {entry["data"]["variant"] for entry in report.require_unique()}
            missing = set(required) - available
            if missing:
                raise ValueError("missing " + ", ".join(sorted(missing)))
        except (OSError, ValueError, TypeError, KeyError) as error:
            plan.issues.append(f"Seed-0 single-encoder selection unavailable for {encoder_key}: {error}")


def plan_benchmark(*, benchmark, config, original_encoders, original_folders,
                   original_destination, swaps, swap_root, selection_seed=0,
                   positive_lambdas=(0.01, 0.5, 1.0), checkpoint_validator=None):
    """Plan all references and selected swaps for one manually chosen training seed.

    ``swaps`` contains (modality, encoder_key, needs_grid) triples. ``swap_root``
    is this benchmark's Phase B directory; each replacement uses its own child.
    Reuse may point into another original folder or to a native appendix fallback.
    The caller must skip train.run for reuse rows and consume ``source_path``.
    Every benchmark plan should pass require_ready before any costly operation.
    With no checkpoint_validator, checkpoint-only work remains blocked. A compute
    caller can supply a validator(path, requested_identity) that returns None only
    after checking the checkpoint schema and recorded recipe; it must not train.
    """
    config, original_encoders = dict(config), dict(original_encoders)
    swaps, penalties = tuple(swaps), tuple(positive_lambdas)
    _validate_request(benchmark, config, selection_seed, original_encoders, swaps, penalties)
    original_destination, swap_root = Path(original_destination), Path(swap_root)
    # Include the publication location when references have historically lived elsewhere.
    original_folders = tuple(dict.fromkeys(
        Path(p).resolve() for p in (*original_folders, original_destination)
    ))
    plan = ExecutionPlan(benchmark, config["seed"], selection_seed, None)
    try:
        selection = swap_results.read_results(
            original_folders, benchmark=benchmark, config={**config, "seed": selection_seed},
            encoders=original_encoders, variants=("mil",), mil_lambdas=penalties,
        )
        records = selection.require_unique()
        available = {entry["data"]["config"]["entropy_lambda"] for entry in records}
        if available != set(penalties):
            raise ValueError("Complete seed-0 positive MIL grid required; missing penalties "
                             + str(sorted(set(penalties) - available)))
        plan.selection_lambda = selection.positive_lambda()
    except (OSError, ValueError, TypeError, KeyError) as error:
        plan.issues.append("Seed-0 MIL selection unavailable: " + str(error))
        return plan
    if config["seed"] != selection_seed:
        _check_selection_singles(
            plan, config=config, original_encoders=original_encoders,
            original_folders=original_folders, swaps=swaps, swap_root=swap_root,
        )
    for variant in ORIGINAL_VARIANTS:
        recipe = {**config, "entropy_lambda": plan.selection_lambda if variant == "mil" else 0.0}
        plan.runs.append(_plan_fit(
            benchmark=benchmark, role="original", encoder_key=None,
            modality=variant.split(":")[1] if ":" in variant else None, variant=variant,
            config=recipe, encoders=original_encoders, folders=original_folders,
            destination_directory=original_destination, native_fallback=True,
            checkpoint_validator=checkpoint_validator,
        ))
    for modality, encoder_key, grid in swaps:
        mapping = {**original_encoders, modality: encoder_key}
        directory = swap_root / encoder_key
        for variant in swap_results.swap_variants(modality, grid=grid):
            recipe = {**config, "entropy_lambda": plan.selection_lambda if variant == "mil" else 0.0}
            plan.runs.append(_plan_fit(
                benchmark=benchmark, role="replacement", encoder_key=encoder_key,
                modality=modality, variant=variant,
                config=recipe, encoders=mapping, folders=(directory,),
                destination_directory=directory, native_fallback=False,
                checkpoint_validator=checkpoint_validator,
            ))
    return plan
