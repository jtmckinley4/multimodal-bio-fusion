"""Read compatible Newton BLF baselines in place; never create replacement fits."""

from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from . import run_identity, swap_results, train


BUNDLES = {
    "original": {"dna": "ntv2_100m", "rna": "rnafm", "prot": "esm2_8m"},
    "mrnafm": {"dna": "ntv2_100m", "rna": "mrnafm", "prot": "esm2_8m"},
}


def read_reference(root, benchmark, bundle, model, seed, config, *,
                   streams=None, cache_records=None):
    """Return one verified saved result, or None when that exact fit is absent.

    Root contains the original ``outputs/phaseA`` and ``outputs/phaseB`` trees.
    A single uses only its named modality: unchanged DNA/protein singles can be
    shared with the replacement bundle. Current input/cache verification requires
    streams and cache_records together. A reading-only call validates recorded
    identity, but does not imply that present inputs have been checked.
    """
    if bundle not in BUNDLES or model not in swap_results.VARIANTS:
        raise ValueError("Unknown encoder bundle or reference model.")
    if benchmark == "mrna_stability":
        raise ValueError("Published-split stability references cannot serve the grouped study.")
    if (streams is None) != (cache_records is None):
        raise ValueError("Supply current streams and cache records together.")
    recipe = asdict(config) if is_dataclass(config) else dict(config)
    if type(seed) is not int or seed < 0 or recipe.get("seed") != seed:
        raise ValueError("Reference request and training seed differ.")
    encoders = BUNDLES[bundle]
    single = model.startswith("unimodal")
    modalities = (model.split(":", 1)[1],) if single else tuple(BUNDLES["original"])
    original = bundle == "original" or (single and modalities[0] != "rna")
    mapping = BUNDLES["original"] if original else encoders
    folder = Path(root) / "outputs" / ("phaseA" if original else "phaseB") / benchmark
    if not original:
        folder /= "mrnafm"
    # Inspect only the requested seed/model. A malformed candidate is an error;
    # it must not look like permission to replace the completed fit.
    paths = []
    for path in sorted(folder.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("variant") != model or data.get("config", {}).get("seed") != seed:
            continue
        if model == "mil" and data["config"].get("entropy_lambda") != recipe["entropy_lambda"]:
            continue
        paths.append(path)
    if not paths:
        return None
    selected = swap_results.read_results(
        [folder], benchmark=benchmark, config=recipe, encoders=mapping,
        native_fallback=single and model.startswith("unimodal:"),
        variants=(model,), mil_lambda=recipe["entropy_lambda"],
    ).require_unique()
    if len(selected) != 1:
        raise ValueError(f"Existing {model} seed {seed} reference is incompatible; preserve it.")
    entry = selected[0]
    record, path = entry["data"], entry["path"]
    signature = run_identity.recorded_blf_inputs(record)
    issue = swap_results.prediction_issue(record, task=signature["targets"]["task"])
    if issue:
        raise ValueError(f"Reference predictions: {issue}")
    dtype = np.int64 if signature["targets"]["task"] == "classification" else np.float32
    if run_identity.array_record(record["test_truth"], dtype) != signature["split_targets"]["test"]:
        raise ValueError("Reference test targets differ from their recorded row identity.")
    if streams is not None:
        if streams.key != benchmark or streams.encoders != encoders:
            raise ValueError("Reference request differs from the current stream bundle.")
        requested = run_identity.blf_identity(
            rows=streams.rows, benchmark=benchmark, variant=model, config=record["config"],
            encoders=streams.encoders, caches=cache_records, task=streams.task,
            split_column=streams.split_column,
        )
        for name in ("rows", "targets", "splits", "split_targets"):
            if signature[name] != requested["inputs"][name]:
                raise ValueError(f"Reference {name} differ from current inputs.")
        actual_model, wanted_model = record["identity"]["model"], requested["model"]
        for name in set(wanted_model) - {"widths"}:
            if actual_model.get(name) != wanted_model[name]:
                raise ValueError(f"Reference model {name} differs from current settings.")
        for modality in modalities:
            if (signature["encoders"][modality] != encoders[modality]
                    or actual_model["widths"][modality] != wanted_model["widths"][modality]
                    or record["identity"]["inputs"]["caches"][modality] != cache_records[modality]):
                raise ValueError(f"Reference {modality} cache or encoder differs.")
        if record["identity"]["loader"] != requested["loader"]:
            raise ValueError("Reference loader settings differ.")
    result = train.Result.load(path)
    result.extra_fields["reference_source"] = {
        "path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "current_inputs_verified": streams is not None,
    }
    return result


def collect_reference_diagnostics(root, benchmark, bundle, seed, config, *, streams,
                                  cache_records, output_stem, device, producer):
    """Explicit MIL best-weight inference with its original identity; never retrain.

    Missing fit/weight evidence is a hold. The new diagnostic record names the
    unchanged source fit and captures the inference producer separately.
    """
    from . import controlled_experiment
    from .heads import TEMPERATURE_RANGE
    result = read_reference(root, benchmark, bundle, "mil", seed, config,
                            streams=streams, cache_records=cache_records)
    if result is None:
        raise FileNotFoundError("The requested MIL reference is missing; diagnostics cannot create a fit.")
    settings = result.extra_fields["identity"]["model"]["kwargs"]
    gate_settings = {"projection_width": settings["out_width"],
                     "scorer_width": settings["attention_width"],
                     "shared_scorer": settings["shared_projection"],
                     "projection_activation": "tanh", "projection_norm": "none",
                     "temperature_range": TEMPERATURE_RANGE}
    return controlled_experiment.collect_diagnostics(
        streams, "sequence_softmax", config, gate_settings, cache_records,
        result_path=result.extra_fields["reference_source"]["path"], output_stem=output_stem,
        device=device, producer=producer, reference_result=result,
    )
