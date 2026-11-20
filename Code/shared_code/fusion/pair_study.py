"""The partner-task panel: which arms are fitted, and how a fit is identified and saved.

``pair_training`` fits one arm. This module is the layer above it, and it exists for the same
reason ``region_training`` does: a notebook should declare a panel and read saved evidence,
not hold the logic that decides whether a result on disk answers today's question.

A saved fit carries its own identity: the table it was fitted on, the split that produced its
rows, the token caches both sides were read from, the architecture, the optimizer settings and
the seed. A reader that finds a result whose identity differs from the one it asked for reports
it rather than using it, and never overwrites it, because that file may be the evidence a
finished comparison was computed from.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from . import pair_tasks, pair_training

SCHEMA = "pair-study.v1"

# The declared panel. Two single-partner baselines establish what one side alone can do; the
# four fusion arms are what has to beat them. Every arm reads the same rows.
PANEL = ("left_only", "right_only", "concat", "bilinear", "cross_attention",
         "interleaved_mamba")

# Optional arms, kept out of the required panel: the gate variants of cross-attention, which
# answer where a gate has to sit rather than whether fusion helps at all.
OPTIONAL = {"cross_attention__sequence_gate": ("cross_attention", {"gate": "sequence"}),
            "cross_attention__position_gate": ("cross_attention", {"gate": "position"}),
            "interleaved_mamba__right_left": ("interleaved_mamba", {"order": "right_left"})}


def arm_of(key):
    """The scorer and model settings one panel key stands for."""
    if key in PANEL:
        return key, {}
    if key in OPTIONAL:
        arm, kwargs = OPTIONAL[key]
        return arm, dict(kwargs)
    raise ValueError(f"{key!r} is not in the panel or the optional arms")


def panel_keys(*, include_optional=False):
    return list(PANEL) + (sorted(OPTIONAL) if include_optional else [])


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def table_record(frame):
    """What the table is, independent of how it was split or fitted."""
    pairs = frame.pairs
    return {"name": frame.name, "kind": frame.kind,
            "modalities": {side: frame.modality(side) for side in pair_tasks.SIDES},
            "pairs": int(len(pairs)),
            "entities": {side: int(len(frame.sequences(side)))
                         for side in pair_tasks.SIDES},
            "provenance": {str(k): int(v)
                           for k, v in pairs["provenance"].value_counts().items()},
            "pairs_sha256": _digest([pairs[column].tolist()
                                     for column in ("left", "right", "label", "provenance")])}


def split_record(split):
    """What the split is, by the rows it produced rather than by the rule that made it."""
    return {"held_out": split["held_out"], "seed": int(split["seed"]),
            "ratios": list(split["ratios"]), "dropped": int(split["dropped"]),
            "rows": {name: int(len(split["indices"][name])) for name in pair_tasks.SPLITS},
            "indices_sha256": {name: _digest(split["indices"][name].tolist())
                               for name in pair_tasks.SPLITS}}


def cache_record(summaries):
    """What each side was read from: the encoder, its revision, and the exact inputs."""
    return {side: {"modality": summary["modality"], "encoder": summary["encoder"],
                   "revision": summary["revision"], "width": summary["width"],
                   "token_limit": summary["token_limit"],
                   "entities": summary["entities"],
                   "at_token_limit": summary["at_token_limit"],
                   "input_sha256": summary["input_sha256"]}
            for side, summary in summaries.items()}


def study_identity(frame, split, key, config, summaries):
    """Everything that has to match for a saved fit to answer today's question."""
    arm, model_kwargs = arm_of(key)
    identity = {"schema": SCHEMA, "key": key, "arm": arm, "model_kwargs": model_kwargs,
                "table": table_record(frame), "split": split_record(split),
                "caches": cache_record(summaries), "config": config.as_record()}
    identity["sha256"] = _digest(identity)
    return identity


def result_path(root, frame, key, config, *, held_out):
    """Where one fit is saved: one file per table, split, arm and seed."""
    return (Path(root) / frame.name / f"held_out_{held_out}"
            / f"{key}__seed{config.seed}.json")


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name, suffix=".partial", delete=False)
    try:
        with handle:
            json.dump(payload, handle, indent=2, default=float)
            handle.write("\n")
        os.replace(handle.name, path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


def strip_model(record):
    """The saved form of a fit: everything but the live module ``fit`` hands back."""
    return {k: v for k, v in record.items() if k != "model"}


def save_result(path, record, identity, *, producer=None, controls=None):
    """Save one fit with the identity it was computed under."""
    payload = {"schema": SCHEMA, "identity": identity, "result": strip_model(record),
               "controls": controls or {}, "producer": producer or {}}
    _write_json(path, payload)
    return payload


def read_result(path, identity=None):
    """Read one saved fit, reporting rather than using a result that does not match.

    Returns ``(payload, issue)``. ``issue`` is ``None`` when the file answers the identity
    asked for, and otherwise names what differs. The producer record is never compared: it
    describes the machine, not the science.
    """
    path = Path(path)
    if not path.exists():
        return None, "missing"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return None, f"unreadable: {error}"
    if payload.get("schema") != SCHEMA:
        return None, f"unsupported schema {payload.get('schema')!r}"
    if identity is None:
        return payload, None
    stored = payload.get("identity", {})
    if stored.get("sha256") == identity["sha256"]:
        return payload, None
    differing = sorted(k for k in set(stored) | set(identity)
                       if k != "sha256" and stored.get(k) != identity.get(k))
    return payload, "incompatible: " + ", ".join(differing or ["sha256"])


def saved_inventory(root, frame, config, summaries, splits, *, include_optional=False):
    """One row per planned fit, saying whether its evidence is on disk and usable."""
    rows = []
    for held_out, split in splits.items():
        for key in panel_keys(include_optional=include_optional):
            identity = study_identity(frame, split, key, config, summaries)
            path = result_path(root, frame, key, config, held_out=held_out)
            payload, issue = read_result(path, identity)
            status = "saved" if issue is None else ("missing" if issue == "missing"
                                                   else "incompatible")
            row = {"table": frame.name, "held_out": held_out, "key": key,
                   "seed": config.seed, "status": status, "issue": issue,
                   "path": str(path)}
            if issue is None:
                row.update({f"test_{k}": v for k, v in payload["result"]["test"].items()})
                row["best_epoch"] = payload["result"]["best_epoch"]
            rows.append(row)
    return rows


def run_panel(frame, split, embeddings, config, summaries, root, *, device="cpu",
              keys=None, include_optional=False, controls=True, producer=None,
              log=print):
    """Fit every arm the panel asks for that is not already saved, and return every record.

    A fit whose evidence is on disk under the same identity is read, not repeated. A fit whose
    saved evidence differs is left alone and reported, so an accidental rerun cannot quietly
    replace the file a comparison was computed from.
    """
    keys = list(keys) if keys else panel_keys(include_optional=include_optional)
    results, reused, refused = {}, [], []
    for key in keys:
        arm, model_kwargs = arm_of(key)
        identity = study_identity(frame, split, key, config, summaries)
        path = result_path(root, frame, key, config, held_out=split["held_out"])
        payload, issue = read_result(path, identity)
        if issue is None:
            reused.append(key)
            results[key] = payload
            continue
        if issue.startswith("incompatible"):
            refused.append({"key": key, "path": str(path), "issue": issue})
            continue
        if log is not None:
            log(f"\n{frame.name} [{split['held_out']} held out] {key}")
        record = pair_training.fit(arm, frame, split, embeddings, config, device=device,
                                   model_kwargs=model_kwargs, log=log)
        scored = (pair_training.control_scores(record, frame, split, embeddings, config,
                                               device=device, seed=config.seed)
                  if controls else {})
        results[key] = save_result(path, record, identity, producer=producer,
                                   controls=scored)
        # A fitted arm is finished with once its scores are saved, and the record holds the
        # live module. On a unified-memory GPU the allocator keeps what that module used in
        # its own pool rather than returning it, so without this each arm leaves its
        # footprint behind and the panel runs out of memory partway down the list, on an
        # allocation far smaller than any one arm needed.
        del record
        pair_training.free_memory()
    if refused and log is not None:
        log(f"\n{len(refused)} saved fits do not match this identity and were left alone:")
        for entry in refused:
            log(f"  {entry['key']}: {entry['issue']}")
    return {"results": results, "reused": reused, "refused": refused}


def results_for_comparison(panel):
    """The saved records in the shape ``pair_training.comparison_rows`` reads."""
    return [{**payload["result"], "arm": key} for key, payload in panel["results"].items()]


def panel_table(panel, frame):
    """One row per fitted arm: its test metrics, its controls, and what it read."""
    rows = []
    for key, payload in panel["results"].items():
        result, controls = payload["result"], payload.get("controls", {})
        row = {"key": key, "arm": result["arm"], "best_epoch": result["best_epoch"],
               "selection_metric": result["selection_metric"]}
        row.update({f"test_{k}": v for k, v in result["test"].items()})
        degree = controls.get("degree", {})
        metric = {"binding": "auroc", "affinity": "spearman"}[frame.kind]
        row[f"degree_{metric}"] = degree.get(metric)
        reassigned = controls.get("partner_reassigned")
        if reassigned:
            row["real_minus_reassigned"] = (reassigned["mean_score_real"]
                                            - reassigned["mean_score_reassigned"])
        rows.append(row)
    return rows


def token_limit_note(summaries):
    """One sentence a write-up can quote about how much of each side was read."""
    parts = []
    for side in pair_tasks.SIDES:
        summary = summaries[side]
        parts.append(f"{summary['modality']} through {summary['encoder']}: "
                     f"{summary['at_token_limit']} of {summary['entities']} entities "
                     f"({summary['at_token_limit_share']:.1%}) reached the "
                     f"{summary['token_limit']}-token limit")
    return "; ".join(parts)


def degree_gap(panel, frame):
    """How far the best fusion arm is above the degree baseline on the selection metric.

    The baseline reads no sequence at all, so an arm that does not clear it has not shown that
    it read the pair; this is the first number to look at, before any arm-to-arm comparison.

    On a split that holds out *both* sides the baseline has no training information about
    either entity of a test pair, so it predicts the same value for every row. That is the
    design working rather than a failure, but it means the baseline carries no signal there:
    a constant prediction scores 0.5 on a binding table and has no Spearman at all. This
    reports the baseline as undefined in that case rather than as a number to clear.
    """
    metric = {"binding": "auroc", "affinity": "spearman"}[frame.kind]
    scores = {key: payload["result"]["test"][metric]
              for key, payload in panel["results"].items()
              if payload["result"]["test"].get(metric) is not None}
    degrees = [payload.get("controls", {}).get("degree", {}).get(metric)
               for payload in panel["results"].values()]
    degrees = [float(d) for d in degrees if d is not None and np.isfinite(d)]
    if not scores:
        return None
    best = max(scores, key=lambda key: scores[key])
    record = {"metric": metric, "best_arm": best, "best": scores[best],
              "degree": None, "gap": None, "degree_defined": bool(degrees)}
    if degrees:
        record["degree"] = float(np.mean(degrees))
        record["gap"] = scores[best] - record["degree"]
    return record
