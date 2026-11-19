"""Synthetic grouped-stability protocol checks; no research data or fitting.

Fixtures use invented sequences, cache descriptors and saved predictions. They
exercise grouping, identity and reuse without opening a real token cache or
executing a notebook, model or bootstrap analysis.
"""

from copy import deepcopy
from contextlib import ExitStack, redirect_stdout
from dataclasses import asdict
import hashlib
import itertools
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np  # Keep NumPy before PyTorch in the Windows project environment.
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code"))
from shared_code import datasets, encoders
from shared_code.fusion import grouped_stability as grouped
from shared_code.fusion import blf, run_identity, swap_reporting, token_embeddings as tokens, train


def fixture_rows():
    """Distinct nine-base strings plus repeated observations with different labels."""
    sequences = ["ATG" + "".join(bases) + "A"
                 for bases in itertools.islice(itertools.product("ACGT", repeat=5), 150)]
    sequences.extend([sequences[0], sequences[0].replace("T", "U"), sequences[10]])
    rows = pd.DataFrame({
        "sequence": sequences,
        "length": [len(value) for value in sequences],
        "Value": [float(index % 17) / 7 for index in range(len(sequences))],
        "Split": ["train", "val", "test"] * (len(sequences) // 3),
    }, index=np.arange(len(sequences)) * 3 + 5)
    return rows


def recipe():
    return train.TrainConfig(tconv="equation", seed=0)


def cache_descriptors(rows, mapping):
    """Metadata sufficient for identity checks; no arrays or files are involved."""
    sequences = rows["sequence"].tolist()
    return {
        key: {
            "encoder": key, "width": 8, "n_sequences": len(rows),
            "input_sha256": tokens.input_fingerprint(
                tokens.inputs_for_rows(encoders.ENCODERS[key], sequences)),
            "revision": "synthetic", "shards": [
                {"name": "shard_00000.npz", "sha256": "b" * 64}],
        }
        for modality, key in mapping.items()
    }


def saved_result(rows, split_record, spec, seed=0, val_score=.2, test_score=.3):
    config = grouped.config_for(spec, recipe(), seed)
    identity = grouped.make_identity(rows, split_record, spec, config,
                                     cache_descriptors(rows, spec.encoders))
    return {
        "variant": spec.variant, "benchmark": "mrna_stability",
        "config": asdict(config), "identity": identity,
        "epochs": 2, "best_epoch": 0, "val_score": val_score,
        "test_score": test_score, "seconds": 1,
        "test_truth": split_record["test_truth"],
        "test_prediction": [float(value) + .1 for value in split_record["test_truth"]],
    }


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def notebook_cells():
    notebook = json.loads((ROOT / "Code/stage4/grouped_stability.ipynb").read_text(encoding="utf-8"))
    return {cell["id"]: "".join(cell["source"])
            for cell in notebook["cells"] if cell["cell_type"] == "code"}


class GroupedSplitTests(unittest.TestCase):
    def setUp(self):
        self.rows = fixture_rows()
        self.before = self.rows.copy(deep=True)
        self.grouped_rows, self.record = grouped.build_split(
            self.rows, source_sha256="a" * 64)

    def test_canonical_identity_preserves_synonymous_sequence_distinction(self):
        self.assertEqual(grouped.canonical_sequence("  augaaa\n"), "ATGAAA")
        self.assertEqual(grouped.canonical_sequence("ATGAAA"), "ATGAAA")
        self.assertNotEqual(grouped.canonical_sequence("ATGAAA"),
                            grouped.canonical_sequence("ATGAAG"))
        for invalid in ("", "  ", None, 3, "ATGN", "ATG AAA", "ATG-"):
            with self.subTest(invalid=invalid), self.assertRaises((TypeError, ValueError)):
                grouped.canonical_sequence(invalid)

    def test_groups_do_not_cross_partitions_and_rows_are_not_collapsed(self):
        frame = self.grouped_rows
        self.assertEqual(len(frame), len(self.rows))
        pd.testing.assert_frame_equal(self.rows, self.before)
        pd.testing.assert_frame_equal(frame[self.rows.columns], self.rows)
        self.assertEqual(set(frame[grouped.SPLIT_COLUMN]), {"train", "val", "test"})
        normalized = frame["sequence"].map(grouped.canonical_sequence)
        self.assertTrue((frame.groupby(normalized)[grouped.SPLIT_COLUMN].nunique() == 1).all())
        members = frame[normalized == normalized.iloc[0]]
        self.assertEqual(len(members), 3)
        self.assertGreater(members["Value"].nunique(), 1)

    def test_group_assignment_is_independent_of_order_targets_and_old_splits(self):
        altered = self.rows.iloc[::-1].copy()
        altered["Value"] = altered["Value"] + 100
        altered["Split"] = "test"
        reordered, record = grouped.build_split(altered, source_sha256="a" * 64)
        expected = self.grouped_rows[grouped.SPLIT_COLUMN].sort_index()
        pd.testing.assert_series_equal(reordered[grouped.SPLIT_COLUMN].sort_index(), expected)
        # Membership stays fixed, while the record must retain changed row/target evidence.
        self.assertNotEqual(record, self.record)

    def test_repeated_builds_match_but_a_new_source_or_seed_has_a_new_identity(self):
        same_rows, same = grouped.build_split(self.rows, source_sha256="a" * 64)
        pd.testing.assert_frame_equal(same_rows, self.grouped_rows)
        self.assertEqual(same, self.record)
        _, changed_source = grouped.build_split(self.rows, source_sha256="c" * 64)
        self.assertNotEqual(changed_source, self.record)
        _, changed_seed = grouped.build_split(self.rows, source_sha256="a" * 64, split_seed=43)
        self.assertNotEqual(changed_seed, self.record)

    def test_invalid_rows_or_empty_partitions_are_refused(self):
        for change in (lambda rows: rows.assign(sequence="ATGN"),
                       lambda rows: rows.assign(Value=np.nan),
                       lambda rows: rows.assign(length=1000)):
            with self.subTest(change=change), self.assertRaises((ValueError, TypeError)):
                grouped.build_split(change(self.rows), source_sha256="a" * 64)
        with self.assertRaises(ValueError):
            grouped.build_split(self.rows.iloc[:1], source_sha256="a" * 64)

    def test_split_publication_reuses_identical_bytes_and_refuses_a_new_record(self):
        with tempfile.TemporaryDirectory(prefix=".grouped-fixture-", dir=ROOT / "tests") as folder:
            path = Path(folder) / "split.json"
            self.assertTrue(grouped.publish_split(path, self.record))
            original = path.read_bytes()
            self.assertFalse(grouped.publish_split(path, deepcopy(self.record)))
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(grouped.read_split(path), self.record)
            _, another = grouped.build_split(self.rows, source_sha256="d" * 64)
            with self.assertRaises((ValueError, FileExistsError)):
                grouped.publish_split(path, another)
            self.assertEqual(path.read_bytes(), original)
            with self.assertRaises(ValueError):
                grouped.read_split(path, split_seed=43)

    def test_competing_split_writer_reuses_identical_assignment_only(self):
        _, other = grouped.build_split(self.rows, source_sha256="c" * 64)
        for winner, compatible in ((self.record, True), (other, False)):
            with self.subTest(compatible=compatible), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "split.json"

                def publish_other_then_fail(source, destination):
                    Path(destination).write_text(json.dumps(winner), encoding="utf-8")
                    raise FileExistsError("another seed published the split first")

                with patch.object(grouped.os, "link", side_effect=publish_other_then_fail):
                    if compatible:
                        self.assertFalse(grouped.publish_split(path, self.record))
                    else:
                        with self.assertRaises(ValueError):
                            grouped.publish_split(path, self.record)
                self.assertEqual(json.loads(path.read_text(encoding="utf-8")), winner)
                self.assertEqual(list(Path(folder).glob("*.partial")), [])

    def test_destination_preflight_accepts_another_seed_during_split_publication(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "results"
            cache_root = Path(folder) / "embeddings"
            destination = root / "split.json"

            def observe_publication_then_link(source, target):
                source = Path(source)
                self.assertTrue(source.name.startswith(".grouped-split-"))
                self.assertEqual(source.suffix, ".partial")
                self.assertTrue(source.is_file())
                self.assertFalse(destination.exists())
                # A second seed's directory preflight may happen in this interval.
                grouped.require_result_destination(root, cache_root)
                # Simulate exclusive publication; native hard-link behavior is
                # exercised separately by the immutable publication test.
                with Path(target).open("xb") as handle:
                    handle.write(source.read_bytes())

            with patch.object(grouped.os, "link", side_effect=observe_publication_then_link):
                self.assertTrue(grouped.publish_split(destination, self.record))
            self.assertEqual(grouped.read_split(destination), self.record)
            self.assertEqual(list(root.glob("*.partial")), [])

    def test_destination_refuses_unrelated_partial_files_and_lookalike_directories(self):
        for name, directory in (("unrelated.partial", False),
                                (".grouped-split-fixture.partial", True)):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                root = Path(folder) / "results"
                root.mkdir()
                unexpected = root / name
                if directory:
                    unexpected.mkdir()
                else:
                    unexpected.write_bytes(b"synthetic unrelated work")
                with self.assertRaises(ValueError):
                    grouped.require_result_destination(root, Path(folder) / "embeddings")
                self.assertTrue(unexpected.exists())

    def test_saved_manifest_checks_assignments_even_with_a_recomputed_fingerprint(self):
        changed = deepcopy(self.record)
        previous = changed["partitions"][0]
        changed["partitions"][0] = "test" if previous != "test" else "train"
        with self.assertRaises(ValueError):
            grouped.validate_split(changed)
        # A checksum detects byte changes; content validation must also refuse an
        # assignment that disagrees with the declared deterministic grouping rule.
        payload = {key: value for key, value in changed.items() if key != "fingerprint"}
        changed["fingerprint"] = hashlib.sha256(json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode()).hexdigest()
        with self.assertRaises(ValueError):
            grouped.validate_split(changed)


class FitPanelTests(unittest.TestCase):
    def test_panel_has_twelve_distinct_fits_and_thirty_six_seed_paths(self):
        panel = grouped.fit_panel()
        expected = {
            "ntv2_native", "ntv2_grid", "rnafm_native", "rnafm_grid",
            "mrnafm_native", "esm2_native", "original_concat", "original_mil",
            "original_cross", "mrnafm_concat", "mrnafm_mil", "mrnafm_cross",
        }
        self.assertEqual({spec.key for spec in panel}, expected)
        self.assertEqual(len(panel), 12)
        paths = {grouped.fit_path(Path("fixture"), spec, seed)
                 for spec in panel for seed in (0, 1, 2)}
        self.assertEqual(len(paths), 36)
        for path in paths:
            self.assertTrue(path.is_relative_to(Path("fixture") / "fits"))

    def test_auxiliary_encoders_are_fixed_and_shared_single_fits_are_unique(self):
        panel = grouped.fit_panel()
        dna_singles = [spec for spec in panel if spec.variant == "unimodal:dna"]
        protein_singles = [spec for spec in panel if spec.variant == "unimodal:prot"]
        self.assertEqual(len(dna_singles), 1)
        self.assertEqual(len(protein_singles), 1)
        for spec in panel:
            self.assertEqual(spec.encoders["dna"], "ntv2_100m")
            self.assertEqual(spec.encoders["prot"], "esm2_8m")
            self.assertIn(spec.encoders["rna"], ("rnafm", "mrnafm"))
            required = ((spec.variant.split(":")[1],) if ":" in spec.variant
                        else ("dna", "rna", "prot"))
            self.assertEqual(spec.modalities, required)
        copy = panel[0].encoders
        copy["dna"] = "unexpected"
        self.assertEqual(panel[0].encoders["dna"], "ntv2_100m")

    def test_existing_training_recipe_and_mil_penalty_are_preserved(self):
        base = recipe()
        original = asdict(base)
        for spec in grouped.fit_panel():
            for seed in (0, 1, 2):
                config = grouped.config_for(spec, base, seed)
                expected = {**original, "seed": seed,
                            "entropy_lambda": .01 if spec.variant == "mil" else 0.0}
                self.assertEqual(asdict(config), expected)
        self.assertEqual(asdict(base), original)
        for seed in (True, -1, 3, "0"):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                grouped.config_for(grouped.fit_panel()[0], base, seed)

    def test_run_identity_binds_the_grouped_partition_and_preserves_cache_metadata(self):
        rows, record = grouped.build_split(fixture_rows(), source_sha256="a" * 64)
        spec = next(spec for spec in grouped.fit_panel() if spec.key == "mrnafm_concat")
        config = grouped.config_for(spec, recipe(), 0)
        caches = cache_descriptors(rows, spec.encoders)
        before = deepcopy(caches)
        identity = grouped.make_identity(rows, record, spec, config, caches)
        run_identity.validate_identity(identity, config=asdict(config), variant=spec.variant)
        self.assertEqual(identity["inputs"]["encoders"], spec.encoders)
        for split in ("train", "val", "test"):
            indices = np.flatnonzero(rows[grouped.SPLIT_COLUMN].to_numpy() == split)
            self.assertEqual(identity["inputs"]["splits"][split],
                             run_identity.array_record(indices, np.int64))
        self.assertEqual(caches, before)
        different_rows, different_record = grouped.build_split(
            fixture_rows(), source_sha256="a" * 64, split_seed=43)
        different = grouped.make_identity(different_rows, different_record, spec, config, caches)
        self.assertNotEqual(identity, different)

    def test_identity_refuses_rows_that_disagree_with_the_saved_split_record(self):
        rows, record = grouped.build_split(fixture_rows(), source_sha256="a" * 64)
        spec = grouped.fit_panel()[0]
        caches = cache_descriptors(rows, spec.encoders)
        config = grouped.config_for(spec, recipe(), 0)
        altered = rows.copy()
        altered.loc[altered.index[0], "Value"] += 10
        with self.assertRaises(ValueError):
            grouped.make_identity(altered, record, spec, config, caches)
        altered = rows.copy()
        old = altered.iloc[0][grouped.SPLIT_COLUMN]
        altered.loc[altered.index[0], grouped.SPLIT_COLUMN] = (
            "test" if old != "test" else "train")
        with self.assertRaises(ValueError):
            grouped.make_identity(altered, record, spec, config, caches)


class SavedResultTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix=".grouped-fixture-", dir=ROOT / "tests")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows, self.record = grouped.build_split(fixture_rows(), source_sha256="a" * 64)

    def write(self, spec, seed=0, **changes):
        payload = saved_result(self.rows, self.record, spec, seed)
        payload.update(changes)
        path = grouped.fit_path(self.root, spec, seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_missing_results_remain_missing_without_fitting_or_loading_arrays(self):
        with patch.object(train, "run", side_effect=AssertionError("unexpected fit")), \
                patch.object(tokens, "summarize", side_effect=AssertionError("opened a cache")), \
                patch.object(np, "load", side_effect=AssertionError("opened token arrays")):
            records, issues = grouped.read_results(self.root, self.record, recipe())
        self.assertEqual(records, {})
        self.assertEqual(len(issues), 36)
        self.assertEqual({issue["issue"] for issue in issues}, {"missing"})
        self.assertEqual(list(self.root.iterdir()), [])

    def test_reader_accepts_complete_panel_and_does_not_modify_evidence(self):
        for spec in grouped.fit_panel():
            for seed in (0, 1, 2):
                self.write(spec, seed)
        before = {path: path.read_bytes() for path in self.root.rglob("*.json")}
        with patch.object(train, "run", side_effect=AssertionError("unexpected fit")), \
                patch.object(tokens, "summarize", side_effect=AssertionError("opened a cache")):
            records, issues = grouped.read_results(self.root, self.record, recipe())
        self.assertEqual(len(records), 36)
        self.assertEqual(issues, [])
        self.assertEqual({path: path.read_bytes() for path in before}, before)

    def test_reader_rejects_legacy_wrong_split_config_and_misaligned_targets(self):
        spec = grouped.fit_panel()[0]
        original = saved_result(self.rows, self.record, spec)
        cases = {}
        payload = deepcopy(original)
        payload.pop("identity")
        cases["legacy"] = payload
        payload = deepcopy(original)
        payload["identity"]["inputs"]["grouped_stability"]["split_fingerprint"] = "e" * 64
        cases["other split"] = payload
        payload = deepcopy(original)
        payload["config"]["seed"] = 1
        cases["different training seed"] = payload
        payload = deepcopy(original)
        payload["test_truth"][0] += 100
        cases["misaligned targets"] = payload
        payload = deepcopy(original)
        payload["test_prediction"][0] = float("nan")
        cases["invalid prediction"] = payload
        payload = deepcopy(original)
        payload["identity"]["inputs"]["grouped_stability"]["fit_key"] = "another_fit"
        cases["different panel entry"] = payload
        path = grouped.fit_path(self.root, spec, 0)
        path.parent.mkdir(parents=True)
        for reason, payload in cases.items():
            with self.subTest(reason=reason):
                path.write_text(json.dumps(payload), encoding="utf-8")
                before = path.read_bytes()
                records, issues = grouped.read_results(self.root, self.record, recipe(), seeds=(0,))
                self.assertNotIn((spec.key, 0), records)
                matching = [issue for issue in issues if issue["fit"] == spec.key]
                self.assertEqual(len(matching), 1)
                self.assertNotEqual(matching[0]["issue"], "missing")
                self.assertEqual(path.read_bytes(), before)

    def test_requested_validator_cannot_silently_promote_rejected_producer(self):
        spec = grouped.fit_panel()[0]
        path = self.write(spec)
        before = path.read_bytes()
        validator = unittest.mock.Mock(side_effect=ValueError("wrong execution cohort"))
        records, issues = grouped.read_results(self.root, self.record, recipe(),
                                              seeds=(0,), validator=validator)
        self.assertEqual(records, {})
        validator.assert_called_once()
        self.assertIn("wrong execution cohort", str(issues))
        self.assertEqual(path.read_bytes(), before)

    def test_reader_refuses_invalid_completion_epoch_and_score_metadata(self):
        spec = grouped.fit_panel()[0]
        original = saved_result(self.rows, self.record, spec)
        config = grouped.config_for(spec, recipe(), 0)
        invalid = {
            "epochs": (0, -1, 1.5, True, "2", None),
            "best_epoch": (-1, 2, 1.5, True, "0", None),
            "test_score": (float("nan"), float("inf"), -float("inf"), True, None, ".3"),
            "val_score": (float("nan"), float("inf"), -float("inf"), True, ".2"),
        }
        for field, values in invalid.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    payload = deepcopy(original)
                    payload[field] = value
                    with self.assertRaises(ValueError):
                        grouped.validate_result(payload, self.record, spec, config)
            with self.subTest(missing_field=field):
                payload = deepcopy(original)
                payload.pop(field)
                with self.assertRaises(ValueError):
                    grouped.validate_result(payload, self.record, spec, config)

    def test_none_validation_score_is_readable_but_cannot_select_a_baseline(self):
        spec = grouped.fit_panel()[0]
        payload = saved_result(self.rows, self.record, spec, val_score=None)
        config = grouped.config_for(spec, recipe(), 0)
        self.assertEqual(grouped.validate_result(payload, self.record, spec, config), payload)
        with self.assertRaises(ValueError):
            grouped.selection_and_comparisons({(spec.key, 0): {"data": payload}})


    def test_reader_rejects_changed_model_or_loader_under_same_training_config(self):
        spec = next(spec for spec in grouped.fit_panel() if spec.key == "original_cross")
        original = saved_result(self.rows, self.record, spec)
        cases = []
        payload = deepcopy(original)
        payload["identity"]["model"]["kwargs"]["n_heads"] = 8
        cases.append(payload)
        payload = deepcopy(original)
        payload["identity"]["loader"]["batch_size"] = 999
        cases.append(payload)
        payload = deepcopy(original)
        payload["identity"]["model"]["n_outputs"] = 2
        cases.append(payload)
        path = grouped.fit_path(self.root, spec, 0)
        path.parent.mkdir(parents=True)
        for index, payload in enumerate(cases):
            with self.subTest(case=index):
                path.write_text(json.dumps(payload), encoding="utf-8")
                records, issues = grouped.read_results(self.root, self.record, recipe(), seeds=(0,))
                self.assertNotIn((spec.key, 0), records)
                matching = [issue for issue in issues if issue["fit"] == spec.key]
                self.assertEqual(len(matching), 1)
                self.assertNotEqual(matching[0]["issue"], "missing")

    def test_trainer_reuses_completed_grouped_fit_and_refuses_changed_split_before_build(self):
        spec = grouped.fit_panel()[0]
        path = self.write(spec)
        saved = json.loads(path.read_text(encoding="utf-8"))
        before = path.read_bytes()
        config = grouped.config_for(spec, recipe(), 0)
        streams = SimpleNamespace(key="mrna_stability")
        changed_rows, changed_record = grouped.build_split(
            fixture_rows(), source_sha256="a" * 64, split_seed=43)
        changed_identity = grouped.make_identity(
            changed_rows, changed_record, spec, config,
            cache_descriptors(changed_rows, spec.encoders))
        with patch.object(train, "build", side_effect=AssertionError("constructed a model")), \
                patch.object(train, "fit", side_effect=AssertionError("trained a model")), \
                patch.object(tokens, "summarize", side_effect=AssertionError("opened cache")):
            result = train.run(streams, spec.variant, config, "cpu", path.parent,
                               tag=path.stem, log=None, require_predictions=True,
                               run_identity=saved["identity"])
            self.assertEqual(result.test_prediction, saved["test_prediction"])
            with self.assertRaises(ValueError):
                train.run(streams, spec.variant, config, "cpu", path.parent,
                          tag=path.stem, log=None, require_predictions=True,
                          run_identity=changed_identity)
        self.assertEqual(path.read_bytes(), before)


class SavedReportTests(unittest.TestCase):
    write = SavedResultTests.write

    def setUp(self):
        SavedResultTests.setUp(self)
        for spec in grouped.fit_panel():
            for seed in (0, 1, 2):
                self.write(spec, seed)
        self.records, issues = grouped.read_results(self.root, self.record, recipe())
        self.assertEqual(issues, [])

    def report(self):
        selected, comparisons = grouped.selection_and_comparisons(self.records)
        positions = np.flatnonzero(np.asarray(self.record["partitions"]) == "test")
        groups = np.asarray(self.record["groups"])[positions].tolist()
        identity = {
            "schema": "swap.paired-report.v1", "benchmark": "mrna_stability",
            "config": asdict(recipe()), "seeds": [0, 1, 2], "selection_seed": 0,
            "best_singles": selected, "mil_lambda": .01, "comparisons": comparisons,
            "context": grouped.report_context(self.record), "draws": 2000,
            "bootstrap_seed": 0, "task": "regression", "scoring_target_precision": "float32",
            "n_classes": None, "truth": run_identity.array_record(
                np.asarray(self.record["test_truth"], dtype=np.float32)),
            "groups_sha256": digest(groups),
            "sources": {f"/cluster/project/fits/{key}/seed{seed}.json": entry["sha256"]
                        for (key, seed), entry in self.records.items()},
        }
        analysis = {"seed_summary": [{"synthetic": True}], "issues": []}
        return {"identity": identity, "analysis": analysis, "analysis_sha256": digest(analysis)}

    def write_report(self, payload, name="fixture"):
        path = self.root / "reports" / f"mrna_stability__{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_saved_report_reads_portable_fit_bindings_without_rescoring(self):
        saved = self.report()
        path = self.write_report(saved)
        before = path.read_bytes()
        with patch.object(swap_reporting, "calculate", side_effect=AssertionError("rescored")), \
                patch.object(swap_reporting, "report", side_effect=AssertionError("rewrote report")):
            read, located = grouped.read_report(self.root, self.records, self.record, recipe())
        self.assertEqual(read, saved)
        self.assertEqual(located, path)
        self.assertEqual(path.read_bytes(), before)

    def test_other_grouping_metric_or_bootstrap_identity_is_not_reused(self):
        for field, value in (("groups_sha256", "e" * 64), ("task", "classification"),
                             ("draws", 10), ("bootstrap_seed", 99)):
            with self.subTest(field=field):
                saved = self.report()
                saved["identity"][field] = value
                path = self.write_report(saved)
                self.assertEqual(grouped.read_report(
                    self.root, self.records, self.record, recipe()), (None, None))
                path.unlink()

    def test_corrupt_analysis_or_fit_binding_is_refused_and_preserved(self):
        for kind in ("analysis", "source"):
            with self.subTest(kind=kind):
                saved = self.report()
                if kind == "analysis":
                    saved["analysis"]["seed_summary"] = []
                else:
                    key = next(iter(saved["identity"]["sources"]))
                    saved["identity"]["sources"][key] = "0" * 64
                path = self.write_report(saved)
                before = path.read_bytes()
                with self.assertRaises(ValueError):
                    grouped.read_report(self.root, self.records, self.record, recipe())
                self.assertEqual(path.read_bytes(), before)
                path.unlink()

    def test_ambiguous_reports_and_incomplete_panel_are_not_silently_selected(self):
        saved = self.report()
        self.write_report(saved, "first")
        self.write_report(saved, "second")
        with self.assertRaises(ValueError):
            grouped.read_report(self.root, self.records, self.record, recipe())
        incomplete = dict(self.records)
        incomplete.pop(("original_cross", 2))
        self.assertEqual(grouped.read_report(self.root, incomplete, self.record, recipe()), (None, None))



class SelectionTests(unittest.TestCase):
    def setUp(self):
        rows, record = grouped.build_split(fixture_rows(), source_sha256="a" * 64)
        self.records = {}
        for index, spec in enumerate(grouped.fit_panel()):
            for seed in (0, 1, 2):
                self.records[spec.key, seed] = {
                    "data": saved_result(rows, record, spec, seed,
                                         val_score=index / 20 if seed == 0 else 1 - index / 20,
                                         test_score=1 - index / 20),
                    "path": f"fixture/{spec.key}/seed{seed}.json", "sha256": "f" * 64,
                }

    def test_selection_uses_new_seed_zero_validation_not_test_or_other_seeds(self):
        self.records["mrnafm_native", 0]["data"]["val_score"] = .9
        self.records["rnafm_grid", 0]["data"]["val_score"] = .8
        selected, comparisons = grouped.selection_and_comparisons(self.records)
        self.assertEqual(selected, {"original": "rnafm_grid", "mrnafm": "mrnafm_native",
                                    "all": "mrnafm_native"})
        self.assertTrue(any(row["first"] == "original_cross" and
                            row["second"] == "original_concat" for row in comparisons))
        self.assertTrue(any(row["first"] == "mrnafm_cross" and
                            row["second"] == "mrnafm_native" for row in comparisons))
        before = deepcopy(selected)
        for (key, seed), entry in self.records.items():
            entry["data"]["test_score"] = 100 if key == "ntv2_native" else -100
            if seed != 0:
                entry["data"]["val_score"] = 200 if key == "ntv2_native" else -200
        self.assertEqual(grouped.selection_and_comparisons(self.records)[0], before)

    def test_selection_refuses_incomplete_or_nonfinite_candidate_pool(self):
        for replacement in (None, float("nan"), float("inf"), True):
            with self.subTest(replacement=replacement):
                records = deepcopy(self.records)
                records["ntv2_grid", 0]["data"]["val_score"] = replacement
                with self.assertRaises(ValueError):
                    grouped.selection_and_comparisons(records)
        missing = deepcopy(self.records)
        missing.pop(("ntv2_grid", 0))
        with self.assertRaises(ValueError):
            grouped.selection_and_comparisons(missing)
        with self.assertRaises(ValueError):
            grouped.selection_and_comparisons(self.records, selection_seed=1)

    def test_report_requires_complete_three_seed_panel_and_keeps_seed_pairing(self):
        bundle = grouped.report_bundle(self.records, [], recipe())
        self.assertEqual(set(bundle["models"]), {spec.key for spec in grouped.fit_panel()})
        self.assertTrue(all(set(values) == {0, 1, 2} for values in bundle["models"].values()))
        with self.assertRaises(ValueError):
            grouped.report_bundle(self.records, [{"issue": "missing"}], recipe())
        missing = dict(self.records)
        missing.pop(("original_cross", 2))
        with self.assertRaises(ValueError):
            grouped.report_bundle(missing, [], recipe())
        seed_zero = {key: value for key, value in self.records.items() if key[1] == 0}
        with self.assertRaises(ValueError):
            grouped.report_bundle(seed_zero, [], recipe(), seeds=(0,))


class NotebookBoundaryTests(unittest.TestCase):
    """Execute isolated source with synthetic paths and real producers blocked."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix=".grouped-fixture-", dir=ROOT / "tests")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.code = self.root / "Code"
        self.results = self.code / "stage4/results/grouped_stability_v1"
        self.cells = notebook_cells()

    def namespace(self):
        return {"CODE_DIR": self.code, "Path": Path, "np": np, "pd": pd, "torch": torch,
                "datasets": datasets, "blf": blf, "grouped": grouped,
                "run_identity": run_identity, "tokens": tokens, "train": train,
                "swap_reporting": swap_reporting, "asdict": asdict,
                "display": lambda *args, **kwargs: None}

    def run_reading_flow(self):
        namespace = self.namespace()
        with ExitStack() as stack:
            blocked = (
                (datasets, "blf_benchmark_rows"), (grouped, "build_split"),
                (grouped, "publish_split"), (run_identity, "cache_record"),
                (tokens, "summarize"), (train, "run"), (train, "fit"),
                (swap_reporting, "report"), (swap_reporting, "calculate"),
                (np, "load"), (Path, "mkdir"), (Path, "write_text"),
                (torch.cuda, "is_available"),
            )
            for owner, name in blocked:
                stack.enter_context(patch.object(owner, name, side_effect=AssertionError(name)))
            stack.enter_context(redirect_stdout(StringIO()))
            for name, source in self.cells.items():
                if name not in ("imports", "project-path"):
                    exec(compile(source, f"synthetic:{name}", "exec"), namespace)
        return namespace

    def test_default_reading_of_missing_evidence_does_not_create_or_compute_anything(self):
        namespace = self.run_reading_flow()
        self.assertTrue(namespace["READ_SAVED_RESULTS"])
        self.assertFalse(namespace["RUN_TRAINING"])
        self.assertFalse(namespace["RUN_COMPARISONS"])
        self.assertIsNone(namespace["split_record"])
        self.assertEqual(namespace["records"], {})
        self.assertEqual(list(self.root.iterdir()), [])

    def test_complete_saved_fit_reading_does_not_recreate_missing_report(self):
        rows, record = grouped.build_split(fixture_rows(), source_sha256="a" * 64)
        self.results.mkdir(parents=True)
        (self.results / "split.json").write_text(json.dumps(record), encoding="utf-8")
        for spec in grouped.fit_panel():
            for seed in (0, 1, 2):
                path = grouped.fit_path(self.results, spec, seed)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(saved_result(rows, record, spec, seed)), encoding="utf-8")
        before = {path: path.read_bytes() for path in self.results.rglob("*.json")}
        namespace = self.run_reading_flow()
        self.assertEqual(len(namespace["records"]), 36)
        self.assertEqual(namespace["issues"], [])
        self.assertIsNone(namespace["saved_report"])
        self.assertFalse((self.results / "reports").exists())
        self.assertEqual({path: path.read_bytes() for path in self.results.rglob("*.json")}, before)

    def test_mode_guard_refuses_mixed_modes_and_training_without_launcher(self):
        namespace = self.namespace()
        exec(self.cells["execution-settings"], namespace)
        namespace["RUN_TRAINING"] = True
        with self.assertRaises(ValueError):
            exec(self.cells["mode-guard"], namespace)
        namespace["READ_SAVED_RESULTS"] = False
        with self.assertRaises(ValueError):
            exec(self.cells["mode-guard"], namespace)

    def test_preflight_preserves_incompatible_records_and_skips_cache_work_when_complete(self):
        namespace = self.namespace()
        namespace.update(PANEL=grouped.fit_panel(), SEED=0, BASE_CONFIG=recipe(),
                         RESULTS_ROOT=self.results,
                         verified_cache_records=Mock(side_effect=AssertionError("opened caches")))
        exec(self.cells["preflight-helper"], namespace)
        preflight = namespace["preflight_missing_fits"]
        with self.assertRaises(ValueError):
            preflight(None, None, {}, [{"issue": "wrong experiment"}])
        complete = {(spec.key, 0): object() for spec in grouped.fit_panel()}
        self.assertEqual(preflight(None, None, complete, []), [])
        namespace["verified_cache_records"].assert_not_called()

    def test_preflight_refuses_orphan_weights_and_bad_checkpoint_without_training(self):
        spec = grouped.fit_panel()[0]
        namespace = self.namespace()
        validator = Mock(side_effect=ValueError("incompatible checkpoint"))
        namespace.update(PANEL=(spec,), SEED=0, BASE_CONFIG=recipe(), RESULTS_ROOT=self.results,
                         verified_cache_records=Mock(return_value={}), CHECKPOINT_VALIDATOR=validator)
        exec(self.cells["preflight-helper"], namespace)
        preflight = namespace["preflight_missing_fits"]
        path = grouped.fit_path(self.results, spec, 0)
        path.parent.mkdir(parents=True)
        weights = path.with_suffix(".weights.pt")
        weights.write_bytes(b"synthetic orphan")
        with patch.object(grouped, "make_identity", return_value={"synthetic": True}), \
                patch.object(train, "run", side_effect=AssertionError("unexpected fit")):
            with self.assertRaises(FileExistsError):
                preflight(None, None, {}, [])
            self.assertEqual(weights.read_bytes(), b"synthetic orphan")
            weights.unlink()
            checkpoint = path.with_suffix(".checkpoint")
            checkpoint.write_bytes(b"synthetic checkpoint")
            with self.assertRaisesRegex(ValueError, "incompatible checkpoint"):
                preflight(None, None, {}, [])
            validator.assert_called_once_with(checkpoint, {"synthetic": True})
            self.assertEqual(checkpoint.read_bytes(), b"synthetic checkpoint")

    def test_fit_cleanup_preserves_primary_failure_and_reports_cleanup_only_failure(self):
        spec = grouped.fit_panel()[0]
        for fitting_fails, cleanup_fails in ((False, False), (True, False), (True, True), (False, True)):
            with self.subTest(fitting=fitting_fails, cleanup=cleanup_fails):
                namespace = self.namespace()
                primary = RuntimeError("synthetic fit failure")
                cleanup = OSError("synthetic close failure")
                stream = SimpleNamespace(close=Mock(side_effect=cleanup if cleanup_fails else None))
                runner = Mock(side_effect=primary if fitting_fails else None, return_value="synthetic result")
                namespace.update(RESULTS_ROOT=self.results, DEVICE="cpu",
                                 open_fit_streams=Mock(return_value=stream),
                                 FIT_PRODUCER=lambda: {"host": "synthetic"},
                                 train=SimpleNamespace(run=runner))
                exec(self.cells["single-fit-helper"], namespace)
                with redirect_stdout(StringIO()):
                    if fitting_fails or cleanup_fails:
                        with self.assertRaises(RuntimeError if fitting_fails else OSError) as raised:
                            namespace["fit_one"](None, spec, recipe(), {})
                        self.assertIs(raised.exception, primary if fitting_fails else cleanup)
                        if fitting_fails and cleanup_fails:
                            self.assertIn("synthetic close failure", " ".join(primary.__notes__))
                    else:
                        self.assertEqual(namespace["fit_one"](None, spec, recipe(), {}), "synthetic result")
                stream.close.assert_called_once()

    def test_partial_stream_open_failure_closes_prior_cache_without_masking_error(self):
        spec = grouped.fit_panel()[0]
        primary = RuntimeError("synthetic cache open failure")
        cache = SimpleNamespace(close=Mock(side_effect=OSError("synthetic close failure")))
        namespace = self.namespace()
        namespace.update(CACHE_ROOT=self.root / "caches",
                         tokens=SimpleNamespace(summarize=Mock(side_effect=[cache, primary])))
        exec(self.cells["stream-helper"], namespace)
        with self.assertRaises(RuntimeError) as raised:
            namespace["open_fit_streams"](None, spec)
        self.assertIs(raised.exception, primary)
        cache.close.assert_called_once()
        self.assertIn("synthetic close failure", " ".join(primary.__notes__))

    def test_result_destination_cannot_overlap_caches_or_historical_artifacts(self):
        cache_root = self.root / "embeddings"
        for root in (cache_root, cache_root / "child", self.root):
            with self.subTest(root=root), self.assertRaises(ValueError):
                grouped.require_result_destination(root, cache_root)
        historical = self.root / "historical"
        historical.mkdir()
        (historical / "old_result.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(ValueError):
            grouped.require_result_destination(historical, cache_root)
        proposed = self.root / "new_results"
        grouped.require_result_destination(proposed, cache_root)
        self.assertFalse(proposed.exists())


if __name__ == "__main__":
    unittest.main()
