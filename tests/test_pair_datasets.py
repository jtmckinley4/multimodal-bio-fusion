"""Check that a saved pair table says what each of its sides actually holds.

The first version of this format wrote every left side to ``rna.fasta`` and every right side to
``protein.fasta``, whatever they held, so a DNA table claimed to hold RNA and a ligand table
claimed its SMILES were a protein. Nothing downstream could then pick an encoder without
guessing from the table's name. The format now records each side's modality, and a directory
written under the old one is still read, as the RNA-and-protein table it was.
"""

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code"))

from shared_code.fusion import pair_sources as PS, pair_tasks as PT  # noqa: E402


def binding_frame(left_modality="rna", right_modality="protein"):
    lefts = {f"R{i:02d}": "ACGU" * 4 for i in range(4)}
    rights = {f"P{i:02d}": "MKVLA" * 3 for i in range(3)}
    rows = [{"left": a, "right": b, "label": (i + j) % 2,
             "provenance": "measured_positive" if (i + j) % 2 else "measured_negative"}
            for i, a in enumerate(lefts) for j, b in enumerate(rights)]
    return PT.PairFrame("binding-stub", pd.DataFrame(rows), lefts, rights, {}, {},
                        "binding", left_modality, right_modality)


def sparse_binding_frame(left_modality="dna", right_modality="protein"):
    """A table with gaps, so a positive pair has a partner nobody recorded.

    Partner reassignment needs an unrecorded partner to reassign to, and refuses a table dense
    enough that none exists. The last right entity is left out of the first two left entities'
    rows, which is what makes those rows reassignable.
    """
    lefts = {f"R{i:02d}": "ACGT" * 4 for i in range(4)}
    rights = {f"P{i:02d}": "MKVLA" * 3 for i in range(4)}
    rows = [{"left": a, "right": b, "label": (i + j) % 2,
             "provenance": "measured_positive" if (i + j) % 2 else "measured_negative"}
            for i, a in enumerate(lefts) for j, b in enumerate(rights)
            if not (j == 3 and i < 2)]
    return PT.PairFrame("sparse-stub", pd.DataFrame(rows), lefts, rights, {}, {},
                        "binding", left_modality, right_modality)


def affinity_frame():
    proteins = {f"K{i:02d}": "MKVLA" * 3 for i in range(3)}
    ligands = {f"L{i:02d}": "Cc1ccccc1O" for i in range(2)}
    rng = np.random.default_rng(0)
    rows = [{"left": a, "right": b, "label": float(rng.normal(7, 1)),
             "provenance": "measured_positive", "endpoint": "pKd"}
            for a in proteins for b in ligands]
    return PT.PairFrame("affinity-stub", pd.DataFrame(rows), proteins, ligands, {}, {},
                        "affinity", "protein", "ligand")


class SideFileTests(unittest.TestCase):
    def test_each_modality_has_a_file_name_and_a_stated_alphabet(self):
        for modality in PT.MODALITIES:
            self.assertIn(modality, PS.SIDE_FILES)

    def test_a_ligand_side_is_not_written_to_a_fasta(self):
        files = PS.side_files(affinity_frame())
        self.assertEqual(files["left"], "protein.fasta")
        self.assertEqual(files["right"], "ligand.smi")

    def test_a_dna_side_is_named_for_what_it_holds(self):
        self.assertEqual(PS.side_files(binding_frame("dna", "protein"))["left"], "dna.fasta")

    def test_two_sides_of_one_modality_do_not_overwrite_each_other(self):
        files = PS.side_files(binding_frame("protein", "protein"))
        self.assertNotEqual(files["left"], files["right"])


class RoundTripTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.addCleanup(self.directory.cleanup)

    def test_a_binding_table_survives_a_round_trip_with_its_modalities(self):
        original = binding_frame("dna", "protein")
        PS.write_pair_dataset(original, self.root / original.name)
        restored = PS.read_pair_dataset(self.root / original.name)
        self.assertEqual(restored.modality("left"), "dna")
        self.assertEqual(restored.modality("right"), "protein")
        self.assertEqual(restored.kind, "binding")
        self.assertEqual(len(restored.pairs), len(original.pairs))
        self.assertEqual(restored.left_sequences, original.left_sequences)
        self.assertEqual(restored.right_sequences, original.right_sequences)

    def test_an_affinity_table_survives_a_round_trip_with_its_endpoint(self):
        original = affinity_frame()
        PS.write_pair_dataset(original, self.root / original.name)
        restored = PS.read_pair_dataset(self.root / original.name)
        self.assertEqual(restored.kind, "affinity")
        self.assertEqual(restored.modality("right"), "ligand")
        self.assertEqual(set(restored.pairs["endpoint"]), {"pKd"})
        np.testing.assert_allclose(np.sort(restored.pairs["label"]),
                                   np.sort(original.pairs["label"]))

    def test_a_smiles_string_comes_back_with_its_case_intact(self):
        """Lower-case atoms mark aromatic rings, so a case change is a different molecule."""
        original = affinity_frame()
        PS.write_pair_dataset(original, self.root / original.name)
        restored = PS.read_pair_dataset(self.root / original.name)
        self.assertEqual(restored.right_sequences["L00"], "Cc1ccccc1O")

    def test_the_metadata_names_each_sides_file_and_modality(self):
        original = binding_frame("dna", "protein")
        metadata = PS.write_pair_dataset(original, self.root / original.name)
        self.assertEqual(metadata["schema"], "pair-dataset.v2")
        self.assertEqual(metadata["sides"]["left"]["modality"], "dna")
        self.assertEqual(metadata["sides"]["left"]["file"], "dna.fasta")
        self.assertEqual(metadata["sides"]["left"]["entities"], 4)
        self.assertTrue((self.root / original.name / "dna.fasta").exists())

    def test_the_written_summary_matches_the_table(self):
        original = binding_frame()
        metadata = PS.write_pair_dataset(original, self.root / original.name)
        self.assertEqual(metadata["summary"]["pairs"], len(original.pairs))
        self.assertEqual(metadata["summary"]["left_entities"], 4)
        self.assertEqual(metadata["summary"]["right_entities"], 3)

    def test_a_directory_written_under_the_old_format_is_still_read(self):
        original = binding_frame()
        directory = self.root / "legacy"
        PS.write_pair_dataset(original, directory)
        # Put the directory back into the first version of the format.
        metadata = json.loads((directory / "dataset.json").read_text())
        metadata["schema"] = "pair-dataset.v1"
        metadata.pop("sides")
        (directory / "dataset.json").write_text(json.dumps(metadata))
        restored = PS.read_pair_dataset(directory)
        self.assertEqual(restored.modality("left"), "rna")
        self.assertEqual(restored.modality("right"), "protein")
        self.assertEqual(len(restored.pairs), len(original.pairs))

    def test_an_unknown_schema_is_refused_rather_than_guessed(self):
        directory = self.root / "future"
        PS.write_pair_dataset(binding_frame(), directory)
        metadata = json.loads((directory / "dataset.json").read_text())
        metadata["schema"] = "pair-dataset.v9"
        (directory / "dataset.json").write_text(json.dumps(metadata))
        with self.assertRaises(ValueError):
            PS.read_pair_dataset(directory)


class ModalityTests(unittest.TestCase):
    def test_an_unknown_modality_is_refused_when_the_table_is_built(self):
        with self.assertRaises(ValueError):
            binding_frame("peptide", "protein")

    def test_a_reassigned_partner_table_keeps_the_sides_modalities(self):
        """The control is scored through the same caches, so it must declare the same sides."""
        original = sparse_binding_frame("dna", "protein")
        rows = np.flatnonzero(original.pairs["label"].to_numpy() == 1)
        control = PT.partner_reassignment(original, rows, seed=0)
        self.assertEqual(control.modality("left"), "dna")
        self.assertEqual(control.modality("right"), "protein")
        self.assertEqual(set(control.pairs["provenance"]), {"artificial_reassigned"})

    def test_a_table_with_no_unrecorded_partner_refuses_the_control(self):
        """A fully dense table cannot support reassignment, and says so rather than returning
        an empty set to score."""
        dense = binding_frame("dna", "protein")
        rows = np.flatnonzero(dense.pairs["label"].to_numpy() == 1)
        with self.assertRaises(ValueError):
            PT.partner_reassignment(dense, rows, seed=0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
