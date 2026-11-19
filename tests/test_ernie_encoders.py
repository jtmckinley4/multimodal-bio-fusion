"""Pinned ERNIE adapters with local vocabulary and tiny invented weights; no downloads."""

import ast
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from transformers import BertConfig, BertModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code"))
from shared_code import encoders
from shared_code.fusion import token_embeddings as tokens


class ErnieEncoderTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name)
        vocabulary = "[PAD]\n[UNK]\n[CLS]\n[EOS]\n[SEP]\n[MASK]\nA\nU\nC\nG\nN\n"
        (self.path / "vocab.txt").write_text(vocabulary, encoding="utf-8")
        (self.path / "tokenizer_config.json").write_text(json.dumps({
            "tokenizer_class": "RNAErnieTokenizer", "model_max_length": 2048,
        }), encoding="utf-8")
        self.encoder = replace(encoders.ENCODERS["rnaernie"], checkpoint=str(self.path), revision=None)

    def tokenizer(self):
        return encoders.load_tokenizer(self.encoder)

    def tiny_backbone(self):
        config = BertConfig(vocab_size=11, hidden_size=8, intermediate_size=16,
                            num_hidden_layers=1, num_attention_heads=2,
                            max_position_embeddings=2048, pad_token_id=0)
        model = BertModel(config, add_pooling_layer=False)
        state = {"bert." + key: value.clone() for key, value in model.state_dict().items()}
        # Published BertForMaskedLM adds this unused prediction head.
        state["cls.predictions.bias"] = torch.zeros(11)
        return config, model, state

    def test_registry_distinguishes_author_checkpoint_and_converted_base(self):
        author = encoders.ENCODERS["rnaernie"]
        self.assertEqual(author.checkpoint, "LLM-EDA/RNAErnie")
        self.assertEqual(author.revision, "df26681ed44bb671b61536c5e2a196d6058a44f7")
        self.assertEqual(author.max_len, 2048)
        converted = encoders.ENCODERS["ernierna"]
        self.assertEqual(converted.checkpoint, "multimolecule/ernierna")
        self.assertEqual(converted.revision, "9f6099b85d61f7567d4e1ed1e92e1de85fdfd292")
        self.assertEqual(converted.max_len, 1026)
        for encoder in (author, converted):
            self.assertEqual(tokens.grid_resolution(encoder.key), "nucleotide")
            self.assertEqual(tokens.expected_tokens(encoder, "ATG" * 333), 999)

    def test_author_token_ids_framing_case_and_unknown_are_preserved(self):
        tokenizer = self.tokenizer()
        self.assertEqual(tokenizer("atcgn?")["input_ids"], [2, 6, 7, 8, 9, 10, 1, 3])
        self.assertEqual(tokenizer("AT", return_special_tokens_mask=True)["special_tokens_mask"],
                         [1, 0, 0, 1])
        self.assertEqual(tokens.structural_token_ids(tokenizer), {0, 2, 3, 4})
        self.assertNotIn(tokenizer.unk_token_id, tokens.structural_token_ids(tokenizer))

    def test_author_pair_framing_matches_release(self):
        tokenizer = self.tokenizer()
        self.assertEqual(tokenizer("AU", "CG")["input_ids"], [2, 6, 7, 4, 8, 9, 3])

    def test_full_ecoli_length_uses_1001_tokens_without_truncation(self):
        ids = self.tokenizer()("ATG" * 333)["input_ids"]
        self.assertEqual(len(ids), 1001)
        self.assertEqual(ids[-4:], [6, 7, 9, 3])

    def test_changed_vocabulary_is_refused(self):
        (self.path / "vocab.txt").write_text("[PAD]\n[UNK]\nA\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "vocabulary differs"):
            self.tokenizer()

    def test_backbone_loading_is_complete_and_frozen(self):
        config, reference, state = self.tiny_backbone()
        with patch("transformers.AutoConfig.from_pretrained", return_value=config), \
             patch.object(encoders, "_load_checkpoint_state", return_value=state) as read:
            loaded = encoders._load_rnaernie_model(self.encoder)
        read.assert_called_once_with(str(self.path), "model.safetensors", None)
        self.assertIsNone(loaded.pooler)
        self.assertTrue(all(not parameter.requires_grad for parameter in loaded.parameters()))
        for key, value in reference.state_dict().items():
            torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)

    def test_ernierna_loads_pinned_base_model(self):
        encoder = encoders.ENCODERS["ernierna"]
        tokenizer, model = object(), Mock()
        model.to.return_value = model
        model.eval.return_value = model
        with patch.dict(sys.modules, {"multimolecule": SimpleNamespace()}), \
             patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer) as tokenize, \
             patch("transformers.AutoModel.from_pretrained", return_value=model) as load:
            self.assertEqual(encoders.load_encoder(encoder, "cpu"), (tokenizer, model))
        expected = dict(trust_remote_code=True, revision=encoder.revision)
        tokenize.assert_called_once_with(encoder.checkpoint, **expected)
        load.assert_called_once_with(encoder.checkpoint, **expected)

    def test_old_short_context_config_is_refused_before_weight_read(self):
        config, _, _ = self.tiny_backbone()
        config.max_position_embeddings = 513
        with patch("transformers.AutoConfig.from_pretrained", return_value=config), \
             patch.object(encoders, "_load_checkpoint_state") as read, \
             self.assertRaisesRegex(ValueError, "configuration differs"):
            encoders._load_rnaernie_model(self.encoder)
        read.assert_not_called()

    def test_missing_or_unexpected_backbone_weights_are_refused(self):
        config, _, state = self.tiny_backbone()
        missing = dict(state)
        missing.pop("bert.embeddings.word_embeddings.weight")
        unexpected = dict(state, unrelated_weight=torch.ones(1))
        for weights in (missing, unexpected):
            with self.subTest(weights="missing" if weights is missing else "unexpected"), \
                 patch("transformers.AutoConfig.from_pretrained", return_value=config), \
                 patch.object(encoders, "_load_checkpoint_state", return_value=weights), \
                 self.assertRaises(RuntimeError):
                encoders._load_rnaernie_model(self.encoder)

    def test_unused_pooler_does_not_require_random_initialization(self):
        config, _, state = self.tiny_backbone()
        state["bert.pooler.dense.weight"] = torch.zeros(8, 8)
        with patch("transformers.AutoConfig.from_pretrained", return_value=config), \
             patch.object(encoders, "_load_checkpoint_state", return_value=state):
            self.assertIsNone(encoders._load_rnaernie_model(self.encoder).pooler)

    def test_small_backbone_states_preserve_all_content_positions(self):
        tokenizer = self.tokenizer()
        _, model, _ = self.tiny_backbone()
        model.eval()
        with torch.no_grad():
            expected = model(**tokenizer("AUN?", return_tensors="pt")).last_hidden_state[0, 1:-1]
        actual = tokens.token_states("AUN?", tokenizer, model, 2048, dtype=np.float32)
        np.testing.assert_allclose(actual, expected.numpy(), rtol=0, atol=0)
        self.assertEqual(actual.shape, (4, 8))

    def test_overlength_input_is_refused_before_model_call(self):
        _, model, _ = self.tiny_backbone()
        with patch.object(model, "forward", side_effect=AssertionError("must not run")), \
             self.assertRaisesRegex(ValueError, "token limit"):
            tokens.token_states("A" * 2047, self.tokenizer(), model, 2048)

    def test_mlm_logits_cannot_be_written_as_token_states(self):
        _, model, _ = self.tiny_backbone()
        logits = SimpleNamespace(logits=torch.zeros(1, 5, 11))
        with patch.object(model, "forward", return_value=logits), \
             self.assertRaisesRegex(ValueError, "not prediction logits"):
            tokens.token_states("AUG", self.tokenizer(), model, 2048)

    def test_tuple_logits_with_wrong_hidden_width_are_refused(self):
        _, model, _ = self.tiny_backbone()
        with patch.object(model, "forward", return_value=(torch.zeros(1, 5, 11),)), \
             self.assertRaisesRegex(ValueError, "hidden_size"):
            tokens.token_states("AUG", self.tokenizer(), model, 2048)

    def test_original_mrfp_panel_does_not_expand_with_registry(self):
        notebook = json.loads((ROOT / "Code/stage1/mrfp_representation_control.ipynb").read_text(
            encoding="utf-8"))
        keys = None
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            for node in ast.parse("".join(cell["source"])).body:
                if isinstance(node, ast.Assign) and any(
                        isinstance(target, ast.Name) and target.id == "ENCODER_KEYS"
                        for target in node.targets):
                    keys = ast.literal_eval(node.value)
        self.assertEqual(keys, ["nt500m", "ntv2_100m", "dnabert2", "hyenadna", "rnafm",
                                "rinalmo", "mrnafm", "calm", "esm2_8m", "esm2_35m",
                                "esm2_150m", "protbert"])


if __name__ == "__main__":
    unittest.main()
