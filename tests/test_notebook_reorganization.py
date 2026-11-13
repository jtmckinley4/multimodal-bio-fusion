"""Static notebook-location and artifact-path contracts; no cells are executed.

Historical result directories and recovery scope are deliberately independent of
the notebook reading layout. These checks never open result arrays or fit models.
"""

import ast
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "Code"
ACTIVE = {
    "stage1/Stage1_stability.ipynb", "stage1/Stage1_gtex.ipynb",
    "stage1/mrfp_representation_control.ipynb",
    "stage1/lucaone_representation_comparison.ipynb",
    "stage3/blf_datasets.ipynb", "stage3/blf_token_embeddings.ipynb",
    "stage3/blf_build_check.ipynb", "stage3/blf_reference_benchmarks.ipynb",
    "stage3/blf_appendix_comparison.ipynb", "stage3/isoformer_gtex_audit.ipynb",
    "stage3/gtex_frozen_fusion_reference.ipynb", "stage3/coupled_mamba_reference.ipynb",
    "stage4/encoder_swaps.ipynb", "stage4/halflife_translation_efficiency.ipynb",
}
REPLACED = {
    "stage3/01_blf_datasets.ipynb", "stage3/02_token_embeddings.ipynb",
    "stage3/03_blf_build_check.ipynb", "stage3/04_lucaone_isoformer_check.ipynb",
    "stage4/tier1_mrfp_control.ipynb", "stage4/tier2_blf_phaseA.ipynb",
    "stage4/tier2_blf_phaseA_appendix.ipynb", "stage4/tier2_encoder_swaps.ipynb",
    "stage4/tier2_isoformer_gtex_audit.ipynb", "stage4/tier2_lucaone_isoformer.ipynb",
    "stage4/tier3_halflife_te.ipynb", "stage4/tier4_designs.ipynb",
}


def assignments(notebook, cell_id, requested):
    document = json.loads((CODE / notebook).read_text(encoding="utf-8"))
    matches = [cell for cell in document["cells"] if cell.get("id") == cell_id]
    if len(matches) != 1 or matches[0]["cell_type"] != "code":
        raise AssertionError(f"Expected one settings code cell: {notebook}:{cell_id}")
    values = {}
    for node in ast.parse("".join(matches[0]["source"])).body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in requested:
                if target.id in values:
                    raise AssertionError(f"Repeated setting {target.id} in {cell_id}")
                values[target.id] = ast.dump(node.value, include_attributes=False)
    return values


class NotebookReorganizationTests(unittest.TestCase):
    def test_active_notebooks_have_one_current_reading_location(self):
        actual = {path.relative_to(CODE).as_posix() for path in CODE.glob("stage*/*.ipynb")}
        self.assertEqual(actual, ACTIVE)

    def test_replaced_paths_do_not_leave_duplicate_active_notebooks(self):
        self.assertEqual([name for name in sorted(REPLACED) if (CODE / name).exists()], [])

    def test_existing_artifact_paths_do_not_follow_notebook_moves(self):
        # Compare expressions without executing settings, resolving inputs or creating paths.
        contracts = [
            ("stage1/mrfp_representation_control.ipynb", "3ccd3a1d", {
                "EMBEDDING_DIR": "CODE_DIR / 'stage1' / 'embeddings' / 'mrfp_expression_n1000_seed42'",
                "RESULTS_DIR": "CODE_DIR / 'stage4' / 'results' / 'tier1_mrfp'",
            }),
            ("stage1/lucaone_representation_comparison.ipynb", "d88fc413", {
                "STAGE3_DIR": "CODE_DIR / 'stage3'", "STAGE4_DIR": "CODE_DIR / 'stage4'",
                "CACHE_ROOT": "STAGE3_DIR / 'embeddings'",
                "LUCAONE_DIR": "CACHE_ROOT / BENCHMARK / 'lucaone'",
                "RESULTS_DIR": "STAGE4_DIR / 'results' / 'lucaone'",
                "LUCAONE_COMPARISON_PATH": "RESULTS_DIR / 'complementarity_v1' / f'{BENCHMARK}.json'",
            }),
            ("stage3/blf_token_embeddings.ipynb", "token-settings", {
                "CACHE_ROOT": "Path('embeddings')",
            }),
            ("stage3/blf_build_check.ipynb", "blf-build-settings", {
                "CACHE_ROOT": "Path('embeddings')", "RESULTS_DIR": "Path('results') / BENCHMARK",
            }),
            ("stage3/blf_reference_benchmarks.ipynb", "0965876d", {
                "CACHE_ROOT": "CODE_DIR / 'stage3' / 'embeddings'",
                "RESULTS_ROOT": "CODE_DIR / 'stage4' / 'results' / 'phaseA'",
            }),
            ("stage3/blf_appendix_comparison.ipynb", "settings", {
                "STAGE3_DIR": "CODE_DIR / 'stage3'", "CACHE_ROOT": "STAGE3_DIR / 'embeddings'",
                "RESULT_DIRECTORY": "STAGE3_DIR / 'results' / BENCHMARK / 'reproducible' "
                                    "if BENCHMARK == 'fungal_expression' else "
                                    "CODE_DIR / 'stage4' / 'results' / 'blf_phaseA' / BENCHMARK",
                "SAVED_ANALYSIS_DIRECTORY": "RESULT_DIRECTORY / 'records' / 'analysis-20261020' "
                                            "if BENCHMARK == 'ecoli_proteins' else None",
            }),
            ("stage3/coupled_mamba_reference.ipynb", "36da38f2", {
                "CACHE_ROOT": "CODE_DIR / 'stage3' / 'embeddings'",
                "RESULTS_ROOT": "CODE_DIR / 'stage4' / 'results' / 'tier4_designs'",
                "PHASE_A": "CODE_DIR / 'stage4' / 'results' / 'phaseA'",
                "LEGACY_REFERENCE_RECORD": "CODE_DIR / 'stage4' / 'legacy_reference_reuse.json'",
            }),
            ("stage3/isoformer_gtex_audit.ipynb", "settings", {
                "CSV_PATH": "REPO_ROOT / '.data-cache' / 'GTEx_final.csv'",
                "RESULT_DIR": "CODE_DIR / 'stage4' / 'results' / 'isoformer_cohort_audit'",
                "RESULT_PATH": "RESULT_DIR / 'gtex_audit_v2.json'",
                "PREVIOUS_RECORD": "RESULT_DIR / 'cohort_audit.json'",
                "HISTORICAL_RECORD": "RESULT_DIR / 'full_csv_2026-11-09.json'",
            }),
            ("stage3/gtex_frozen_fusion_reference.ipynb", "s4-pilot-settings", {
                "CACHE_ROOT": "Path('embeddings')", "RESULTS_DIR": "Path('results') / BENCHMARK",
                "GTEX_CACHE_RECOVERY": "CODE_DIR / 'stage3' / 'gtex_cache_recovery.json'",
            }),
            ("stage3/gtex_frozen_fusion_reference.ipynb", "gtex-repeat-setup", {
                "STAGE3_DIR": "CODE_DIR / 'stage3'", "STAGE4_DIR": "CODE_DIR / 'stage4'",
                "REPEAT_CACHE_ROOT": "STAGE3_DIR / 'embeddings'",
                "REPEAT_RESULTS_DIR": "STAGE4_DIR / 'results' / 'gtex_isoformer'",
                "REPEAT_CACHE_RECOVERY": "STAGE3_DIR / 'gtex_cache_recovery.json'",
            }),
            ("stage4/encoder_swaps.ipynb", "swaps-locations", {
                "CACHE_ROOT": "CODE_DIR / 'stage3' / 'embeddings'",
                "RESULTS_ROOT": "CODE_DIR / 'stage4' / 'results' / 'phaseB'",
                "COMPARISON_DIR": "RESULTS_ROOT / 'analyses' / 'paired-report-v1'",
            }),
            ("stage4/halflife_translation_efficiency.ipynb", "6880b3af", {
                "RESULTS_DIR": "CODE_DIR / 'stage4' / 'results' / 'halflife_te'",
                "EMBED_DIR": "CODE_DIR / 'stage3' / 'embeddings' / 'mrnabench'",
                "MATCHED_RESULTS_DIR": "RESULTS_DIR / 'matched_probes_v1'",
            }),
        ]
        for notebook, cell_id, expected in contracts:
            with self.subTest(notebook=notebook, cell=cell_id):
                actual = assignments(notebook, cell_id, expected)
                for name, expression in expected.items():
                    self.assertIn(name, actual)
                    self.assertEqual(actual[name], ast.dump(ast.parse(expression, mode="eval").body,
                                                           include_attributes=False), name)

    def test_legacy_recovery_record_keeps_its_original_scope(self):
        record = json.loads((CODE / "stage4/legacy_reference_reuse.json").read_text(encoding="utf-8"))
        self.assertEqual(record["schema"], "blf.reference-reuse.v1")
        self.assertEqual(record["scope"], "tier4_designs.references")
        # The notebook's new title must not silently rename an immutable evidence contract.
        self.assertEqual(len(record["references"]), 24)


if __name__ == "__main__":
    unittest.main()
