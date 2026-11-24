"""
Explain which inputs travel with an experiment and which to locate on Newton.

prepare.py uses this module to show what belongs in the upload and which files
the experiment needs separately. The printed commands help you check those
files from your own Newton account before submitting jobs.
"""

from textwrap import dedent


# -------- Inputs for each experiment --------


def describe_inputs(recipe, experiment="full"):
    """
    List what the upload contains, what Newton needs, and how to check it.
    """
    if recipe == "partner_tasks":
        return {
            "packaged": [
                "Partner-task Python helpers, command-line launcher, Slurm wrapper, "
                "nine-row batch plan, source/input identity records and Python package versions. "
                "No notebook, dataset, embeddings or previous fit is copied."
            ],
            "newton_needs": [
                "Stage the complete built RPI2241 table at inputs/rpi2241/, including "
                "dataset.json, pairs.csv, both FASTA files and groups.json if supplied.",
                "Stage the complete entity caches at "
                "inputs/pair_entities/rpi2241/left__rna__rinalmo/ and "
                "inputs/pair_entities/rpi2241/right__protein__esm2_150m/. "
                "Each directory needs entities.json and every recorded shard.",
                "Run the staged-input check on a compute node. It verifies the recorded "
                "input identities and complete caches without extracting embeddings or training.",
                "New fit evidence belongs under outputs/rpi2241-newton/ and logs under "
                "outputs/logs/. Preparation updates manage source files only; staged inputs "
                "and outputs remain outside the update inventory.",
            ],
            "verify": [
                "python -B run_partner_tasks.py --verify-only",
                "python -B run_partner_tasks.py --verify-inputs",
            ],
        }
    if recipe == "grouped_stability":
        return {
            "packaged": [
                "Grouped-stability notebook, stability CSV, Python helpers, launcher "
                "and package versions."
            ],
            "newton_needs": [
                "Set ARCC_CACHE_ROOT to the directory containing mrna_stability/. "
                "That benchmark needs ntv2_100m, rnafm, mrnafm and esm2_8m.",
                "Set ARCC_KERNEL to the name of your registered notebook kernel.",
                "The sample check tells you whether you can read two files per encoder. "
                "Training also checks which rows the embeddings describe and verifies "
                "the contents of the embedding files against their recorded hashes.",
            ],
            "verify": [
                'python -B run_grouped_stability.py --verify-only --kernel "${ARCC_KERNEL:?}"',
                _sample_embedding_check_command(
                    "mrna_stability", "ntv2_100m rnafm mrnafm esm2_8m"
                ),
            ],
        }
    if recipe == "fusion_studies":
        return {
            "packaged": [
                "Four study notebooks, Python helpers, launchers, a list of expected "
                "input files and hashes, and Python package versions."
            ],
            "newton_needs": [
                "Correction, gates and attention reuse benchmark CSVs, embeddings and "
                "saved Newton reference predictions. Set ARCC_REFERENCE_ROOT to the "
                "workspace with the completed encoder swaps.",
                "Correction and gates also use saved weights for the prediction heads. "
                "For stability correction, supply the grouped split and its matching fits.",
                "For region-aware fusion (A.6), point REGION_DATA_ROOT at the two human "
                "Parquet tables and the homology map. Use the recorded versions of "
                "RNA-FM, RiNALMo-mega and ESM-2 8M.",
                "Run the input check for the study you chose. For A.6, prepare the cohort "
                "and extract the token embeddings once.",
            ],
            "verify": [
                "python -B run_fusion_study.py --verify-only",
                'python -B verify_inputs.py --scope references --reference-root "${ARCC_REFERENCE_ROOT:?}"',
                'python -B verify_inputs.py --scope regions --region-data-root '
                '"${REGION_DATA_ROOT:-$ARCC_PROJECT_ROOT/inputs/mrnabench}" '
                '--hf-hub-root "${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}"',
            ],
        }
    if recipe != "encoder_swaps":
        raise ValueError(f"Unknown ARCC recipe: {recipe}")

    package_check = "python -B run_encoder_swaps.py --verify-only"
    if experiment == "ecoli_ernie":
        return {
            "packaged": [
                "Encoder-swap notebook, Python helpers, launcher, package versions, "
                "E. coli CSV and original validation records for the gated head."
            ],
            "newton_needs": [
                "Set ARCC_CACHE_ROOT to the embeddings directory containing the saved "
                "E. coli DNA and protein embeddings.",
                "Set ARCC_REFERENCE_ROOT to the completed encoder-swap workspace "
                "containing the RiNALMo comparison.",
                "Stage the nine RNAErnie and ERNIE-RNA files at the versions listed in "
                "the encoder-swaps README. Extract both sets of RNA embeddings once "
                "and reuse them across seeds.",
                "The commands check the upload, sample embedding files, the reference "
                "directory and the pretrained model filenames.",
            ],
            "verify": [
                package_check,
                _sample_embedding_check_command("ecoli_proteins", "ntv2_100m esm2_8m"),
                'test -d "${ARCC_REFERENCE_ROOT:?}/outputs/phaseB/ecoli_proteins/rinalmo"',
                'hub="${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}"',
                'ls -lL "$hub"/models--LLM-EDA--RNAErnie/snapshots/df26681ed44bb671b61536c5e2a196d6058a44f7/'
                '{config.json,tokenizer_config.json,special_tokens_map.json,vocab.txt,model.safetensors}',
                'ls -lL "$hub"/models--multimolecule--ernierna/snapshots/9f6099b85d61f7567d4e1ed1e92e1de85fdfd292/'
                '{config.json,tokenizer_config.json,vocab.txt,model.safetensors}',
            ],
        }
    if experiment not in (None, "full"):
        raise ValueError(f"Unknown encoder-swap panel: {experiment}")
    return {
        "packaged": [
            "Encoder-swap notebook, Python helpers, launcher, package versions, "
            "four benchmark CSVs and the original validation selection records."
        ],
        "newton_needs": [
            "Set ARCC_CACHE_ROOT to the saved embeddings for the original and replacement "
            "encoders used by your benchmarks.",
            "Include recovery-records.json for older caches that require it.",
            "The first command verifies the uploaded files. The second lists the metadata "
            "files in the embeddings directory. Training checks whether those embeddings "
            "match the requested fit.",
        ],
        "verify": [
            package_check,
            'find "${ARCC_CACHE_ROOT:?}" -mindepth 3 -maxdepth 3 -name meta.json -print',
        ],
    }


# -------- Build the sample embedding file check --------


def _sample_embedding_check_command(benchmark, encoders):
    """
    Return a terminal command that checks whether Newton can read each encoder's
    metadata file and its first file of saved embeddings.
    """
    # The parentheses keep a failed check from closing the user's Newton shell.
    return dedent(
        f"""
        (
            cache_root="${{ARCC_CACHE_ROOT:?Set the embedding cache path}}"
            for encoder in {encoders}; do
                directory="$cache_root/{benchmark}/$encoder"
                for filename in meta.json shard_00000.npz; do
                    if ! test -r "$directory/$filename"; then
                        echo "Unreadable: $directory/$filename"
                        exit 1
                    fi
                done
            done
        )
        """
    ).strip()
