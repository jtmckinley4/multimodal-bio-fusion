"""Stage 3 fusion models, imported by the Stage 3 build checks and the Stage 4 experiments.

Planned modules, built in the order the research plan sets (see Code/stage3/README.md):
    codon_grid       resample DNA, RNA, and protein token streams to one codon grid
    heads            codon-aware concatenation, gated MIL attention, cross-modal attention
    textcnn          the fixed TextCNN prediction head
    blf              the BioLangFusion pipeline, reproduced exactly (research plan item 1)
    lucaone          LucaOne frozen embeddings (research plan item 2)
    isoformer        IsoFormer's cross-attention aggregator, reproduced on GTEx
    train            one training and scoring loop shared by every model
    coupled_mamba    the project's first design (round 2, January)
    cross_attention  the project's second design (round 2, January)

Empty until the Stage 3 build starts. Keep model code here rather than in notebooks so that
Stage 4 imports exactly what Stage 3 checked.
"""
