"""Shared code for the multimodal-bio-fusion project.

Modules:
    sequences   sequence checks, translation, composition features, motif scanning
    encoders    registry of frozen pretrained encoders, and model loading
    datasets    registry of labeled datasets, and the inputs each encoder reads
    splits      deterministic train and test splits shared by every stage
    embeddings  frozen-encoder embeddings, with a cache
    analysis    probes, representation similarity, composition control, tables, figures
    notebook    showing shared source code inside notebooks

Importing the package does not import PyTorch. Only shared_code.embeddings and the model-loading
functions in shared_code.encoders need it.
"""
