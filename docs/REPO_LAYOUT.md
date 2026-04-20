# Repository Layout (Class Project)

This repository is organized into two clearly separated tracks:

1. **Pipeline track**: SciFact -> embeddings -> clustering -> retrieval (+ optional FHE interface)
2. **Architecture track**: lightweight plaintext kernels for simulator analysis

## Top-Level Structure

- `data/`
  - `raw/`: original downloaded datasets
  - `processed/`: pipeline artifacts (`*.npy`, ids, clusters)
  - `exported/`: compact benchmark exports for architecture experiments
- `src/`
  - `data/`: data loading/processing
  - `embed/`: embedding stage
  - `cluster/`: K-means clustering stage
  - `retrieval/`: plaintext retrieval stage
  - `fhe/`: FHE interface layer (mock + OpenFHE backend hook)
- `scripts/`: stage entry points and verification scripts
- `arch/`: plaintext architecture kernels only
- `reference/`: notes and links to external/original implementations
- `docs/`: concise design notes

## Stage Entry Points

- Stage 1: `scripts/download_scifact.py`
- Stage 1 verify: `scripts/verify_stage1_data.py`
- Stage 2: `scripts/build_embeddings.py`
- Stage 3: `scripts/cluster_db.py`
- Stage 4: `scripts/run_plaintext_retrieval.py`
- Stage 4 verify: `scripts/verify_stage4_plaintext.py`
- FHE consistency verify: `scripts/verify_fhe_clustered_consistency.py`
- Export benchmark package: `scripts/export_arch_bench_data.py`

## Collaboration Guidance

- Collaborators focused on architecture should only need:
  - `arch/`
  - exported arrays under `data/exported/`
  - (optionally) retrieval scoring conventions in `src/retrieval/plaintext.py`
