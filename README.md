# Toy Retrieval Pipeline for Architecture Simulation

## Project Overview

This repository contains a small, SciFact-based, RAG-inspired retrieval pipeline.
The focus is **not** full RAG generation; it is the **similarity search stage** (query-to-vector distance/similarity).

The class-project goal is to study how architecture choices (cache, memory latency, etc.) affect distance computation performance.

## Repository Structure

- `src/` - core pipeline modules (data, embedding, clustering, retrieval, FHE interface layer)
- `scripts/` - stage entry points and verification/export utilities
- `data/` - raw inputs, processed artifacts, and exported benchmark packages
- `arch/` - minimal plaintext C++ kernels for architecture simulation

## Pipeline Summary

1. **Data loading**  
   Prepare processed SciFact document/query records used by later stages.

2. **Embedding**  
   Convert document/query text to dense vectors (`.npy`) with a sentence-transformer.

3. **Clustering**  
   K-means cluster document embeddings, producing centroids and assignments.

4. **Retrieval (plaintext + FHE interface)**  
   - Plaintext: non-clustered full scan or clustered centroid-first retrieval  
   - FHE: interface/backends for encrypted query vs plaintext centroid scoring

## Architecture Benchmark (Important)

The architecture kernels are in `arch/kernels/`:

- `pt_nonclustered.cpp` = **full scan** kernel  
  Computes query-to-all-database-vector scores (`N x d` work).

- `pt_clustered.cpp` = **centroid-only** kernel  
  Computes query-to-all-centroid scores (`K x d` work).

These represent two retrieval patterns:

- non-clustered baseline (higher memory/computation)
- clustered gating stage (smaller candidate-selection computation)

Expected inputs (flat float arrays, row-major):

- `query`: shape `[d]`
- `db`: shape `[N, d]` (non-clustered kernel)
- `centroids`: shape `[K, d]` (clustered kernel)

Metric convention in kernels:

- `0` -> dot product
- `1` -> negative squared L2 distance

## How to Run

### 1) Install dependencies

Python (pipeline/export scripts):

```bash
pip install numpy sentence-transformers pyyaml datasets
```

C++ (architecture kernels):

- any C++17 compiler (`g++`/`clang++`)
- `make`

### 2) Generate embeddings

If you already have `data/processed/scifact_processed.json`, run:

```bash
python scripts/build_embeddings.py
```

If you need to create/check stage-1 data:

```bash
python scripts/verify_stage1_data.py
```

### 3) Cluster document embeddings

```bash
python scripts/cluster_db.py --k 64 --seed 42
```

### 4) Export architecture benchmark data

```bash
python scripts/export_arch_data.py --n-docs 512 --k-centroids 64 --dim 128
```

Outputs (default): `data/exported/arch_minimal/`

- `query.npy`
- `db.npy`
- `centroids.npy`
- `metadata.json`

### 5) Compile and run kernels

```bash
make -C arch
./arch/build/pt_nonclustered 512 128 1 5
./arch/build/pt_clustered 64 128 1 20
```

Arguments:

- non-clustered: `N d metric iters`
- clustered: `K d metric iters`

Each binary prints elapsed time and a checksum.

## Suggested Experiments

- Compare **non-clustered** vs **clustered** kernel behavior (runtime and memory footprint)
- Sweep cache sizes / memory latency settings in your simulator
- Track cycles, IPC, cache misses, and memory stall behavior
- Vary `N`, `K`, and `d` to study scaling trends

For the final plaintext/FHE architecture simulation split, see:

- `docs/ARCH_SIM_SCAN_README.md`

## Design Notes

- The dataset and exported benchmark sizes are intentionally small to keep simulator runs practical.
- We isolate retrieval-distance computation because that is the architecture-sensitive hotspot relevant to this project.
- Full RAG/generation complexity is intentionally out of scope.
