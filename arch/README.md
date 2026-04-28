# Architecture Kernels (Plaintext Only)

This folder is intentionally minimal for architecture experiments (e.g., gem5).

## Files

- `kernels/pt_nonclustered.cpp`: standalone query-vs-all-docs kernel + test `main()`
- `kernels/pt_clustered.cpp`: standalone query-vs-all-centroids kernel + test `main()`
- `Makefile`: builds standalone executables under `build/`

## Kernel API

Metric enum in both files:

- `0` => dot product
- `1` => negative L2 squared distance

Each kernel function operates on flat row-major float arrays and writes scores to output arrays:

- `pt_nonclustered_scores(query, docs, N, d, metric, out_scores)`
- `pt_clustered_scores(query, centroids, K, d, metric, out_scores)`

## Build

From repo root:

```bash
make -C arch
```

Outputs:

- `arch/build/pt_nonclustered`
- `arch/build/pt_clustered`

Quick run examples:

```bash
# Non-clustered full scan: N=4096 document vectors, d=384.
./arch/build/pt_nonclustered 4096 384 1 3

# Clustered centroid scan: K=64 centroids, d=384.
./arch/build/pt_clustered 64 384 1 10
```

Argument order:

- non-clustered: `N d metric iters`
- clustered: `K d metric iters`

Each program prints elapsed milliseconds and a checksum for quick validation.
