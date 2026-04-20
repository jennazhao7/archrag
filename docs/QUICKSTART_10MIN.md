# 10-Minute Quickstart

## 1) Run the core pipeline

```bash
python scripts/download_scifact.py
python scripts/build_embeddings.py
python scripts/cluster_db.py --k 64 --seed 42
```

## 2) Run plaintext retrieval

```bash
python scripts/run_plaintext_retrieval.py --mode non-clustered --query "aspirin reduces heart attack risk" --top-k 5
python scripts/run_plaintext_retrieval.py --mode clustered --query "aspirin reduces heart attack risk" --top-k 5 --num-clusters 1
```

## 3) Export compact architecture benchmark data

```bash
python scripts/export_arch_bench_data.py --centroid-count 16 --include-docs --doc-count 64
```

Exports land in `data/exported/arch_bench/`.

## 4) Build plaintext architecture kernels

```bash
make -C arch
```

Static library output:

- `arch/build/libptkernels.a`
