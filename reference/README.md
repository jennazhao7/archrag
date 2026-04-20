# Reference

This folder is for optional reference material and external pointers.

## OpenFHE Reference Source

The original OpenFHE distance pipeline used for porting lives outside this repo:

- `/home/jzhao7/RAGPIR/openfhe_core`

Minimal code path that was ported into `src/fhe/fhe_wrapper.py`:

- `openfhe_keygen`
- `openfhe_encrypt_query`
- `openfhe_compute_distances`
- `openfhe_decrypt_topk`

This repo intentionally keeps only the minimal interface and orchestration logic,
not the full legacy project layout.
