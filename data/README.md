# Data Layout

- `raw/`: source datasets (downloaded/original form)
- `processed/`: pipeline artifacts (embeddings, IDs, clusters)
- `exported/`: compact benchmark packages for architecture experiments

Notes:

- Local runtime caches (e.g., Hugging Face cache, FHE runtime scratch) are
  treated as non-source artifacts and ignored via `.gitignore`.
