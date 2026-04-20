"""CLI for standalone document-embedding clustering stage.

Usage:
    python scripts/cluster_db.py
    python scripts/cluster_db.py --k 128 --seed 7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    """Create command-line parser for clustering stage."""
    parser = argparse.ArgumentParser(
        description="Cluster document embeddings into K centroids and save artifacts."
    )
    parser.add_argument(
        "--embeddings",
        default="data/processed/doc_embeddings.npy",
        help="Path to document embeddings .npy file.",
    )
    parser.add_argument(
        "--doc-ids",
        default="data/processed/doc_ids.json",
        help="Path to ordered document IDs JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory where clustering artifacts are written.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=64,
        help="Number of clusters.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic centroid initialization.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=100,
        help="Maximum K-means iterations.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-6,
        help="Convergence tolerance on max centroid shift.",
    )
    return parser


def main() -> int:
    """Run clustering stage only (no retrieval/index build)."""
    project_root = Path(__file__).resolve().parents[1]
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    args = _build_parser().parse_args()

    from cluster.kmeans_cluster import cluster_and_save

    result = cluster_and_save(
        embeddings_path=args.embeddings,
        doc_ids_path=args.doc_ids,
        output_dir=args.output_dir,
        k=args.k,
        seed=args.seed,
        max_iter=args.max_iter,
        tol=args.tol,
    )

    print(
        "Clustering complete "
        f"(k={result['k']}, seed={result['seed']}, docs={result['num_documents']}, dim={result['embedding_dim']})"
    )
    print("Saved clustering artifacts:")
    for name, path in result["artifacts"].items():
        print(f"- {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
