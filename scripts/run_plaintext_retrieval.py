"""CLI for plaintext retrieval stage (non-clustered and clustered).

Usage:
    python scripts/run_plaintext_retrieval.py --query "covid treatment efficacy"
    python scripts/run_plaintext_retrieval.py --mode clustered --num-clusters 2 --query "..."
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run plaintext retrieval over document embeddings."
    )
    parser.add_argument(
        "--mode",
        choices=("non-clustered", "clustered"),
        default="non-clustered",
        help="Retrieval mode.",
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Raw query text to embed and retrieve against.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top documents to return.",
    )
    parser.add_argument(
        "--metric",
        choices=("dot", "l2"),
        default="dot",
        help="Similarity metric.",
    )
    parser.add_argument(
        "--doc-embeddings",
        default="data/processed/doc_embeddings.npy",
        help="Path to document embeddings .npy file.",
    )
    parser.add_argument(
        "--doc-ids",
        default="data/processed/doc_ids.json",
        help="Path to ordered document IDs JSON file.",
    )
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-transformer model name for query embedding.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional model device (e.g., cpu, cuda).",
    )
    parser.add_argument(
        "--normalize-query",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to normalize query embedding before scoring.",
    )
    parser.add_argument(
        "--cluster-centroids",
        default="data/processed/cluster_centroids.npy",
        help="Path to cluster centroid matrix (.npy). Used for clustered mode.",
    )
    parser.add_argument(
        "--cluster-doc-ids",
        default="data/processed/cluster_doc_ids.json",
        help="Path to per-cluster doc IDs mapping JSON. Used for clustered mode.",
    )
    parser.add_argument(
        "--num-clusters",
        type=int,
        default=1,
        help="Number of best clusters to search in clustered mode.",
    )
    return parser


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    args = _build_parser().parse_args()

    if args.mode == "non-clustered":
        from retrieval.plaintext import retrieve_non_clustered

        result = retrieve_non_clustered(
            query_text=args.query,
            doc_embeddings_path=args.doc_embeddings,
            doc_ids_path=args.doc_ids,
            model_name=args.model_name,
            device=args.device,
            normalize_query_embedding=args.normalize_query,
            metric=args.metric,
            top_k=args.top_k,
        )
    else:
        from retrieval.clustered import retrieve_clustered

        result = retrieve_clustered(
            query_text=args.query,
            doc_embeddings_path=args.doc_embeddings,
            doc_ids_path=args.doc_ids,
            cluster_centroids_path=args.cluster_centroids,
            cluster_doc_ids_path=args.cluster_doc_ids,
            model_name=args.model_name,
            device=args.device,
            normalize_query_embedding=args.normalize_query,
            metric=args.metric,
            top_k=args.top_k,
            num_clusters=args.num_clusters,
        )

    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
