"""Verify Stage 4 plaintext retrieval behavior.

This script runs one query through:
1) non-clustered retrieval
2) clustered retrieval

Then it prints top-5 outputs, compares score ranges, reports overlap, and
asserts basic ranked-output validity for both modes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify Stage 4 plaintext retrieval outputs."
    )
    parser.add_argument(
        "--query",
        default="aspirin reduces heart attack risk",
        help="Query text used for both retrieval modes.",
    )
    parser.add_argument(
        "--metric",
        choices=("dot", "l2"),
        default="dot",
        help="Scoring metric used for both retrieval modes.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top results to display/check.",
    )
    parser.add_argument(
        "--num-clusters",
        type=int,
        default=1,
        help="Number of top clusters for clustered retrieval.",
    )
    parser.add_argument(
        "--doc-embeddings",
        default="data/processed/doc_embeddings.npy",
        help="Path to document embeddings.",
    )
    parser.add_argument(
        "--doc-ids",
        default="data/processed/doc_ids.json",
        help="Path to document IDs.",
    )
    parser.add_argument(
        "--cluster-centroids",
        default="data/processed/cluster_centroids.npy",
        help="Path to cluster centroids.",
    )
    parser.add_argument(
        "--cluster-doc-ids",
        default="data/processed/cluster_doc_ids.json",
        help="Path to per-cluster document IDs map.",
    )
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-transformer model name used for query encoding.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional model device (cpu/cuda).",
    )
    parser.add_argument(
        "--normalize-query",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to normalize query embedding before scoring.",
    )
    return parser


def _assert_valid_ranked_output(name: str, result: dict[str, Any], expected_top_k: int) -> None:
    records = result.get("results")
    if not isinstance(records, list):
        raise ValueError(f"{name}: 'results' must be a list.")
    if len(records) == 0:
        raise ValueError(f"{name}: 'results' must be non-empty.")
    if len(records) > expected_top_k:
        raise ValueError(f"{name}: returned more than requested top-k.")

    prev_score: float | None = None
    for idx, row in enumerate(records):
        if not isinstance(row, dict):
            raise ValueError(f"{name}: result at index {idx} must be an object.")
        if "doc_id" not in row or "score" not in row:
            raise ValueError(f"{name}: result at index {idx} must include doc_id and score.")
        doc_id = str(row["doc_id"]).strip()
        if not doc_id:
            raise ValueError(f"{name}: result at index {idx} has empty doc_id.")
        score = float(row["score"])
        if prev_score is not None and score > prev_score + 1e-8:
            raise ValueError(f"{name}: scores are not ranked in descending order.")
        prev_score = score


def _print_top5(name: str, result: dict[str, Any]) -> None:
    records = result["results"]
    print(f"\n{name} top-{len(records)}:")
    for idx, row in enumerate(records, start=1):
        print(f"{idx:>2}. doc_id={row['doc_id']} score={float(row['score']):.6f}")


def _score_range(result: dict[str, Any]) -> tuple[float, float]:
    scores = [float(row["score"]) for row in result["results"]]
    return min(scores), max(scores)


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    args = _build_parser().parse_args()

    try:
        from retrieval.clustered import retrieve_clustered
        from retrieval.plaintext import retrieve_non_clustered

        non_clustered = retrieve_non_clustered(
            query_text=args.query,
            doc_embeddings_path=args.doc_embeddings,
            doc_ids_path=args.doc_ids,
            model_name=args.model_name,
            device=args.device,
            normalize_query_embedding=args.normalize_query,
            metric=args.metric,
            top_k=args.top_k,
        )
        clustered = retrieve_clustered(
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

        _assert_valid_ranked_output("non-clustered", non_clustered, args.top_k)
        _assert_valid_ranked_output("clustered", clustered, args.top_k)

        _print_top5("Non-clustered", non_clustered)
        _print_top5("Clustered", clustered)

        nc_min, nc_max = _score_range(non_clustered)
        c_min, c_max = _score_range(clustered)
        print("\nScore ranges:")
        print(f"- non-clustered: min={nc_min:.6f} max={nc_max:.6f}")
        print(f"- clustered:     min={c_min:.6f} max={c_max:.6f}")

        nc_ids = [str(row["doc_id"]) for row in non_clustered["results"]]
        c_ids = [str(row["doc_id"]) for row in clustered["results"]]
        overlap = sorted(set(nc_ids).intersection(c_ids))
        print(f"\nTop-{args.top_k} overlap count: {len(overlap)}")
        print(f"Overlap doc_ids: {overlap}")

        print("\nStage 4 plaintext verification: SUCCESS")
        return 0
    except Exception as exc:  # broad by design for user-facing verification
        print(f"Stage 4 plaintext verification: FAILURE - {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
