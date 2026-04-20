"""Compare plaintext vs FHE centroid scoring for one clustered query.

This script runs:
1) plaintext clustered retrieval
2) FHE clustered retrieval (same query + centroids)
3) direct centroid-score comparison between plaintext and decrypted FHE scores

It reports max/mean absolute error and top-k centroid overlap, and asserts the
error is within a configurable tolerance for toy validation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify FHE clustered centroid scores against plaintext baseline."
    )
    parser.add_argument(
        "--query",
        default="aspirin reduces heart attack risk",
        help="Query text to evaluate.",
    )
    parser.add_argument(
        "--metric",
        choices=("dot", "l2"),
        default="l2",
        help="Distance/similarity metric for centroid scoring.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-k centroids used for overlap comparison.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-5,
        help="Maximum allowed absolute error for toy verification.",
    )
    parser.add_argument(
        "--backend",
        default="mock",
        help="FHE backend name ('mock' or 'openfhe').",
    )
    parser.add_argument(
        "--doc-embeddings",
        default="data/processed/doc_embeddings.npy",
        help="Path to document embeddings for clustered retrieval call.",
    )
    parser.add_argument(
        "--doc-ids",
        default="data/processed/doc_ids.json",
        help="Path to document ids.",
    )
    parser.add_argument(
        "--cluster-centroids",
        default="data/processed/cluster_centroids.npy",
        help="Path to centroid matrix.",
    )
    parser.add_argument(
        "--cluster-doc-ids",
        default="data/processed/cluster_doc_ids.json",
        help="Path to cluster doc-id lists.",
    )
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-transformer model for query embedding.",
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
        help="Whether query embedding should be normalized.",
    )
    parser.add_argument(
        "--num-clusters",
        type=int,
        default=1,
        help="Clusters selected in clustered retrieval calls (not vector compare length).",
    )
    parser.add_argument(
        "--openfhe-bin-dir",
        default=None,
        help="Optional OpenFHE binary dir when --backend openfhe.",
    )
    parser.add_argument(
        "--openfhe-work-dir",
        default="data/fhe_runtime",
        help="Working directory for OpenFHE runtime artifacts.",
    )
    parser.add_argument(
        "--openfhe-poly-modulus-degree",
        type=int,
        default=16384,
        help="OpenFHE CKKS ring dimension (e.g., 16384).",
    )
    parser.add_argument(
        "--openfhe-coeff-mod-bit-sizes",
        default="60,40,40,60",
        help="Comma-separated OpenFHE CKKS coeff mod bit sizes.",
    )
    parser.add_argument(
        "--openfhe-batch-size",
        type=int,
        default=64,
        help="Batch size for OpenFHE distance kernel.",
    )
    parser.add_argument(
        "--openfhe-num-threads",
        type=int,
        default=0,
        help="OpenFHE distance kernel threads (0 lets runtime decide).",
    )
    return parser


def _topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        raise ValueError("top-k must be > 0.")
    k_eff = min(k, scores.shape[0])
    if k_eff == scores.shape[0]:
        return np.argsort(scores)[::-1]
    part = np.argpartition(scores, -k_eff)[-k_eff:]
    return part[np.argsort(scores[part])[::-1]]


def _print_topk(name: str, scores: np.ndarray, k: int) -> None:
    idx = _topk_indices(scores, k)
    print(f"\n{name} top-{len(idx)} centroid scores:")
    for rank, centroid_idx in enumerate(idx.tolist(), start=1):
        print(f"{rank:>2}. centroid={centroid_idx} score={float(scores[centroid_idx]):.6f}")


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    args = _build_parser().parse_args()

    try:
        from fhe.fhe_clustered_retrieval import fhe_clustered_retrieve, load_cluster_centroids
        from fhe.fhe_wrapper import create_backend
        from retrieval.clustered import retrieve_clustered
        from retrieval.plaintext import compute_similarity_scores, encode_query

        # 1) Run plaintext clustered retrieval
        plaintext_clustered = retrieve_clustered(
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
        if not isinstance(plaintext_clustered.get("results"), list) or len(plaintext_clustered["results"]) == 0:
            raise ValueError("Plaintext clustered retrieval returned empty/invalid results.")

        # 2) Run FHE clustered retrieval (same query/centroids), returning centroid stage only
        backend_options: dict[str, Any] = {}
        if args.backend.lower() == "openfhe":
            coeff_sizes = [
                int(part.strip())
                for part in str(args.openfhe_coeff_mod_bit_sizes).split(",")
                if part.strip()
            ]
            backend_options["work_dir"] = args.openfhe_work_dir
            if args.openfhe_bin_dir:
                backend_options["binary_dir"] = args.openfhe_bin_dir
            backend_options["poly_modulus_degree"] = int(args.openfhe_poly_modulus_degree)
            backend_options["coeff_mod_bit_sizes"] = coeff_sizes
            backend_options["batch_size"] = int(args.openfhe_batch_size)
            backend_options["num_threads"] = int(args.openfhe_num_threads)

        fhe_clustered = fhe_clustered_retrieve(
            query_embedding=encode_query(
                args.query,
                model_name=args.model_name,
                device=args.device,
                normalize_embedding=args.normalize_query,
            ),
            doc_embeddings_path=args.doc_embeddings,
            doc_ids_path=args.doc_ids,
            cluster_centroids_path=args.cluster_centroids,
            cluster_doc_ids_path=args.cluster_doc_ids,
            metric=args.metric,
            top_k=args.top_k,
            num_clusters=args.num_clusters,
            backend_name=args.backend,
            backend_options=backend_options,
            rank_within_selected_clusters=False,
        )
        if not isinstance(fhe_clustered.get("selected_cluster_ids"), list):
            raise ValueError("FHE clustered retrieval returned invalid selected_cluster_ids.")

        # 3) Compare full centroid score vectors: plaintext vs decrypted FHE
        query_embedding = encode_query(
            args.query,
            model_name=args.model_name,
            device=args.device,
            normalize_embedding=args.normalize_query,
        )
        centroids = load_cluster_centroids(args.cluster_centroids)
        plaintext_scores = compute_similarity_scores(query_embedding, centroids, metric=args.metric)

        backend = create_backend(args.backend, **backend_options)
        context = backend.setup_context()
        encrypted_query = backend.encrypt_query_embedding(query_embedding, context=context)
        encrypted_scores = backend.encrypted_similarity_to_plaintext(
            encrypted_query,
            centroids,
            metric=args.metric,
            context=context,
        )
        decrypted_scores = backend.decrypt_scores(encrypted_scores, context=context)
        if plaintext_scores.shape != decrypted_scores.shape:
            raise ValueError("Plaintext and decrypted score vectors have different shapes.")

        abs_err = np.abs(plaintext_scores - decrypted_scores)
        max_abs_error = float(np.max(abs_err))
        mean_abs_error = float(np.mean(abs_err))

        pt_topk = _topk_indices(plaintext_scores, args.top_k)
        fhe_topk = _topk_indices(decrypted_scores, args.top_k)
        overlap = sorted(set(pt_topk.tolist()).intersection(fhe_topk.tolist()))

        _print_topk("Plaintext", plaintext_scores, args.top_k)
        _print_topk("FHE decrypted", decrypted_scores, args.top_k)

        print("\nVector comparison metrics:")
        print(f"- max absolute error:  {max_abs_error:.8f}")
        print(f"- mean absolute error: {mean_abs_error:.8f}")
        print(f"- top-{min(args.top_k, plaintext_scores.shape[0])} centroid overlap: {len(overlap)}")
        print(f"- overlap centroid ids: {overlap}")

        if max_abs_error > args.tolerance:
            raise AssertionError(
                f"Max absolute error {max_abs_error:.8f} exceeds tolerance {args.tolerance:.8f}."
            )

        print("\nFHE vs plaintext centroid-score verification: SUCCESS")
        return 0
    except Exception as exc:  # broad by design for verification script UX
        print(f"FHE vs plaintext centroid-score verification: FAILURE - {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
