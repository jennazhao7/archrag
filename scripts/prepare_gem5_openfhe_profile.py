"""Prepare real OpenFHE artifacts for gem5 distance-kernel profiling.

This script runs the native pipeline steps that are practical outside gem5:
query embedding, OpenFHE context/key generation, query encryption, and centroid
sampling. The generated manifest/env file can be passed to run_gem5_openfhe.sh
to simulate only the encrypted distance hotspot.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare OpenFHE query/centroid artifacts for gem5 profiling."
    )
    parser.add_argument(
        "--query",
        default="aspirin reduces heart attack risk",
        help="Query text to embed and encrypt when precomputed embeddings are disabled.",
    )
    parser.add_argument(
        "--use-precomputed-query-embedding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use a row from query_embeddings.npy instead of loading a sentence-transformer.",
    )
    parser.add_argument(
        "--query-embeddings",
        default="data/processed/query_embeddings.npy",
        help="Path to precomputed query embeddings .npy file.",
    )
    parser.add_argument(
        "--query-ids",
        default="data/processed/query_ids.json",
        help="Path to query IDs JSON file used with --query-id.",
    )
    parser.add_argument(
        "--query-index",
        type=int,
        default=0,
        help="Query embedding row to use when --query-id is not provided.",
    )
    parser.add_argument(
        "--query-id",
        default=None,
        help="Optional query ID to select from query_ids.json.",
    )
    parser.add_argument(
        "--cluster-centroids",
        default="data/processed/cluster_centroids.npy",
        help="Path to full centroid matrix.",
    )
    parser.add_argument(
        "--sample-centroids",
        type=int,
        default=1,
        help="Number of centroid rows to include in the gem5 workload.",
    )
    parser.add_argument(
        "--centroid-start",
        type=int,
        default=0,
        help="First centroid row to include when sampling a contiguous block.",
    )
    parser.add_argument(
        "--centroid-indices",
        default=None,
        help="Optional comma-separated centroid indices. Overrides contiguous sampling.",
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
        help="Whether to normalize the query embedding.",
    )
    parser.add_argument(
        "--openfhe-bin-dir",
        default=os.environ.get("OPENFHE_BIN_DIR", "bin/openfhe"),
        help="Directory containing openfhe_keygen and openfhe_encrypt_query.",
    )
    parser.add_argument(
        "--work-dir",
        default="data/fhe_runtime_gem5_profile",
        help="Output directory for context, encrypted query, and sampled centroids.",
    )
    parser.add_argument(
        "--poly-modulus-degree",
        type=int,
        default=16384,
        help="OpenFHE CKKS ring dimension.",
    )
    parser.add_argument(
        "--coeff-mod-bit-sizes",
        default="60,40,40,60",
        help="Comma-separated OpenFHE CKKS coeff mod bit sizes.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size to record for the gem5 distance workload.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Thread count to record for the gem5 distance workload.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional manifest JSON path. Defaults to <work-dir>/gem5_profile_manifest.json.",
    )
    parser.add_argument(
        "--env-file",
        default=None,
        help="Optional shell env file path. Defaults to <work-dir>/gem5_profile.env.",
    )
    return parser


def _parse_coeff_sizes(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("At least one coeff mod bit size is required.")
    return values


def _select_centroids(centroids: np.ndarray, *, args: argparse.Namespace) -> tuple[np.ndarray, list[int]]:
    if centroids.ndim != 2:
        raise ValueError("Centroids must be a 2D matrix.")
    if args.centroid_indices:
        indices = [int(part.strip()) for part in args.centroid_indices.split(",") if part.strip()]
    else:
        if args.sample_centroids <= 0:
            raise ValueError("--sample-centroids must be > 0.")
        indices = list(range(args.centroid_start, args.centroid_start + args.sample_centroids))
    if not indices:
        raise ValueError("No centroid indices selected.")
    if min(indices) < 0 or max(indices) >= centroids.shape[0]:
        raise ValueError(
            f"Centroid index out of range for matrix with {centroids.shape[0]} rows."
        )
    return np.asarray(centroids[indices], dtype=np.float64), indices


def _write_env_file(path: Path, values: dict[str, str | int]) -> None:
    lines = [f"export {key}={shlex.quote(str(value))}" for key, value in values.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_precomputed_query_embedding(args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    query_embeddings = np.load(Path(args.query_embeddings))
    if query_embeddings.ndim != 2:
        raise ValueError("Query embeddings must be a 2D matrix.")

    query_index = int(args.query_index)
    query_id = args.query_id
    if query_id is not None:
        query_ids = json.loads(Path(args.query_ids).read_text(encoding="utf-8"))
        if not isinstance(query_ids, list):
            raise ValueError("Query IDs file must contain a JSON list.")
        query_ids = [str(value) for value in query_ids]
        try:
            query_index = query_ids.index(str(query_id))
        except ValueError as exc:
            raise ValueError(f"Query ID not found in {args.query_ids}: {query_id}") from exc
    if query_index < 0 or query_index >= query_embeddings.shape[0]:
        raise ValueError(
            f"Query index {query_index} out of range for {query_embeddings.shape[0]} query rows."
        )

    metadata = {
        "source": "precomputed",
        "query_embeddings": str(Path(args.query_embeddings)),
        "query_ids": str(Path(args.query_ids)),
        "query_index": query_index,
        "query_id": str(query_id) if query_id is not None else None,
    }
    return np.asarray(query_embeddings[query_index], dtype=np.float32), metadata


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    args = _build_parser().parse_args()

    from fhe.fhe_wrapper import OpenFHECiphertextBundle, create_backend

    work_dir = Path(args.work_dir)
    sample_dir = work_dir / "gem5_profile"
    sample_dir.mkdir(parents=True, exist_ok=True)

    centroids = np.load(Path(args.cluster_centroids))
    sampled_centroids, sampled_indices = _select_centroids(centroids, args=args)
    centroids_file = sample_dir / "centroids.txt"
    with centroids_file.open("w", encoding="utf-8") as handle:
        for row in sampled_centroids:
            handle.write(" ".join(f"{float(value):.12g}" for value in row))
            handle.write("\n")

    if args.use_precomputed_query_embedding:
        query_embedding, query_metadata = _load_precomputed_query_embedding(args)
    else:
        from retrieval.plaintext import encode_query

        query_embedding = encode_query(
            args.query,
            model_name=args.model_name,
            device=args.device,
            normalize_embedding=args.normalize_query,
        )
        query_metadata = {
            "source": "encoded_text",
            "query_text": args.query,
            "model_name": args.model_name,
            "normalize_query": bool(args.normalize_query),
        }

    coeff_sizes = _parse_coeff_sizes(args.coeff_mod_bit_sizes)
    backend = create_backend(
        "openfhe",
        binary_dir=args.openfhe_bin_dir,
        work_dir=work_dir,
        poly_modulus_degree=args.poly_modulus_degree,
        coeff_mod_bit_sizes=coeff_sizes,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
    )
    context = backend.setup_context()
    encrypted_query = backend.encrypt_query_embedding(query_embedding, context=context)
    if not isinstance(encrypted_query.payload, OpenFHECiphertextBundle):
        raise ValueError("Unexpected OpenFHE encrypted-query payload.")

    gem5_output_dir = sample_dir / "gem5_distances"
    gem5_output_dir.mkdir(parents=True, exist_ok=True)

    env_values: dict[str, str | int] = {
        "OPENFHE_BIN_DIR": str(Path(args.openfhe_bin_dir).resolve()),
        "OPENFHE_CONTEXT_DIR": str(Path(context.params["context_dir"])),
        "OPENFHE_CENTROIDS_FILE": str(centroids_file),
        "OPENFHE_ENCRYPTED_QUERY": str(encrypted_query.payload.encrypted_query_path),
        "OPENFHE_ENCRYPTED_NORM": str(encrypted_query.payload.encrypted_norm_path),
        "OPENFHE_OUTPUT_DIR": str(gem5_output_dir),
        "OPENFHE_BATCH_SIZE": max(1, int(args.batch_size)),
        "OPENFHE_NUM_THREADS": max(0, int(args.num_threads)),
    }

    manifest_path = Path(args.manifest) if args.manifest else work_dir / "gem5_profile_manifest.json"
    env_file = Path(args.env_file) if args.env_file else work_dir / "gem5_profile.env"

    manifest: dict[str, Any] = {
        "query": args.query,
        "query_embedding": query_metadata,
        "model_name": args.model_name,
        "normalize_query": bool(args.normalize_query),
        "cluster_centroids": str(Path(args.cluster_centroids)),
        "sampled_centroid_indices": sampled_indices,
        "sampled_centroid_shape": list(sampled_centroids.shape),
        "query_embedding_dim": int(query_embedding.shape[0]),
        "openfhe": {
            "binary_dir": str(Path(args.openfhe_bin_dir).resolve()),
            "context_dir": env_values["OPENFHE_CONTEXT_DIR"],
            "poly_modulus_degree": int(args.poly_modulus_degree),
            "coeff_mod_bit_sizes": coeff_sizes,
            "batch_size": env_values["OPENFHE_BATCH_SIZE"],
            "num_threads": env_values["OPENFHE_NUM_THREADS"],
        },
        "gem5_env": env_values,
        "env_file": str(env_file),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    _write_env_file(env_file, env_values)

    print("Prepared OpenFHE gem5 profile artifacts:")
    print(f"- manifest: {manifest_path}")
    print(f"- env file: {env_file}")
    print(f"- sampled centroids: {centroids_file}")
    print(f"- encrypted query: {encrypted_query.payload.encrypted_query_path}")
    print(f"- encrypted norm: {encrypted_query.payload.encrypted_norm_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
