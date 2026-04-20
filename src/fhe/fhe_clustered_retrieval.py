"""Clustered retrieval orchestration with an FHE backend boundary.

This module wires the "pipeline shape" for FHE usage:
1) setup context/keys
2) encrypt query embedding
3) score encrypted query against plaintext centroids
4) decrypt centroid scores
5) (optional) rank docs in selected clusters in plaintext
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fhe.fhe_wrapper import FHEBackend, Metric, create_backend
from retrieval.plaintext import (
    DEFAULT_DOC_EMBEDDINGS_PATH,
    DEFAULT_DOC_IDS_PATH,
    compute_similarity_scores,
    load_doc_embeddings,
    load_doc_ids,
    top_k_indices,
    validate_metric,
)

DEFAULT_CLUSTER_CENTROIDS_PATH = Path("data/processed/cluster_centroids.npy")
DEFAULT_CLUSTER_DOC_IDS_PATH = Path("data/processed/cluster_doc_ids.json")


def load_cluster_centroids(
    centroids_path: str | Path = DEFAULT_CLUSTER_CENTROIDS_PATH,
) -> np.ndarray:
    """Load clustering centroid matrix."""
    centroids = np.load(Path(centroids_path))
    if centroids.ndim != 2:
        raise ValueError("Cluster centroids must be a 2D numpy array.")
    return np.asarray(centroids, dtype=np.float32)


def load_cluster_doc_ids(
    cluster_doc_ids_path: str | Path = DEFAULT_CLUSTER_DOC_IDS_PATH,
) -> dict[str, list[str]]:
    """Load per-cluster document IDs mapping."""
    payload = json.loads(Path(cluster_doc_ids_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Cluster doc IDs file must contain a JSON object.")
    parsed: dict[str, list[str]] = {}
    for cluster_id, value in payload.items():
        if not isinstance(value, list):
            raise ValueError("Each cluster entry must be a list of doc IDs.")
        parsed[str(cluster_id)] = [str(doc_id) for doc_id in value]
    return parsed


def _select_top_clusters(scores: np.ndarray, *, num_clusters: int) -> list[str]:
    cluster_indices = top_k_indices(scores, k=min(num_clusters, scores.shape[0]))
    return [str(idx) for idx in cluster_indices.tolist()]


def fhe_clustered_retrieve(
    *,
    query_embedding: np.ndarray,
    doc_embeddings_path: str | Path = DEFAULT_DOC_EMBEDDINGS_PATH,
    doc_ids_path: str | Path = DEFAULT_DOC_IDS_PATH,
    cluster_centroids_path: str | Path = DEFAULT_CLUSTER_CENTROIDS_PATH,
    cluster_doc_ids_path: str | Path = DEFAULT_CLUSTER_DOC_IDS_PATH,
    metric: str = "dot",
    top_k: int = 10,
    num_clusters: int = 1,
    backend_name: str = "mock",
    backend_options: dict[str, object] | None = None,
    rank_within_selected_clusters: bool = True,
) -> dict[str, object]:
    """Run centroid-gated retrieval with an FHE backend abstraction.

    Args:
        query_embedding: Plaintext query vector produced upstream.
        metric: 'dot' or 'l2' for both centroid and document scoring.
        top_k: Number of docs to return.
        num_clusters: Number of top centroids/clusters to inspect.
        backend_name: Backend selector ('mock' or 'openfhe').
        backend_options: Optional constructor kwargs passed to create_backend.
        rank_within_selected_clusters: If False, return only selected clusters
            and their decrypted scores.
    """
    metric_name: Metric = validate_metric(metric)
    if top_k <= 0:
        raise ValueError("top_k must be > 0.")
    if num_clusters <= 0:
        raise ValueError("num_clusters must be > 0.")

    backend: FHEBackend = create_backend(backend_name, **(backend_options or {}))
    context = backend.setup_context()

    centroids = load_cluster_centroids(cluster_centroids_path)
    if query_embedding.shape[0] != centroids.shape[1]:
        raise ValueError("Query embedding dimension must match centroid dimension.")

    encrypted_query = backend.encrypt_query_embedding(
        np.asarray(query_embedding, dtype=np.float32),
        context=context,
    )
    encrypted_centroid_scores = backend.encrypted_similarity_to_plaintext(
        encrypted_query,
        centroids,
        metric=metric_name,
        context=context,
    )
    centroid_scores = backend.decrypt_scores(encrypted_centroid_scores, context=context)
    selected_cluster_ids = _select_top_clusters(centroid_scores, num_clusters=num_clusters)

    base_output: dict[str, object] = {
        "mode": "fhe-clustered",
        "backend": backend.backend_name,
        "metric": metric_name,
        "selected_cluster_ids": selected_cluster_ids,
        "centroid_scores": {cluster_id: float(centroid_scores[int(cluster_id)]) for cluster_id in selected_cluster_ids},
        "pipeline": {
            "context_setup": True,
            "query_encrypted": True,
            "centroid_scoring": "encrypted_query_vs_plaintext_centroids",
            "score_decryption": True,
        },
    }

    if not rank_within_selected_clusters:
        base_output["top_k"] = 0
        base_output["results"] = []
        base_output["ranking_stage"] = "skipped"
        return base_output

    doc_embeddings = load_doc_embeddings(doc_embeddings_path)
    doc_ids = load_doc_ids(doc_ids_path)
    cluster_doc_ids = load_cluster_doc_ids(cluster_doc_ids_path)
    if len(doc_ids) != doc_embeddings.shape[0]:
        raise ValueError(
            f"Document IDs count ({len(doc_ids)}) must match embedding rows ({doc_embeddings.shape[0]})."
        )
    if query_embedding.shape[0] != doc_embeddings.shape[1]:
        raise ValueError("Query embedding dimension must match document embedding dimension.")

    doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
    candidate_indices: list[int] = []
    seen: set[int] = set()
    for cluster_id in selected_cluster_ids:
        for doc_id in cluster_doc_ids.get(cluster_id, []):
            idx = doc_id_to_idx.get(doc_id)
            if idx is not None and idx not in seen:
                seen.add(idx)
                candidate_indices.append(idx)

    if not candidate_indices:
        base_output["top_k"] = 0
        base_output["results"] = []
        base_output["ranking_stage"] = "plaintext_in_selected_clusters"
        return base_output

    candidate_array = np.asarray(candidate_indices, dtype=np.int32)
    candidate_embeddings = doc_embeddings[candidate_array]
    candidate_scores = compute_similarity_scores(
        np.asarray(query_embedding, dtype=np.float32),
        candidate_embeddings,
        metric=metric_name,
    )
    local_ranked = top_k_indices(candidate_scores, k=min(top_k, candidate_scores.shape[0]))
    ranked_global = candidate_array[local_ranked]

    results = [
        {
            "doc_id": doc_ids[int(doc_idx)],
            "score": float(candidate_scores[int(local_idx)]),
        }
        for local_idx, doc_idx in zip(local_ranked.tolist(), ranked_global.tolist())
    ]

    base_output["top_k"] = min(top_k, len(candidate_indices))
    base_output["results"] = results
    base_output["ranking_stage"] = "plaintext_in_selected_clusters"
    return base_output
