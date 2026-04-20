"""Cluster-aware plaintext retrieval.

This stage first scores the query against cluster centroids, then ranks
documents only within the top cluster(s).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from retrieval.plaintext import (
    DEFAULT_DOC_EMBEDDINGS_PATH,
    DEFAULT_DOC_IDS_PATH,
    DEFAULT_MODEL_NAME,
    compute_similarity_scores,
    encode_query,
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
    """Load centroid matrix from .npy file."""
    centroids = np.load(Path(centroids_path))
    if centroids.ndim != 2:
        raise ValueError("Cluster centroids must be a 2D numpy array.")
    return np.asarray(centroids, dtype=np.float32)


def load_cluster_doc_ids(
    cluster_doc_ids_path: str | Path = DEFAULT_CLUSTER_DOC_IDS_PATH,
) -> dict[str, list[str]]:
    """Load per-cluster ordered document ID lists."""
    payload = json.loads(Path(cluster_doc_ids_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Cluster doc IDs file must contain a JSON object.")

    parsed: dict[str, list[str]] = {}
    for key, value in payload.items():
        if not isinstance(value, list):
            raise ValueError("Each cluster entry must be a list of doc IDs.")
        parsed[str(key)] = [str(doc_id) for doc_id in value]
    return parsed


def retrieve_clustered(
    *,
    query_text: str,
    doc_embeddings_path: str | Path = DEFAULT_DOC_EMBEDDINGS_PATH,
    doc_ids_path: str | Path = DEFAULT_DOC_IDS_PATH,
    cluster_centroids_path: str | Path = DEFAULT_CLUSTER_CENTROIDS_PATH,
    cluster_doc_ids_path: str | Path = DEFAULT_CLUSTER_DOC_IDS_PATH,
    model_name: str = DEFAULT_MODEL_NAME,
    device: str | None = None,
    normalize_query_embedding: bool = True,
    metric: str = "dot",
    top_k: int = 10,
    num_clusters: int = 1,
) -> dict[str, object]:
    """Run centroid-first retrieval and rank docs in top cluster(s)."""
    metric_name = validate_metric(metric)
    if num_clusters <= 0:
        raise ValueError("num_clusters must be > 0.")

    doc_embeddings = load_doc_embeddings(doc_embeddings_path)
    doc_ids = load_doc_ids(doc_ids_path)
    if len(doc_ids) != doc_embeddings.shape[0]:
        raise ValueError(
            f"Document IDs count ({len(doc_ids)}) must match embedding rows ({doc_embeddings.shape[0]})."
        )

    centroids = load_cluster_centroids(cluster_centroids_path)
    cluster_doc_ids = load_cluster_doc_ids(cluster_doc_ids_path)
    if centroids.shape[1] != doc_embeddings.shape[1]:
        raise ValueError("Centroid embedding dimension must match document embedding dimension.")

    query_embedding = encode_query(
        query_text,
        model_name=model_name,
        device=device,
        normalize_embedding=normalize_query_embedding,
    )

    centroid_scores = compute_similarity_scores(query_embedding, centroids, metric=metric_name)
    top_cluster_idx = top_k_indices(centroid_scores, k=min(num_clusters, centroids.shape[0]))
    selected_cluster_ids = [str(cluster_idx) for cluster_idx in top_cluster_idx.tolist()]

    doc_id_to_index = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
    candidate_indices: list[int] = []
    seen: set[int] = set()
    for cluster_id in selected_cluster_ids:
        for doc_id in cluster_doc_ids.get(cluster_id, []):
            idx = doc_id_to_index.get(doc_id)
            if idx is not None and idx not in seen:
                seen.add(idx)
                candidate_indices.append(idx)

    if not candidate_indices:
        return {
            "mode": "clustered",
            "metric": metric_name,
            "selected_cluster_ids": selected_cluster_ids,
            "top_k": 0,
            "results": [],
        }

    candidate_array = np.asarray(candidate_indices, dtype=np.int32)
    candidate_embeddings = doc_embeddings[candidate_array]
    candidate_scores = compute_similarity_scores(
        query_embedding,
        candidate_embeddings,
        metric=metric_name,
    )
    local_ranked = top_k_indices(candidate_scores, k=min(top_k, candidate_scores.shape[0]))
    ranked_global_indices = candidate_array[local_ranked]
    global_scores = np.asarray(candidate_scores[local_ranked], dtype=np.float32)
    ranked_doc_ids = [doc_ids[idx] for idx in ranked_global_indices.tolist()]

    results = [
        {"doc_id": doc_id, "score": float(score)}
        for doc_id, score in zip(ranked_doc_ids, global_scores.tolist())
    ]

    return {
        "mode": "clustered",
        "metric": metric_name,
        "selected_cluster_ids": selected_cluster_ids,
        "top_k": min(top_k, len(candidate_indices)),
        "results": results,
    }
