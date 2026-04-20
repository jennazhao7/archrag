"""Standalone deterministic K-means clustering for document embeddings.

This module is intentionally lightweight and focused on the clustering stage.
It reads embedding artifacts produced by the embedding stage, clusters document
embeddings into K clusters, and saves clustering artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_EMBEDDINGS_PATH = Path("data/processed/doc_embeddings.npy")
DEFAULT_DOC_IDS_PATH = Path("data/processed/doc_ids.json")
DEFAULT_OUTPUT_DIR = Path("data/processed")
DEFAULT_K = 64
DEFAULT_RANDOM_SEED = 42
DEFAULT_MAX_ITER = 100
DEFAULT_TOL = 1e-6


def load_doc_embeddings(embeddings_path: str | Path = DEFAULT_EMBEDDINGS_PATH) -> np.ndarray:
    """Load document embeddings matrix from .npy artifact."""
    embeddings = np.load(Path(embeddings_path))
    if embeddings.ndim != 2:
        raise ValueError("Document embeddings must be a 2D numpy array.")
    return np.asarray(embeddings, dtype=np.float32)


def load_doc_ids(doc_ids_path: str | Path = DEFAULT_DOC_IDS_PATH) -> list[str]:
    """Load ordered document IDs from JSON artifact."""
    payload = json.loads(Path(doc_ids_path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Document IDs file must contain a JSON list.")
    return [str(doc_id) for doc_id in payload]


def _validate_k(k: int, n_samples: int) -> None:
    if k <= 0:
        raise ValueError("K must be > 0.")
    if k > n_samples:
        raise ValueError(f"K ({k}) cannot exceed number of samples ({n_samples}).")


def _initialize_centroids(embeddings: np.ndarray, *, k: int, seed: int) -> np.ndarray:
    """Pick K unique points as initial centroids using a fixed seed."""
    rng = np.random.default_rng(seed)
    indices = rng.choice(embeddings.shape[0], size=k, replace=False)
    return embeddings[indices].copy()


def _assign_clusters(embeddings: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Assign each point to nearest centroid (KNN-style nearest-center step)."""
    # Squared Euclidean distance from each point to each centroid.
    distances = np.sum((embeddings[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    return np.argmin(distances, axis=1).astype(np.int32)


def _handle_empty_cluster(
    embeddings: np.ndarray,
    centroids: np.ndarray,
    assignments: np.ndarray,
    cluster_idx: int,
) -> np.ndarray:
    """Deterministically repair empty clusters by re-seeding from hardest point."""
    distances = np.sum((embeddings - centroids[assignments]) ** 2, axis=1)
    farthest_point_idx = int(np.argmax(distances))
    return embeddings[farthest_point_idx]


def _update_centroids(
    embeddings: np.ndarray,
    assignments: np.ndarray,
    centroids: np.ndarray,
) -> np.ndarray:
    """Recompute centroids as means of assigned points."""
    k = centroids.shape[0]
    new_centroids = centroids.copy()
    for cluster_idx in range(k):
        member_mask = assignments == cluster_idx
        if np.any(member_mask):
            new_centroids[cluster_idx] = embeddings[member_mask].mean(axis=0)
        else:
            new_centroids[cluster_idx] = _handle_empty_cluster(
                embeddings, centroids, assignments, cluster_idx
            )
    return new_centroids


def run_kmeans(
    embeddings: np.ndarray,
    *,
    k: int = DEFAULT_K,
    seed: int = DEFAULT_RANDOM_SEED,
    max_iter: int = DEFAULT_MAX_ITER,
    tol: float = DEFAULT_TOL,
) -> tuple[np.ndarray, np.ndarray]:
    """Run deterministic K-means and return (centroids, assignments)."""
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D matrix.")
    if max_iter <= 0:
        raise ValueError("max_iter must be > 0.")
    if tol < 0:
        raise ValueError("tol must be >= 0.")

    n_samples = embeddings.shape[0]
    _validate_k(k, n_samples)

    centroids = _initialize_centroids(embeddings, k=k, seed=seed)
    assignments = np.zeros(n_samples, dtype=np.int32)

    for _ in range(max_iter):
        assignments = _assign_clusters(embeddings, centroids)
        updated = _update_centroids(embeddings, assignments, centroids)
        shift = float(np.max(np.linalg.norm(updated - centroids, axis=1)))
        centroids = updated
        if shift <= tol:
            break

    return centroids.astype(np.float32), assignments


def build_cluster_doc_lists(assignments: np.ndarray, doc_ids: list[str], *, k: int) -> dict[str, list[str]]:
    """Create cluster_id -> ordered list of document IDs mapping."""
    if len(doc_ids) != int(assignments.shape[0]):
        raise ValueError("Document IDs length must match assignments length.")

    cluster_to_doc_ids: dict[str, list[str]] = {str(cluster_idx): [] for cluster_idx in range(k)}
    for row_idx, cluster_idx in enumerate(assignments.tolist()):
        cluster_to_doc_ids[str(int(cluster_idx))].append(doc_ids[row_idx])
    return cluster_to_doc_ids


def save_cluster_outputs(
    *,
    output_dir: str | Path,
    centroids: np.ndarray,
    assignments: np.ndarray,
    cluster_to_doc_ids: dict[str, list[str]],
) -> dict[str, Path]:
    """Write clustering artifacts and return resolved output paths."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    centroids_path = output / "cluster_centroids.npy"
    assignments_path = output / "doc_cluster_assignments.json"
    cluster_doc_ids_path = output / "cluster_doc_ids.json"

    np.save(centroids_path, centroids)
    assignments_path.write_text(
        json.dumps(assignments.astype(int).tolist(), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    cluster_doc_ids_path.write_text(
        json.dumps(cluster_to_doc_ids, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    return {
        "centroids": centroids_path.resolve(),
        "assignments": assignments_path.resolve(),
        "cluster_doc_ids": cluster_doc_ids_path.resolve(),
    }


def cluster_and_save(
    *,
    embeddings_path: str | Path = DEFAULT_EMBEDDINGS_PATH,
    doc_ids_path: str | Path = DEFAULT_DOC_IDS_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    k: int = DEFAULT_K,
    seed: int = DEFAULT_RANDOM_SEED,
    max_iter: int = DEFAULT_MAX_ITER,
    tol: float = DEFAULT_TOL,
) -> dict[str, Any]:
    """Run clustering stage from embedding artifacts and persist outputs."""
    embeddings = load_doc_embeddings(embeddings_path)
    doc_ids = load_doc_ids(doc_ids_path)
    if len(doc_ids) != embeddings.shape[0]:
        raise ValueError(
            f"doc_ids ({len(doc_ids)}) must match number of embedding rows ({embeddings.shape[0]})."
        )

    centroids, assignments = run_kmeans(
        embeddings,
        k=k,
        seed=seed,
        max_iter=max_iter,
        tol=tol,
    )
    cluster_to_doc_ids = build_cluster_doc_lists(assignments, doc_ids, k=k)
    artifact_paths = save_cluster_outputs(
        output_dir=output_dir,
        centroids=centroids,
        assignments=assignments,
        cluster_to_doc_ids=cluster_to_doc_ids,
    )

    return {
        "k": int(k),
        "seed": int(seed),
        "num_documents": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "artifacts": artifact_paths,
    }
