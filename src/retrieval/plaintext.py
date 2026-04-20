"""Plaintext retrieval utilities over document embeddings.

This module supports non-clustered retrieval by scoring a query embedding
against all document embeddings and returning top-k results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
from sentence_transformers import SentenceTransformer

Metric = Literal["dot", "l2"]

DEFAULT_DOC_EMBEDDINGS_PATH = Path("data/processed/doc_embeddings.npy")
DEFAULT_DOC_IDS_PATH = Path("data/processed/doc_ids.json")
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_doc_embeddings(doc_embeddings_path: str | Path = DEFAULT_DOC_EMBEDDINGS_PATH) -> np.ndarray:
    """Load document embedding matrix from disk."""
    embeddings = np.load(Path(doc_embeddings_path))
    if embeddings.ndim != 2:
        raise ValueError("Document embeddings must be a 2D numpy array.")
    return np.asarray(embeddings, dtype=np.float32)


def load_doc_ids(doc_ids_path: str | Path = DEFAULT_DOC_IDS_PATH) -> list[str]:
    """Load ordered document IDs from JSON."""
    payload = json.loads(Path(doc_ids_path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Document IDs file must contain a JSON list.")
    return [str(doc_id) for doc_id in payload]


def encode_query(
    query_text: str,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    device: str | None = None,
    normalize_embedding: bool = True,
) -> np.ndarray:
    """Encode query text into a 1D embedding vector."""
    model = SentenceTransformer(model_name, device=device)
    vector = model.encode(
        [query_text],
        normalize_embeddings=normalize_embedding,
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0]
    return np.asarray(vector, dtype=np.float32)


def validate_metric(metric: str) -> Metric:
    """Validate and normalize metric name."""
    metric_value = metric.strip().lower()
    if metric_value not in {"dot", "l2"}:
        raise ValueError("metric must be one of: 'dot', 'l2'")
    return metric_value  # type: ignore[return-value]


def compute_similarity_scores(
    query_embedding: np.ndarray,
    doc_embeddings: np.ndarray,
    *,
    metric: Metric,
) -> np.ndarray:
    """Compute query-to-document similarity scores."""
    if query_embedding.ndim != 1:
        raise ValueError("Query embedding must be 1D.")
    if doc_embeddings.ndim != 2:
        raise ValueError("Document embeddings must be 2D.")
    if query_embedding.shape[0] != doc_embeddings.shape[1]:
        raise ValueError("Query embedding dimension must match document embedding dimension.")

    if metric == "dot":
        return np.asarray(doc_embeddings @ query_embedding, dtype=np.float32)

    # For L2 mode we return negative squared distance so higher is better.
    distances = np.sum((doc_embeddings - query_embedding[None, :]) ** 2, axis=1)
    return np.asarray(-distances, dtype=np.float32)


def top_k_indices(scores: np.ndarray, *, k: int) -> np.ndarray:
    """Return indices of top-k highest scores in descending score order."""
    if k <= 0:
        raise ValueError("k must be > 0.")
    if scores.ndim != 1:
        raise ValueError("Scores must be 1D.")

    k_eff = min(k, scores.shape[0])
    if k_eff == scores.shape[0]:
        return np.argsort(scores)[::-1]

    part = np.argpartition(scores, -k_eff)[-k_eff:]
    return part[np.argsort(scores[part])[::-1]]


def format_ranked_results(
    *,
    indices: np.ndarray,
    scores: np.ndarray,
    doc_ids: list[str],
) -> list[dict[str, float | str]]:
    """Convert ranked index list into doc-id/score records."""
    results: list[dict[str, float | str]] = []
    for idx in indices.tolist():
        results.append({"doc_id": doc_ids[idx], "score": float(scores[idx])})
    return results


def retrieve_non_clustered(
    *,
    query_text: str,
    doc_embeddings_path: str | Path = DEFAULT_DOC_EMBEDDINGS_PATH,
    doc_ids_path: str | Path = DEFAULT_DOC_IDS_PATH,
    model_name: str = DEFAULT_MODEL_NAME,
    device: str | None = None,
    normalize_query_embedding: bool = True,
    metric: str = "dot",
    top_k: int = 10,
) -> dict[str, object]:
    """Run non-clustered retrieval against all documents."""
    metric_name = validate_metric(metric)
    doc_embeddings = load_doc_embeddings(doc_embeddings_path)
    doc_ids = load_doc_ids(doc_ids_path)
    if len(doc_ids) != doc_embeddings.shape[0]:
        raise ValueError(
            f"Document IDs count ({len(doc_ids)}) must match embedding rows ({doc_embeddings.shape[0]})."
        )

    query_embedding = encode_query(
        query_text,
        model_name=model_name,
        device=device,
        normalize_embedding=normalize_query_embedding,
    )
    scores = compute_similarity_scores(query_embedding, doc_embeddings, metric=metric_name)
    ranked_indices = top_k_indices(scores, k=top_k)
    results = format_ranked_results(indices=ranked_indices, scores=scores, doc_ids=doc_ids)

    return {
        "mode": "non-clustered",
        "metric": metric_name,
        "top_k": min(top_k, doc_embeddings.shape[0]),
        "results": results,
    }
