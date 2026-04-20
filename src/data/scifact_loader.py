"""Utilities for downloading and processing the SciFact dataset.

This module keeps the SciFact loading stage self-contained and reproducible.
It exposes functions to:
1. download/load SciFact from Hugging Face datasets,
2. return normalized document and query records,
3. persist processed output as JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import load_dataset

SCIFACT_DATASET_NAME = "allenai/scifact"
DEFAULT_DATASET_REVISION = "main"
DEFAULT_OUTPUT_PATH = Path("data/processed/scifact_processed.json")


def _normalize_document_text(title: str, abstract_sentences: list[str]) -> str:
    """Build a single text string for a SciFact document.

    Args:
        title: Document title.
        abstract_sentences: Abstract represented as sentence chunks.

    Returns:
        A normalized text string combining title and abstract.
    """
    title = (title or "").strip()
    abstract = " ".join((sentence or "").strip() for sentence in abstract_sentences).strip()
    if title and abstract:
        return f"{title}\n\n{abstract}"
    return title or abstract


def load_scifact_documents(
    *,
    cache_dir: str | None = None,
    revision: str = DEFAULT_DATASET_REVISION,
) -> list[dict[str, Any]]:
    """Load SciFact corpus documents as a list of id/text records.

    Args:
        cache_dir: Optional local cache directory for Hugging Face datasets.
        revision: Dataset revision/tag/commit to make loading reproducible.

    Returns:
        A list of dictionaries with keys:
            - id: string document id
            - text: normalized document text (title + abstract)
    """
    corpus = load_dataset(
        SCIFACT_DATASET_NAME,
        "corpus",
        split="train",
        revision=revision,
        cache_dir=cache_dir,
    )

    documents: list[dict[str, Any]] = []
    for row in corpus:
        doc_id = str(row["doc_id"])
        text = _normalize_document_text(row.get("title", ""), row.get("abstract", []))
        documents.append({"id": doc_id, "text": text})

    return documents


def load_scifact_queries(
    *,
    cache_dir: str | None = None,
    revision: str = DEFAULT_DATASET_REVISION,
) -> list[dict[str, Any]]:
    """Load SciFact claims as query id/text records from all claim splits.

    Args:
        cache_dir: Optional local cache directory for Hugging Face datasets.
        revision: Dataset revision/tag/commit to make loading reproducible.

    Returns:
        A list of dictionaries with keys:
            - id: string query id
            - text: claim text
    """
    split_names = ("train", "validation", "test")
    queries_by_id: dict[str, str] = {}

    for split in split_names:
        claims = load_dataset(
            SCIFACT_DATASET_NAME,
            "claims",
            split=split,
            revision=revision,
            cache_dir=cache_dir,
        )
        for row in claims:
            query_id = str(row["id"])
            queries_by_id[query_id] = row["claim"]

    return [{"id": query_id, "text": text} for query_id, text in sorted(queries_by_id.items())]


def build_processed_scifact_payload(
    *,
    cache_dir: str | None = None,
    revision: str = DEFAULT_DATASET_REVISION,
) -> dict[str, Any]:
    """Build the processed SciFact payload with documents and queries.

    Args:
        cache_dir: Optional local cache directory for Hugging Face datasets.
        revision: Dataset revision/tag/commit for reproducible loads.

    Returns:
        A dictionary ready to be serialized as JSON.
    """
    documents = load_scifact_documents(cache_dir=cache_dir, revision=revision)
    queries = load_scifact_queries(cache_dir=cache_dir, revision=revision)

    return {
        "dataset": SCIFACT_DATASET_NAME,
        "revision": revision,
        "documents": documents,
        "queries": queries,
    }


def save_processed_scifact(
    *,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    cache_dir: str | None = None,
    revision: str = DEFAULT_DATASET_REVISION,
) -> Path:
    """Save processed SciFact data as a deterministic JSON file.

    Args:
        output_path: Destination JSON path.
        cache_dir: Optional local cache directory for Hugging Face datasets.
        revision: Dataset revision/tag/commit for reproducible loads.

    Returns:
        The resolved output path that was written.
    """
    payload = build_processed_scifact_payload(cache_dir=cache_dir, revision=revision)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return output.resolve()
