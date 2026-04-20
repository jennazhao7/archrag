"""Embedding stage for processed SciFact documents and queries.

This module is intentionally standalone for the embedding stage only.
It reads processed SciFact JSON, embeds documents and queries with a
sentence-transformer model configured via YAML, and writes:
    - doc_embeddings.npy
    - query_embeddings.npy
    - doc_ids.json
    - query_ids.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_INPUT_JSON = Path("data/processed/scifact_processed.json")
DEFAULT_OUTPUT_DIR = Path("data/processed")
DEFAULT_CONFIG_PATH = Path("config/embedding.yaml")


def load_yaml_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load YAML config file used by the embedding stage.

    Args:
        config_path: Path to YAML config.

    Returns:
        Parsed config dictionary.

    Raises:
        ImportError: If PyYAML is not installed.
        FileNotFoundError: If config path does not exist.
        ValueError: If config content is empty or invalid.
    """
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PyYAML is required for YAML config loading. Install with: pip install pyyaml"
        ) from exc

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Embedding config not found: {path}")

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML config at {path}: expected a mapping/object.")

    return config


def _get_embedding_settings(config: dict[str, Any]) -> dict[str, Any]:
    """Extract and validate embedding settings from config."""
    embedding_cfg = config.get("embedding", {})
    if not isinstance(embedding_cfg, dict):
        raise ValueError("Config key 'embedding' must be a mapping/object.")

    model_name = embedding_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
    batch_size = int(embedding_cfg.get("batch_size", 64))
    normalize_embeddings = bool(embedding_cfg.get("normalize_embeddings", True))
    device = embedding_cfg.get("device", None)
    show_progress_bar = bool(embedding_cfg.get("show_progress_bar", True))

    if not model_name:
        raise ValueError("Config 'embedding.model_name' must be a non-empty string.")
    if batch_size <= 0:
        raise ValueError("Config 'embedding.batch_size' must be > 0.")

    return {
        "model_name": model_name,
        "batch_size": batch_size,
        "normalize_embeddings": normalize_embeddings,
        "device": device,
        "show_progress_bar": show_progress_bar,
    }


def load_processed_scifact(
    input_path: str | Path = DEFAULT_INPUT_JSON,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load processed SciFact documents and queries from JSON.

    Args:
        input_path: Path to processed SciFact JSON.

    Returns:
        Tuple of (documents, queries), where each item has id/text fields.
    """
    path = Path(input_path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    documents = payload.get("documents", [])
    queries = payload.get("queries", [])
    if not isinstance(documents, list) or not isinstance(queries, list):
        raise ValueError("Processed SciFact JSON must contain list fields: documents, queries.")
    return documents, queries


def _extract_ids_and_text(records: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """Extract ids/text from records and preserve order."""
    ids: list[str] = []
    texts: list[str] = []
    for row in records:
        ids.append(str(row["id"]))
        texts.append(str(row["text"]))
    return ids, texts


def _encode_texts(
    model: SentenceTransformer,
    texts: list[str],
    *,
    batch_size: int,
    normalize_embeddings: bool,
    show_progress_bar: bool,
) -> np.ndarray:
    """Encode text list into a 2D numpy array of embeddings."""
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        convert_to_numpy=True,
        show_progress_bar=show_progress_bar,
    )
    return np.asarray(embeddings)


def build_and_save_embeddings(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    input_path: str | Path = DEFAULT_INPUT_JSON,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    """Build embeddings for SciFact docs/queries and save output artifacts.

    Args:
        config_path: YAML config path with embedding model settings.
        input_path: Processed SciFact JSON path.
        output_dir: Destination directory for embeddings and ID files.

    Returns:
        Mapping of artifact names to absolute output paths.
    """
    config = load_yaml_config(config_path)
    settings = _get_embedding_settings(config)
    documents, queries = load_processed_scifact(input_path)

    doc_ids, doc_texts = _extract_ids_and_text(documents)
    query_ids, query_texts = _extract_ids_and_text(queries)

    model = SentenceTransformer(
        settings["model_name"],
        device=settings["device"],
    )

    doc_embeddings = _encode_texts(
        model,
        doc_texts,
        batch_size=settings["batch_size"],
        normalize_embeddings=settings["normalize_embeddings"],
        show_progress_bar=settings["show_progress_bar"],
    )
    query_embeddings = _encode_texts(
        model,
        query_texts,
        batch_size=settings["batch_size"],
        normalize_embeddings=settings["normalize_embeddings"],
        show_progress_bar=settings["show_progress_bar"],
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    doc_embeddings_path = output / "doc_embeddings.npy"
    query_embeddings_path = output / "query_embeddings.npy"
    doc_ids_path = output / "doc_ids.json"
    query_ids_path = output / "query_ids.json"

    np.save(doc_embeddings_path, doc_embeddings)
    np.save(query_embeddings_path, query_embeddings)
    doc_ids_path.write_text(json.dumps(doc_ids, indent=2, ensure_ascii=True), encoding="utf-8")
    query_ids_path.write_text(
        json.dumps(query_ids, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    return {
        "doc_embeddings": doc_embeddings_path.resolve(),
        "query_embeddings": query_embeddings_path.resolve(),
        "doc_ids": doc_ids_path.resolve(),
        "query_ids": query_ids_path.resolve(),
    }
