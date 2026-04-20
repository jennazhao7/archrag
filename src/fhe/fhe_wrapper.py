"""FHE backend abstraction layer for retrieval pipeline stages.

This module defines a clean interface for FHE-like operations without locking
the codebase to any single FHE implementation today. A mock backend is included
so the pipeline shape can be exercised before integrating OpenFHE/TenSEAL.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable
from uuid import uuid4

import numpy as np

Metric = Literal["dot", "l2"]


@dataclass
class FHEContext:
    """Opaque context holder used by a backend for keys/params/state."""

    backend_name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class EncryptedQuery:
    """Opaque encrypted query embedding handle."""

    payload: Any


@dataclass
class EncryptedScores:
    """Opaque encrypted score vector handle."""

    payload: Any


@dataclass
class OpenFHECiphertextBundle:
    """Paths to OpenFHE ciphertext artifacts for one encrypted query."""

    encrypted_query_path: Path
    encrypted_norm_path: Path
    run_dir: Path


@dataclass
class OpenFHEScoreBundle:
    """Paths/metadata for encrypted centroid-distance results."""

    distances_dir: Path
    metric: Metric
    n_centroids: int


@runtime_checkable
class FHEBackend(Protocol):
    """Backend contract for FHE query-time scoring."""

    backend_name: str

    def setup_context(self) -> FHEContext:
        """Create backend context (keys/parameters)."""

    def encrypt_query_embedding(
        self,
        query_embedding: np.ndarray,
        *,
        context: FHEContext,
    ) -> EncryptedQuery:
        """Encrypt a plaintext query embedding into backend ciphertext form."""

    def encrypted_similarity_to_plaintext(
        self,
        encrypted_query: EncryptedQuery,
        plaintext_matrix: np.ndarray,
        *,
        metric: Metric,
        context: FHEContext,
    ) -> EncryptedScores:
        """Compute encrypted query vs plaintext matrix scores."""

    def decrypt_scores(
        self,
        encrypted_scores: EncryptedScores,
        *,
        context: FHEContext,
    ) -> np.ndarray:
        """Decrypt score vector back to plaintext numpy array."""


def _validate_inputs(query_embedding: np.ndarray, matrix: np.ndarray) -> None:
    if query_embedding.ndim != 1:
        raise ValueError("query_embedding must be a 1D vector.")
    if matrix.ndim != 2:
        raise ValueError("plaintext_matrix must be a 2D matrix.")
    if query_embedding.shape[0] != matrix.shape[1]:
        raise ValueError("Query dimension must match matrix column dimension.")


def _compute_scores_plaintext(
    query_embedding: np.ndarray,
    matrix: np.ndarray,
    *,
    metric: Metric,
) -> np.ndarray:
    if metric == "dot":
        return np.asarray(matrix @ query_embedding, dtype=np.float32)
    if metric == "l2":
        distances = np.sum((matrix - query_embedding[None, :]) ** 2, axis=1)
        return np.asarray(-distances, dtype=np.float32)
    raise ValueError("metric must be one of: 'dot', 'l2'")


def _ensure_backend_context(context: FHEContext, backend_name: str) -> None:
    if context.backend_name != backend_name:
        raise ValueError("Context backend mismatch.")


def _run_checked(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Backend command failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )


class OpenFHECoreSubprocessBackend:
    """Minimal OpenFHE subprocess backend for centroid distance computation.

    This mirrors the old encrypted-query-vs-plaintext-centroids flow but is
    fully isolated from the legacy repo structure. It expects OpenFHE binaries
    to be provided in a standalone directory.
    """

    backend_name = "openfhe"

    def __init__(
        self,
        *,
        binary_dir: str | Path | None = None,
        work_dir: str | Path = "data/fhe_runtime",
        poly_modulus_degree: int = 8192,
        coeff_mod_bit_sizes: list[int] | None = None,
        batch_size: int = 64,
        num_threads: int = 0,
    ) -> None:
        self.binary_dir = Path(
            binary_dir or os.environ.get("OPENFHE_BIN_DIR", "bin/openfhe")
        ).resolve()
        self.work_dir = Path(work_dir).resolve()
        self.context_dir = self.work_dir / "context"
        self.poly_modulus_degree = int(poly_modulus_degree)
        self.coeff_mod_bit_sizes = coeff_mod_bit_sizes or [60, 40, 40, 60]
        self.batch_size = int(batch_size)
        self.num_threads = int(num_threads)

    def _binary(self, name: str) -> Path:
        path = self.binary_dir / name
        if not path.exists():
            raise FileNotFoundError(
                f"Required OpenFHE binary not found: {path}. "
                "Set OPENFHE_BIN_DIR or pass binary_dir to create_backend(...)."
            )
        return path

    def setup_context(self) -> FHEContext:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.context_dir.mkdir(parents=True, exist_ok=True)
        keygen = self._binary("openfhe_keygen")
        _run_checked(
            [
                str(keygen),
                "--context-dir",
                str(self.context_dir),
                "--poly-modulus-degree",
                str(self.poly_modulus_degree),
                "--coeff-mod-bit-sizes",
                ",".join(str(x) for x in self.coeff_mod_bit_sizes),
            ]
        )
        return FHEContext(
            backend_name=self.backend_name,
            params={
                "context_dir": str(self.context_dir),
                "binary_dir": str(self.binary_dir),
                "poly_modulus_degree": self.poly_modulus_degree,
                "coeff_mod_bit_sizes": list(self.coeff_mod_bit_sizes),
            },
        )

    def encrypt_query_embedding(
        self,
        query_embedding: np.ndarray,
        *,
        context: FHEContext,
    ) -> EncryptedQuery:
        _ensure_backend_context(context, self.backend_name)
        query = np.asarray(query_embedding, dtype=np.float64)
        if query.ndim != 1:
            raise ValueError("query_embedding must be a 1D vector.")

        run_dir = self.work_dir / f"query_{uuid4().hex}"
        run_dir.mkdir(parents=True, exist_ok=True)
        input_vector_path = run_dir / "query_vector.txt"
        np.savetxt(input_vector_path, query, fmt="%.12g")

        encrypt_query = self._binary("openfhe_encrypt_query")
        _run_checked(
            [
                str(encrypt_query),
                "--context-dir",
                str(self.context_dir),
                "--input-vector",
                str(input_vector_path),
                "--output-dir",
                str(run_dir),
            ]
        )
        payload = OpenFHECiphertextBundle(
            encrypted_query_path=run_dir / "encrypted_query.bin",
            encrypted_norm_path=run_dir / "encrypted_norm_squared.bin",
            run_dir=run_dir,
        )
        return EncryptedQuery(payload=payload)

    def encrypted_similarity_to_plaintext(
        self,
        encrypted_query: EncryptedQuery,
        plaintext_matrix: np.ndarray,
        *,
        metric: Metric,
        context: FHEContext,
    ) -> EncryptedScores:
        _ensure_backend_context(context, self.backend_name)
        if metric != "l2":
            raise ValueError(
                "OpenFHE subprocess backend currently supports only 'l2' metric. "
                "Use mock backend for 'dot' until an encrypted dot path is added."
            )

        bundle = encrypted_query.payload
        if not isinstance(bundle, OpenFHECiphertextBundle):
            raise ValueError("Encrypted query payload type mismatch for OpenFHE backend.")

        centroids = np.asarray(plaintext_matrix, dtype=np.float64)
        if centroids.ndim != 2:
            raise ValueError("plaintext_matrix must be a 2D matrix.")

        distances_dir = bundle.run_dir / "distances"
        distances_dir.mkdir(parents=True, exist_ok=True)
        centroids_txt = distances_dir / "centroids.txt"
        with centroids_txt.open("w", encoding="utf-8") as handle:
            for row in centroids:
                handle.write(" ".join(f"{float(v):.12g}" for v in row))
                handle.write("\n")

        compute_distances = self._binary("openfhe_compute_distances")
        cmd = [
            str(compute_distances),
            "--context-dir",
            str(self.context_dir),
            "--centroids-file",
            str(centroids_txt),
            "--encrypted-query",
            str(bundle.encrypted_query_path),
            "--encrypted-norm",
            str(bundle.encrypted_norm_path),
            "--output-dir",
            str(distances_dir),
            "--batch-size",
            str(max(1, self.batch_size)),
        ]
        if self.num_threads > 0:
            cmd.extend(["--num-threads", str(self.num_threads)])
        _run_checked(cmd)

        return EncryptedScores(
            payload=OpenFHEScoreBundle(
                distances_dir=distances_dir,
                metric=metric,
                n_centroids=int(centroids.shape[0]),
            )
        )

    def decrypt_scores(
        self,
        encrypted_scores: EncryptedScores,
        *,
        context: FHEContext,
    ) -> np.ndarray:
        _ensure_backend_context(context, self.backend_name)
        bundle = encrypted_scores.payload
        if not isinstance(bundle, OpenFHEScoreBundle):
            raise ValueError("Encrypted scores payload type mismatch for OpenFHE backend.")

        decrypt_topk = self._binary("openfhe_decrypt_topk")
        output_json = bundle.distances_dir / "all_scores.json"
        _run_checked(
            [
                str(decrypt_topk),
                "--context-dir",
                str(self.context_dir),
                "--encrypted-distances-dir",
                str(bundle.distances_dir),
                "--top-k",
                str(bundle.n_centroids),
                "--output-json",
                str(output_json),
            ]
        )

        data = json.loads(output_json.read_text(encoding="utf-8"))
        distances = np.asarray(data.get("distances", []), dtype=np.float32)
        centroid_indices = np.asarray(data.get("centroid_indices", []), dtype=np.int32)
        if distances.shape[0] != centroid_indices.shape[0]:
            raise ValueError("Decryption output has mismatched distances/indices lengths.")

        # decrypt_topk returns sorted best-first entries; reconstruct dense order.
        dense_scores = np.full((bundle.n_centroids,), -np.inf, dtype=np.float32)
        dense_scores[centroid_indices] = -distances  # similarity-style score for L2
        return dense_scores


class MockFHEBackend:
    """Mock backend that preserves pipeline shape with plaintext internals.

    The mock backend intentionally does not perform real encryption. It only
    wraps arrays in opaque dataclasses to emulate backend boundaries.
    """

    backend_name = "mock"

    def setup_context(self) -> FHEContext:
        return FHEContext(
            backend_name=self.backend_name,
            params={"scheme": "mock", "note": "not cryptographically secure"},
        )

    def encrypt_query_embedding(
        self,
        query_embedding: np.ndarray,
        *,
        context: FHEContext,
    ) -> EncryptedQuery:
        _ensure_backend_context(context, self.backend_name)
        vector = np.asarray(query_embedding, dtype=np.float32)
        if vector.ndim != 1:
            raise ValueError("query_embedding must be a 1D vector.")
        return EncryptedQuery(payload=vector.copy())

    def encrypted_similarity_to_plaintext(
        self,
        encrypted_query: EncryptedQuery,
        plaintext_matrix: np.ndarray,
        *,
        metric: Metric,
        context: FHEContext,
    ) -> EncryptedScores:
        _ensure_backend_context(context, self.backend_name)
        query_embedding = np.asarray(encrypted_query.payload, dtype=np.float32)
        matrix = np.asarray(plaintext_matrix, dtype=np.float32)
        _validate_inputs(query_embedding, matrix)
        scores = _compute_scores_plaintext(query_embedding, matrix, metric=metric)
        return EncryptedScores(payload=scores)

    def decrypt_scores(
        self,
        encrypted_scores: EncryptedScores,
        *,
        context: FHEContext,
    ) -> np.ndarray:
        _ensure_backend_context(context, self.backend_name)
        scores = np.asarray(encrypted_scores.payload, dtype=np.float32)
        if scores.ndim != 1:
            raise ValueError("Decrypted score payload must be a 1D vector.")
        return scores


def create_backend(backend: str = "mock", **kwargs: Any) -> FHEBackend:
    """Factory for FHE backends.

    Extend this with real backends later (e.g., TenSEAL/OpenFHE wrappers)
    without changing retrieval-stage orchestration.
    """
    name = backend.strip().lower()
    if name == "mock":
        return MockFHEBackend()
    if name == "openfhe":
        return OpenFHECoreSubprocessBackend(**kwargs)
    raise ValueError(f"Unsupported FHE backend: {backend}")
