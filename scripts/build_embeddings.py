"""CLI for the SciFact embedding stage.

Usage:
    python scripts/build_embeddings.py
    python scripts/build_embeddings.py --config config/embedding.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    """Create command-line parser for embedding stage."""
    parser = argparse.ArgumentParser(
        description="Build SciFact document/query embeddings from processed JSON."
    )
    parser.add_argument(
        "--config",
        default="config/embedding.yaml",
        help="Path to YAML config containing embedding model settings.",
    )
    parser.add_argument(
        "--input",
        default="data/processed/scifact_processed.json",
        help="Path to processed SciFact JSON input.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory where embedding artifacts are written.",
    )
    return parser


def main() -> int:
    """Run embedding stage only (no clustering or retrieval)."""
    project_root = Path(__file__).resolve().parents[1]
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    args = _build_parser().parse_args()

    from embed.embedder import build_and_save_embeddings

    outputs = build_and_save_embeddings(
        config_path=args.config,
        input_path=args.input,
        output_dir=args.output_dir,
    )

    print("Saved embedding artifacts:")
    for name, path in outputs.items():
        print(f"- {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
