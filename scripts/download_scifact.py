"""CLI for downloading and processing the SciFact dataset.

Usage:
    python scripts/download_scifact.py
    python scripts/download_scifact.py --output data/processed/scifact.json --revision main
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser for SciFact download pipeline."""
    parser = argparse.ArgumentParser(
        description="Download/load SciFact and save processed document/query JSON output."
    )
    parser.add_argument(
        "--output",
        default="data/processed/scifact_processed.json",
        help="Destination JSON path for processed data.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face datasets cache directory.",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Dataset revision/tag/commit for reproducible loading.",
    )
    return parser


def main() -> int:
    """Run the standalone SciFact loading stage."""
    project_root = Path(__file__).resolve().parents[1]
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from data.scifact_loader import save_processed_scifact

    args = _build_parser().parse_args()
    output_path = save_processed_scifact(
        output_path=args.output,
        cache_dir=args.cache_dir,
        revision=args.revision,
    )
    print(f"Saved SciFact processed data to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
