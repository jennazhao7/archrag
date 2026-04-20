"""Verify Stage 1 processed SciFact data artifacts.

Checks:
1) processed SciFact JSON exists and is loadable
2) documents/queries collections are present
3) each record has non-empty id and text fields

Prints counts and one sample record for each split.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify Stage 1 processed SciFact data."
    )
    parser.add_argument(
        "--input",
        default="data/processed/scifact_processed.json",
        help="Path to processed SciFact JSON file.",
    )
    return parser


def _ensure_records(records: Any, *, name: str) -> list[dict[str, Any]]:
    if not isinstance(records, list):
        raise ValueError(f"Field '{name}' must be a list.")
    if not records:
        raise ValueError(f"Field '{name}' is empty.")
    for idx, item in enumerate(records):
        if not isinstance(item, dict):
            raise ValueError(f"{name}[{idx}] must be an object.")
        if "id" not in item or "text" not in item:
            raise ValueError(f"{name}[{idx}] must contain 'id' and 'text'.")
        if str(item["id"]).strip() == "":
            raise ValueError(f"{name}[{idx}] has empty 'id'.")
        if str(item["text"]).strip() == "":
            raise ValueError(f"{name}[{idx}] has empty 'text'.")
    return records


def main() -> int:
    args = _build_parser().parse_args()
    input_path = Path(args.input)

    try:
        if not input_path.exists():
            raise FileNotFoundError(f"Processed file not found: {input_path}")

        payload = json.loads(input_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Processed JSON must be an object at top level.")

        documents = _ensure_records(payload.get("documents"), name="documents")
        queries = _ensure_records(payload.get("queries"), name="queries")

        print(f"Documents: {len(documents)}")
        print(f"Queries: {len(queries)}")

        sample_doc = documents[0]
        sample_query = queries[0]
        print("\nSample document:")
        print(f"- id: {sample_doc['id']}")
        print(f"- text: {str(sample_doc['text'])[:300]}")

        print("\nSample query:")
        print(f"- id: {sample_query['id']}")
        print(f"- text: {str(sample_query['text'])[:300]}")

        print(f"\nStage 1 data verification: SUCCESS ({input_path.resolve()})")
        return 0
    except Exception as exc:  # broad by design for clear failure reporting
        print(f"Stage 1 data verification: FAILURE - {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
