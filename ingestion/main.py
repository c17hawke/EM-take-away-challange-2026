#!/usr/bin/env python3
"""
Ingestion entrypoint — run this container once to populate ChromaDB.
"""
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    stream=sys.stdout,
)

from ingestion.workflow import run_ingestion  # noqa: E402

if __name__ == "__main__":
    state = run_ingestion()
    errors = state.get("errors", [])
    if errors:
        sys.exit(1)
