"""Fetch/refresh the catalog and warm the embedding cache.

Run once before deploying so the container ships with the catalog and (if
embeddings are enabled) a precomputed vector cache, avoiding cold-start cost.

Usage:  python -m scripts.build_catalog
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import requests  # noqa: E402

from app.catalog import load_catalog  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.retrieval import HybridRetriever  # noqa: E402


def refresh_catalog() -> None:
    settings = get_settings()
    print(f"Fetching catalog from {settings.catalog_url}")
    resp = requests.get(settings.catalog_url, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    settings.catalog_file.parent.mkdir(parents=True, exist_ok=True)
    settings.catalog_file.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved {len(data)} raw entries to {settings.catalog_file}")


def main() -> None:
    refresh_catalog()
    settings = get_settings()
    catalog = load_catalog(settings)
    print(f"Normalized {len(catalog)} assessments.")
    retriever = HybridRetriever(catalog, settings)
    print(f"Retriever ready in '{retriever.mode}' mode (embedding cache warmed if enabled).")


if __name__ == "__main__":
    main()
