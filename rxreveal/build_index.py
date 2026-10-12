"""Build the search index from the PDFs in the data folder.

Usage:
    python -m rxreveal.build_index
    python -m rxreveal.build_index --data data --out index
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import get_settings
from .index import EmbedderError, VectorIndex, make_embedder
from .ingest import load_folder


def build(data_dir: Path, index_dir: Path, embedder=None) -> VectorIndex:
    settings = get_settings()
    chunks = load_folder(data_dir, size=settings.chunk_size, overlap=settings.chunk_overlap)
    embedder = embedder or make_embedder(settings)
    index = VectorIndex.build(chunks, embedder)
    index.save(index_dir)
    return index


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Build the RxReveal search index.")
    parser.add_argument("--data", type=Path, default=settings.data_dir, help="Folder with PDF files")
    parser.add_argument("--out", type=Path, default=settings.index_dir, help="Folder to save the index")
    args = parser.parse_args(argv)

    if not any(args.data.rglob("*.pdf")):
        print(f"No PDF files found in '{args.data}'. Add PDFs there and run again.", file=sys.stderr)
        return 1

    try:
        index = build(args.data, args.out)
    except (EmbedderError, ValueError) as exc:
        print(f"Could not build the index: {exc}", file=sys.stderr)
        return 1

    print(
        f"Indexed {index.meta['chunks']} chunks from {len(index.meta['sources'])} file(s) "
        f"into '{args.out}' using {index.meta['embed_model']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
