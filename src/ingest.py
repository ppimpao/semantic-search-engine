"""Embed the cached corpus and upsert it into ChromaDB.

Run: `python -m src.ingest`. Idempotent (upsert keyed by arXiv id), so it's
safe to re-run after fetching more abstracts.

Application-managed vectorization: we embed here and hand Chroma the vectors,
rather than letting Chroma embed. This keeps the model choice ours and visible.
"""

import json

from . import config
from .embedder import embed
from .store import get_collection

BATCH = 256


def doc_text(record: dict) -> str:
    """What we actually embed: title + abstract together."""
    return f"{record['title']}. {record['abstract']}"


def ingest(records: list[dict], collection=None, embed_fn=embed) -> int:
    collection = collection if collection is not None else get_collection()
    for i in range(0, len(records), BATCH):
        batch = records[i : i + BATCH]
        collection.upsert(
            ids=[r["id"] for r in batch],
            embeddings=embed_fn([doc_text(r) for r in batch]),
            documents=[r["abstract"] for r in batch],
            metadatas=[
                {"title": r["title"], "url": r["url"], "published": r["published"]}
                for r in batch
            ],
        )
    return len(records)


def main() -> None:
    if not config.RAW_CORPUS.exists():
        raise SystemExit("No corpus cache. Run `python -m src.fetch_corpus` first.")
    records = json.loads(config.RAW_CORPUS.read_text())
    n = ingest(records)
    print(f"Ingested {n} abstracts into collection '{config.COLLECTION}'.")


if __name__ == "__main__":
    main()
