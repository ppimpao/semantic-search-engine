"""Step 2 of the pipeline: embed the cached corpus and store it in ChromaDB.

Position in the workflow
------------------------
The write path. It reads the JSON cache produced by `fetch_corpus`, turns each
document into a vector via the shared `embedder.embed`, and upserts the vectors
(plus the raw text and metadata) into the persisted collection from `store`.
After this runs, the database is ready for `search` to query.

    data/raw/corpus.json  ->  ingest --(embed)--> data/chroma/  ->  search

"Application-managed vectorization": we embed here, in our own code, and hand
Chroma the finished vectors — rather than letting Chroma embed internally. This
keeps the model choice ours and visible, and lets tests inject a fake embedder.

Depends on
----------
- `config`            : RAW_CORPUS (input file), COLLECTION (for the log line)
- `embedder.embed`    : to vectorize each document (the default `embed_fn`)
- `store.get_collection` : the destination collection
- `fetch_corpus`      : (data dependency) must have produced corpus.json first

Provides for
------------
- `search`  : the populated ChromaDB collection it reads
"""

import json

from . import config
from .embedder import embed
from .store import get_collection

BATCH = 256  # embed/upsert this many at a time — far faster than one-by-one


def doc_text(record: dict) -> str:
    """Build the single string we embed for a document: title + abstract.

    Joining both gives the model more signal than the abstract alone. Whatever
    this returns is what the document's vector represents, so the same recipe
    must be used everywhere a document is embedded.
    """
    return f"{record['title']}. {record['abstract']}"


def ingest(records: list[dict], collection=None, embed_fn=embed) -> int:
    """Embed `records` and upsert them into the collection; return the count.

    `collection` and `embed_fn` are injectable: production passes neither and
    gets the real persisted collection + real model, while tests pass an
    in-memory collection and a fake embedder to run this offline.

    `upsert` (not `add`) keys on the arXiv id, so re-running overwrites existing
    documents instead of duplicating them — the whole step is idempotent and
    safe to repeat after fetching more abstracts.
    """
    collection = collection if collection is not None else get_collection()
    # Walk the records in BATCH-sized windows.
    for i in range(0, len(records), BATCH):
        batch = records[i : i + BATCH]
        # The four lists below are parallel (index N describes the same doc):
        collection.upsert(
            ids=[r["id"] for r in batch],                          # primary key
            embeddings=embed_fn([doc_text(r) for r in batch]),     # the vectors
            documents=[r["abstract"] for r in batch],              # raw text, for snippets
            metadatas=[                                            # shown / filterable
                {"title": r["title"], "url": r["url"], "published": r["published"]}
                for r in batch
            ],
        )
    return len(records)


def main() -> None:
    """CLI entry point: load the cache and ingest it into the persisted DB."""
    # Fail clearly if step 1 hasn't been run yet.
    if not config.RAW_CORPUS.exists():
        raise SystemExit("No corpus cache. Run `python -m src.fetch_corpus` first.")
    records = json.loads(config.RAW_CORPUS.read_text())
    n = ingest(records)
    print(f"Ingested {n} abstracts into collection '{config.COLLECTION}'.")


if __name__ == "__main__":
    main()
