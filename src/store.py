"""Thin wrapper that opens the persisted ChromaDB collection.

Position in the workflow
------------------------
The single door to the vector database. Both `ingest` (which writes) and
`search` (which reads) call `get_collection()`, so they always open the *same*
collection, at the same on-disk path, with the same distance metric — they
cannot disagree about where the data lives or how it is compared.

Depends on
----------
- `config`  : CHROMA_DIR (where the DB persists), COLLECTION (its name),
              DISTANCE_SPACE (the similarity metric)
- chromadb  : the vector database itself

Provides for
------------
- `ingest`  : a collection handle to upsert vectors into
- `search`  : the same collection handle to query
"""

import chromadb

from . import config


def get_collection(persist_dir=config.CHROMA_DIR):
    """Open (creating if needed) the on-disk astro_ph collection.

    `PersistentClient` is what makes the database durable: it reads/writes the
    Chroma files under `persist_dir` (data/chroma/), so vectors written by
    ingest survive the process and are still there when search runs later.
    (Contrast with eval, which uses an in-memory EphemeralClient instead.)
    """
    client = chromadb.PersistentClient(path=str(persist_dir))
    # get_or_create: ingest's first run creates the collection; every later run
    # (including all searches) reuses it. The metadata sets the index's
    # similarity metric to cosine explicitly — Chroma would otherwise default to
    # l2. With normalized vectors the ranking matches either way, but stating it
    # documents intent and is safe if the model is ever swapped.
    return client.get_or_create_collection(
        name=config.COLLECTION,
        metadata={"hnsw:space": config.DISTANCE_SPACE},
    )
