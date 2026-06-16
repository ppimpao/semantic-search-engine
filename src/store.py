"""Thin wrapper over the persisted ChromaDB collection.

One place creates/opens the collection so the cosine metric and persist path
are set consistently for both ingest and search.
"""

import chromadb

from . import config


def get_collection(persist_dir=config.CHROMA_DIR):
    client = chromadb.PersistentClient(path=str(persist_dir))
    # Explicit cosine: Chroma defaults to l2. With normalized vectors the
    # ranking matches, but stating it avoids surprises if the model changes.
    return client.get_or_create_collection(
        name=config.COLLECTION,
        metadata={"hnsw:space": config.DISTANCE_SPACE},
    )
