"""Embed a query and return the top-k most relevant abstracts.

Run: `python -m src.search "how do galaxies form?"`
"""

import sys
import textwrap

from .embedder import embed
from .store import get_collection


def search(query: str, k: int = 5, collection=None, embed_fn=embed) -> list[dict]:
    collection = collection if collection is not None else get_collection()
    res = collection.query(query_embeddings=embed_fn([query]), n_results=k)
    hits = []
    for doc_id, doc, meta, dist in zip(
        res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        hits.append(
            {
                "id": doc_id,
                "score": 1.0 - dist,  # cosine distance -> similarity
                "title": meta["title"],
                "url": meta["url"],
                "snippet": doc,
            }
        )
    return hits


def _print(query: str, hits: list[dict]) -> None:
    print(f'\nTop {len(hits)} for: "{query}"\n')
    for rank, h in enumerate(hits, 1):
        snippet = textwrap.shorten(h["snippet"], width=200, placeholder=" …")
        print(f"{rank}. [{h['score']:.3f}] {h['title']}")
        print(f"   {h['url']}")
        print(f"   {snippet}\n")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python -m src.search "your query"')
    query = sys.argv[1]
    _print(query, search(query))


if __name__ == "__main__":
    main()
