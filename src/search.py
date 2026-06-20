"""Step 3 of the pipeline: embed a query and return the top-k abstracts.

Position in the workflow
------------------------
The read path, and the only part a user runs repeatedly. It embeds the incoming
query with the *same* `embedder.embed` used at ingest, asks the `store`
collection for the nearest stored vectors, and prints them. All the expensive
work (embedding every document) already happened in `ingest`, so a query is just
one forward pass + an HNSW lookup — sub-second.

    user query  ->  search --(embed + collection.query)-->  ranked hits

Depends on
----------
- `embedder.embed`       : to vectorize the query (must match ingest's model)
- `store.get_collection` : the populated collection to search
- `ingest`               : (data dependency) must have populated the DB first

Provides for
------------
- the end user (CLI output); `search()` also returns structured hits that a
  future API/RAG layer could consume directly.
"""

import sys
import textwrap

from .embedder import embed
from .store import get_collection


def search(query: str, k: int = 5, collection=None, embed_fn=embed) -> list[dict]:
    """Return the k most relevant documents for `query`, best first.

    Mirror image of ingest: there we embedded documents and stored them; here we
    embed the one query the same way and ask Chroma for its nearest neighbours.
    `collection`/`embed_fn` are injectable for the same reason as in ingest
    (tests supply fakes; production uses the real DB + model).
    """
    collection = collection if collection is not None else get_collection()
    # query_embeddings expects a list, hence embed([query]); n_results = top-k.
    res = collection.query(query_embeddings=embed_fn([query]), n_results=k)
    hits = []
    # Chroma returns parallel lists wrapped one level deep (one slot per query);
    # we sent a single query, so everything we need is at index [0]. zip walks
    # the k results together: id, raw text, metadata, and distance per hit.
    for doc_id, doc, meta, dist in zip(
        res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        hits.append(
            {
                "id": doc_id,
                # Chroma reports cosine *distance* (0 = identical); 1 - dist
                # converts it back to a human-friendly similarity score.
                "score": 1.0 - dist,
                "title": meta["title"],
                "url": meta["url"],
                "snippet": doc,
            }
        )
    return hits


def _print(query: str, hits: list[dict]) -> None:
    """Pretty-print the hits to the terminal: rank, score, title, link, snippet."""
    print(f'\nTop {len(hits)} for: "{query}"\n')
    for rank, h in enumerate(hits, 1):
        # Trim the abstract to a single readable line ending in an ellipsis.
        snippet = textwrap.shorten(h["snippet"], width=200, placeholder=" …")
        print(f"{rank}. [{h['score']:.3f}] {h['title']}")
        print(f"   {h['url']}")
        print(f"   {snippet}\n")


def main() -> None:
    """CLI entry point: read the query from argv and print its top-k results."""
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python -m src.search "your query"')
    query = sys.argv[1]
    _print(query, search(query))


if __name__ == "__main__":
    main()
