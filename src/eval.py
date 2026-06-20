"""Retrieval eval harness: Hit@1, Hit@5, MRR over a labelled query set.

Position in the workflow
------------------------
A self-contained side branch — the "how do I know it works" evidence. Unlike the
main pipeline it does NOT touch the persisted DB or the cached corpus: it builds
its own throwaway in-memory collection from the committed labelled files, so it
is fully reproducible and runs independently of fetch/ingest.

    eval/corpus.yaml + eval/queries.yaml  ->  eval  --(embed + score)-->  metrics

Run: `python -m src.eval`            # evaluate the QA model
     `python -m src.eval --compare`  # QA model vs pure-similarity model

--compare is the architecture-decision evidence: the same labelled queries
scored with multi-qa-MiniLM (relevance) vs all-MiniLM (similarity). Because the
queries are questions and the passages are answers (asymmetric), the QA model
wins.

Depends on
----------
- `config`         : MODEL_NAME, SIMILARITY_MODEL_NAME, DISTANCE_SPACE, ROOT
- `embedder.embed` : to vectorize corpus + queries (the model under test)
- chromadb / yaml  : ephemeral collection + reading the labelled files
- eval/corpus.yaml, eval/queries.yaml : the gold data

Provides for
------------
- the developer / README : printed retrieval metrics (nothing downstream
  consumes this; it is measurement, not part of the serving path).
"""

import sys

import chromadb
import yaml

from . import config
from .embedder import embed

EVAL_DIR = config.ROOT / "eval"


def hit_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> int:
    """1 if any relevant id appears in the top k results, else 0 (per query)."""
    return int(any(i in relevant for i in ranked_ids[:k]))


def reciprocal_rank(ranked_ids: list[str], relevant: set[str]) -> float:
    """1/rank of the first relevant hit (1.0 if #1, 0.5 if #2, ...); 0 if none.

    Averaged over all queries this is MRR — it rewards ranking the right answer
    higher, not just having it somewhere in the list.
    """
    for rank, doc_id in enumerate(ranked_ids, 1):  # enumerate from 1, not 0
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def evaluate(queries: list[dict], collection, embed_fn=embed, k: int = 5) -> dict:
    """Run every query against `collection` and average the metrics.

    For each query we embed it, fetch the top-k ids, and accumulate Hit@1,
    Hit@k, and reciprocal rank; then divide by the number of queries to get the
    means. Returns a dict ready for printing.
    """
    h1 = h5 = 0    # hit counters
    rr = 0.0       # summed reciprocal rank
    for q in queries:
        relevant = set(q["relevant"])  # the gold id(s) for this query
        res = collection.query(query_embeddings=embed_fn([q["query"]]), n_results=k)
        ranked = res["ids"][0]  # the ranked ids for our single query
        h1 += hit_at_k(ranked, relevant, 1)
        h5 += hit_at_k(ranked, relevant, k)
        rr += reciprocal_rank(ranked, relevant)
    n = len(queries)
    return {"Hit@1": h1 / n, f"Hit@{k}": h5 / n, "MRR": rr / n, "n": n}


def load_corpus() -> list[dict]:
    """Read the labelled gold passages from eval/corpus.yaml."""
    return yaml.safe_load((EVAL_DIR / "corpus.yaml").read_text())["documents"]


def load_queries() -> list[dict]:
    """Read the labelled queries (+ their relevant ids) from eval/queries.yaml."""
    return yaml.safe_load((EVAL_DIR / "queries.yaml").read_text())["queries"]


def build_collection(corpus: list[dict], embed_fn=embed):
    """Embed the labelled corpus into a throwaway in-memory collection.

    EphemeralClient (not PersistentClient) means this never touches disk or the
    real astro_ph DB — each eval run starts from a clean, isolated index so the
    numbers are reproducible. Same cosine metric as production.
    """
    client = chromadb.EphemeralClient()
    # The in-memory client is shared in-process, so a previous build (e.g. the
    # first model under --compare) leaves an "eval" collection behind. Drop it
    # first to guarantee a *fresh* index — reusing it would mix the two models'
    # vectors and corrupt the second model's score.
    client.get_or_create_collection("eval")
    client.delete_collection("eval")
    coll = client.create_collection(
        name="eval", metadata={"hnsw:space": config.DISTANCE_SPACE}
    )
    coll.add(
        ids=[d["id"] for d in corpus],
        # Same title + text recipe as ingest.doc_text, so eval mirrors production.
        embeddings=embed_fn([f"{d['title']}. {d['text']}" for d in corpus]),
        metadatas=[{"title": d["title"]} for d in corpus],
    )
    return coll


def _fmt(name: str, m: dict) -> str:
    """Format one model's metrics as a single aligned line for the terminal."""
    return f"{name:<34} Hit@1={m['Hit@1']:.2f}  Hit@5={m['Hit@5']:.2f}  MRR={m['MRR']:.3f}"


def main() -> None:
    """CLI entry point: score the QA model, or both models under `--compare`."""
    corpus, queries = load_corpus(), load_queries()
    if "--compare" in sys.argv:
        print(f"{len(queries)} queries over {len(corpus)} labelled passages:\n")
        # Score each model on its OWN embeddings of the same corpus + queries.
        for label, model in [
            ("multi-qa-MiniLM (relevance/QA)", config.MODEL_NAME),
            ("all-MiniLM (similarity)", config.SIMILARITY_MODEL_NAME),
        ]:
            # Bind `model` now (default arg) so the lambda doesn't capture the
            # loop variable by reference and end up using the last model twice.
            embed_fn = lambda t, _m=model: embed(t, _m)
            m = evaluate(queries, build_collection(corpus, embed_fn), embed_fn=embed_fn)
            print(_fmt(label, m))
    else:
        # Default: just the production QA model.
        m = evaluate(queries, build_collection(corpus))
        print(_fmt(config.MODEL_NAME, m))


if __name__ == "__main__":
    main()
