"""Retrieval eval harness: Hit@1, Hit@5, MRR over a labelled query set.

Position in the workflow
------------------------
A self-contained side branch — the "how do I know it works" evidence. Unlike the
main pipeline it does NOT touch the persisted DB or the cached corpus: it builds
its own throwaway in-memory collection from the committed labelled files, so it
is fully reproducible and runs independently of fetch/ingest.

    eval/corpus.yaml + eval/queries.yaml  ->  eval  --(embed + score)-->  metrics

Run: `python -m src.eval`            # score the default model (all-MiniLM)
     `python -m src.eval --compare`  # all-MiniLM (default) vs multi-qa (QA)
     `python -m src.eval --debug`    # per-query top-1 for both models
     `python -m src.eval --corpus`   # both models on the REAL abstracts (length probe)

The corpus includes deliberate *hard negatives* (passages c31+) that echo a
query's wording without answering it. They make the set discriminate: --compare
showed the general similarity model (all-MiniLM) beats the QA model on this
short-abstract corpus, and --debug shows why (see README "How I measured it").

Depends on
----------
- `config`         : MODEL_NAME, QA_MODEL_NAME, DISTANCE_SPACE, ROOT
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


def run_debug() -> None:
    """Per-query diagnostic: show each model's top-1 hit, side by side.

    Aggregate Hit@k hides *which* queries fail at n=15. For every query we print
    the gold id(s) and each model's top-1 (with ✓/✗), so disagreements and the
    specific hard negatives a model falls for are visible.
    """
    corpus, queries = load_corpus(), load_queries()
    models = [("all-MiniLM", config.MODEL_NAME), ("multi-qa", config.QA_MODEL_NAME)]
    # Process one model at a time: build_collection reuses the name "eval", so we
    # can't hold two collections at once. Collect each model's top-1 per query.
    tops: dict[str, list[str]] = {}
    for name, model in models:
        embed_fn = lambda t, _m=model: embed(t, _m)
        coll = build_collection(corpus, embed_fn)
        tops[name] = [
            coll.query(query_embeddings=embed_fn([q["query"]]), n_results=1)["ids"][0][0]
            for q in queries
        ]
    # Print a row per query: gold, then each model's top-1 with a hit marker.
    print(f"{'query':<50} {'gold':<8} {'all-MiniLM':<12} multi-qa")
    for i, q in enumerate(queries):
        gold = set(q["relevant"])
        cells = []
        for name, _ in models:
            hit = tops[name][i]
            cells.append(f"{hit} {'✓' if hit in gold else '✗'}")
        print(f"{q['query'][:48]:<50} {','.join(sorted(gold)):<8} {cells[0]:<12} {cells[1]}")


def _fresh_collection():
    """A clean, empty in-memory cosine collection named "eval".

    The EphemeralClient is shared in-process, so a previous build leaves an
    "eval" collection behind; we drop it first to guarantee a fresh index —
    reusing it would mix two models' vectors and corrupt the second's score.
    """
    client = chromadb.EphemeralClient()
    client.get_or_create_collection("eval")
    client.delete_collection("eval")
    return client.create_collection(
        name="eval", metadata={"hnsw:space": config.DISTANCE_SPACE}
    )


def build_collection(corpus: list[dict], embed_fn=embed):
    """Embed the labelled corpus into a throwaway in-memory collection.

    EphemeralClient (not PersistentClient) means this never touches disk or the
    real astro_ph DB — each eval run starts from a clean, isolated index so the
    numbers are reproducible. Same cosine metric as production.
    """
    coll = _fresh_collection()
    coll.add(
        ids=[d["id"] for d in corpus],
        # Same title + text recipe as ingest.doc_text, so eval mirrors production.
        embeddings=embed_fn([f"{d['title']}. {d['text']}" for d in corpus]),
        metadatas=[{"title": d["title"]} for d in corpus],
    )
    return coll


def run_corpus_eval(k: int = 5) -> None:
    """Probe both models on the REAL fetched abstracts (production length).

    The curated eval uses ~50-word passages, but production docs are ~200-word
    abstracts — and all-MiniLM truncates at 256 tokens while multi-qa reads 512.
    This tests both models at production length with *no manual labelling*, using
    known-item retrieval: embed each abstract (abstract text only, so the query
    is not a substring of the doc), query with that paper's TITLE, and check
    whether its own abstract comes back. Gold = the abstract's own arXiv id.
    """
    import json

    if not config.RAW_CORPUS.exists():
        raise SystemExit("No corpus cache. Run `python -m src.fetch_corpus` first.")
    records = json.loads(config.RAW_CORPUS.read_text())
    lengths = sorted(len(r["abstract"].split()) for r in records)
    avg = sum(lengths) // len(lengths)
    print(f"Known-item retrieval over {len(records)} real abstracts "
          f"(title -> abstract).")
    print(f"Abstract length: avg {avg}, median {lengths[len(lengths)//2]}, "
          f"max {lengths[-1]} words. all-MiniLM truncates ~256 tokens, "
          f"multi-qa ~512.\n")
    for label, model in [
        ("all-MiniLM (default)", config.MODEL_NAME),
        ("multi-qa-MiniLM (QA)", config.QA_MODEL_NAME),
    ]:
        embed_fn = lambda t, _m=model: embed(t, _m)
        coll = _fresh_collection()
        coll.add(
            ids=[r["id"] for r in records],
            embeddings=embed_fn([r["abstract"] for r in records]),  # abstract only
        )
        h1 = h5 = 0
        rr = 0.0
        for r in records:
            ranked = coll.query(
                query_embeddings=embed_fn([r["title"]]), n_results=k
            )["ids"][0]
            gold = {r["id"]}
            h1 += hit_at_k(ranked, gold, 1)
            h5 += hit_at_k(ranked, gold, k)
            rr += reciprocal_rank(ranked, gold)
        n = len(records)
        print(_fmt(label, {"Hit@1": h1 / n, f"Hit@{k}": h5 / n, "MRR": rr / n}))


def _fmt(name: str, m: dict) -> str:
    """Format one model's metrics as a single aligned line for the terminal."""
    return f"{name:<34} Hit@1={m['Hit@1']:.2f}  Hit@5={m['Hit@5']:.2f}  MRR={m['MRR']:.3f}"


def main() -> None:
    """CLI entry point: --compare, --debug, --corpus, or (default) score it."""
    if "--debug" in sys.argv:
        run_debug()
        return
    if "--corpus" in sys.argv:
        run_corpus_eval()
        return
    corpus, queries = load_corpus(), load_queries()
    if "--compare" in sys.argv:
        print(f"{len(queries)} queries over {len(corpus)} labelled passages:\n")
        # Score each model on its OWN embeddings of the same corpus + queries.
        for label, model in [
            ("all-MiniLM (similarity, default)", config.MODEL_NAME),
            ("multi-qa-MiniLM (relevance/QA)", config.QA_MODEL_NAME),
        ]:
            # Bind `model` now (default arg) so the lambda doesn't capture the
            # loop variable by reference and end up using the last model twice.
            embed_fn = lambda t, _m=model: embed(t, _m)
            m = evaluate(queries, build_collection(corpus, embed_fn), embed_fn=embed_fn)
            print(_fmt(label, m))
    else:
        # Default: just the production model.
        m = evaluate(queries, build_collection(corpus))
        print(_fmt(config.MODEL_NAME, m))


if __name__ == "__main__":
    main()
