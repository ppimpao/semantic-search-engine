"""Retrieval eval harness: Hit@1, Hit@5, MRR over a labelled query set.

Run: `python -m src.eval`            # evaluate the QA model
     `python -m src.eval --compare`  # QA model vs pure-similarity model

The eval is self-contained and reproducible: a committed labelled corpus
(eval/corpus.yaml) of distinct space-domain passages, and queries
(eval/queries.yaml) each labelled with the passage id(s) that answer it. Every
query's gold passage sits among the others as distractors, so ranking it to the
top is a real retrieval task — not a lookup.

--compare is the architecture-decision evidence: the same labelled queries
scored with multi-qa-MiniLM (relevance) vs all-MiniLM (similarity). Because the
queries are questions and the passages are answers (asymmetric), the QA model
wins.
"""

import sys

import chromadb
import yaml

from . import config
from .embedder import embed

EVAL_DIR = config.ROOT / "eval"


def hit_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> int:
    return int(any(i in relevant for i in ranked_ids[:k]))


def reciprocal_rank(ranked_ids: list[str], relevant: set[str]) -> float:
    for rank, doc_id in enumerate(ranked_ids, 1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def evaluate(queries: list[dict], collection, embed_fn=embed, k: int = 5) -> dict:
    h1 = h5 = 0
    rr = 0.0
    for q in queries:
        relevant = set(q["relevant"])
        res = collection.query(query_embeddings=embed_fn([q["query"]]), n_results=k)
        ranked = res["ids"][0]
        h1 += hit_at_k(ranked, relevant, 1)
        h5 += hit_at_k(ranked, relevant, k)
        rr += reciprocal_rank(ranked, relevant)
    n = len(queries)
    return {"Hit@1": h1 / n, f"Hit@{k}": h5 / n, "MRR": rr / n, "n": n}


def load_corpus() -> list[dict]:
    return yaml.safe_load((EVAL_DIR / "corpus.yaml").read_text())["documents"]


def load_queries() -> list[dict]:
    return yaml.safe_load((EVAL_DIR / "queries.yaml").read_text())["queries"]


def build_collection(corpus: list[dict], embed_fn=embed):
    """Embed the labelled corpus into a throwaway in-memory collection."""
    coll = chromadb.EphemeralClient().create_collection(
        name="eval", metadata={"hnsw:space": config.DISTANCE_SPACE}
    )
    coll.add(
        ids=[d["id"] for d in corpus],
        embeddings=embed_fn([f"{d['title']}. {d['text']}" for d in corpus]),
        metadatas=[{"title": d["title"]} for d in corpus],
    )
    return coll


def _fmt(name: str, m: dict) -> str:
    return f"{name:<34} Hit@1={m['Hit@1']:.2f}  Hit@5={m['Hit@5']:.2f}  MRR={m['MRR']:.3f}"


def main() -> None:
    corpus, queries = load_corpus(), load_queries()
    if "--compare" in sys.argv:
        print(f"{len(queries)} queries over {len(corpus)} labelled passages:\n")
        for label, model in [
            ("multi-qa-MiniLM (relevance/QA)", config.MODEL_NAME),
            ("all-MiniLM (similarity)", config.SIMILARITY_MODEL_NAME),
        ]:
            embed_fn = lambda t, _m=model: embed(t, _m)
            m = evaluate(queries, build_collection(corpus, embed_fn), embed_fn=embed_fn)
            print(_fmt(label, m))
    else:
        m = evaluate(queries, build_collection(corpus))
        print(_fmt(config.MODEL_NAME, m))


if __name__ == "__main__":
    main()
