# Semantic Search Engine (Space-Domain)

Semantic search over a space-domain corpus (arXiv `astro-ph` abstracts): embed
the documents once, store them in ChromaDB, embed an incoming query, and return
the top-_k_ abstracts that best **answer** it. Foundation for a later Space
Document RAG system, so it's built to extend.

## What problem this solves

Keyword search misses paraphrase and conceptual matches: ask *"why do stars at
the edge of a galaxy move too fast?"* and a literal index won't surface the
abstract about *dark matter halos and flat rotation curves* unless the words
line up. This engine retrieves by **meaning** — it returns the passages whose
content answers the natural-language query, not the ones that merely repeat its
words.

## Architecture & decisions (and why)

**1. Bi-encoder, dense retrieval.** Documents are embedded once at ingest and
stored; queries are embedded at request time; ranking is a cosine-similarity
search over the stored vectors. The bi-encoder precomputes the expensive side,
so query time is one forward pass plus an approximate-nearest-neighbour lookup.
No cross-encoder reranker in v1 — it's O(N) per query and unnecessary at this
scale (see *Future work*).

**2. Model: `multi-qa-MiniLM-L6-cos-v1` — a relevance/QA model, not a
similarity model.** This is the decision worth explaining. Sentence-embedding
models target two different goals:

- **Similarity (symmetric):** *"find text that looks like the query."* A generic
  model like `all-MiniLM-L6-v2` optimises for this and will rank a near-verbatim
  restatement of the query #1.
- **Relevance / QA (asymmetric):** *"find the passage that answers the query."* A
  question and its answer rarely share surface form, so a similarity model
  systematically under-ranks the true answer.

Search is a relevance task, so we use `multi-qa-MiniLM-L6-cos-v1`, trained on
~215M question/answer pairs (MS MARCO and friends) specifically for
query→document retrieval. It's the same small, fast, free MiniLM size class as
`all-MiniLM`, so there's no real cost to the better-aligned choice. Outputs are
L2-normalized → cosine is the natural metric. The eval harness ships a
`--compare` mode that scores both models on the same labelled queries so the
choice is backed by numbers, not assertion (see *How I measured it*).

**3. Vector store: ChromaDB (a vector database, not a raw index).** Chosen over
raw FAISS because it gives metadata storage + filtering, persistence, and a
clean Python SDK out of the box — the scaffolding that makes iteration fast and
that the follow-on RAG project needs (e.g. *"only abstracts after 2020"*). It
wraps an HNSW index, so search is sub-linear automatically; no index tuning at
this scale.

**4. Cosine metric, set explicitly.** The collection is created with
`metadata={"hnsw:space": "cosine"}`. ChromaDB defaults to L2; with normalized
vectors the ranking is identical, but stating cosine signals intent and avoids
surprises if the model is ever swapped.

**5. Application-managed vectorization.** We embed in our own code and pass
vectors to Chroma rather than using Chroma's auto-embedding. This keeps the
embedding step visible and preserves model-choice flexibility. The critical
invariant — ingest and query must use the *same* model — is enforced by routing
both through one place (`src/embedder.py`, model name in `src/config.py`), so
they can't drift.

**6. Dense-only, monolingual, no hybrid — on purpose.** The corpus is English
and homogeneous, so pure dense retrieval is the right v1. Hybrid (BM25 + dense)
and multilingual are deferred, not built — adding them now would be premature
optimisation (see *Future work* for the trigger conditions).

## How I measured it

`eval/queries.yaml` is a hand-written set of 15 realistic space-domain
questions, each labelled with the passage id(s) that answer it. The gold
passages live in `eval/corpus.yaml` (30 distinct space-domain passages), so each
query's answer must be ranked above 29 distractors — a genuine retrieval task,
not a lookup. `python -m src.eval` reports **Hit@1**, **Hit@5**, and **MRR**.

`python -m src.eval --compare` runs the same query set through the QA model and
the pure-similarity model — this is the evidence for decision #2.

| Model | Hit@1 | Hit@5 | MRR |
|---|---|---|---|
| `multi-qa-MiniLM-L6-cos-v1` (relevance/QA) | _run `--compare`_ | _run `--compare`_ | _run `--compare`_ |
| `all-MiniLM-L6-v2` (similarity) | _run `--compare`_ | _run `--compare`_ | _run `--compare`_ |
| keyword bag-of-words (reference baseline) | 0.47 | 0.73 | 0.55 |

The keyword baseline row is produced by the offline test embedder and is shown
as a floor: a model that understands the *question→answer* mapping should beat
it clearly. Run `python -m src.eval --compare` to populate the model rows (it
downloads the two MiniLM models on first run).

## Run it

```bash
pip install -r requirements.txt

python -m src.fetch_corpus            # pull + cache arXiv astro-ph abstracts -> data/raw
python -m src.ingest                  # embed + upsert into persisted ChromaDB (cosine)
python -m src.search "how do galaxies form?"   # top-k by meaning
python -m src.eval                    # Hit@1 / Hit@5 / MRR on the labelled set
python -m src.eval --compare          # QA vs similarity model, same queries

pytest                                # offline smoke + metrics tests (no model download)
```

Network note: `fetch_corpus` reaches `export.arxiv.org` and the eval/search
steps download the MiniLM models from `huggingface.co` on first run; both must be
reachable. The corpus fetch is cached to `data/raw/corpus.json`, so re-ingesting
never re-hits the API.

## Project layout

```
src/config.py        MODEL_NAME, COLLECTION, paths — single source of truth
src/embedder.py      the one place the model is loaded and called (ingest == query)
src/fetch_corpus.py  pull + cache arXiv astro-ph abstracts
src/store.py         open/create the persisted cosine collection
src/ingest.py        embed docs, upsert into Chroma
src/search.py        embed query, query Chroma, pretty-print top-k
src/eval.py          labelled queries -> Hit@k, MRR (+ --compare)
eval/corpus.yaml     30 labelled space-domain passages
eval/queries.yaml    15 queries, each tagged with the answering passage id(s)
tests/test_pipeline.py  offline smoke test + metrics, via an injected fake embedder
```

## Future work

- **Hybrid search (BM25 + dense)** — add only if eval shows exact-term queries
  failing (mission names, instrument codes like `MWIR`, paper ids). ChromaDB has
  no native hybrid; run `rank_bm25` separately and fuse with RRF. Likely the right
  default for the RAG follow-on.
- **Cross-encoder reranker** — add if dense Hit@k plateaus; rerank the top ~50.
- **Multilingual model (`multilingual-e5-base`)** — switch when the corpus or
  queries include non-English material.
