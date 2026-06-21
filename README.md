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
passages live in `eval/corpus.yaml`, and `python -m src.eval` reports **Hit@1**,
**Hit@5**, and **MRR**.

### Designing an eval that actually discriminates

My first version of this set was 30 passages, each on a *distinct* topic. It
turned out to measure the wrong thing: telling "Mars water" apart from 29
unrelated topics is **topic separation**, not **relevance**, and on that easy
set both small models scored near-ceiling — the pure-similarity model even edged
ahead (a one-query difference at n=15, i.e. noise). A retrieval eval has to
contain the case the model can get *wrong*, or it proves nothing.

So the set now includes deliberate **hard negatives** (`c31`+): passages that
echo a query's wording while answering a *different* question — water ice on the
*Moon* for a Mars-water query, *cosmic inflation* for a dark-energy query,
*gamma-ray* bursts for a *radio*-burst query. These are the lexical traps a
similarity-only model falls for. (Sanity check: the keyword bag-of-words floor
drops from 0.47→0.40 Hit@1 once the traps are added — the set got harder in the
intended way.)

`python -m src.eval --compare` scores both models over this harder set:

| Model | Hit@1 | Hit@5 | MRR |
|---|---|---|---|
| `multi-qa-MiniLM-L6-cos-v1` (relevance/QA) | _run `--compare`_ | _run `--compare`_ | _run `--compare`_ |
| `all-MiniLM-L6-v2` (similarity) | _run `--compare`_ | _run `--compare`_ | _run `--compare`_ |
| keyword bag-of-words (reference floor) | 0.40 | 0.60 | 0.48 |

### The cleanest isolation: answer vs. reworded question

`python -m src.eval --demo` is the sharpest version of decision #2. Each case
gives a model a question plus two candidates — the **answer**, and the **same
question reworded** — and asks which it ranks higher. A similarity model is
pulled toward the look-alike question; a relevance/QA model prefers the answer:

| Model | Answer ranked above the reworded question |
|---|---|
| `multi-qa-MiniLM-L6-cos-v1` (relevance/QA) | _run `--demo`_ |
| `all-MiniLM-L6-v2` (similarity) | _run `--demo`_ |

Run `--compare` and `--demo` (they download the two MiniLM models on first use)
to populate the cells above.

## How it works

The system is a straight pipeline with one shared component in the middle. Data
flows left to right; each stage hands its output to the next through a file or
the database, so the stages are decoupled and you only re-run what changed.

```
                      ┌──────────────────────────────────────────────┐
                      │  src/config.py  (names, paths, model, metric) │  <- every stage reads this
                      └──────────────────────────────────────────────┘

  [arXiv API]
      │  fetch_corpus.py            ingest.py                      search.py
      ▼                                                                ▲
  data/raw/corpus.json  ───────►  embed each doc  ───►  data/chroma/  ─┘
                                       ▲                  (vectors +     embed query → query()
                                       │                   text +        → ranked top-k
                                  embedder.py              metadata)
                                   embed()  ◄───────────────────────────────┘
                                (the SAME function vectorizes documents AND queries)
```

**Stage 0 — `config.py` (configuration, read by everyone).** Defines the model
name, the collection name, the cosine metric, and the on-disk paths. It is the
top of the dependency graph: changing the model or storage location here changes
it everywhere, which is what keeps ingest and search in agreement.

**Stage 1 — `fetch_corpus.py` (acquire the data).** Calls the public arXiv API,
flattens each result into `{id, title, abstract, url, published}`, and writes the
list to `data/raw/corpus.json`. This is the only stage that uses the network for
the corpus; everything after it reads the cache, so you fetch once.

**Stage 2 — `ingest.py` (build the index).** Reads `corpus.json`, and for each
document calls `embedder.embed()` to turn `title + abstract` into a vector. It
`upsert`s those vectors — together with the raw text and metadata — into the
ChromaDB collection opened by `store.py`, which persists under `data/chroma/`.
Because it keys on the arXiv id, re-running it updates rather than duplicates.
This is the expensive step, and it runs offline against the cache.

**The shared core — `embedder.py` (text → vectors).** Both ingest and search go
through its single `embed()` function, so a document and a query are always
mapped into the *same* vector space by the *same* model. This is the invariant
the whole design protects: if the two sides used different models, the cosine
comparison would be meaningless. `store.py` plays the analogous role for the
database — one function both sides call to open the same collection the same way.

**Stage 3 — `search.py` (serve queries).** The only stage you run repeatedly. It
embeds the incoming query with the same `embed()`, asks the collection for the
`k` nearest stored vectors, converts cosine distance back to a 0–1 similarity
score, and prints title / link / snippet. All the heavy lifting happened at
ingest, so this is one forward pass + an index lookup — sub-second.

**Side branch — `eval.py` (measure quality).** Independent of the serving path:
it builds its *own* in-memory index from the committed `eval/corpus.yaml` +
`eval/queries.yaml` and reports Hit@k / MRR, so you can quantify retrieval
quality (and compare models) without disturbing the real database.

### Order of operations

Run the stages in dependency order — each needs the artifact the previous one
produced:

1. `fetch_corpus` **must** run before `ingest` (ingest reads `corpus.json`; it
   exits with a clear message if the cache is missing).
2. `ingest` **must** run before `search` (search reads the populated
   `data/chroma/`; an empty DB just returns nothing).
3. After the first full run the cache and DB persist, so day-to-day you only run
   `search`. Re-run `fetch_corpus` to refresh the corpus, then `ingest` again to
   re-index. `eval` can be run any time — it depends on neither.

## Run it

```bash
pip install -r requirements.txt

python -m src.fetch_corpus            # pull + cache arXiv astro-ph abstracts -> data/raw
python -m src.ingest                  # embed + upsert into persisted ChromaDB (cosine)
python -m src.search "how do galaxies form?"   # top-k by meaning
python -m src.eval                    # Hit@1 / Hit@5 / MRR on the labelled set
python -m src.eval --compare          # QA vs similarity model, same queries
python -m src.eval --demo             # answer vs. reworded-question isolation

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
src/eval.py          labelled queries -> Hit@k, MRR (+ --compare, --demo)
eval/corpus.yaml     40 passages: 30 gold answers + 10 hard-negative traps
eval/queries.yaml    15 queries, each tagged with the answering passage id(s)
eval/demo.yaml       answer vs. reworded-question cases for --demo
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
