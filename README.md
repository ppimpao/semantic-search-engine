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

**2. Model: `all-MiniLM-L6-v2` — chosen by measurement, against my first
instinct.** This is the decision worth explaining, because the eval overturned
my hypothesis. Sentence-embedding models split into two families:

- **Similarity (symmetric):** *"find text that looks like the query."*
  `all-MiniLM-L6-v2` is a strong general model of this kind.
- **Relevance / QA (asymmetric):** *"find the passage that answers the query."*
  `multi-qa-MiniLM-L6-cos-v1` is trained on ~215M question/answer pairs (MS MARCO
  and friends) for query→document retrieval.

The textbook heuristic says *"search is a relevance task, so use the QA model,"*
and that's what I picked first. **My own eval disagreed.** On a labelled set with
hard negatives, `all-MiniLM` beat `multi-qa` (Hit@1 **0.93 vs 0.80**), and the
per-query diagnostic showed *why*: arXiv abstracts are short, dense, and
topic-restating, so queries reuse the answer's vocabulary — the task is
effectively **symmetric**. The QA model's trained habit of discounting surface
form then *backfired*, pulling it to topically-adjacent-but-wrong passages (it
answered a redshift question with "radio galaxies"). The QA heuristic is sound
for *web-passage QA*; abstract retrieval isn't that. So I follow the evidence and
default to `all-MiniLM`. `multi-qa` stays wired in (`eval --compare`) and is the
expected default for **Phase 3's long-document RAG**, where retrieval is genuinely
asymmetric. Full numbers and reasoning in *How I measured it*.

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

### Building an eval that actually discriminates

My first set was 30 passages, each on a *distinct* topic. It measured the wrong
thing: telling "Mars water" apart from 29 unrelated topics is **topic
separation**, not **relevance**, and both small models scored near-ceiling. A
retrieval eval has to contain the case a model can get *wrong*, or it proves
nothing.

So the set adds deliberate **hard negatives** (`c31`+): passages that echo a
query's wording while answering a *different* question — water ice on the *Moon*
for a Mars-water query, *cosmic inflation* for a dark-energy query, *gamma-ray*
bursts for a *radio*-burst query. (Sanity check: the keyword bag-of-words floor
drops from 0.47→0.40 Hit@1 once the traps are added — the set got harder in the
intended way.)

### What `--compare` found — and why I switched models

`python -m src.eval --compare` scores both models over the harder 40-passage set:

| Model | Hit@1 | Hit@5 | MRR |
|---|---|---|---|
| **`all-MiniLM-L6-v2`** (similarity, **default**) | **0.93** | **1.00** | **0.947** |
| `multi-qa-MiniLM-L6-cos-v1` (relevance/QA) | 0.80 | 0.93 | 0.867 |
| keyword bag-of-words (reference floor) | 0.40 | 0.60 | 0.48 |

The similarity model won — the opposite of decision #2's original instinct. The
entire gap was two queries `multi-qa` dropped that `all-MiniLM` got, and
`python -m src.eval --debug` (per-query top-1 for both models) showed exactly
what happened:

- *"What makes a star **wobble** so we can weigh an unseen planet?"* — the gold
  passage literally contains "wobble." `all-MiniLM` matched it; `multi-qa`,
  discounting surface form, wandered to a white-dwarf passage.
- *"How far away is a galaxy from the **colour of its light**?"* — `all-MiniLM`
  found the redshift passage; `multi-qa` was pulled to a *radio-galaxies* hard
  negative ("galaxy + wavelengths"), a plausible-but-wrong leap.

The lesson: **arXiv abstracts are short and topic-restating, so queries reuse
their vocabulary — abstract search is effectively symmetric.** The QA model's
asymmetric specialism is wasted here and even hurts. It remains the right tool
for the asymmetric, long-document retrieval coming in Phase 3, which is why it
stays available behind `--compare`.

> Takeaway: the value wasn't confirming a guess — it was the eval *refuting* one.
> The hypothesis was the QA model; the measurement said otherwise; the diagnostic
> said why; the default changed. That loop is the point of building the harness.

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
python -m src.eval --compare          # all-MiniLM (default) vs multi-qa
python -m src.eval --debug            # per-query top-1 for both models

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
src/eval.py          labelled queries -> Hit@k, MRR (+ --compare, --debug)
eval/corpus.yaml     40 passages: 30 gold answers + 10 hard-negative traps
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
