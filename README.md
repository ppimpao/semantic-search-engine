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
default to `all-MiniLM` — a choice then re-tested at production length (200-word
abstracts) and on the QA model's best-case task, where it still held. `multi-qa`
stays wired in (`eval --compare`) and is the expected default for **Phase 3's
full-document RAG**, where the asymmetry should finally pay off. The full
three-test arc and numbers are in *How I measured it*.

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

The model choice (decision #2) was settled by measurement, not by the textbook
heuristic. The harness reports **Hit@1 / Hit@5 / MRR**, and I used it to test the
QA-vs-similarity question in three regimes of increasing difficulty. The story
below is the actual sequence — including the wrong turn it corrected.

### Test 1 — curated short passages (`eval --compare`)

`eval/queries.yaml` is 15 hand-written space questions labelled to passages in
`eval/corpus.yaml`. My *first* version was 30 distinct-topic passages, and it
measured the wrong thing: telling "Mars water" apart from 29 unrelated topics is
**topic separation**, not **relevance**, and both models scored near-ceiling. A
retrieval eval has to contain the case a model can get *wrong*. So I added 10
**hard negatives** (`c31`+) — passages that echo a query's wording while
answering a *different* question (water ice on the *Moon* for a Mars-water query,
*cosmic inflation* for a dark-energy query). The keyword bag-of-words floor
dropped 0.47→0.40 Hit@1, confirming the set got harder in the intended way.

| Model (40 passages, hard negatives) | Hit@1 | Hit@5 | MRR |
|---|---|---|---|
| **`all-MiniLM-L6-v2`** (similarity) | **0.93** | **1.00** | **0.947** |
| `multi-qa-MiniLM-L6-cos-v1` (relevance/QA) | 0.80 | 0.93 | 0.867 |
| keyword bag-of-words (reference floor) | 0.40 | 0.60 | 0.48 |

The similarity model won — the opposite of my instinct. `eval --debug` (per-query
top-1) showed *why*: the gold passage for *"what makes a star **wobble**…"*
literally contains "wobble"; `all-MiniLM` matched it, while `multi-qa`,
discounting surface form, wandered to a white-dwarf passage. **arXiv abstracts
are short and topic-restating, so queries reuse their vocabulary — abstract
search is effectively symmetric**, and the QA model's asymmetric specialism is
wasted (it even hurts). I switched the default to `all-MiniLM` here.

### Test 2 — real abstracts, known-item (`eval --corpus`)

Test 1 used ~50-word passages, but production docs are ~200-word **abstracts**
(avg 206, max 319 words). That length gap matters: `all-MiniLM` truncates at
**256 tokens**, `multi-qa` at **512** — so on long abstracts the QA model
literally reads more text. To probe production length with no manual labelling, I
ran known-item retrieval (query with each paper's *title*, retrieve its abstract)
over all 500 real abstracts:

| Model (500 real abstracts, title → abstract) | Hit@1 | Hit@5 | MRR |
|---|---|---|---|
| `all-MiniLM-L6-v2` | 0.90 | 0.98 | 0.933 |
| `multi-qa-MiniLM-L6-cos-v1` | 0.90 | 0.98 | 0.933 |

A dead tie — *identical* to three decimals. When two different models score
identically, the task isn't exercising their difference: title→abstract is
near-symmetric and front-loaded (the title's match lives in the abstract's head,
not the truncated tail), so it tests neither asymmetry nor truncation. Inconclusive
by design — which is why I built Test 3.

### Test 3 — real abstracts, deep analytical questions (`eval --analytical`)

15 hand-written questions (`eval/analytical_queries.yaml`) over 100 real
abstracts, each one engineered to be the hard case: it targets a detail in the
**middle or end** of an abstract (probing truncation) and is **paraphrased** to
share little surface form with it (probing asymmetry). E.g. *"Which observed
evolved star was used as a stand-in for the Sun's future mass loss?"* → an
abstract whose answer (**L2 Pup**) is in its final sentence.

| Model (100 real abstracts, deep paraphrased Qs) | Hit@1 | Hit@5 | MRR |
|---|---|---|---|
| `all-MiniLM-L6-v2` | **0.80** | 0.87 | **0.833** |
| `multi-qa-MiniLM-L6-cos-v1` | 0.73 | **0.93** | 0.822 |

A near-tie — and a revealing one. The models split in opposite directions:
`all-MiniLM` takes **Hit@1** (better at ranking its best guess #1), while
`multi-qa` takes **Hit@5** (better at getting the answer *somewhere* in the
top-5). That precision-vs-recall split is exactly the predicted shape of a QA
model's advantage — and it's the *first time in the whole investigation* that
`multi-qa` beat `all-MiniLM` at anything. Every gap is one query (n=15), so it's
within noise — but it points the way theory says it should.

### The arc, and the verdict

Lined up by how *asymmetric* the task is, `multi-qa`'s relative standing climbs
monotonically — yet never crosses over:

| Task | asymmetry | all-MiniLM | multi-qa |
|---|---|---|---|
| Curated short passages | low | **0.93** Hit@1 | 0.80 |
| Real abstracts, title → abstract | lowest | 0.90 | 0.90 (tie) |
| Real abstracts, deep paraphrased Qs | **highest** | 0.80 Hit@1 / 0.833 MRR | 0.73 / 0.822 (**wins Hit@5**) |

**Verdict: `all-MiniLM` is the default, now confirmed across three regimes, not
assumed.** The QA model's advantage is real in *direction* (it improves with
asymmetry, and takes Hit@5 on the hardest set) but too small in *magnitude* to
matter at this corpus and scale. It stays wired in (`eval --compare`) as the
expected default for **Phase 3's full-document RAG**, where the asymmetry — and
the 512-token window — should finally pay off.

> The point was never to confirm a guess. The hypothesis (QA model) was *refuted*
> on short text, the diagnostic explained *why*, and re-testing at production
> length and on the QA model's best-case task pinned down exactly *where* the
> advantage lives and how big it is. The default is evidence, not a default.

### Corrections and things I didn't anticipate

This project changed shape as the evidence came in. The honest list:

- **arXiv category bug.** The first fetch used `cat:astro-ph`, which only matches
  the *legacy* pre-2009 category and misses modern papers. Fixed by OR-ing the six
  `astro-ph.*` subcategories (see `fetch_corpus.search_query`).
- **Token truncation.** I didn't initially consider that `all-MiniLM` caps at 256
  tokens while `multi-qa` reads 512 — directly relevant once the real docs turned
  out to average ~270 tokens. It became a central reason to test at production
  length (Test 2/3).
- **Eval representativeness.** My first eval used passages far shorter than the
  abstracts actually served, so its conclusion didn't automatically transfer; the
  abstract-length tests exist to close that gap.
- **A flawed test, removed.** An earlier "answer vs. reworded-question" demo
  scored 0/6 for *both* models — it measured near-duplicate-text dominance, not
  relevance, and couldn't separate the models. It was deleted rather than left in
  as misleading evidence.
- **Model swap ⇒ re-ingest.** Because ingest and query must share a model, changing
  the default means rebuilding `data/chroma/` with `python -m src.ingest`. The
  same-model invariant (decision #5) is what makes this a one-line, foolproof change.

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
python -m src.eval --compare          # all-MiniLM (default) vs multi-qa (Test 1)
python -m src.eval --debug            # per-query top-1 for both models
python -m src.eval --corpus           # both models on real abstracts, title->abstract (Test 2)
python -m src.eval --freeze           # snapshot real abstracts -> eval/abstracts.json
python -m src.eval --analytical       # both models on deep questions over real abstracts (Test 3)

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
src/eval.py          Hit@k / MRR (+ --compare, --debug, --corpus, --freeze, --analytical)
eval/corpus.yaml     40 passages: 30 gold answers + 10 hard-negative traps (Test 1)
eval/queries.yaml    15 queries, each tagged with the answering passage id(s) (Test 1)
eval/analytical_queries.yaml  15 deep, paraphrased questions over real abstracts (Test 3)
eval/abstracts.json  frozen real-abstract corpus for Test 3 (created by --freeze)
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
