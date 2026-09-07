# Semantic Search Engine (Space-Domain)

This engine searches a space-domain corpus. The corpus holds arXiv `astro-ph`
abstracts. The engine embeds each document one time and stores the vectors in
ChromaDB. The engine embeds each incoming query and returns the top-_k_
abstracts that answer the query best. This engine is the foundation for a
later Space Document RAG system, so the design allows easy extension.

## What problem this solves

Keyword search misses paraphrases and concept matches. For example, ask *"why
do stars at the edge of a galaxy move too fast?"* A literal index will not
find the abstract about *dark matter halos and flat rotation curves* unless
the words match exactly. This engine retrieves by **meaning**. It returns the
passages that answer the query, not the passages that only repeat the query's
words.

## Architecture & decisions (and why)

**1. Bi-encoder, dense retrieval.** The system embeds each document one time
at ingest and stores the vector. The system embeds each query at request time.
The ranking step runs a cosine-similarity search over the stored vectors. The
bi-encoder precomputes the expensive part, so a query needs only one forward
pass and one approximate-nearest-neighbour lookup. Version 1 has no
cross-encoder reranker. A reranker costs O(N) per query, and this scale does
not need it. See *Future work* for details.

**2. Model: `all-MiniLM-L6-v2` — chosen by measurement, not by first
instinct.** This decision needs an explanation, because the evaluation
overturned the original hypothesis. Sentence-embedding models fall into two
families:

- **Similarity (symmetric):** this family finds text that looks like the
  query. `all-MiniLM-L6-v2` is a strong general model in this family.
- **Relevance / QA (asymmetric):** this family finds the passage that answers
  the query. `multi-qa-MiniLM-L6-cos-v1` trains on about 215 million
  question-and-answer pairs (MS MARCO and similar sets) for query-to-document
  retrieval.

The common heuristic states: search is a relevance task, so use the QA model.
I picked the QA model first for this reason. **My own evaluation disagreed.**
On a labelled set with hard negatives, `all-MiniLM` beat `multi-qa`: Hit@1 was
**0.93 versus 0.80**. The per-query diagnostic showed the reason. ArXiv
abstracts are short, dense, and topic-restating. Queries reuse the vocabulary
of the answer. The task is effectively **symmetric**. The QA model's habit of
discounting surface form then worked against it. The QA model matched
topically-adjacent but wrong passages — for example, it answered a redshift
question with "radio galaxies". The QA heuristic works well for *web-passage
QA*, but abstract retrieval is a different task. I followed the evidence and
set `all-MiniLM` as the default. I re-tested this choice at production length
(200-word abstracts) and on the QA model's best-case task, and the choice
still held. `multi-qa` stays wired into the code (`eval --compare`) as the
expected default for **Phase 3's full-document RAG**, where the asymmetry
should finally give an advantage. The full three-test sequence and numbers
appear in *How I measured it*.

**3. Vector store: ChromaDB (a vector database, not a raw index).** I chose
ChromaDB over raw FAISS. ChromaDB gives metadata storage, metadata filtering,
persistence, and a clean Python SDK. This scaffolding speeds up iteration and
supports the needs of the follow-on RAG project — for example, a filter like
*"only abstracts after 2020"*. ChromaDB wraps an HNSW index, so search runs
sub-linear automatically. This scale needs no index tuning.

**4. Cosine metric, set explicitly.** The code creates the collection with
`metadata={"hnsw:space": "cosine"}`. ChromaDB defaults to the L2 metric. With
normalized vectors, L2 and cosine give the same ranking, but an explicit
cosine setting states the intent clearly. This setting avoids surprises if the
model changes later.

**5. Application-managed vectorization.** The code embeds text and passes the
vectors to Chroma. The code does not use Chroma's auto-embedding feature. This
approach keeps the embedding step visible and preserves flexibility in model
choice. One rule is critical: ingest and query must use the *same* model. The
code enforces this rule by routing both steps through one place
(`src/embedder.py`, with the model name in `src/config.py`). This design stops
the two steps from drifting apart.

**6. Dense-only, monolingual, no hybrid — by design.** The corpus is English
and uniform in style, so pure dense retrieval is the right choice for version
1. Hybrid search (BM25 plus dense) and multilingual support are deferred, not
built. Adding them now would be premature optimization. See *Future work* for
the trigger conditions.

## How I measured it

The model choice (decision #2) was settled by measurement, not by the common
heuristic. The test harness reports **Hit@1 / Hit@5 / MRR**. I used the
harness to test the QA-versus-similarity question in three regimes of
increasing difficulty. The section below describes the actual sequence,
including the wrong turn that the evidence corrected.

### Test 1 — curated short passages (`eval --compare`)

`eval/queries.yaml` contains 15 hand-written space questions. Each question is
labelled with its answer passage in `eval/corpus.yaml`. My first version had
30 passages, each on a distinct topic, and it measured the wrong thing.
Telling "Mars water" apart from 29 unrelated topics tests **topic
separation**, not **relevance**, so both models scored near-ceiling. A
retrieval evaluation must contain cases where a model can get the answer
*wrong*. So I added 10 **hard negatives** (`c31` and later). These passages
echo a query's wording but answer a *different* question — for example, water
ice on the *Moon* for a Mars-water query, or *cosmic inflation* for a
dark-energy query. The keyword bag-of-words floor dropped from 0.47 to 0.40
Hit@1. This drop confirmed that the set became harder in the intended way.

| Model (40 passages, hard negatives) | Hit@1 | Hit@5 | MRR |
|---|---|---|---|
| **`all-MiniLM-L6-v2`** (similarity) | **0.93** | **1.00** | **0.947** |
| `multi-qa-MiniLM-L6-cos-v1` (relevance/QA) | 0.80 | 0.93 | 0.867 |
| keyword bag-of-words (reference floor) | 0.40 | 0.60 | 0.48 |

The similarity model won. This result was the opposite of my first instinct.
`eval --debug` shows the per-query top-1 result and explains the reason. The
gold passage for *"what makes a star **wobble**…"* contains the word "wobble"
directly. `all-MiniLM` matched this passage. `multi-qa` discounted the surface
form and matched a white-dwarf passage instead — the wrong answer. **ArXiv
abstracts are short and topic-restating, so queries reuse the abstract's
vocabulary. Abstract search is effectively symmetric.** The QA model's
asymmetric specialism is wasted here, and it even hurts accuracy. I switched
the default to `all-MiniLM` at this point.

### Test 2 — real abstracts, known-item (`eval --corpus`)

Test 1 used passages of about 50 words. Production documents are
**abstracts** of about 200 words (average 206, maximum 319 words). This
length gap matters. `all-MiniLM` truncates text at **256 tokens**, and
`multi-qa` truncates at **512 tokens**. So on long abstracts, the QA model
reads more text. To test production length without manual labelling, I ran
known-item retrieval: query with each paper's *title*, and retrieve its
abstract. I ran this test over all 500 real abstracts:

| Model (500 real abstracts, title → abstract) | Hit@1 | Hit@5 | MRR |
|---|---|---|---|
| `all-MiniLM-L6-v2` | 0.90 | 0.98 | 0.933 |
| `multi-qa-MiniLM-L6-cos-v1` | 0.90 | 0.98 | 0.933 |

A dead tie — identical to three decimal places. When two different models
score identically, the task does not exercise their difference.
Title-to-abstract retrieval is near-symmetric and front-loaded: the title's
match lives in the abstract's opening, not in the truncated tail. So this test
checks neither asymmetry nor truncation. The result is inconclusive by
design. This is why I built Test 3.

### Test 3 — real abstracts, deep analytical questions (`eval --analytical`)

This test uses 15 hand-written questions (`eval/analytical_queries.yaml`)
over 100 real abstracts. Each question targets the hard case by design: it
points to a detail in the **middle or end** of an abstract, to test
truncation, and it uses **paraphrase** to share little surface form with the
abstract, to test asymmetry. For example: *"Which observed evolved star was
used as a stand-in for the Sun's future mass loss?"* The answer, **L2 Pup**,
appears in the abstract's final sentence.

| Model (100 real abstracts, deep paraphrased Qs) | Hit@1 | Hit@5 | MRR |
|---|---|---|---|
| `all-MiniLM-L6-v2` | **0.80** | 0.87 | **0.833** |
| `multi-qa-MiniLM-L6-cos-v1` | 0.73 | **0.93** | 0.822 |

A near-tie — and a revealing one. The two models split in opposite
directions. `all-MiniLM` wins on **Hit@1**: it ranks its best guess higher.
`multi-qa` wins on **Hit@5**: it places the correct answer somewhere in the
top 5 more often. This precision-versus-recall split matches the predicted
shape of a QA model's advantage. This is also the first point in the whole
investigation where `multi-qa` beats `all-MiniLM` at anything. Each gap
represents one query (n=15), so the difference stays within noise, but it
points in the direction that theory predicts.

### The arc, and the verdict

When I line up the tests by task asymmetry, `multi-qa`'s relative standing
climbs step by step, but it never overtakes `all-MiniLM`:

| Task | asymmetry | all-MiniLM | multi-qa |
|---|---|---|---|
| Curated short passages | low | **0.93** Hit@1 | 0.80 |
| Real abstracts, title → abstract | lowest | 0.90 | 0.90 (tie) |
| Real abstracts, deep paraphrased Qs | **highest** | 0.80 Hit@1 / 0.833 MRR | 0.73 / 0.822 (**wins Hit@5**) |

**Verdict: `all-MiniLM` is the default. Three regimes now confirm this
choice; the choice is not an assumption.** The QA model's advantage is real
in *direction*: it improves as asymmetry increases, and it wins Hit@5 on the
hardest set. But the advantage is too small in *magnitude* to matter at this
corpus and this scale. `multi-qa` stays wired into the code (`eval
--compare`) as the expected default for **Phase 3's full-document RAG**,
where the asymmetry and the 512-token window should finally give an
advantage.

> The goal was never to confirm a guess. The evidence *refuted* the
> hypothesis (the QA model) on short text. The diagnostic explained the
> reason. Re-testing at production length, and on the QA model's best-case
> task, pinned down exactly where the advantage lives and how large it is.
> Evidence set the default; assumption did not.

### Corrections and things I didn't anticipate

This project changed shape as the evidence came in. Here is the honest list:

- **ArXiv category bug.** The first fetch used `cat:astro-ph`. This query
  matches only the *legacy* pre-2009 category and misses modern papers. I
  fixed the bug by joining the six `astro-ph.*` subcategories with OR (see
  `fetch_corpus.search_query`).
- **Token truncation.** At first, I did not consider that `all-MiniLM` caps
  text at 256 tokens while `multi-qa` reads up to 512 tokens. This gap became
  directly relevant once the real documents turned out to average about 270
  tokens. It became a central reason to test at production length (Test 2
  and Test 3).
- **Evaluation representativeness.** My first evaluation used passages far
  shorter than the abstracts the system actually serves. So its conclusion
  did not automatically transfer to production. The abstract-length tests
  close that gap.
- **A flawed test, removed.** An earlier "answer versus reworded-question"
  demo scored 0 out of 6 for *both* models. This test measured
  near-duplicate-text dominance, not relevance, and it could not separate the
  models. I deleted the test instead of leaving it as misleading evidence.
- **Model swap requires re-ingest.** Ingest and query must share one model.
  So a change to the default model requires a rebuild of `data/chroma/` with
  `python -m src.ingest`. The same-model invariant (decision #5) makes this
  change a one-line, foolproof step.

## How it works

The system is a straight pipeline with one shared component in the middle.
Data flows from left to right. Each stage hands its output to the next stage
through a file or the database. This design decouples the stages, so you
re-run only the stage that changed.

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

**Stage 0 — `config.py` (configuration, read by every stage).** This file
defines the model name, the collection name, the cosine metric, and the
on-disk paths. It sits at the top of the dependency graph. A change to the
model or storage location here changes it everywhere, and this keeps ingest
and search in agreement.

**Stage 1 — `fetch_corpus.py` (get the data).** This script calls the public
arXiv API. It flattens each result into `{id, title, abstract, url,
published}` and writes the list to `data/raw/corpus.json`. This is the only
stage that uses the network for the corpus. Every later stage reads the cache
instead, so you fetch the data once.

**Stage 2 — `ingest.py` (build the index).** This script reads
`corpus.json`. For each document, it calls `embedder.embed()` to turn `title +
abstract` into a vector. It upserts each vector, together with the raw text
and metadata, into the ChromaDB collection. `store.py` opens this collection
and persists it under `data/chroma/`. The script keys each entry on the
arXiv id, so a re-run updates existing entries instead of duplicating them.
This is the expensive step, and it runs offline against the cache.

**The shared core — `embedder.py` (text to vectors).** Both ingest and search
call its single `embed()` function. So the system always maps a document and
a query into the *same* vector space with the *same* model. The whole design
protects this invariant: if the two sides used different models, the cosine
comparison would carry no meaning. `store.py` plays the same role for the
database. One function opens the same collection the same way for both
sides.

**Stage 3 — `search.py` (serve queries).** This is the only stage you run
repeatedly. It embeds the incoming query with the same `embed()` function,
asks the collection for the `k` nearest stored vectors, converts cosine
distance back to a 0–1 similarity score, and prints the title, link, and
snippet for each result. Ingest already did the heavy work, so a search runs
one forward pass plus one index lookup — under one second.

**Side branch — `eval.py` (measure quality).** This script runs independently
of the serving path. It builds its *own* in-memory index from the committed
`eval/corpus.yaml` and `eval/queries.yaml` files and reports Hit@k and MRR.
So you can measure retrieval quality, and compare models, without disturbing
the real database.

### Order of operations

Run the stages in dependency order. Each stage needs the artifact that the
previous stage produced:

1. `fetch_corpus` **must** run before `ingest`. Ingest reads `corpus.json`
   and exits with a clear message if the cache is missing.
2. `ingest` **must** run before `search`. Search reads the populated
   `data/chroma/` database; an empty database returns nothing.
3. After the first full run, the cache and the database persist. So
   day-to-day, you run only `search`. Re-run `fetch_corpus` to refresh the
   corpus, then run `ingest` again to re-index. Run `eval` at any time — it
   depends on neither stage.

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

Network note: `fetch_corpus` reaches `export.arxiv.org`. The eval and search
steps download the MiniLM models from `huggingface.co` on the first run.
Both addresses must stay reachable. The corpus fetch caches its result to
`data/raw/corpus.json`, so re-ingesting never calls the API again.

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

## A note on process
AI assistance helped build this project, as a collaborator on the
implementation. The architecture, the evaluation design, the interpretation
of results, and every decision documented above are mine. The evaluation
design includes the decision to add hard negatives and to re-test at
production length (200-word abstracts). The evaluation that overturned my own
model hypothesis is the clearest evidence of which parts are mine.

## Future work

- **Hybrid search (BM25 + dense).** Add this only if the evaluation shows
  exact-term queries failing — for example, mission names, instrument codes
  like `MWIR`, or paper ids. ChromaDB has no native hybrid mode. Run
  `rank_bm25` separately and fuse the results with RRF. This is likely the
  right default for the RAG follow-on.
- **Cross-encoder reranker.** Add this if dense Hit@k plateaus. Rerank the
  top 50 results.
- **Multilingual model (`multilingual-e5-base`).** Switch to this model when
  the corpus or the queries include non-English material.
