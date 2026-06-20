"""Central configuration — the single source of truth for the whole pipeline.

Position in the workflow
------------------------
This is the root of the dependency graph. Almost every other module imports it
(`embedder`, `store`, `fetch_corpus`, `ingest`, `search`, `eval`), and it imports
nothing of our own. Change a name or path here and the entire pipeline follows.

Provides for
------------
- `embedder`     : MODEL_NAME / SIMILARITY_MODEL_NAME (which model to load)
- `store`        : COLLECTION, DISTANCE_SPACE, CHROMA_DIR (which DB to open, how)
- `fetch_corpus` : RAW_CORPUS (where to cache the download)
- `ingest`       : RAW_CORPUS (what to read), COLLECTION (where it lands)
- `eval`         : MODEL_NAME, SIMILARITY_MODEL_NAME, ROOT (to find eval/*.yaml)

Depends on
----------
Nothing internal — only the standard library.
"""

from pathlib import Path

# --- Model ---------------------------------------------------------------
# A *relevance / QA* model (asymmetric query->document), not a pure-similarity
# model. See README "Architecture & decisions" for why this beats
# all-MiniLM-L6-v2 for search. Outputs are L2-normalized, so cosine is natural.
# `embedder.embed()` defaults to this; ingest and search therefore share it.
MODEL_NAME = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"

# The pure-similarity baseline. Used *only* by `eval --compare` to demonstrate,
# with numbers, why the QA model above is the right choice for search.
SIMILARITY_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Vector store --------------------------------------------------------
COLLECTION = "astro_ph"            # the named collection inside the Chroma DB
# ChromaDB defaults to l2; set cosine explicitly to signal intent. `store` reads
# this when creating the collection so ingest and search agree on the metric.
DISTANCE_SPACE = "cosine"

# --- Paths ---------------------------------------------------------------
# ROOT is the repo root: this file is src/config.py, so two parents up.
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_CORPUS = DATA_DIR / "raw" / "corpus.json"   # fetch_corpus writes, ingest reads
CHROMA_DIR = DATA_DIR / "chroma"                # ingest writes, search reads
