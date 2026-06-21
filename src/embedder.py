"""The one place a sentence-transformer model is loaded and called.

Position in the workflow
------------------------
This is the heart of the system: text-in, vectors-out. Both sides of the
pipeline route through `embed()`, which guarantees they use the *same* model —
the critical invariant, because a query vector can only be compared against
document vectors produced by the identical model.

Depends on
----------
- `config`             : MODEL_NAME (the default model to load)
- sentence-transformers: the actual neural model (imported lazily, see below)

Provides for
------------
- `ingest`  : `embed()` to vectorize documents at write time
- `search`  : `embed()` to vectorize the query at read time
- `eval`    : `embed()` for both models in the comparison
"""

from functools import lru_cache

from . import config


@lru_cache(maxsize=2)
def _load(model_name: str):
    """Load (and cache) a SentenceTransformer model by name.

    Loading is the slow, memory-heavy step (downloads weights on first use),
    so `@lru_cache` keeps each model in memory and returns the same instance on
    every later call — the model loads once per process, not once per batch.
    maxsize=2 covers the two models the eval comparison uses.
    """
    # Imported here, not at module top, so the rest of the pipeline (and the
    # tests, which inject their own embedder) can import this module without
    # pulling in torch/transformers just to be importable.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed(texts: list[str], model_name: str = config.MODEL_NAME) -> list[list[float]]:
    """Turn a list of strings into a list of normalized dense vectors.

    This is the function the whole pipeline shares. `ingest` calls it on
    documents, `search` calls it on the query; same code, same model, so the
    vectors live in the same space and cosine comparison is meaningful.

    We ask for L2-normalized output and let cosine similarity be the metric.
    Documents and queries go through the identical forward pass (the bi-encoder
    is mechanically symmetric); for abstract search that symmetry is a feature,
    which is why the general all-MiniLM model wins here (see config.MODEL_NAME).
    """
    model = _load(model_name)
    # normalize_embeddings=True -> unit-length vectors (cosine-ready);
    # convert_to_numpy=True -> a single array we can cheaply turn into lists.
    vectors = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    # Chroma wants plain Python lists, not a numpy array.
    return vectors.tolist()
