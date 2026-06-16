"""The one place a sentence-transformer model gets loaded and called.

Both ingest and search go through `embed()`, so they are guaranteed to use the
same model (the critical invariant). The model is loaded lazily and cached, so
importing this module is cheap and tests can avoid it entirely by passing their
own embed function.
"""

from functools import lru_cache

from . import config


@lru_cache(maxsize=2)
def _load(model_name: str):
    # Imported lazily so the rest of the pipeline (and the tests, which inject
    # their own embedder) doesn't pull in torch just to be importable.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed(texts: list[str], model_name: str = config.MODEL_NAME) -> list[list[float]]:
    """Embed texts into normalized dense vectors.

    The model is a *-cos-v1 variant, so we normalize and let cosine similarity
    be the metric. Documents and queries are embedded the same way — the
    bi-encoder is symmetric in mechanics; the *training* is what makes it
    asymmetric (query<->answer aware).
    """
    model = _load(model_name)
    vectors = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return vectors.tolist()
