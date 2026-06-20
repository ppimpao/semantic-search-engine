"""Offline tests for the retrieval plumbing.

These never load the neural model: because vectorization is application-managed,
we inject a tiny deterministic bag-of-words embedder and exercise the real
ChromaDB store, the search ranking, and the eval metrics end to end.
"""

import hashlib
import math
import re

import chromadb
import pytest

from src import eval as evalmod
from src.ingest import ingest
from src.search import search

DIM = 1024


def _bucket(tok: str) -> int:
    # Deterministic across runs (unlike Python's salted hash()).
    return int.from_bytes(hashlib.md5(tok.encode()).digest()[:4], "big") % DIM


def fake_embed(texts):
    """Hashing bag-of-words: shared vocabulary -> high cosine similarity."""
    out = []
    for text in texts:
        vec = [0.0] * DIM
        for tok in re.findall(r"[a-z]+", text.lower()):
            vec[_bucket(tok)] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        out.append([v / norm for v in vec])
    return out


@pytest.fixture
def collection():
    client = chromadb.EphemeralClient()
    client.get_or_create_collection(name="smoke")  # reset across runs
    client.delete_collection(name="smoke")
    return client.create_collection(
        name="smoke", metadata={"hnsw:space": "cosine"}
    )


FIXTURE = [
    {"id": "p1", "title": "Mars rovers find ancient riverbeds",
     "abstract": "Dry channels and clay minerals show liquid water once flowed on Mars.",
     "url": "u1", "published": "2024-01-01"},
    {"id": "p2", "title": "Detecting exoplanets by transit",
     "abstract": "A planet crossing its star dims the starlight in periodic transits.",
     "url": "u2", "published": "2024-01-02"},
    {"id": "p3", "title": "Pulsars are spinning neutron stars",
     "abstract": "Collapsed stellar cores sweep radio beams past Earth as regular pulses.",
     "url": "u3", "published": "2024-01-03"},
]


def test_ingest_then_search_returns_expected_top1(collection):
    assert ingest(FIXTURE, collection=collection, embed_fn=fake_embed) == 3
    hits = search("water flowing on the planet Mars",
                  collection=collection, embed_fn=fake_embed)
    assert hits[0]["id"] == "p1"
    assert 0.0 < hits[0]["score"] <= 1.0  # cosine similarity from shared vocabulary


def test_metrics_math():
    assert evalmod.hit_at_k(["a", "b", "c"], {"b"}, 1) == 0
    assert evalmod.hit_at_k(["a", "b", "c"], {"b"}, 5) == 1
    assert evalmod.reciprocal_rank(["a", "b", "c"], {"b"}) == pytest.approx(0.5)
    assert evalmod.reciprocal_rank(["a", "b"], {"z"}) == 0.0


def test_eval_queries_reference_real_corpus_ids():
    ids = {d["id"] for d in evalmod.load_corpus()}
    for q in evalmod.load_queries():
        assert q["relevant"], f"query has no labels: {q['query']}"
        assert set(q["relevant"]) <= ids, f"unknown id in: {q['query']}"


def test_parse_arxiv_entry():
    import xml.etree.ElementTree as ET

    from src.fetch_corpus import NS, _parse_entry

    entry = ET.fromstring(
        """
        <entry xmlns="http://www.w3.org/2005/Atom">
          <id>http://arxiv.org/abs/2401.01234v2</id>
          <title>A   Study   of\nGalaxies</title>
          <summary>We   measure\nthings.</summary>
          <published>2024-01-02T10:00:00Z</published>
        </entry>
        """.strip()
    )
    rec = _parse_entry(entry)
    assert rec == {
        "id": "2401.01234",
        "title": "A Study of Galaxies",
        "abstract": "We measure things.",
        "url": "https://arxiv.org/abs/2401.01234",
        "published": "2024-01-02",
    }
    assert NS["atom"]  # namespace map is wired up


def test_evaluate_runs_over_eval_corpus():
    corpus = evalmod.load_corpus()
    coll = evalmod.build_collection(corpus, embed_fn=fake_embed)
    metrics = evalmod.evaluate(evalmod.load_queries(), coll, embed_fn=fake_embed)
    assert metrics["n"] == len(evalmod.load_queries())
    assert 0.0 <= metrics["MRR"] <= 1.0
