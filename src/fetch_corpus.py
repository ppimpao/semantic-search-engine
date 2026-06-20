"""Step 1 of the pipeline: download the corpus and cache it to disk.

Position in the workflow
------------------------
The very first stage. It talks to the public arXiv API, normalizes each result
into a flat record, and writes them all to a JSON cache. Nothing downstream
touches the network — `ingest` reads only the cached file — so the corpus is
fetched once and reused.

    [arXiv API]  ->  fetch_corpus  ->  data/raw/corpus.json  ->  ingest

Depends on
----------
- `config`  : RAW_CORPUS (where to write the cache)
- requests  : HTTP calls to the arXiv API
- network   : reachable `export.arxiv.org`

Provides for
------------
- `ingest`  : the cached corpus.json it reads at the next step

Each cached record is a flat dict: {id, title, abstract, url, published}.
"""

import json
import sys
import time
import xml.etree.ElementTree as ET

import requests

from . import config

ARXIV_API = "https://export.arxiv.org/api/query"
# The arXiv API returns Atom XML; this namespace prefix lets ElementTree find
# <entry>, <title>, etc. (every tag is namespaced under this URI).
NS = {"atom": "http://www.w3.org/2005/Atom"}
# Per the API manual a single call returns at most 2000 results per slice (and
# 30000 total); we page in chunks of this size, well under the cap.
PAGE_SIZE = 200

# In 2009 the astro-ph archive was split into these six subcategories. The bare
# token `cat:astro-ph` only matches the *legacy* category and misses modern
# papers, so we OR the subcategories to cover all of astrophysics.
ASTRO_PH_SUBCATS = (
    "astro-ph.GA",  # Astrophysics of Galaxies
    "astro-ph.CO",  # Cosmology and Nongalactic Astrophysics
    "astro-ph.EP",  # Earth and Planetary Astrophysics
    "astro-ph.HE",  # High Energy Astrophysical Phenomena
    "astro-ph.IM",  # Instrumentation and Methods for Astrophysics
    "astro-ph.SR",  # Solar and Stellar Astrophysics
)


def search_query() -> str:
    """Build the search_query: every astro-ph subcategory OR'd together.

    e.g. "cat:astro-ph.GA OR cat:astro-ph.CO OR ...". requests URL-encodes this;
    arXiv treats the OR as a boolean union, so we get all of astrophysics.
    """
    return " OR ".join(f"cat:{c}" for c in ASTRO_PH_SUBCATS)


def _parse_entry(entry: ET.Element) -> dict:
    """Turn one Atom <entry> element into our flat record dict.

    Pulls out the id/title/abstract/date and tidies them: the arXiv id is
    extracted from its full URL and stripped of the version suffix, and the text
    fields are whitespace-collapsed (the API pretty-prints them across lines).
    """
    raw_id = entry.findtext("atom:id", default="", namespaces=NS)
    # raw_id looks like "http://arxiv.org/abs/2401.01234v1":
    #   rsplit("/", 1)[-1] -> "2401.01234v1"; split("v")[0] -> "2401.01234"
    arxiv_id = raw_id.rsplit("/", 1)[-1].split("v")[0]
    return {
        "id": arxiv_id,
        # " ".join(text.split()) collapses newlines/runs of spaces into one space.
        "title": " ".join(entry.findtext("atom:title", "", NS).split()),
        "abstract": " ".join(entry.findtext("atom:summary", "", NS).split()),
        "url": f"https://arxiv.org/abs/{arxiv_id}",
        "published": entry.findtext("atom:published", "", NS)[:10],  # YYYY-MM-DD
    }


def fetch(total: int = 500) -> list[dict]:
    """Pull `total` most-recent astro-ph abstracts, paging through the API.

    Loops in PAGE_SIZE-sized windows, parsing each page's entries and stopping
    early if the API runs out of results. Returns the combined list of records.
    """
    records: list[dict] = []
    for start in range(0, total, PAGE_SIZE):
        params = {
            "search_query": search_query(),                  # all astro-ph subcats
            "start": start,                                  # paging offset
            "max_results": min(PAGE_SIZE, total - start),    # don't overshoot total
            "sortBy": "submittedDate",
            "sortOrder": "descending",                       # newest first
        }
        resp = requests.get(ARXIV_API, params=params, timeout=30)
        resp.raise_for_status()  # turn any HTTP error into an exception
        entries = ET.fromstring(resp.text).findall("atom:entry", NS)
        if not entries:
            break  # no more results — stop even if we haven't reached `total`
        records.extend(_parse_entry(e) for e in entries)
        time.sleep(3)  # arXiv asks callers to leave a 3s gap between requests
    return records


def main() -> None:
    """CLI entry point: fetch and write the cache to data/raw/corpus.json.

    Optional first arg overrides how many abstracts to pull (default 500).
    """
    total = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    records = fetch(total)
    # Create data/raw/ if it doesn't exist yet, then write the cache.
    config.RAW_CORPUS.parent.mkdir(parents=True, exist_ok=True)
    config.RAW_CORPUS.write_text(json.dumps(records, indent=2))
    print(f"Cached {len(records)} abstracts -> {config.RAW_CORPUS}")


if __name__ == "__main__":
    main()
