"""Fetch arXiv astro-ph abstracts and cache them to disk.

Run once: `python -m src.fetch_corpus`. The raw result is cached to
data/raw/corpus.json so re-ingesting never re-hits the API.

Each record: {id, title, abstract, url, published}.
"""

import json
import sys
import time
import xml.etree.ElementTree as ET

import requests

from . import config

ARXIV_API = "http://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom"}
PAGE_SIZE = 100


def _parse_entry(entry: ET.Element) -> dict:
    raw_id = entry.findtext("atom:id", default="", namespaces=NS)
    # http://arxiv.org/abs/2401.01234v1 -> 2401.01234
    arxiv_id = raw_id.rsplit("/", 1)[-1].split("v")[0]
    return {
        "id": arxiv_id,
        "title": " ".join(entry.findtext("atom:title", "", NS).split()),
        "abstract": " ".join(entry.findtext("atom:summary", "", NS).split()),
        "url": f"https://arxiv.org/abs/{arxiv_id}",
        "published": entry.findtext("atom:published", "", NS)[:10],
    }


def fetch(total: int = 500) -> list[dict]:
    """Pull `total` recent astro-ph abstracts, paging through the API."""
    records: list[dict] = []
    for start in range(0, total, PAGE_SIZE):
        params = {
            "search_query": "cat:astro-ph",
            "start": start,
            "max_results": min(PAGE_SIZE, total - start),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        resp = requests.get(ARXIV_API, params=params, timeout=30)
        resp.raise_for_status()
        entries = ET.fromstring(resp.text).findall("atom:entry", NS)
        if not entries:
            break
        records.extend(_parse_entry(e) for e in entries)
        time.sleep(3)  # arXiv asks for a 3s gap between calls
    return records


def main() -> None:
    total = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    records = fetch(total)
    config.RAW_CORPUS.parent.mkdir(parents=True, exist_ok=True)
    config.RAW_CORPUS.write_text(json.dumps(records, indent=2))
    print(f"Cached {len(records)} abstracts -> {config.RAW_CORPUS}")


if __name__ == "__main__":
    main()
