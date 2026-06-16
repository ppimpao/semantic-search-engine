"""Single source of truth for paths, model, and collection.

The model name lives here and nowhere else: ingest and search both import it,
so the embedding model can never drift between the two sides of the pipeline.
"""

from pathlib import Path

# --- Model ---------------------------------------------------------------
# A *relevance / QA* model (asymmetric query->document), not a pure-similarity
# model. See README "Architecture & decisions" for why this beats
# all-MiniLM-L6-v2 for search. Outputs are L2-normalized, so cosine is natural.
MODEL_NAME = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"

# Baseline used only by the eval comparison (similarity vs relevance demo).
SIMILARITY_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Vector store --------------------------------------------------------
COLLECTION = "astro_ph"
# ChromaDB defaults to l2; set cosine explicitly to signal intent.
DISTANCE_SPACE = "cosine"

# --- Paths ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_CORPUS = DATA_DIR / "raw" / "corpus.json"   # cached fetch
CHROMA_DIR = DATA_DIR / "chroma"                 # persisted collection
