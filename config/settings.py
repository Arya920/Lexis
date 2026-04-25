import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

UPLOAD_DIR = os.path.join(BASE_DIR, "data/uploads")
VECTOR_DB_PATH = os.path.join(BASE_DIR, "data/vectordb")
CHUNK_STORE_PATH = os.path.join(VECTOR_DB_PATH, "chunks.json")

# Files here are NOT indexed into RAG — they are used by agents only.
DATASETS_DIR = os.path.join(BASE_DIR, "data", "datasets")
LOG_DIR = os.path.join(BASE_DIR, "Logs")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# GENERATION_MODEL_NAME = "groq:llama-3.1-8b-instant"
GENERATION_MODEL_NAME = "groq:llama-3.3-70b-versatile"

SEMANTIC_CHUNK_MIN_SENTENCES = 3
SEMANTIC_CHUNK_MAX_SENTENCES = 8
SEMANTIC_CHUNK_MAX_CHARS = 1500
SEMANTIC_CHUNK_BREAKPOINT_PERCENTILE = 85

HYBRID_SEMANTIC_WEIGHT = 0.65
HYBRID_KEYWORD_WEIGHT = 0.35
HYBRID_CANDIDATE_COUNT = 12
FINAL_TOP_K =1

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_DB_PATH, exist_ok=True)
os.makedirs(DATASETS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
