"""Shared configuration for Lab 18."""

import os
from dotenv import load_dotenv

# override=True để giá trị trong .env thắng các biến môi trường OS cũ
# (tránh trường hợp OPENAI_API_KEY cũ trong shell ghi đè key trong .env).
load_dotenv(override=True)

# --- API Keys / LLM Gateway ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "") or None
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Đẩy ngược vào os.environ để các thư viện đọc env trực tiếp (openai SDK,
# langchain/RAGAS) cũng dùng đúng gateway thay vì api.openai.com mặc định.
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
if OPENAI_BASE_URL:
    os.environ["OPENAI_BASE_URL"] = OPENAI_BASE_URL
    os.environ["OPENAI_API_BASE"] = OPENAI_BASE_URL  # langchain-openai cũ dùng tên này

# --- Qdrant ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "lab18_production"
NAIVE_COLLECTION = "lab18_naive"

# --- Embedding ---
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# --- Chunking ---
HIERARCHICAL_PARENT_SIZE = 2048
HIERARCHICAL_CHILD_SIZE = 256
SEMANTIC_THRESHOLD = 0.85

# --- Search ---
BM25_TOP_K = 20
DENSE_TOP_K = 20
HYBRID_TOP_K = 20
RERANK_TOP_K = 3

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set.json")
