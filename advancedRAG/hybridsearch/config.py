import os

from dotenv import load_dotenv


load_dotenv()


MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    raise ValueError(
        "MISTRAL_API_KEY is not set."
    )


CHROMA_PATH = "chroma_db"

EMBEDDING_MODEL = "mistral-embed"

LLM_MODEL = "mistral-small-latest"

RERANKER_MODEL = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

VECTOR_TOP_K = 10
BM25_TOP_K = 10

HYBRID_CANDIDATE_K = 30
FINAL_TOP_K = 5

RRF_K = 60

LLM_TEMPERATURE = 0.2