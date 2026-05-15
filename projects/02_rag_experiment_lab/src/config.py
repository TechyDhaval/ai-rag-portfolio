"""
src/config.py
Centralised configuration loaded from environment variables (.env file).
All other modules import from here — change settings in one place.

This project supports runtime overrides from the Streamlit UI. The Config
class holds the defaults; the UI passes explicit values to each module so
you can experiment without restarting the app.

Supported LLM providers (LLM_PROVIDER env var):
  openai  — OpenAI API directly (default)
  azure   — Azure OpenAI Service via Azure AI Foundry

Supported embedding providers (EMBEDDING_PROVIDER env var):
  huggingface — sentence-transformers, runs 100% locally (default)
  ollama      — local models served by Ollama daemon (must be running)
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # ── Provider switch ────────────────────────────────────────────────────────
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai").lower()

    # ── OpenAI (used when LLM_PROVIDER=openai) ────────────────────────────────
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")

    # ── Azure OpenAI (used when LLM_PROVIDER=azure) ───────────────────────────
    AZURE_OPENAI_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    AZURE_OPENAI_DEPLOYMENT: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
    AZURE_OPENAI_API_VERSION: str = os.getenv(
        "AZURE_OPENAI_API_VERSION", "2024-12-01-preview"
    )

    # ── Shared LLM settings ────────────────────────────────────────────────────
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))

    # ── Embeddings ─────────────────────────────────────────────────────────────
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "huggingface").lower()
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

    # ── Vector store ──────────────────────────────────────────────────────────
    FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH", "faiss_index")

    # ── Text splitting ────────────────────────────────────────────────────────
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # ── Retrieval defaults ────────────────────────────────────────────────────
    TOP_K: int = int(os.getenv("TOP_K", "6"))
    SEARCH_TYPE: str = os.getenv("SEARCH_TYPE", "mmr")  # mmr | similarity | similarity_score_threshold
    SCORE_THRESHOLD: float = float(os.getenv("SCORE_THRESHOLD", "0.4"))
    MMR_LAMBDA: float = float(os.getenv("MMR_LAMBDA", "0.5"))

    # ── Reranker ──────────────────────────────────────────────────────────────
    RERANKER_ENABLED: bool = os.getenv("RERANKER_ENABLED", "false").lower() == "true"
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    RERANKER_TOP_N: int = int(os.getenv("RERANKER_TOP_N", "4"))

    # ── Experiments ───────────────────────────────────────────────────────────
    EXPERIMENTS_DIR: str = os.getenv("EXPERIMENTS_DIR", "experiments")
    TEST_SETS_DIR: str = os.getenv("TEST_SETS_DIR", "test_sets")

    def validate(self) -> None:
        """Raise early if required keys are missing for the active provider."""
        if self.LLM_PROVIDER == "azure":
            missing = [
                name
                for name, val in [
                    ("AZURE_OPENAI_API_KEY", self.AZURE_OPENAI_API_KEY),
                    ("AZURE_OPENAI_ENDPOINT", self.AZURE_OPENAI_ENDPOINT),
                    ("AZURE_OPENAI_DEPLOYMENT", self.AZURE_OPENAI_DEPLOYMENT),
                ]
                if not val
            ]
            if missing:
                raise EnvironmentError(
                    f"Azure provider requires these env vars: {', '.join(missing)}. "
                    "Copy .env.example → .env and fill in the Azure section."
                )
        else:
            if not self.OPENAI_API_KEY:
                raise EnvironmentError(
                    "OPENAI_API_KEY is not set. "
                    "Copy .env.example → .env and add your API key."
                )


config = Config()
