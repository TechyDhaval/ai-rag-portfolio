"""
src/config.py
Centralised configuration loaded from environment variables (.env file).
All other modules import from here — change settings in one place.

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
    # Set to "openai" or "azure"
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai").lower()

    # ── OpenAI (used when LLM_PROVIDER=openai) ────────────────────────────────
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")

    # ── Azure OpenAI (used when LLM_PROVIDER=azure) ───────────────────────────
    AZURE_OPENAI_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    # e.g. https://my-resource.openai.azure.com/
    AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    # Deployment name created in Azure AI Foundry (can differ from model name)
    AZURE_OPENAI_DEPLOYMENT: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
    AZURE_OPENAI_API_VERSION: str = os.getenv(
        "AZURE_OPENAI_API_VERSION", "2024-12-01-preview"
    )

    # ── Shared LLM settings ────────────────────────────────────────────────────
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))

    # ── Embeddings (local, no API key needed) ─────────────────────────────────
    # "huggingface" (default) → sentence-transformers, fully local
    # "ollama"                → Ollama local server (must be running: `ollama serve`)
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "huggingface").lower()
    # HuggingFace model name  OR  Ollama model name, depending on EMBEDDING_PROVIDER.
    # Default is BAAI/bge-small-en-v1.5 — strictly better than all-MiniLM-L6-v2:
    #   • 512-token limit (vs 256) so long chunks are not truncated
    #   • Higher MTEB retrieval scores at the same speed and size
    # See .env.example for a full catalogue of recommended free models.
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

    # ── Vector store ──────────────────────────────────────────────────────────
    FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH", "faiss_index")

    # ── Text splitting ────────────────────────────────────────────────────────
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

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
                    f"Azure provider requires these env vars to be set: "
                    f"{', '.join(missing)}. "
                    "Copy .env.example → .env and fill in the Azure section."
                )
        else:
            if not self.OPENAI_API_KEY:
                raise EnvironmentError(
                    "OPENAI_API_KEY is not set. "
                    "Copy .env.example → .env and add your key."
                )

        valid_embedding_providers = {"huggingface", "ollama"}
        if self.EMBEDDING_PROVIDER not in valid_embedding_providers:
            raise EnvironmentError(
                f"EMBEDDING_PROVIDER='{self.EMBEDDING_PROVIDER}' is not supported. "
                f"Choose one of: {', '.join(sorted(valid_embedding_providers))}."
            )


config = Config()
