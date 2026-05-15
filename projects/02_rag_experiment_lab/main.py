"""
main.py — CLI entry point for the RAG Experiment Lab

Usage:
    python main.py              Run the Streamlit web app (recommended)
    python main.py --cli        Run a quick CLI test to verify setup

For the full interactive experience, use: streamlit run app.py
"""

import subprocess
import sys


def main():
    if "--cli" in sys.argv:
        _run_cli_test()
    else:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])


def _run_cli_test():
    """Quick smoke test: validates config and prints available models."""
    from src.config import config
    from src.embeddings import EMBEDDING_MODELS

    print("RAG Experiment Lab — CLI Smoke Test")
    print("=" * 50)

    try:
        config.validate()
        print(f"✓ LLM provider: {config.LLM_PROVIDER}")
        print(f"✓ LLM model:    {config.LLM_MODEL}")
    except EnvironmentError as e:
        print(f"✗ Config error: {e}")
        sys.exit(1)

    print(f"\nAvailable embedding models ({len(EMBEDDING_MODELS)}):")
    for name, info in EMBEDDING_MODELS.items():
        print(f"  {name:<40} {info['dim']}d  {info['size_mb']}MB")

    print(f"\n✓ Default embedding: {config.EMBEDDING_MODEL}")
    print(f"✓ Default search:    {config.SEARCH_TYPE} (top_k={config.TOP_K})")
    print(f"✓ Reranker:          {'enabled' if config.RERANKER_ENABLED else 'disabled'}")
    print("\nSetup OK. Run `streamlit run app.py` for the full UI.")


if __name__ == "__main__":
    main()
