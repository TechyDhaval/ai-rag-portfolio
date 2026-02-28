"""
main.py — CLI entry point for the PDF RAG Assistant
=====================================================
Usage:
    # Ingest one or more PDFs and start chatting
    python main.py --pdf docs/report.pdf
    python main.py --pdf docs/a.pdf docs/b.pdf docs/c.pdf

    # Ingest every PDF in a directory
    python main.py --pdf-dir docs/

    # Chat using an already-ingested index
    python main.py

    # Ingest only (no interactive session)
    python main.py --pdf docs/report.pdf --ingest-only

    # Adjust retrieval breadth
    python main.py --top-k 6
"""

import argparse
import sys
from pathlib import Path
from typing import List

from src.document_loader import load_and_split_pdfs
from src.vector_store import build_vector_store, load_vector_store
from src.rag_chain import build_rag_chain

_SESSION_ID = "cli"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PDF RAG Assistant — ask questions about any PDF document(s)."
    )
    parser.add_argument(
        "--pdf",
        type=str,
        nargs="+",
        default=None,
        metavar="PATH",
        help="Path(s) to PDF file(s) to ingest (e.g. docs/a.pdf docs/b.pdf).",
    )
    parser.add_argument(
        "--pdf-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Directory containing PDF files to ingest (all *.pdf files).",
    )
    parser.add_argument(
        "--ingest-only",
        action="store_true",
        help="Only ingest the PDF(s) and build the index; do not start chat.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=6,
        help="Number of document chunks to retrieve per query (default: 6).",
    )
    return parser.parse_args()


def _collect_pdf_paths(args: argparse.Namespace) -> List[str]:
    """
    Resolve PDF paths from --pdf and/or --pdf-dir arguments.

    Returns:
        Deduplicated list of PDF path strings.

    Raises:
        SystemExit: If --pdf-dir is not a directory or no PDFs are found.
    """
    paths: List[str] = []

    if args.pdf:
        paths.extend(args.pdf)

    if args.pdf_dir:
        dir_path = Path(args.pdf_dir)
        if not dir_path.is_dir():
            print(f"[Error] --pdf-dir '{args.pdf_dir}' is not a directory.")
            sys.exit(1)
        found = sorted(str(p) for p in dir_path.glob("*.pdf"))
        if not found:
            print(f"[Error] No PDF files found in '{args.pdf_dir}'.")
            sys.exit(1)
        print(f"[Main] Found {len(found)} PDF(s) in '{args.pdf_dir}': "
              f"{', '.join(Path(p).name for p in found)}")
        paths.extend(found)

    # Deduplicate while preserving order
    seen = set()
    unique: List[str] = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def ingest(pdf_paths: List[str]):
    """Load PDF(s) → split → embed → save FAISS index."""
    chunks = load_and_split_pdfs(pdf_paths)
    return build_vector_store(chunks)


def chat_loop(vector_store, top_k: int) -> None:
    """Interactive Q&A loop with conversation memory."""
    chain = build_rag_chain(vector_store, top_k=top_k)

    print("\n" + "=" * 60)
    print("  PDF RAG Assistant — ready!")
    print("  The assistant remembers your conversation history.")
    print("  Type your question and press Enter.")
    print("  Type 'exit' or press Ctrl+C to quit.")
    print("=" * 60 + "\n")

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break

        print("\nAssistant: ", end="", flush=True)
        try:
            result = chain.invoke(
                {"input": question},
                config={"configurable": {"session_id": _SESSION_ID}},
            )
            print(result["answer"])
        except Exception as exc:
            print(f"[Error] {exc}")
        print()


def main() -> None:
    args = parse_args()

    # ── Step 1: Get or build the vector store ─────────────────────────────────
    pdf_paths = _collect_pdf_paths(args)

    if pdf_paths:
        vector_store = ingest(pdf_paths)
    else:
        try:
            vector_store = load_vector_store()
        except FileNotFoundError as exc:
            print(f"[Error] {exc}")
            print("Hint: pass --pdf <path> or --pdf-dir <dir> to ingest documents first.")
            sys.exit(1)

    if args.ingest_only:
        print("[Done] Ingestion complete. Run without --ingest-only to chat.")
        return

    # ── Step 2: Start the chat loop ────────────────────────────────────────────
    chat_loop(vector_store, top_k=args.top_k)


if __name__ == "__main__":
    main()
