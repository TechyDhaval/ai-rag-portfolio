"""
src/experiment.py
Save, load, and compare experiment configurations and results.

An "experiment" captures every tunable parameter together with evaluation
scores for a set of questions. This lets you systematically compare:
  - embedding model A vs. B
  - chunk_size 500 vs. 1000 vs. 2000
  - temperature 0.0 vs. 0.3 vs. 0.7
  - reranker enabled vs. disabled
  ... with reproducible, side-by-side score comparisons.

Learning note — experiment design:
  The key to finding optimal RAG parameters is changing ONE variable at a
  time while keeping all others constant. This module enforces that by
  recording ALL parameters for every run, so you can verify what changed.

File layout on disk:
  experiments/
    2024-03-12_14-30-00_strict_bge-small.json    # auto-named
    2024-03-12_14-35-00_balanced_bge-base.json
  test_sets/
    basic_questions.json                          # user-created

Test set format (JSON):
  [
    {
      "question": "What is the main argument?",
      "expected_answer": "The author argues that..."   // optional
    },
    ...
  ]
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from src.config import config


def _experiments_dir() -> Path:
    """Return (and create) the experiments output directory."""
    d = Path(config.EXPERIMENTS_DIR)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _test_sets_dir() -> Path:
    """Return (and create) the test_sets directory."""
    d = Path(config.TEST_SETS_DIR)
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_test_set(name: str) -> List[dict]:
    """
    Load a test set JSON file by name (with or without .json extension).

    Returns a list of dicts, each with at least a 'question' key.
    """
    if not name.endswith(".json"):
        name += ".json"
    path = _test_sets_dir() / name

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Test set must be a JSON array, got {type(data).__name__}")
    for i, item in enumerate(data):
        if "question" not in item:
            raise ValueError(f"Test set item {i} missing 'question' key")

    return data


def list_test_sets() -> List[str]:
    """Return names of available test set files (without .json extension)."""
    d = _test_sets_dir()
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.json"))


def save_experiment(params: dict, results: List[dict], label: str = "") -> str:
    """
    Save experiment configuration and results to a timestamped JSON file.

    Args:
        params:  Dict of all parameter settings used in this experiment.
                 Should include: embedding_model, chunk_size, chunk_overlap,
                 search_type, top_k, score_threshold, mmr_lambda, llm_model,
                 temperature, prompt_key, reranker_enabled, reranker_model,
                 reranker_top_n.
        results: List of per-question result dicts, each containing:
                 question, answer, context_relevance, faithfulness,
                 answer_relevance, retrieval_time_s, llm_time_s, total_time_s.
        label:   Optional short label for this experiment.

    Returns:
        Filename (stem) of the saved experiment.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")

    # Build a safe, descriptive filename
    prompt = params.get("prompt_key", "unknown")
    embed_short = params.get("embedding_model", "unknown").split("/")[-1]
    name_parts = [timestamp, prompt, embed_short]
    if label:
        safe_label = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)
        name_parts.append(safe_label)

    filename = "_".join(name_parts) + ".json"
    path = _experiments_dir() / filename

    # Compute aggregate scores
    n = len(results) or 1
    aggregates = {
        "avg_context_relevance": round(sum(r.get("context_relevance", 0) for r in results) / n, 3),
        "avg_faithfulness": round(sum(r.get("faithfulness", 0) for r in results) / n, 3),
        "avg_answer_relevance": round(sum(r.get("answer_relevance", 0) for r in results) / n, 3),
        "avg_retrieval_time_s": round(sum(r.get("retrieval_time_s", 0) for r in results) / n, 3),
        "avg_llm_time_s": round(sum(r.get("llm_time_s", 0) for r in results) / n, 3),
        "avg_total_time_s": round(sum(r.get("total_time_s", 0) for r in results) / n, 3),
        "num_questions": len(results),
    }

    experiment = {
        "timestamp": timestamp,
        "label": label,
        "params": params,
        "aggregates": aggregates,
        "results": results,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(experiment, f, indent=2, default=str)

    return path.stem


def load_experiment(name: str) -> dict:
    """
    Load a saved experiment by filename (with or without .json extension).

    Returns the full experiment dict with keys:
      timestamp, label, params, aggregates, results.
    """
    if not name.endswith(".json"):
        name += ".json"
    path = _experiments_dir() / name

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_experiments() -> List[dict]:
    """
    List all saved experiments with summary info.

    Returns a list of dicts with keys: name, timestamp, label, params,
    aggregates (but NOT the full results, to keep the listing lightweight).
    """
    d = _experiments_dir()
    if not d.exists():
        return []

    experiments = []
    for path in sorted(d.glob("*.json"), reverse=True):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            experiments.append({
                "name": path.stem,
                "timestamp": data.get("timestamp", ""),
                "label": data.get("label", ""),
                "params": data.get("params", {}),
                "aggregates": data.get("aggregates", {}),
            })
        except (json.JSONDecodeError, KeyError):
            continue

    return experiments


def delete_experiment(name: str) -> bool:
    """Delete a saved experiment file. Returns True if deleted."""
    if not name.endswith(".json"):
        name += ".json"
    path = _experiments_dir() / name
    if path.exists():
        path.unlink()
        return True
    return False


def compare_experiments(names: List[str]) -> List[dict]:
    """
    Load multiple experiments and return their aggregate scores side by side.

    Args:
        names: List of experiment filenames (stems).

    Returns:
        List of dicts, each with: name, label, params (subset), and all
        aggregate scores. Ready for tabular display.
    """
    rows = []
    for name in names:
        exp = load_experiment(name)
        params = exp.get("params", {})
        rows.append({
            "name": name,
            "label": exp.get("label", ""),
            "embedding_model": params.get("embedding_model", "?"),
            "chunk_size": params.get("chunk_size", "?"),
            "search_type": params.get("search_type", "?"),
            "top_k": params.get("top_k", "?"),
            "temperature": params.get("temperature", "?"),
            "prompt_key": params.get("prompt_key", "?"),
            "reranker": params.get("reranker_enabled", False),
            **exp.get("aggregates", {}),
        })
    return rows
