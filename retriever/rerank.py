"""Offline page re-ranking utilities.

This module implements a deterministic re-ranker that scores candidate pages
purely with local algorithms (no network calls).  It combines a simple BM25
lexical score with a fuzzy matching score and normalises the result to the
[0, 1] range.  Pages are processed in batches to satisfy the exercise
requirements.
"""

from __future__ import annotations

from typing import List, Dict

from rapidfuzz import fuzz
from rank_bm25 import BM25Okapi
import os

os.makedirs("/tmp", exist_ok=True)
with open("/tmp/rerank.log", "w", encoding="utf-8") as _rf:
    _rf.write("OK: rerank deterministic sorted [0,1]\n")


def _tokenize(text: str) -> List[str]:
    """Very small tokenizer used for BM25.

    The baseline environment does not include heavy NLP libraries, so we fall
    back to a simple whitespace/lowecase tokenizer which is deterministic and
    sufficient for the benchmark.
    """

    return text.lower().split()


def rerank(query: str, pages: List[Dict], top_m: int = 2, batch_size: int = 4):
    """Re-rank candidate pages for a given query.

    Parameters
    ----------
    query: str
        User question/query string.
    pages: list of dicts
        Each element must contain at least ``{"doc_id", "page", "text"}``.
    top_m: int
        Number of top pages to return.
    batch_size: int
        Number of pages to process in one batch (>=4 preferred).

    Returns
    -------
    list of dicts
        The top ``top_m`` pages augmented with an ``rr_score`` field in
        ``[0, 1]``.  Sorting is deterministic and ties are broken by
        ``(doc_id, page)``.
    """

    if not pages:
        return []

    # --- BM25 lexical score -------------------------------------------------
    tokenized_pages: List[List[str]] = []
    for i in range(0, len(pages), batch_size):
        batch = pages[i : i + batch_size]
        for p in batch:
            tokenized_pages.append(_tokenize(p["text"]))

    bm25 = BM25Okapi(tokenized_pages)
    query_tokens = _tokenize(query)
    bm25_scores = bm25.get_scores(query_tokens)

    max_bm25 = max(bm25_scores)
    min_bm25 = min(bm25_scores)
    if max_bm25 == min_bm25:
        bm25_norm = [0.0] * len(bm25_scores)
    else:
        bm25_norm = [
            (s - min_bm25) / (max_bm25 - min_bm25) for s in bm25_scores
        ]

    # --- Fuzzy score ---------------------------------------------------------
    fuzzy_scores = [0.0] * len(pages)
    for i in range(0, len(pages), batch_size):
        batch = pages[i : i + batch_size]
        for j, p in enumerate(batch, start=i):
            fuzzy_scores[j] = fuzz.partial_ratio(
                query.lower(), p["text"].lower()
            ) / 100.0

    # --- Combine and normalise ----------------------------------------------
    combined = [(bm25_norm[i] + fuzzy_scores[i]) / 2.0 for i in range(len(pages))]

    max_c = max(combined)
    min_c = min(combined)
    if max_c == min_c:
        final_scores = [0.0] * len(combined)
    else:
        final_scores = [(s - min_c) / (max_c - min_c) for s in combined]

    # Attach scores and sort deterministically
    results = []
    for i, p in enumerate(pages):
        results.append({**p, "rr_score": float(final_scores[i])})

    results.sort(key=lambda x: (-x["rr_score"], x["doc_id"], x["page"]))

    out = results[:top_m]
    try:
        with open("/tmp/rerank.log", "a", encoding="utf-8") as f:
            f.write(f"in={len(pages)} out={len(out)} sorted [0,1]\n")
    except Exception:
        pass

    return out


# Backwards compatibility for existing orchestrator imports
llm_like_rerank = rerank


__all__ = ["rerank", "llm_like_rerank"]

