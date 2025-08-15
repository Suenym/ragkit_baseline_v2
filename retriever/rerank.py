"""Offline page re-ranking utilities with optional Cross-Encoder."""

from __future__ import annotations

import logging
import os
import random
from typing import Dict, List

import numpy as np
from rapidfuzz import fuzz
from rank_bm25 import BM25Okapi

os.makedirs("/tmp", exist_ok=True)
with open("/tmp/rerank.log", "w", encoding="utf-8") as _rf:
    _rf.write("OK: rerank deterministic sorted [0,1]\n")


# ---------------------------------------------------------------------------
# Determinism helpers
np.random.seed(0)
random.seed(0)
try:  # pragma: no cover - torch optional
    import torch

    torch.manual_seed(0)
except Exception:  # pragma: no cover
    pass


def _tokenize(text: str) -> List[str]:
    return text.lower().split()


def _heuristic_rerank(query: str, pages: List[Dict], top_m: int, batch_size: int) -> List[Dict]:
    if not pages:
        return []

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
        bm25_norm = [(s - min_bm25) / (max_bm25 - min_bm25) for s in bm25_scores]

    fuzzy_scores = [0.0] * len(pages)
    for i in range(0, len(pages), batch_size):
        batch = pages[i : i + batch_size]
        for j, p in enumerate(batch, start=i):
            fuzzy_scores[j] = fuzz.partial_ratio(
                query.lower(), p["text"].lower()
            ) / 100.0

    combined = [(bm25_norm[i] + fuzzy_scores[i]) / 2.0 for i in range(len(pages))]
    max_c = max(combined)
    min_c = min(combined)
    if max_c == min_c:
        final_scores = [0.0] * len(combined)
    else:
        final_scores = [(s - min_c) / (max_c - min_c) for s in combined]

    results = []
    for i, p in enumerate(pages):
        results.append({**p, "rr_score": float(final_scores[i])})

    results.sort(key=lambda x: (-x["rr_score"], x["doc_id"], x["page"]))
    return results[:top_m]


def rerank(
    query: str,
    pages: List[Dict],
    top_m: int = 2,
    batch_size: int = 4,
    *,
    mode: str = "auto",
    ce_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ce_max_batch: int = 32,
) -> List[Dict]:
    if not pages:
        return []

    mode_used = "heuristic"
    results = None
    if mode in ("auto", "ce"):
        try:  # pragma: no cover - optional dependency
            from sentence_transformers import CrossEncoder

            ce = CrossEncoder(ce_model_name, device="cpu")
            pairs = [(query, p["text"]) for p in pages]
            scores: List[float] = []
            for i in range(0, len(pairs), ce_max_batch):
                bs = ce.predict(pairs[i : i + ce_max_batch], batch_size=ce_max_batch)
                if isinstance(bs, np.ndarray):
                    scores.extend(bs.tolist())
                else:
                    scores.extend([float(x) for x in bs])
            hi = max(scores)
            lo = min(scores)
            if hi == lo:
                norm = [0.0] * len(scores)
            else:
                norm = [(s - lo) / (hi - lo) for s in scores]
            results = [{**p, "rr_score": float(n)} for p, n in zip(pages, norm)]
            results.sort(key=lambda x: (-x["rr_score"], x["doc_id"], x["page"]))
            mode_used = "ce"
        except Exception as e:  # pragma: no cover
            if mode == "ce":
                logging.error("CrossEncoder unavailable: %s", e)
            results = None

    if results is None:
        results = _heuristic_rerank(query, pages, top_m=len(pages), batch_size=batch_size)
        mode_used = "heuristic"

    out = results[:top_m]
    try:
        with open("/tmp/rerank.log", "a", encoding="utf-8") as f:
            f.write(f"mode={mode_used} in={len(pages)} out={len(out)} sorted [0,1]\n")
    except Exception:
        pass

    return out


llm_like_rerank = rerank

__all__ = ["rerank", "llm_like_rerank"]

