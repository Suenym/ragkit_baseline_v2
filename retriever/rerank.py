"""Offline page re-ranking utilities with optional CrossEncoder support."""

from __future__ import annotations

from typing import Dict, List, Tuple

import os
import random
import numpy as np
from rapidfuzz import fuzz
from rank_bm25 import BM25Okapi

os.makedirs("/tmp", exist_ok=True)
open("/tmp/rerank.log", "w", encoding="utf-8").close()

random.seed(0)
np.random.seed(0)
try:  # pragma: no cover - torch is optional
    import torch

    torch.manual_seed(0)
except Exception:  # pragma: no cover
    pass

_CE_MODEL = None
_CE_FAILED = False


def _tokenize(text: str) -> List[str]:
    return text.lower().split()


def _load_ce(model_name: str) -> Tuple[object, bool]:
    global _CE_MODEL, _CE_FAILED
    if _CE_MODEL is None and not _CE_FAILED:
        try:  # pragma: no cover - optional dependency
            from sentence_transformers import CrossEncoder

            _CE_MODEL = CrossEncoder(model_name, device="cpu")
            _CE_MODEL.eval()
        except Exception:
            _CE_FAILED = True
    return _CE_MODEL, not _CE_FAILED


def rerank(
    query: str,
    pages: List[Dict],
    top_m: int = 2,
    batch_size: int = 4,
    *,
    mode: str = "auto",
    ce_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ce_max_batch: int = 32,
):
    if not pages:
        return []

    ce_model = None
    mode_used = "heuristic"
    if mode in {"auto", "ce"}:
        ce_model, ok = _load_ce(ce_model_name)
        if ok and ce_model is not None:
            mode_used = "ce"
        elif mode == "ce":
            try:
                import logging

                logging.error("CrossEncoder unavailable, falling back to heuristic")
            except Exception:
                pass

    if mode_used == "ce" and ce_model is not None:
        pairs = [(query, p["text"]) for p in pages]
        scores: List[float] = []
        for i in range(0, len(pairs), ce_max_batch):
            part = pairs[i : i + ce_max_batch]
            preds = ce_model.predict(part, convert_to_numpy=True, show_progress_bar=False)
            scores.extend(preds.tolist())

        arr = np.array(scores, dtype=float)
        if arr.size:
            lo = arr.min()
            hi = arr.max()
            if hi - lo < 1e-9:
                arr = np.zeros_like(arr)
            else:
                arr = (arr - lo) / (hi - lo)
        final_scores = arr.tolist()

        results = []
        for p, s in zip(pages, final_scores):
            results.append({**p, "rr_score": float(s)})

    else:
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
        combined = [
            (bm25_norm[i] + fuzzy_scores[i]) / 2.0 for i in range(len(pages))
        ]

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
    out = results[:top_m]

    try:
        with open("/tmp/rerank.log", "a", encoding="utf-8") as f:
            f.write(
                f"mode={mode_used} in={len(pages)} out={len(out)} sorted [0,1]\n"
            )
    except Exception:
        pass

    return out


# Backwards compatibility for existing orchestrator imports
llm_like_rerank = rerank

__all__ = ["rerank", "llm_like_rerank"]

