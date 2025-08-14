from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List

from rapidfuzz import fuzz

_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)


def _tokenize(text: str) -> List[str]:
    """Lowercase tokenization keeping only word characters."""
    return _WORD_RE.findall(text.lower())


def rerank(query: str, pages: List[Dict], top_m: int = 2, batch_size: int = 4):
    """Rerank pages for a query using lexical + fuzzy scoring.

    Args:
        query: question text.
        pages: list of dicts with at least ``{"doc_id", "page", "text"}``.
        top_m: number of items to return.
        batch_size: number of pages processed per batch (>=1).

    Returns:
        List of pages with added ``rr_score`` in [0, 1], sorted by score desc.
    """
    if not pages:
        return []

    batch_size = max(1, int(batch_size))
    query_tokens = _tokenize(query)
    doc_tokens = [_tokenize(p.get("text", "")) for p in pages]
    N = len(doc_tokens)

    # Document frequencies for BM25
    df = Counter()
    for toks in doc_tokens:
        df.update(set(toks))
    avgdl = sum(len(t) for t in doc_tokens) / N if N else 0.0
    k1, b = 1.5, 0.75

    # Pre-compute idf for query tokens
    idf = {}
    for tok in query_tokens:
        n = df.get(tok, 0)
        idf[tok] = math.log(1 + (N - n + 0.5) / (n + 0.5))

    bm25_raw: List[float] = []
    fuzzy_scores: List[float] = []

    # Process pages in batches to emulate batched inference
    for start in range(0, N, batch_size):
        pages_batch = pages[start : start + batch_size]
        tokens_batch = doc_tokens[start : start + batch_size]
        for p, toks in zip(pages_batch, tokens_batch):
            counter = Counter(toks)
            dl = len(toks)
            score = 0.0
            for tok in query_tokens:
                if tok in counter:
                    tf = counter[tok]
                    denom = tf + k1 * (1 - b + b * dl / avgdl)
                    score += idf[tok] * tf * (k1 + 1) / denom
            bm25_raw.append(score)
            fscore = fuzz.partial_ratio(query.lower(), p.get("text", "").lower()) / 100.0
            fuzzy_scores.append(fscore)

    # Normalize BM25 scores to [0,1]
    if bm25_raw:
        min_b, max_b = min(bm25_raw), max(bm25_raw)
        if max_b > min_b:
            bm25_norm = [(s - min_b) / (max_b - min_b) for s in bm25_raw]
        else:
            bm25_norm = [0.0 for _ in bm25_raw]
    else:
        bm25_norm = []

    scored = []
    for p, bnorm, fscore in zip(pages, bm25_norm, fuzzy_scores):
        rr = 0.7 * bnorm + 0.3 * fscore
        rr = max(0.0, min(1.0, rr))
        scored.append({**p, "rr_score": rr})

    scored.sort(key=lambda x: (-x["rr_score"], str(x.get("doc_id", "")), x.get("page", 0)))
    return scored[:top_m]


# Backwards compatibility for the pipeline
llm_like_rerank = rerank
