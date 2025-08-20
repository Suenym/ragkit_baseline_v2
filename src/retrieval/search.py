import os
import re
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz


class Searcher:
    """Simple dense/BM25 search with optional hybrid RRF fusion."""

    def __init__(self, index_dir: str, hybrid: bool = False):
        self.index_dir = index_dir
        self.hybrid = hybrid
        chunks_path = os.path.join(index_dir, "chunks.parquet")
        self.chunks = pd.read_parquet(chunks_path)
        texts = self.chunks["text"].astype(str).tolist()
        self._tokenized = [self._tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(self._tokenized)

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower(), flags=re.UNICODE)

    def _dense_search(self, query: str, top_n: int) -> List[Dict[str, Any]]:
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)
        idx = np.argsort(scores)[::-1][:top_n]
        hits = []
        for rank, i in enumerate(idx, start=1):
            row = self.chunks.iloc[i]
            hits.append(
                {
                    "id": int(i),
                    "doc_name": row.get("doc_name") or row.get("document"),
                    "page_number": int(row.get("page_number") or row.get("page", 0)),
                    "text": row.get("text", ""),
                    "score": float(scores[i]),
                }
            )
        return hits

    def _bm25_search(self, query: str, top_n: int) -> List[Dict[str, Any]]:
        # In this baseline implementation dense search already uses BM25; reuse
        return self._dense_search(query, top_n)

    def search(self, query: str, k: int, overfetch: int, hybrid: bool | None = None) -> List[Dict[str, Any]]:
        if hybrid is None:
            hybrid = self.hybrid
        top_n = max(k, overfetch)
        dense_hits = self._dense_search(query, top_n)
        if hybrid:
            bm25_hits = self._bm25_search(query, top_n)
            fused: Dict[int, Dict[str, Any]] = {}
            for rank, h in enumerate(dense_hits, start=1):
                score = 1.0 / (60 + rank)
                fused.setdefault(h["id"], {**h, "score": 0.0})
                fused[h["id"]]["score"] += score
            for rank, h in enumerate(bm25_hits, start=1):
                score = 1.0 / (60 + rank)
                fused.setdefault(h["id"], {**h, "score": 0.0})
                fused[h["id"]]["score"] += score
            hits = sorted(fused.values(), key=lambda x: x["score"], reverse=True)[:top_n]
        else:
            hits = dense_hits[:top_n]
        return hits

    def rerank(self, query: str, hits: List[Dict[str, Any]], top_n: int) -> List[Dict[str, Any]]:
        subset = hits[:top_n]
        for h in subset:
            h["rerank_score"] = fuzz.partial_ratio(query.lower(), h["text"].lower()) / 100.0
        subset.sort(key=lambda x: x["rerank_score"], reverse=True)
        return subset + hits[top_n:]


