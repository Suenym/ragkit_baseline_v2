"""Utilities for combining dense and BM25 retrieval results."""

from index.build_dense import search_faiss, _load_meta
from index.build_bm25 import search_bm25
import json, os, threading

_META_LOGGED = False
_LOG_LOCK = threading.Lock()
os.makedirs("/tmp", exist_ok=True)
open("/tmp/hybrid.log", "w", encoding="utf-8").close()


def _normalize(records, score_key):
    """Min-max normalisation of scores in ``records``.

    Returns a mapping keyed by ``(doc_id, page, chunk_id)`` with normalised scores in
    ``[0, 1]``.  When all scores are equal (including the degenerate case of all
    zeros) the normalised value for every record is ``0.0`` to avoid division by
    zero and keep deterministic behaviour.
    """

    if not records:
        return {}

    scores = [r[score_key] for r in records]
    lo = min(scores)
    hi = max(scores)

    if hi - lo < 1e-9:
        norm = [0.0] * len(records)
    else:
        denom = hi - lo
        norm = [(s - lo) / denom for s in scores]

    normalised = {}
    for r, n in zip(records, norm):
        key = (r["doc_id"], r["page"], r.get("chunk_id", 0))
        normalised[key] = n
    return normalised


def hybrid_search(
    query,
    dense_index,
    dense_meta,
    bm25_index,
    top_k_dense: int = 12,
    top_k_bm25: int = 6,
    *,
    alpha: float = 0.7,
    k_out: int = 24,
):
    """Hybrid retrieval combining dense and BM25 results.

    Parameters are kept backward compatible with the previous implementation.
    ``alpha`` controls the balance between dense and sparse scores. ``k_out`` is
    the number of final candidates to return (defaults to 24 as per config).
    """

    global _META_LOGGED
    dense = search_faiss(dense_index, dense_meta, query, top_k_dense)
    sparse = search_bm25(bm25_index, query, top_k_bm25)

    dense_norm = _normalize(dense, "score_dense")
    sparse_norm = _normalize(sparse, "score_bm25")

    has_dense = bool(dense_norm)
    has_sparse = bool(sparse_norm)

    merged = {}

    for r in dense:
        key = (r["doc_id"], r["page"], r.get("chunk_id", 0))
        merged.setdefault(
            key,
            {
                "doc_id": r["doc_id"],
                "page": r["page"],
                "chunk_id": r.get("chunk_id", 0),
                "dense": 0.0,
                "bm25": 0.0,
            },
        )
        merged[key]["dense"] = dense_norm.get(key, 0.0)

    for r in sparse:
        key = (r["doc_id"], r["page"], r.get("chunk_id", 0))
        merged.setdefault(
            key,
            {
                "doc_id": r["doc_id"],
                "page": r["page"],
                "chunk_id": r.get("chunk_id", 0),
                "dense": 0.0,
                "bm25": 0.0,
            },
        )
        merged[key]["bm25"] = sparse_norm.get(key, 0.0)

    chunk_results = []
    for key, vals in merged.items():
        d = vals["dense"]
        b = vals["bm25"]
        if has_dense and has_sparse:
            score = alpha * d + (1 - alpha) * b
        elif has_dense:
            score = d
        else:
            score = b
        chunk_results.append(
            {
                "doc_id": vals["doc_id"],
                "page": vals["page"],
                "chunk_id": vals.get("chunk_id", 0),
                "score": score,
            }
        )

    page_best = {}
    for r in chunk_results:
        key = (r["doc_id"], r["page"])
        if key not in page_best or r["score"] > page_best[key]["score"]:
            page_best[key] = r

    results = list(page_best.values())
    results.sort(key=lambda x: (-x["score"], x["doc_id"], x["page"]))
    if k_out is not None:
        results = results[:k_out]

    try:
        os.makedirs("/tmp", exist_ok=True)
        with _LOG_LOCK, open("/tmp/hybrid.log", "a", encoding="utf-8") as f:
            if not _META_LOGGED:
                _, emb = _load_meta(dense_meta)
                f.write(
                    json.dumps(
                        {
                            "method": emb.get("method"),
                            "model": emb.get("model"),
                            "normalize": emb.get("normalize"),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                _META_LOGGED = True
            top = results[0] if results else {}
            f.write(
                "OK hybrid: "
                + json.dumps(
                    {
                        "N_dense": len(dense),
                        "N_bm25": len(sparse),
                        "merged": len(results),
                        "top1": {
                            "doc_id": top.get("doc_id"),
                            "page": top.get("page"),
                            "score": round(top.get("score", 0.0), 6),
                        },
                    }
                )
                + "\n"
            )
    except Exception:
        pass

    return results
