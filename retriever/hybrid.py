"""Utilities for combining dense and BM25 retrieval results."""

from index.build_dense import search_faiss
from index.build_bm25 import search_bm25


def _normalize(records, score_key):
    """Min-max normalisation of scores in ``records``.

    Returns a mapping keyed by ``(doc_id, page)`` with normalised scores in
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
        key = (r["doc_id"], r["page"])
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

    dense = search_faiss(dense_index, dense_meta, query, top_k_dense)
    sparse = search_bm25(bm25_index, query, top_k_bm25)

    dense_norm = _normalize(dense, "score_dense")
    sparse_norm = _normalize(sparse, "score_bm25")

    has_dense = bool(dense_norm)
    has_sparse = bool(sparse_norm)

    merged = {}

    for r in dense:
        key = (r["doc_id"], r["page"])
        merged.setdefault(
            key, {"doc_id": r["doc_id"], "page": r["page"], "dense": 0.0, "bm25": 0.0}
        )
        merged[key]["dense"] = dense_norm[key]

    for r in sparse:
        key = (r["doc_id"], r["page"])
        merged.setdefault(
            key, {"doc_id": r["doc_id"], "page": r["page"], "dense": 0.0, "bm25": 0.0}
        )
        merged[key]["bm25"] = sparse_norm[key]

    results = []
    for key, vals in merged.items():
        d = vals["dense"]
        b = vals["bm25"]
        if has_dense and has_sparse:
            score = alpha * d + (1 - alpha) * b
        elif has_dense:
            score = d
        else:
            score = b
        results.append({"doc_id": vals["doc_id"], "page": vals["page"], "score": score})

    results.sort(key=lambda x: (-x["score"], x["doc_id"], x["page"]))
    if k_out is not None:
        results = results[:k_out]
    return results
