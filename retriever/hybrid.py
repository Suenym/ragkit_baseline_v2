from index.build_dense import search_faiss
from index.build_bm25 import search_bm25

def hybrid_search(query, dense_index, dense_meta, bm25_index, top_k_dense=12, top_k_bm25=6):
    dense = search_faiss(dense_index, dense_meta, query, top_k_dense)
    sparse = search_bm25(bm25_index, query, top_k_bm25)
    max_d = max([r["score_dense"] for r in dense] + [1e-9])
    max_s = max([r["score_bm25"] for r in sparse] + [1e-9])
    merged = {}
    for r in dense:
        key = (r["doc_id"], r["page"])
        merged[key] = {"doc_id": r["doc_id"], "page": r["page"], "score": (r["score_dense"]/max_d)*0.7}
    for r in sparse:
        key = (r["doc_id"], r["page"])
        merged.setdefault(key, {"doc_id": r["doc_id"], "page": r["page"], "score": 0.0})
        merged[key]["score"] += (r["score_bm25"]/max_s)*0.3
    return sorted(merged.values(), key=lambda x: x["score"], reverse=True)
