from retriever.rerank import rerank
from retriever.hybrid import hybrid_search
from retriever.parent_page import load_pages, expand_to_pages

Q = "подтверждение сертификации"
DENSE="data/faiss.index"; META="data/faiss.index.meta.json"; BM25="data/bm25.json"; PAGES="data/pages.jsonl"

cands = hybrid_search(Q, DENSE, META, BM25, top_k_dense=8, top_k_bm25=8)
store = load_pages(PAGES)
ctx = expand_to_pages(cands, store, max_pages=8)

a = rerank(Q, ctx, top_m=3, batch_size=2)
b = rerank(Q, ctx, top_m=3, batch_size=8)

assert a == b, "batch_size changed ranking (should be invariant)"
assert all(0.0 <= p["rr_score"] <= 1.0 for p in a), "rr_score out of [0,1]"
assert all(a[i]["rr_score"] >= a[i+1]["rr_score"] for i in range(len(a)-1)), "rerank not sorted"
print("OK: rerank deterministic, sorted, [0,1], batch-invariant; top:", a)
