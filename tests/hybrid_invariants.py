from retriever.hybrid import hybrid_search
import json

DENSE="data/faiss.index"; META="data/faiss.index.meta.json"; BM25="data/bm25.json"
queries = ["выручка 2024", "сертификация подтверждена", "параметр A"]

for q in queries:
    res = hybrid_search(q, DENSE, META, BM25, top_k_dense=8, top_k_bm25=8, alpha=0.7)
    assert all(0.0 <= x["score"] <= 1.0 for x in res), "scores not in [0,1]"
    assert all(res[i]["score"] >= res[i+1]["score"] for i in range(len(res)-1)), "not sorted desc"
    pairs = [(x["doc_id"], x["page"]) for x in res]
    assert len(pairs) == len(set(pairs)), "duplicates (doc_id,page)"
    res2 = hybrid_search(q, DENSE, META, BM25, top_k_dense=8, top_k_bm25=8, alpha=0.7)
    assert res == res2, "non-deterministic output"
    print(f'OK hybrid: "{q}" -> {len(res)} candidates; top1={res[0] if res else None}')

print("OK: hybrid invariants passed.")
