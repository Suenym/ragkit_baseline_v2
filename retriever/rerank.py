from rapidfuzz import fuzz

def llm_like_rerank(query: str, pages: list, top_m=2):
    scored = []
    for p in pages:
        s = fuzz.partial_ratio(query.lower(), p["text"].lower())/100.0
        scored.append({**p, "rr_score": s})
    scored.sort(key=lambda x: x["rr_score"], reverse=True)
    return scored[:top_m]
