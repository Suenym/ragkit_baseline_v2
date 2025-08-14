import json, faiss, numpy as np, os

IDX = "data/faiss.index"
META = "data/faiss.index.meta.json"

# --- check vectors ---
idx = faiss.read_index(IDX)
n = idx.ntotal
xb = np.vstack([idx.reconstruct(i) for i in range(n)]) if n>0 else np.zeros((0,0), dtype="float32")
print("ntotal:", n, "dim:", (xb.shape[1] if n>0 else None))

if n>0 and xb.size>0:
    norms = np.linalg.norm(xb, axis=1)
    print("norms mean/min/max:", round(norms.mean(),3), round(norms.min(),3), round(norms.max(),3))
    assert (np.abs(norms - 1.0) <= 1e-2).all(), "L2 norms are not ~1.0"

# --- check meta (support old/new shapes) ---
if os.path.exists(META):
    meta = json.load(open(META, "r", encoding="utf-8"))
    # Try new shape
    method = meta.get("method"); dim = meta.get("dim"); normalize = meta.get("normalize")
    # Try nested/old shape
    if method is None and isinstance(meta.get("embedding"), dict):
        emb = meta["embedding"]
        method = method or emb.get("type")
        dim = dim or emb.get("dim")
        # normalize флага может не быть  допустимо; нормы мы уже проверили по факту
        normalize = normalize if normalize is not None else True

    print(f"meta: method={method}, dim={dim}, normalize={normalize}")
    assert method in ("sbert","lsa","tfidf-svd","tfidf_lsa","dummy"), "unexpected method/type in meta"
    assert isinstance(dim, int) and dim >= 1, "bad dim in meta"
else:
    print("warn: meta file not found:", META)

print("OK: FAISS vectors L2-normalized; meta parsed successfully.")
