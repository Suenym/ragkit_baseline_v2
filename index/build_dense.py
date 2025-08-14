import json, faiss, numpy as np, os, hashlib
from tqdm import tqdm

def dummy_embed(texts):
    vecs = []
    for t in texts:
        h = np.frombuffer(hashlib.sha1(t.encode("utf-8")).digest(), dtype=np.uint8).astype("float32")
        vecs.append(h)
    X = np.vstack(vecs)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    return X

def build_faiss_index(pages_jsonl, out_index_path, out_meta_path):
    texts, meta = [], []
    with open(pages_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            texts.append(rec["text"])
            meta.append({"doc_id": rec["doc_id"], "page": rec["page"]})
    X = dummy_embed(texts)
    dim = X.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(X)
    faiss.write_index(index, out_index_path)
    with open(out_meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return out_index_path, out_meta_path

def search_faiss(index_path, meta_path, query, top_k=10):
    index = faiss.read_index(index_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    qv = dummy_embed([query])
    D, I = index.search(qv, top_k)
    res = []
    for score, idx in zip(D[0], I[0]):
        m = meta[int(idx)]
        res.append({**m, "score_dense": float(score)})
    return res

if __name__ == "__main__":
    import argparse, os
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", required=True)
    ap.add_argument("--out_index", required=True)
    ap.add_argument("--out_meta", required=True)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out_index), exist_ok=True)
    build_faiss_index(args.pages, args.out_index, args.out_meta)
