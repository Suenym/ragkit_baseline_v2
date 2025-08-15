import json, os
from rank_bm25 import BM25Okapi
from tqdm import tqdm

def build_bm25(pages_jsonl, out_path):
    docs, metas = [], []
    with open(pages_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            tokens = rec["text"].lower().split()
            docs.append(tokens)
            metas.append(
                {
                    "doc_id": rec["doc_id"],
                    "page": rec["page"],
                    "chunk_id": rec.get("chunk_id", 0),
                }
            )
    bm25 = BM25Okapi(docs)
    data = {"metas": metas, "docs": docs}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return out_path

def search_bm25(index_path, query, top_k=10):
    with open(index_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    docs = data["docs"]
    metas = data["metas"]
    bm25 = BM25Okapi(docs)
    scores = bm25.get_scores(query.lower().split())
    idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [{**metas[i], "score_bm25": float(scores[i])} for i in idxs]

if __name__ == "__main__":
    import argparse, os
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    build_bm25(args.pages, args.out)
