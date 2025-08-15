import json


def load_pages(jsonl_path):
    pages = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            key = (rec["doc_id"], rec["page"])
            pages.setdefault(key, []).append(rec.get("text", ""))
    return {k: "\n".join(v) for k, v in pages.items()}

def expand_to_pages(candidates, pages_store, max_pages=2):
    uniq, seen = [], set()
    for r in candidates:
        key = (str(r["doc_id"]), int(r["page"]))
        if key in seen:
            continue
        if key in pages_store:
            uniq.append({"doc_id": key[0], "page": key[1], "text": pages_store[key], "score": r.get("score", 0.0)})
            seen.add(key)
        if len(uniq) >= max_pages:
            break
    return uniq
