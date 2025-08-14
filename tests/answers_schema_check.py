import sys, json

ANS = sys.argv[1] if len(sys.argv) > 1 else "answers.json"
PAGES = "data/pages.jsonl"

pages = {}
with open(PAGES, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        obj = json.loads(line)
        pages[(str(obj.get("doc_id")), int(obj.get("page")))] = obj.get("text","")

ans = json.load(open(ANS, "r", encoding="utf-8"))
assert isinstance(ans, list) and len(ans) > 0, "answers: must be non-empty list"

ok = 0
for o in ans:
    assert {"question_id","answer","sources"} <= o.keys(), "missing keys"
    assert isinstance(o["sources"], list) and len(o["sources"]) >= 1, "sources empty"
    if any((str(s.get("document")), int(s.get("page"))) in pages for s in o["sources"]):
        ok += 1

print(f"OK: {ok}/{len(ans)} answers reference an existing (document,page)")
