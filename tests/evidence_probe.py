import sys, json, re

ANS = sys.argv[1] if len(sys.argv)>1 else "answers.json"
PAGES = "data/pages.jsonl"

# pages map
pages={}
with open(PAGES,"r",encoding="utf-8") as f:
    for line in f:
        if line.strip():
            o=json.loads(line); pages[(str(o["doc_id"]), int(o["page"]))]=o.get("text","")

def gen_numeric_patterns(val: float):
    # 1234.5 ; 1234,5 ; 1 234,5 ; 1 234.5
    s_dot = f"{val}".replace(",",".")
    # с одной цифрой после запятой, если есть дробная часть
    if "." in s_dot:
        intp, frac = s_dot.split(".",1)
        s1 = f"{intp}.{frac}"
        s2 = f"{intp},{frac}"
        s3 = f"{int(intp):,}.{frac}".replace(","," ")
        s4 = f"{int(intp):,},{frac}".replace(","," ")
        return [s1, s2, s3, s4]
    else:
        s1 = s_dot
        s2 = f"{int(s_dot):,}".replace(","," ")
        return [s1, s2]

def text_has_answer(txt, ans):
    t = txt.lower()
    if isinstance(ans,(int,float)):
        for p in gen_numeric_patterns(float(ans)):
            if p.lower() in t:
                return True, f"found number pattern: {p}"
        return False, "number not found"
    if isinstance(ans,str):
        a = ans.strip().lower()
        if not a: return False, "empty string answer"
        if a in t: return True, f"found substring: {a}"
        return False, "substring not found"
    return False, "unsupported answer type"

ans = json.load(open(ANS,"r",encoding="utf-8"))
for o in ans:
    qid=o["question_id"]; a=o["answer"]
    hits=[]
    for s in o["sources"]:
        key=(str(s["document"]), int(s["page"]))
        txt=pages.get(key,"")
        ok, why = text_has_answer(txt,a)
        hits.append((key, ok, why))
    print(f"Q{qid} -> answer={a!r}")
    for h in hits:
        print("  ",h)
