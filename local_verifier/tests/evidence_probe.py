
import sys, json, argparse
from typing import Dict, Tuple, List, Any
from utils_io import load_pages_map, _norm_ws, normalize_number_str, find_number_mentions, YES_NO_TOKENS, contains_boolean_signal

def str_present_on_any_source(ans: str, pages_map, sources) -> bool:
    if not isinstance(ans, str) or not ans.strip():
        return False
    needle = _norm_ws(ans)
    for s in sources:
        key = (str(s["document"]), int(s["page"]))
        text = pages_map.get(key, "")
        if needle and _norm_ws(text).find(needle) >= 0:
            return True
    return False

def number_present_on_any_source(ans_num: float, pages_map, sources, tol_pct=0.5) -> bool:
    for s in sources:
        key = (str(s["document"]), int(s["page"]))
        text = pages_map.get(key, "")
        nums = find_number_mentions(text)
        for v in nums:
            if ans_num == 0:
                if abs(v) < 1e-9:
                    return True
            else:
                if abs(v - ans_num) / max(abs(ans_num), 1e-9) * 100.0 <= tol_pct:
                    return True
    return False

def list_covered(ans_list: List[str], pages_map, sources) -> List[bool]:
    covered = []
    for item in ans_list:
        ok = str_present_on_any_source(item, pages_map, sources)
        covered.append(ok)
    return covered

def boolean_supported(ans_bool, pages_map, sources) -> bool:
    val = str(ans_bool)
    for s in sources:
        key = (str(s["document"]), int(s["page"]))
        text = pages_map.get(key, "")
        if contains_boolean_signal(text, val):
            return True
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("answers", help="answers.json path")
    ap.add_argument("--pages", default="./data/pages.jsonl", help="pages.jsonl (or pages.parquet if jsonl missing)")
    ap.add_argument("--tol_pct", type=float, default=0.5, help="numeric tolerance in percent")
    args = ap.parse_args()

    pages_map = load_pages_map(args.pages, "./data/pages.parquet")

    with open(args.answers, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = data if isinstance(data, list) else [data]

    total = len(items)
    ok = 0
    failures: List[str] = []

    for i, obj in enumerate(items):
        qid = obj.get("question_id")
        ans = obj.get("answer")
        srcs = obj.get("sources") or []

        if ans == "N/A":
            ok += 1
            continue

        if not srcs:
            failures.append(f"[{i}|{qid}] No sources provided")
            continue

        verdict = False

        if isinstance(ans, str):
            verdict = str_present_on_any_source(ans, pages_map, srcs)
        elif isinstance(ans, (int, float)):
            verdict = number_present_on_any_source(float(ans), pages_map, srcs, tol_pct=args.tol_pct)
        elif isinstance(ans, list) and all(isinstance(x, str) for x in ans):
            covered = list_covered(ans, pages_map, srcs)
            verdict = all(covered)
            if not verdict:
                missing = [ans[j] for j, c in enumerate(covered) if not c]
                failures.append(f"[{i}|{qid}] list elements not covered: {missing}")
        elif isinstance(ans, bool) or (isinstance(ans, str) and ans.lower() in {"yes","no","true","false"}):
            verdict = boolean_supported(ans, pages_map, srcs)
        else:
            failures.append(f"[{i}|{qid}] Unsupported answer type: {type(ans).__name__} ({ans})")

        if verdict:
            ok += 1
        else:
            if isinstance(ans, str):
                failures.append(f"[{i}|{qid}] string not found on any cited page")
            elif isinstance(ans, (int,float)):
                failures.append(f"[{i}|{qid}] number not found within tolerance on cited pages")
            elif isinstance(ans, (bool,str)):
                failures.append(f"[{i}|{qid}] boolean signal not found on cited pages")

    print(f"EVIDENCE PROBE: {ok}/{total} answers pass")
    if failures:
        print("Details (first 50):")
        for e in failures[:50]:
            print(" -", e)
        sys.exit(1)
    else:
        print("OK: All answers have supporting evidence or are 'N/A'")

if __name__ == "__main__":
    main()
