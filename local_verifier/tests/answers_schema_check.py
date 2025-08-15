
import sys, json
from typing import Any, Dict, List

def type_ok(v):
    if v == "N/A":
        return True
    if isinstance(v, (int, float, bool, str)):
        return True
    if isinstance(v, list) and all(isinstance(x, str) for x in v):
        # unique elements for list[string]
        return len(v) == len(set(v))
    return False

def validate_one(obj: Dict[str, Any], idx: int) -> List[str]:
    errs: List[str] = []
    if not isinstance(obj, dict):
        return [f"[{idx}] not an object"]
    if "question_id" not in obj:
        errs.append(f"[{idx}] missing question_id")
    if "answer" not in obj:
        errs.append(f"[{idx}] missing answer")
    if "sources" not in obj:
        errs.append(f"[{idx}] missing sources")

    # question_id type
    qid = obj.get("question_id")
    if not isinstance(qid, (str,int)):
        errs.append(f"[{idx}] question_id must be string|int, got {type(qid).__name__}")

    # answer type
    ans = obj.get("answer")
    if not type_ok(ans):
        errs.append(f"[{idx}] answer has invalid type/value: {repr(ans)}")

    # sources
    srcs = obj.get("sources")
    if not isinstance(srcs, list) or len(srcs) == 0:
        errs.append(f"[{idx}] sources must be a non-empty list")
    else:
        for j, s in enumerate(srcs):
            if not isinstance(s, dict):
                errs.append(f"[{idx}] sources[{j}] must be object")
                continue
            if "document" not in s or "page" not in s:
                errs.append(f"[{idx}] sources[{j}] missing document/page")
                continue
            if not isinstance(s["document"], (str,int)):
                errs.append(f"[{idx}] sources[{j}].document must be string|int")
            if not isinstance(s["page"], int):
                errs.append(f"[{idx}] sources[{j}].page must be int")

    return errs

def main():
    if len(sys.argv) < 2:
        print("Usage: python tests/answers_schema_check.py answers.json")
        sys.exit(2)
    path = sys.argv[1]
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data if isinstance(data, list) else [data]
    all_errs = []
    for i, obj in enumerate(items):
        all_errs.extend(validate_one(obj, i))

    if all_errs:
        print("SCHEMA CHECK: FAIL")
        for e in all_errs[:200]:
            print(" -", e)
        sys.exit(1)
    else:
        print("SCHEMA CHECK: OK – all items valid")

if __name__ == "__main__":
    main()
