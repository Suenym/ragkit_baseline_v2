import re

def number_in_text(num, text):
    if num is None:
        return False
    s = str(num)
    s1 = s.replace(".", ",")
    return (s in text) or (s1 in text)

def check_sources(ans_obj: dict, pages_store: dict) -> bool:
    sources = ans_obj.get("sources", [])
    if not sources:
        return False
    a = ans_obj.get("answer")
    for src in sources:
        key = (str(src["document"]), int(src["page"]))
        page_text = pages_store.get(key, "")
        if isinstance(a, (int, float)):
            if number_in_text(a, page_text):
                return True
        elif isinstance(a, str):
            if a == "N/A":
                continue
            if a.lower() in page_text.lower():
                return True
        elif isinstance(a, list):
            ok = any((isinstance(x, str) and x.lower() in page_text.lower()) for x in a)
            if ok: 
                return True
        elif isinstance(a, bool):
            if (a and re.search(r"\b(да|yes|является)\b", page_text, re.I)) or ((not a) and re.search(r"\b(нет|no|не является)\b", page_text, re.I)):
                return True
    return False
