import re
def detect_doc_id(question: str):
    m = re.search(r"(doc|report|file)[ _-]?(\d+)", question, flags=re.I)
    if m:
        return f"{m.group(1)}{m.group(2)}"
    return None

def detect_answer_type(meta: dict):
    t = meta.get("expected_answer_type")
    if t: 
        return t
    q = meta.get("question_text","").lower()
    if any(k in q for k in ["да", "нет", "yes", "no", "является ли"]):
        return "boolean"
    if any(k in q for k in ["сколько", "сумма", "процент", "%", "млн", "тыс"]):
        return "number"
    if any(k in q for k in ["перечисл", "список", "назовите все"]):
        return "list[string]"
    return "string"
