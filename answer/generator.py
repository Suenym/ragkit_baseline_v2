import re
from .validate import as_json_obj

# Убираем теги вида [PAGE=nn], чтобы не ловить "1" из тега вместо числа из текста
PAGE_TAG_RE = re.compile(r"\[PAGE=\d+\]\s*")
NUM_RE = re.compile(r"([-+]?\d[\d\s.,]*)")

def _strip_tags(text: str) -> str:
    return PAGE_TAG_RE.sub("", text)

def _parse_num(s: str):
    s = s.strip().replace(" ", "")
    s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None

def simple_extract_number(text: str, question: str):
    """
    Если в вопросе подсказка "выруч/доход/revenue/sales", берём ПЕРВОЕ число
    в окне после ключевого слова. Иначе берём первое «нормальное» число в тексте.
    """
    t = _strip_tags(text)
    hints = ["выруч", "доход", "revenue", "sales"]
    qlow = question.lower()
    prioritized = any(h in qlow for h in hints)

    if prioritized:
        for m_kw in re.finditer(r"(выруч\w*|доход\w*|revenue|sales)", t, flags=re.I):
            win = t[m_kw.end(): m_kw.end() + 120]
            m_num = NUM_RE.search(win)
            if m_num:
                n = _parse_num(m_num.group(1))
                if n is not None:
                    return n

    for m in NUM_RE.finditer(t):
        n = _parse_num(m.group(1))
        if n is not None:
            return n
    return None

def _classify_boolean_near_kw(question: str, context_pages: list):
    """
    Ищем Yes/No/Да/Нет в окрестности ключевых слов из вопроса (сертифика.../certif...)
    """
    kw = r"(сертифик\w+|certif\w+)"
    for p in context_pages:
        t = _strip_tags(p["text"])
        for m in re.finditer(kw, t, flags=re.I):
            start = max(0, m.start() - 80)
            end = min(len(t), m.end() + 80)
            window = t[start:end]
            if re.search(r"\b(да|yes|true)\b", window, re.I):
                return "Yes", p
            if re.search(r"\b(нет|no|false)\b", window, re.I):
                return "No", p

    # Фоллбек по всему объединённому тексту контекста
    big = "\n".join(_strip_tags(p["text"]) for p in context_pages)
    if re.search(r"\b(да|yes|true)\b", big, re.I):
        return "Yes", context_pages[0]
    if re.search(r"\b(нет|no|false)\b", big, re.I):
        return "No", context_pages[0]
    return "N/A", context_pages[0]

def generate_answer(question: dict, context_pages: list, answer_type: str):
    """
    Главная точка генерации ответа без LLM:
    - boolean: локатор по ключевому слову и окрестности
    - number: извлечение числа с учётом подсказок из вопроса
    - list[string]: демо-извлечение строк в кавычках
    - string: первая содержательная строка страницы (демо)
    Возвращает валидный JSON-объект под нашу схему.
    """
    qid = question["question_id"]
    qtext = question.get("question_text", "")

    if not context_pages:
        return {"question_id": qid, "answer": "N/A", "sources": []}

    # BOOLEAN
    if answer_type == "boolean":
        ans, src_page = _classify_boolean_near_kw(qtext, context_pages)
        return {
            "question_id": qid,
            "answer": ans,
            "sources": [{"document": src_page["doc_id"], "page": src_page["page"]}],
        }

    # NUMBER
    if answer_type == "number":
        for p in context_pages:
            val = simple_extract_number(p["text"], qtext)
            if val is not None:
                obj = {
                    "question_id": qid,
                    "answer": val,
                    "sources": [{"document": p["doc_id"], "page": p["page"]}],
                }
                # Проверим, что объект валиден по схеме
                return as_json_obj(obj)
        # Если ничего не нашли — N/A с первой страницей как источником-кандидатом
        return as_json_obj({
            "question_id": qid,
            "answer": "N/A",
            "sources": [{"document": context_pages[0]["doc_id"], "page": context_pages[0]["page"]}],
        })

    # LIST[STRING]
    if answer_type == "list[string]":
        items = []
        for p in context_pages:
            t = _strip_tags(p["text"])
            items += re.findall(r"“([^”]{3,50})”|\"([^\"]{3,50})\"", t)
        flat = [a or b for a, b in items]
        flat = list(dict.fromkeys([s.strip() for s in flat if s]))
        obj = {
            "question_id": qid,
            "answer": flat if flat else "N/A",
            "sources": [{"document": p["doc_id"], "page": p["page"]} for p in context_pages],
        }
        return as_json_obj(obj)

    # STRING (default)
    lines = _strip_tags(context_pages[0]["text"]).splitlines()
    line = next((ln.strip() for ln in lines if ln.strip()), "")
    obj = {
        "question_id": qid,
        "answer": line if line else "N/A",
        "sources": [{"document": context_pages[0]["doc_id"], "page": context_pages[0]["page"]}],
    }
    return as_json_obj(obj)
