import json
import os
import re
from typing import List

from .validate import as_json_obj

try:  # Пытаемся импортировать SDK Gemini
    import google.generativeai as genai  # type: ignore

    _HAS_GENAI = True
except Exception:  # pragma: no cover - если пакета нет
    genai = None
    _HAS_GENAI = False

# Убираем теги вида [PAGE=nn], чтобы не ловить "1" из тега вместо числа из текста
PAGE_TAG_RE = re.compile(r"\[PAGE=\d+\]\s*")
NUM_RE = re.compile(r"([-+]?\d[\d\s.,]*)")


# ---------------------------------------------------------------------------
#  LLM helpers
# ---------------------------------------------------------------------------
SYS_PROMPT = (
    "You must answer STRICTLY based on the provided pages. If the answer is not "
    'explicitly present, return "N/A". Output ONLY valid JSON matching the provided schema. Temperature=0.'
)

SCHEMA_PROMPT = (
    "Return ONLY valid JSON matching:\n"
    "{\n"
    '  "question_id": "<string|int>",\n'
    '  "answer": "<number|boolean|string|array[string]|\\"N/A\\">",\n'
    '  "sources": [{ "document": "<string|int>", "page": <int> }]\n'
    "}"
)

PROMPTS = {
    "ru": {
        "number": (
            "{SYS}\n{SCHEMA}\n[КОНТЕКСТ]\n{CTX}\n[ВОПРОС]\n{Q}\n[ИНСТРУКЦИИ]\n"
            "Найди одно число; нормализуй тысячи/миллионы; без валют и знака %; верни только число."
        ),
        "boolean": (
            "{SYS}\n{SCHEMA}\n[КОНТЕКСТ]\n{CTX}\n[ВОПРОС]\n{Q}\n[ИНСТРУКЦИИ]\n"
            'Ответ Yes/No только при явном наличии в тексте; иначе верни "N/A".'
        ),
        "string": (
            "{SYS}\n{SCHEMA}\n[КОНТЕКСТ]\n{CTX}\n[ВОПРОС]\n{Q}\n[ИНСТРУКЦИИ]\n"
            "Верни точную строку без пояснений и дополнений."
        ),
        "list": (
            "{SYS}\n{SCHEMA}\n[КОНТЕКСТ]\n{CTX}\n[ВОПРОС]\n{Q}\n[ИНСТРУКЦИИ]\n"
            "Верни array[string] уникальных элементов, которые явно встречаются в контексте."
        ),
    },
    "en": {
        "number": (
            "{SYS}\n{SCHEMA}\n[CONTEXT]\n{CTX}\n[QUESTION]\n{Q}\n[INSTRUCTIONS]\n"
            "Extract a single number; normalize thousands/millions; no % or currency; return only the number."
        ),
        "boolean": (
            "{SYS}\n{SCHEMA}\n[CONTEXT]\n{CTX}\n[QUESTION]\n{Q}\n[INSTRUCTIONS]\n"
            'Answer Yes/No only if explicitly stated; otherwise return "N/A".'
        ),
        "string": (
            "{SYS}\n{SCHEMA}\n[CONTEXT]\n{CTX}\n[QUESTION]\n{Q}\n[INSTRUCTIONS]\n"
            "Return the exact string with no explanations."
        ),
        "list": (
            "{SYS}\n{SCHEMA}\n[CONTEXT]\n{CTX}\n[QUESTION]\n{Q}\n[INSTRUCTIONS]\n"
            "Return array[string] of unique items that explicitly appear in the context."
        ),
    },
    "kz": {
        "number": (
            "{SYS}\n{SCHEMA}\n[МӘТІН]\n{CTX}\n[СҰРАҚ]\n{Q}\n[НҰСҚАУЛАР]\n"
            "Бір ғана санды қайтар; мың/млн нормализациясы; % және валюта жоқ."
        ),
        "boolean": (
            "{SYS}\n{SCHEMA}\n[МӘТІН]\n{CTX}\n[СҰРАҚ]\n{Q}\n[НҰСҚАУЛАР]\n"
            'Тек анық болса Yes/No; әйтпесе "N/A".'
        ),
        "string": (
            "{SYS}\n{SCHEMA}\n[МӘТІН]\n{CTX}\n[СҰРАҚ]\n{Q}\n[НҰСҚАУЛАР]\n"
            "Дәл жолды қайтар; түсіндірмесіз."
        ),
        "list": (
            "{SYS}\n{SCHEMA}\n[МӘТІН]\n{CTX}\n[СҰРАҚ]\n{Q}\n[НҰСҚАУЛАР]\n"
            "Мәтінде бар бірегей элементтерден array[string] қайтар."
        ),
    },
}


LANG_KZ = re.compile(r"[ӘәҒғҚқҢңӨөҰұҮүҺһІі]")
LANG_RU = re.compile(r"[А-Яа-яЁё]")


def _detect_lang(text: str, order: List[str]) -> str:
    if LANG_KZ.search(text):
        return "kz"
    if LANG_RU.search(text):
        return "ru"
    if re.search(r"[A-Za-z]", text):
        return "en"
    return order[0] if order else "ru"


def _build_ctx(pages: list) -> str:
    parts = []
    for p in pages:
        parts.append(f"[PAGE={p['page']} DOC={p['doc_id']}]\n{p['text']}")
    return "\n".join(parts)


def _try_parse_json(s: str):
    try:
        return json.loads(s)
    except Exception:
        try:
            start = s.find("{")
            end = s.rfind("}") + 1
            if 0 <= start < end:
                return json.loads(s[start:end])
        except Exception:
            return None
    return None


def _gemini_call(prompt: str, cfg: dict):
    if not _HAS_GENAI:
        return None
    key_env = cfg.get("gemini_api_key_env", "GEMINI_API_KEY")
    api_key = os.getenv(key_env)
    if not api_key:
        return None

    try:
        genai.configure(api_key=api_key)
        gen_cfg = {
            "temperature": float(cfg.get("llm_temperature", 0.0)),
            "max_output_tokens": int(cfg.get("llm_max_output_tokens", 512)),
            "response_mime_type": "application/json",
        }
        model = genai.GenerativeModel(cfg.get("llm_model", "gemini-1.5-flash"), generation_config=gen_cfg)
        response = model.generate_content(prompt, request_options={"timeout": int(cfg.get("llm_timeout_s", 60))})
        text = getattr(response, "text", "")
    except Exception:
        return None

    obj = _try_parse_json(text)
    if obj is None:
        return None
    try:
        return as_json_obj(obj)
    except Exception:
        return None


# ---------------------------------------------------------------------------

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

def generate_answer(question: dict, context_pages: list, answer_type: str, cfg: dict | None = None):
    """Главная точка генерации ответа.

    Если в конфиге включён провайдер Gemini и доступен ключ, используем модель.
    При любой ошибке или отсутствии SDK/ключа происходит мягкий фолбэк на
    простые эвристики, присутствовавшие в базовой версии.
    Возвращается валидный JSON-объект под нашу схему.
    """

    cfg = cfg or {}
    qid = question["question_id"]
    qtext = question.get("question_text", "")

    if not context_pages:
        return {"question_id": qid, "answer": "N/A", "sources": []}

    # --- Попытка генерации через Gemini ----------------------------------
    provider = cfg.get("llm_provider", "none")
    if provider == "gemini":
        lang = _detect_lang(qtext, cfg.get("llm_lang_order", ["ru", "en", "kz"]))
        at = "list" if answer_type.startswith("list") else answer_type
        tmpl = PROMPTS.get(lang, {}).get(at)
        if tmpl:
            ctx = _build_ctx(context_pages[: cfg.get("answer_max_pages", 2)])
            prompt = tmpl.format(SYS=SYS_PROMPT, SCHEMA=SCHEMA_PROMPT, CTX=ctx, Q=qtext)
            obj = _gemini_call(prompt, cfg)
            if obj is not None:
                return obj

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
