
import os, json, re
from typing import Dict, Tuple, List, Optional

def _norm_ws(s: str) -> str:
    # Normalize whitespace and common typographic artifacts
    if s is None:
        return ""
    s = s.replace("\u00a0", " ").replace("\u2009", " ").replace("\u202f", " ")
    s = s.replace("«", "\"").replace("»", "\"").replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s.strip())
    return s.lower()

def load_pages_map(pages_path_jsonl="./data/pages.jsonl", pages_path_parquet="./data/pages.parquet") -> Dict[Tuple[str,int], str]:
    # Returns a dict {(doc_id, page): concatenated_text} from pages.jsonl or pages.parquet.
    # For chunked pages, concatenates all chunks belonging to the same (doc_id,page).
    pages: Dict[Tuple[str,int], List[str]] = {}
    if os.path.exists(pages_path_jsonl):
        with open(pages_path_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                doc_id = obj.get("doc_id")
                page = obj.get("page")
                text = obj.get("text") or ""
                if doc_id is None or page is None:
                    continue
                pages.setdefault((str(doc_id), int(page)), []).append(text)
    elif os.path.exists(pages_path_parquet):
        try:
            import pandas as pd
            df = pd.read_parquet(pages_path_parquet)
            for _, row in df.iterrows():
                doc_id = row.get("doc_id")
                page = int(row.get("page"))
                text = row.get("text") or ""
                pages.setdefault((str(doc_id), page), []).append(text)
        except Exception as e:
            print(f"ERROR: Failed to read parquet: {e}")
            return {}
    else:
        print("ERROR: Neither pages.jsonl nor pages.parquet found under ./data/.")
        return {}

    # Concatenate chunks by page
    ret: Dict[Tuple[str,int], str] = {}
    for k, parts in pages.items():
        ret[k] = "\n".join([p for p in parts if isinstance(p, str)])
    return ret

def normalize_number_str(s: str) -> Optional[float]:
    # Converts strings like "1 234,56", "1,234.56", "1234,5", "1.234.567" into float.
    # Returns None if cannot parse.
    if s is None: 
        return None
    s = s.strip()
    # Remove spaces (thousands separators in many locales)
    s = re.sub(r"\s+", "", s)
    # Two main formats: "1,234.56" (US) or "1.234,56" (EU)
    # Heuristic: if both separators exist, last one is decimal separator.
    if "," in s and "." in s:
        last_comma = s.rfind(",")
        last_dot = s.rfind(".")
        if last_dot > last_comma:
            # dot is decimal separator -> remove commas
            s = s.replace(",", "")
        else:
            # comma is decimal separator -> remove dots, replace comma with dot
            s = s.replace(".", "").replace(",", ".")
    else:
        # Only one of them present
        if s.count(",") == 1 and s.count(".") == 0:
            # Comma decimal
            s = s.replace(",", ".")
        # Else assume dot decimal or integer with no separators

    try:
        return float(s)
    except:
        return None

def find_number_mentions(page_text: str) -> List[float]:
    if not page_text:
        return []
    # capture numbers with optional thousand separators and decimal part
    rx = re.compile(r"(?<![\w\-/])(?:\d{1,3}(?:[ ,.\u00a0]\d{3})*|\d+)(?:[.,]\d+)?(?![\w\-/])")
    vals: List[float] = []
    for m in rx.finditer(page_text):
        v = normalize_number_str(m.group(0))
        if v is not None:
            vals.append(v)
    return vals

YES_NO_TOKENS = {"да","нет","иә","жоқ","yes","no"}

def contains_boolean_signal(text: str, answer: str) -> bool:
    if not text:
        return False
    t = _norm_ws(text)
    # True if any token from YES_NO_TOKENS is present
    if isinstance(answer, str) and answer.lower() in {"yes","no"}:
        return any(tok in t for tok in YES_NO_TOKENS)
    # If bool type true/false used, also accept yes/no
    if isinstance(answer, str) and answer.lower() in {"true","false"}:
        return any(tok in t for tok in YES_NO_TOKENS.union({"true","false"}))
    if isinstance(answer, bool):
        return any(tok in t for tok in YES_NO_TOKENS)
    return False
