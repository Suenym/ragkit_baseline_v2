"""Lightweight evidence checker with normalisation and logging."""

import json
import os
import re
from typing import Iterable

import yaml


def _load_cfg() -> dict:
    try:
        with open("configs/default.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


CFG = _load_cfg()
BOOL_TOKENS = {t.lower() for t in CFG.get("boolean_tokens", [])}
NUM_TOL_PCT = float(CFG.get("number_tolerance_pct", 0.5))


def _norm_text(s: str) -> str:
    if s is None:
        return ""
    s = s.lower()
    s = s.replace("\u00a0", " ")
    s = s.replace("«", '"').replace("»", '"')
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s.strip())
    return s


def _normalize_number_str(s: str):
    if s is None:
        return None
    s = s.strip()
    s = re.sub(r"\s+", "", s)
    if "," in s and "." in s:
        if s.rfind(".") > s.rfind(","):
            s = s.replace(",", "")
        else:
            s = s.replace(".", "").replace(",", ".")
    elif s.count(",") == 1 and s.count(".") == 0:
        s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None


_NUM_RX = re.compile(
    r"(?<![\w\-/])(?:\d{1,3}(?:[ ,.\u00a0]\d{3})*|\d+)(?:[.,]\d+)?(?![\w\-/])"
)


def _find_numbers(text: str) -> Iterable[float]:
    for m in _NUM_RX.finditer(text):
        v = _normalize_number_str(m.group(0))
        if v is not None:
            yield v


def _log(reason: str) -> None:
    try:
        os.makedirs("/tmp", exist_ok=True)
        with open("/tmp/ev.log", "a", encoding="utf-8") as f:
            f.write(reason + "\n")
    except Exception:
        pass


def check_sources(ans_obj: dict, pages_store: dict) -> bool:
    sources = ans_obj.get("sources", [])
    if not sources:
        _log("no_sources")
        return False

    ans = ans_obj.get("answer")
    norm_ans = _norm_text(ans) if isinstance(ans, str) else ans

    for src in sources:
        key = (str(src["document"]), int(src["page"]))
        page_text = pages_store.get(key, "")
        norm_page = _norm_text(page_text)

        if isinstance(ans, str):
            if ans == "N/A":
                continue
            if norm_ans and norm_ans in norm_page:
                return True
        elif isinstance(ans, (int, float)):
            for v in _find_numbers(page_text):
                if ans == 0:
                    if abs(v) < 1e-9:
                        return True
                else:
                    if (
                        abs(v - ans) / max(abs(ans), 1e-9) * 100.0
                        <= NUM_TOL_PCT
                    ):
                        return True
        elif isinstance(ans, list) and all(isinstance(x, str) for x in ans):
            covered = [
                any(_norm_text(x) in _norm_text(pages_store.get((str(s["document"]), int(s["page"])), "")) for s in sources)
                for x in ans
            ]
            if all(covered):
                return True
            missing = [ans[i] for i, c in enumerate(covered) if not c]
            _log("list_miss:" + json.dumps(missing, ensure_ascii=False))
            return False
        elif isinstance(ans, bool) or (
            isinstance(ans, str) and ans.lower() in {"yes", "no", "true", "false"}
        ):
            if any(t in norm_page for t in BOOL_TOKENS):
                return True

    if isinstance(ans, str):
        _log("string_not_found")
    elif isinstance(ans, (int, float)):
        _log("number_not_found")
    elif isinstance(ans, bool) or (
        isinstance(ans, str) and ans.lower() in {"yes", "no", "true", "false"}
    ):
        _log("boolean_not_found")
    return False


__all__ = ["check_sources"]

