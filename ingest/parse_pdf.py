import json
import os
import re
from typing import List, Tuple

import fitz  # PyMuPDF
from tqdm import tqdm


def clean_text(txt: str) -> str:
    txt = txt.replace("\r", " ").replace("\t", " ")
    txt = re.sub(r"[ \u00A0]+", " ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()


def parse_pdf(path: str, doc_id: str) -> list[dict]:
    """Парсит PDF в список страниц с текстом."""
    pages: list[dict] = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc):
            text = page.get_text("text") or ""
            text = clean_text(text)
            pages.append(
                {
                    "doc_id": str(doc_id),
                    "page": i + 1,
                    "text": f"[PAGE={i+1}]\n{text}",
                    "lang_guess": "auto",
                }
            )
    return pages


def _collect_pdfs(input_dir: str) -> List[Tuple[str, str]]:
    """Собирает (full_path, relative_path) для всех PDF в дереве input_dir."""
    input_dir = os.path.abspath(input_dir)
    pdfs: List[Tuple[str, str]] = []
    for root, _dirs, files in os.walk(input_dir):
        for fname in files:
            if not fname.lower().endswith(".pdf"):
                continue
            full = os.path.join(root, fname)
            rel = os.path.relpath(full, input_dir)
            pdfs.append((full, rel))
    # Детерминированный порядок по нормализованному пути (OS-агностично)
    pdfs.sort(key=lambda x: x[1].replace(os.sep, "/"))
    return pdfs


def parse_corpus(input_dir: str, out_jsonl: str) -> None:
    """Рекурсивный ingest: doc_id из относительного пути (+ __dupN при коллизиях)."""
    input_dir = os.path.abspath(input_dir)
    out_dir = os.path.dirname(out_jsonl) or "."
    os.makedirs(out_dir, exist_ok=True)

    pdfs = _collect_pdfs(input_dir)
    doc_ids: dict[str, int] = {}  # base_id -> count (для __dupN)

    with open(out_jsonl, "w", encoding="utf-8") as out:
        for full_path, rel_path in tqdm(pdfs, total=len(pdfs)):
            base_id = os.path.splitext(rel_path)[0].replace(os.sep, "__")
            n = doc_ids.get(base_id, 0)
            doc_ids[base_id] = n + 1
            doc_id = base_id if n == 0 else f"{base_id}__dup{n}"

            try:
                pages = parse_pdf(full_path, doc_id)
            except Exception as e:  # не валим весь ingest из-за одного файла
                print(f"warning: failed to parse {rel_path}: {e}")
                continue

            for rec in pages:
                rec.setdefault("meta", {})
                rec["meta"]["original_path"] = rel_path  # для дебага/трассировки
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    parse_corpus(args.input, args.out)
