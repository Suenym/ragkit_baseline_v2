import json, os, fitz, re
from tqdm import tqdm

def clean_text(txt: str) -> str:
    txt = txt.replace("\r", " ").replace("\t", " ")
    txt = re.sub(r"[ \u00A0]+", " ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()

def parse_pdf(path, doc_id):
    pages = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc):
            text = page.get_text("text") or ""
            text = clean_text(text)
            pages.append({
                "doc_id": str(doc_id),
                "page": i+1,
                "text": f"[PAGE={i+1}]\n{text}",
                "lang_guess": "auto"
            })
    return pages

def parse_corpus(input_dir: str, out_jsonl: str):
    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as out:
        for fname in tqdm(sorted(os.walk(input_dir))):
            if not fname.lower().endswith(".pdf"): 
                continue
            path = os.path.join(input_dir, fname)
            doc_id = os.path.splitext(fname)[0]
            for rec in parse_pdf(path, doc_id):
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    parse_corpus(args.input, args.out)
