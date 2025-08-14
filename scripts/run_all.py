import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import argparse, yaml, asyncio, os
from ingest.parse_pdf import parse_corpus
from index.build_dense import build_faiss_index
from index.build_bm25 import build_bm25
from orchestrator.pipeline import run_batch

def load_cfg(path="configs/default.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("ingest")
    p1.add_argument("--input", required=True)
    p1.add_argument("--out", required=True)

    p2 = sub.add_parser("index")
    p2.add_argument("--pages", required=True)
    p2.add_argument("--out_dense", required=True)
    p2.add_argument("--out_dense_meta", default=None)
    p2.add_argument("--out_bm25", required=True)

    p3 = sub.add_parser("answer")
    p3.add_argument("--pages", required=True)
    p3.add_argument("--faiss", required=True)
    p3.add_argument("--faiss_meta", default=None)
    p3.add_argument("--bm25", required=True)
    p3.add_argument("--questions", required=True)
    p3.add_argument("--out", required=True)
    p3.add_argument("--config", default="configs/default.yaml")

    args = ap.parse_args()

    if args.cmd == "ingest":
        parse_corpus(args.input, args.out)
    elif args.cmd == "index":
        dense_meta = args.out_dense + ".meta.json" if args.out_dense_meta is None else args.out_dense_meta
        build_faiss_index(args.pages, args.out_dense, dense_meta)
        build_bm25(args.pages, args.out_bm25)
    elif args.cmd == "answer":
        cfg = load_cfg(args.config)
        asyncio.run(run_batch(args.questions, args.pages, args.faiss, args.faiss_meta or (args.faiss + ".meta.json"), args.bm25, cfg, args.out))

if __name__ == "__main__":
    main()
