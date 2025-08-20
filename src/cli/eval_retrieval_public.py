import argparse
import csv
import json
import os
import re
from typing import List

import pandas as pd

from retrieval.search import Searcher
from validate.match_utils import relaxed_answer_match

try:
    from retrieval.llm_rerank import GeminiReranker
except Exception:  # pragma: no cover
    GeminiReranker = None


def _normalize_answer(text: str) -> str:
    text = str(text)
    text = (
        text.replace("«", '"')
        .replace("»", '"')
        .replace("“", '"')
        .replace("”", '"')
        .replace("\u00A0", " ")
    )
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_qa(path: str) -> pd.DataFrame:
    with open(path, encoding="utf-8-sig") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    q_col = None
    for col in ["question", "query", "full_question"]:
        if col in df.columns:
            q_col = col
            if col != "question":
                print(f"ℹ️ Колонка с вопросом: '{col}'")
                df = df.rename(columns={col: "question"})
            break
    if q_col is None:
        raise ValueError("No question column in answers file")
    cols = ["question_id", "question", "answer"]
    if "relevant_chunks" in df.columns:
        cols.append("relevant_chunks")
    df = df[cols]
    df["answer_raw"] = df["answer"]
    df["answer"] = df["answer"].apply(_normalize_answer)
    return df


def evaluate(args):
    answers_path = os.path.join(args.cache, "answers_public.json")
    if not os.path.exists(answers_path):
        answers_path = "answers_public.json"
    df = load_qa(answers_path)
    searcher = Searcher(args.index, hybrid=args.hybrid)
    overfetch = max(args.rerank_topn, 5 * args.k)
    llm = None
    if args.llm_rerank and GeminiReranker is not None:
        try:
            llm = GeminiReranker(model=args.llm_model)
        except Exception:
            llm = None
    ranks: List[int | None] = []
    report_rows = []
    csv_rows = []
    for _, row in df.iterrows():
        qid = row["question_id"]
        question = row["question"]
        hits = searcher.search(question, k=args.k, overfetch=overfetch, hybrid=args.hybrid)
        hits = searcher.rerank(question, hits, top_n=overfetch)
        if llm is not None:
            hits = llm.rerank(question, hits, top_k=min(overfetch, 5 * args.k))
        final_hits = hits[: args.k]
        patterns = relaxed_answer_match.compile_patterns(row["answer"])
        matched_rank = None
        matched_pattern = ""
        for rank, h in enumerate(final_hits, start=1):
            ok, pat = relaxed_answer_match.any_match(h.get("text", ""), patterns)
            if ok:
                matched_rank = rank
                matched_pattern = pat
                break
        ranks.append(matched_rank)
        if matched_rank is not None:
            top_hit = final_hits[matched_rank - 1]
            report_rows.append(
                {
                    "qid": qid,
                    "rank": matched_rank,
                    "doc_name": top_hit.get("doc_name", ""),
                    "page_number": top_hit.get("page_number", ""),
                    "pattern": matched_pattern,
                    "score": top_hit.get("rerank_score", top_hit.get("score", 0)),
                }
            )
        else:
            report_rows.append(
                {
                    "qid": qid,
                    "rank": "MISS",
                    "doc_name": "",
                    "page_number": "",
                    "pattern": "",
                    "score": "",
                }
            )
        for rank, h in enumerate(final_hits, start=1):
            csv_rows.append(
                {
                    "question_id": qid,
                    "question": question,
                    "answer_raw": row["answer_raw"],
                    "answer_type": type(row["answer_raw"]).__name__,
                    "rank": rank,
                    "score": h.get("score", 0),
                    "rerank_score": h.get("rerank_score", h.get("score", 0)),
                    "doc_name": h.get("doc_name", ""),
                    "page_number": h.get("page_number", ""),
                    "preview": h.get("text", "")[:200].replace("\n", " "),
                    "matched_pattern": matched_pattern if rank == matched_rank else "",
                }
            )
    total = len(ranks)
    def recall_at(k: int) -> float:
        return sum(1 for r in ranks if r is not None and r <= k) / total
    r1 = recall_at(1)
    r3 = recall_at(3)
    r5 = recall_at(5)
    r10 = recall_at(10)
    mrr = sum((1.0 / r) for r in ranks if r is not None and r <= 10) / total
    report_path = os.path.join("reports", "retrieval_public.md")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Retrieval Evaluation\n\n")
        f.write(
            f"R@1 {r1:.3f} | R@3 {r3:.3f} | R@5 {r5:.3f} | R@10 {r10:.3f} | MRR@10 {mrr:.3f}\n\n"
        )
        f.write(
            "| qid | top_hit_rank | doc_name | page_number | matched_pattern | score |\n"
        )
        f.write("|---|---|---|---|---|---|\n")
        for row in report_rows:
            f.write(
                f"| {row['qid']} | {row['rank']} | {row['doc_name']} | {row['page_number']} | {row['pattern']} | {row['score']} |\n"
            )
        if r1 < 0.33:
            f.write("\n## Auto Error Analysis\n")
            misses = [i for i, r in enumerate(ranks) if r is None][:5]
            for mi in misses:
                q = df.iloc[mi]
                cand_hits = searcher.search(q["question"], k=1, overfetch=1, hybrid=args.hybrid)
                cand = cand_hits[0] if cand_hits else {}
                preview = cand.get("text", "")[:200].replace("\n", " ")
                pattern = relaxed_answer_match.compile_patterns(q["answer"])[0].pattern
                f.write(
                    f"- qid {q['question_id']} best {cand.get('doc_name','')}/{cand.get('page_number','')} "
                    f"score {cand.get('score',0):.3f} pattern `{pattern}` preview {preview}\n"
                )
    if args.dump_topk:
        os.makedirs(os.path.dirname(args.dump_topk), exist_ok=True)
        with open(args.dump_topk, "w", encoding="utf-8", newline="") as csvfile:
            fieldnames = [
                "question_id",
                "question",
                "answer_raw",
                "answer_type",
                "rank",
                "score",
                "rerank_score",
                "doc_name",
                "page_number",
                "preview",
                "matched_pattern",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
    msg = (
        f"Saved report to {report_path} • R@1 {r1:.3f} | R@3 {r3:.3f} | R@5 {r5:.3f} | "
        f"R@10 {r10:.3f} | MRR@10 {mrr:.3f}"
    )
    if args.dump_topk:
        msg += f" • Top-K CSV: {args.dump_topk}"
    print(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rerank-model", dest="rerank_model", default=None)
    parser.add_argument("--rerank-topn", dest="rerank_topn", type=int, default=100)
    parser.add_argument("--dump-topk", default=None)
    parser.add_argument("--hybrid", action="store_true")
    parser.add_argument("--llm-rerank", action="store_true")
    parser.add_argument("--llm-model", default="gemini-1.5-flash")
    args = parser.parse_args()
    evaluate(args)
