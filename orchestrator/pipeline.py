import json, asyncio
from retriever.router import detect_answer_type
from retriever.hybrid import hybrid_search
from retriever.parent_page import load_pages, expand_to_pages
from retriever.rerank import llm_like_rerank
from answer.generator import generate_answer
from answer.validate import as_json_obj
from answer.source_check import check_sources

async def answer_one(q, pages_path, faiss_index, faiss_meta, bm25_index, cfg):
    pages_store = load_pages(pages_path)
    qtext = q["question_text"]
    atype = detect_answer_type(q)
    cands = hybrid_search(qtext, faiss_index, faiss_meta, bm25_index, cfg["top_k_dense"], cfg["top_k_bm25"])
    ctx_pages = expand_to_pages(cands, pages_store, max_pages=cfg["answer_max_pages"]*2)
    reranked = llm_like_rerank(qtext, ctx_pages, top_m=cfg["answer_max_pages"])
    ans = generate_answer(q, reranked, atype)
    try:
        ans = as_json_obj(ans)
    except Exception:
        ans = {"question_id": q["question_id"], "answer": "N/A", "sources": [{"document": reranked[0]["doc_id"], "page": reranked[0]["page"]}]}
    ok = check_sources(ans, pages_store)
    if not ok and len(ctx_pages) > cfg["answer_max_pages"]:
        reranked = llm_like_rerank(qtext, ctx_pages, top_m=min(len(ctx_pages), cfg["answer_max_pages"]+1))
        ans = generate_answer(q, reranked, atype)
        try:
            ans = as_json_obj(ans)
        except Exception:
            ans = {"question_id": q["question_id"], "answer": "N/A", "sources": [{"document": reranked[0]["doc_id"], "page": reranked[0]["page"]}]}
    return ans

async def run_batch(questions_path, pages_path, faiss_index, faiss_meta, bm25_index, cfg, out_path):
    tasks = []
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = [json.loads(x) for x in f]
    for q in questions:
        tasks.append(answer_one(q, pages_path, faiss_index, faiss_meta, bm25_index, cfg))
    results = await asyncio.gather(*tasks)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return out_path
