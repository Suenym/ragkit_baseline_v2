import json, asyncio, logging, os, time, hashlib, concurrent.futures
from collections import OrderedDict

from retriever.router import detect_answer_type
from retriever.hybrid import hybrid_search
from retriever.parent_page import load_pages, expand_to_pages
from retriever.rerank import llm_like_rerank
from answer.generator import generate_answer
from answer.validate import as_json_obj
from answer.source_check import check_sources

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module level LRU caches.  They live for the duration of the Python process
# so that repeated invocations of the pipeline within the same process can
# benefit from cache hits.
# ---------------------------------------------------------------------------
retrieval_cache: OrderedDict[str, list] = OrderedDict()
rerank_cache: OrderedDict[tuple, float] = OrderedDict()


def _norm_query(q: str) -> str:
    """Normalise query string for use as a cache key."""

    return " ".join(q.lower().strip().split())


def answer_one(q, pages_path, faiss_index, faiss_meta, bm25_index, cfg, stats):
    start_ts = time.perf_counter()
    pages_store = load_pages(pages_path)
    qtext = q["question_text"]
    norm_q = _norm_query(qtext)
    atype = detect_answer_type(q)

    # --- Retrieval ---------------------------------------------------------
    if cfg.get("enable_caching", True):
        if norm_q in retrieval_cache:
            stats["retrieval_hits"] += 1
            cands = retrieval_cache[norm_q]
            retrieval_cache.move_to_end(norm_q)
        else:
            stats["retrieval_misses"] += 1
            cands = hybrid_search(
                qtext,
                faiss_index,
                faiss_meta,
                bm25_index,
                cfg["top_k_dense"],
                cfg["top_k_bm25"],
            )
            retrieval_cache[norm_q] = cands
            max_size = cfg.get("retrieval_cache_size", 512)
            if len(retrieval_cache) > max_size:
                retrieval_cache.popitem(last=False)
    else:
        cands = hybrid_search(
            qtext,
            faiss_index,
            faiss_meta,
            bm25_index,
            cfg["top_k_dense"],
            cfg["top_k_bm25"],
        )

    ctx_pages = expand_to_pages(
        cands, pages_store, max_pages=cfg["answer_max_pages"] * 2
    )

    # --- Re-rank -----------------------------------------------------------
    query_hash = hashlib.sha1(norm_q[:100].encode("utf-8")).hexdigest()
    reranked = None
    if cfg.get("enable_caching", True):
        keys = [(query_hash, p["doc_id"], p["page"]) for p in ctx_pages]
        if all(k in rerank_cache for k in keys):
            stats["rerank_hits"] += 1
            cached = [
                {**p, "rr_score": rerank_cache[k]} for p, k in zip(ctx_pages, keys)
            ]
            cached.sort(key=lambda x: (-x["rr_score"], x["doc_id"], x["page"]))
            reranked = cached[: cfg["answer_max_pages"]]
        else:
            stats["rerank_misses"] += 1
            reranked_all = llm_like_rerank(
                qtext,
                ctx_pages,
                top_m=len(ctx_pages),
                batch_size=cfg.get("rerank_batch_size", 4),
            )
            for r in reranked_all:
                key = (query_hash, r["doc_id"], r["page"])
                rerank_cache[key] = r["rr_score"]
                max_size = cfg.get("rerank_cache_size", 1024)
                if len(rerank_cache) > max_size:
                    rerank_cache.popitem(last=False)
            reranked = reranked_all[: cfg["answer_max_pages"]]
    else:
        reranked = llm_like_rerank(
            qtext,
            ctx_pages,
            top_m=cfg["answer_max_pages"],
            batch_size=cfg.get("rerank_batch_size", 4),
        )

    ans = generate_answer(q, reranked, atype)
    try:
        ans = as_json_obj(ans)
    except Exception:
        ans = {
            "question_id": q["question_id"],
            "answer": "N/A",
            "sources": [
                {"document": reranked[0]["doc_id"], "page": reranked[0]["page"]}
            ],
        }
    ok = check_sources(ans, pages_store)
    if not ok and len(ctx_pages) > cfg["answer_max_pages"]:
        reranked = llm_like_rerank(
            qtext,
            ctx_pages,
            top_m=min(len(ctx_pages), cfg["answer_max_pages"] + 1),
            batch_size=cfg.get("rerank_batch_size", 4),
        )
        ans = generate_answer(q, reranked, atype)
        try:
            ans = as_json_obj(ans)
        except Exception:
            ans = {
                "question_id": q["question_id"],
                "answer": "N/A",
                "sources": [
                    {"document": reranked[0]["doc_id"], "page": reranked[0]["page"]}
                ],
            }
        # Вторичная проверка источников: если не подтвердили, отдаём N/A
        if not check_sources(ans, pages_store):
            ans = {
                "question_id": q["question_id"],
                "answer": "N/A",
                "sources": [
                    {"document": reranked[0]["doc_id"], "page": reranked[0]["page"]}
                ],
            }

    elapsed_ms = (time.perf_counter() - start_ts) * 1000.0
    stats["latencies"].append(elapsed_ms)
    return ans


async def run_batch(
    questions_path,
    pages_path,
    faiss_index,
    faiss_meta,
    bm25_index,
    cfg,
    out_path,
):
    stats = {
        "retrieval_hits": 0,
        "retrieval_misses": 0,
        "rerank_hits": 0,
        "rerank_misses": 0,
        "latencies": [],
    }

    with open(questions_path, "r", encoding="utf-8") as f:
        questions = [json.loads(x) for x in f]

    concurrency = max(1, int(cfg.get("orchestrator_concurrency", 8)))
    sem = asyncio.Semaphore(concurrency)
    timeout_per_q = cfg.get("timeout_per_q_seconds", 90)
    loop = asyncio.get_running_loop()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=concurrency)

    async def process(q):
        async with sem:
            try:
                return await asyncio.wait_for(
                    loop.run_in_executor(
                        executor,
                        answer_one,
                        q,
                        pages_path,
                        faiss_index,
                        faiss_meta,
                        bm25_index,
                        cfg,
                        stats,
                    ),
                    timeout=timeout_per_q,
                )
            except asyncio.TimeoutError:
                pages_store = load_pages(pages_path)
                cands = hybrid_search(
                    q["question_text"],
                    faiss_index,
                    faiss_meta,
                    bm25_index,
                    cfg["top_k_dense"],
                    cfg["top_k_bm25"],
                )
                ctx_pages = expand_to_pages(
                    cands, pages_store, max_pages=cfg["answer_max_pages"] * 2
                )
                if ctx_pages:
                    src = {
                        "document": ctx_pages[0]["doc_id"],
                        "page": ctx_pages[0]["page"],
                    }
                    sources = [src]
                else:
                    sources = []
                stats["latencies"].append(timeout_per_q * 1000.0)
                return {
                    "question_id": q["question_id"],
                    "answer": "N/A",
                    "sources": sources,
                }
            except Exception:
                pages_store = load_pages(pages_path)
                cands = hybrid_search(
                    q["question_text"],
                    faiss_index,
                    faiss_meta,
                    bm25_index,
                    cfg["top_k_dense"],
                    cfg["top_k_bm25"],
                )
                ctx_pages = expand_to_pages(
                    cands, pages_store, max_pages=cfg["answer_max_pages"] * 2
                )
                if ctx_pages:
                    src = {
                        "document": ctx_pages[0]["doc_id"],
                        "page": ctx_pages[0]["page"],
                    }
                    sources = [src]
                else:
                    sources = []
                stats["latencies"].append(0.0)
                return {
                    "question_id": q["question_id"],
                    "answer": "N/A",
                    "sources": sources,
                }

    tasks = [asyncio.create_task(process(q)) for q in questions]

    batch_start = time.perf_counter()
    results = await asyncio.gather(*tasks)
    total_wall_ms = (time.perf_counter() - batch_start) * 1000.0

    executor.shutdown(wait=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    if cfg.get("enable_metrics_logging", False):
        latencies = stats["latencies"]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        lat_sorted = sorted(latencies)
        if lat_sorted:
            p95_index = max(0, int(0.95 * len(lat_sorted)) - 1)
            p95_latency = lat_sorted[p95_index]
        else:
            p95_latency = 0.0

        metrics = {
            "answers_total": len(results),
            "total_wall_ms": round(total_wall_ms, 2),
            "latency_ms_avg": round(avg_latency, 2),
            "latency_ms_p95": round(p95_latency, 2),
            "cache_retrieval_hits": stats["retrieval_hits"],
            "cache_retrieval_misses": stats["retrieval_misses"],
            "cache_rerank_hits": stats["rerank_hits"],
            "cache_rerank_misses": stats["rerank_misses"],
            "concurrency": concurrency,
            "timestamp": time.time(),
        }

        os.makedirs("logs", exist_ok=True)
        with open("logs/metrics.jsonl", "a", encoding="utf-8") as mf:
            mf.write(json.dumps(metrics, ensure_ascii=False) + "\n")

    logger.info(
        "cache_retrieval_hits=%d cache_rerank_hits=%d avg_latency_ms=%.2f",
        stats["retrieval_hits"],
        stats["rerank_hits"],
        (sum(stats["latencies"]) / len(stats["latencies"]))
        if stats["latencies"]
        else 0.0,
    )
    return out_path
