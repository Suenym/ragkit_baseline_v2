import asyncio, json, pathlib
from orchestrator.pipeline import run_batch
from scripts.run_all import load_cfg

async def main():
    cfg = load_cfg("configs/default.yaml")
    pathlib.Path("logs/metrics.jsonl").unlink(missing_ok=True)
    await run_batch("data/questions.jsonl","data/pages.jsonl","data/faiss.index","data/faiss.index.meta.json","data/bm25.json",cfg,"answers_cache1.json")
    await run_batch("data/questions.jsonl","data/pages.jsonl","data/faiss.index","data/faiss.index.meta.json","data/bm25.json",cfg,"answers_cache2.json")

    lines = pathlib.Path("logs/metrics.jsonl").read_text(encoding="utf-8").splitlines()
    two = [json.loads(x) for x in lines[-2:]] if len(lines)>=2 else []
    for i,m in enumerate(two,1):
        print(f"[run {i}]", {k:m.get(k) for k in ["answers_total","total_wall_ms","cache_retrieval_hits","cache_rerank_hits","concurrency"]})

if __name__ == "__main__":
    asyncio.run(main())
