import asyncio
from orchestrator.pipeline import run_batch
from scripts.run_all import load_cfg
cfg = load_cfg('configs/default.yaml')
async def main():
    await run_batch('data/questions.jsonl','data/pages.jsonl','data/faiss.index','data/faiss.index.meta.json','data/bm25.json',cfg,'answers1.json')
    await run_batch('data/questions.jsonl','data/pages.jsonl','data/faiss.index','data/faiss.index.meta.json','data/bm25.json',cfg,'answers2.json')
asyncio.run(main())
