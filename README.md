# RAGKit Baseline (OCR + Advanced Retrieval)

Минимальный прототип пайплайна под хакатон AI RAG Challenge.
Фокус: скорость, строгий JSON, ссылки на страницы, гибридный ретривал.

## Быстрый старт
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) подготовить корпус (PDF/сканы в ./corpus)
python scripts/run_all.py ingest --input ./corpus --out ./data/pages.jsonl

# 2) индексация
python scripts/run_all.py index --pages ./data/pages.jsonl --out_dense ./data/faiss.index --out_bm25 ./data/bm25.json

# 3) ответы на вопросы (пример ввода questions.jsonl)
python scripts/run_all.py answer --pages ./data/pages.jsonl --faiss ./data/faiss.index --bm25 ./data/bm25.json --questions ./data/questions.jsonl --out ./answers.json
```
