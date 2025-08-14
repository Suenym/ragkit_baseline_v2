# RAGKit Baseline — CodeX Ready Guide (OCR + Advanced Retrieval)

Этот README описывает **минимальные шаги для запуска проекта в среде CodeX/автопроверки** (без UI, без интернета, с ограничениями по времени). Подходит и для локального запуска.

---

## TL;DR (Linux/Bash)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) Ингест PDF -> pages.jsonl
python scripts/run_all.py ingest --input ./corpus --out ./data/pages.jsonl

# 2) Индексация (FAISS + BM25)
python scripts/run_all.py index --pages ./data/pages.jsonl --out_dense ./data/faiss.index --out_bm25 ./data/bm25.json

# 3) Ответы (JSON)
python scripts/run_all.py answer --pages ./data/pages.jsonl --faiss ./data/faiss.index --bm25 ./data/bm25.json --questions ./data/questions.jsonl --out ./answers.json
```

## Что подразумевается под CodeX
- **Запуск из консоли**, без доступа к интернету и без внешних API.
- **Ограничение по времени** на задание (например, *200 вопросов / ≤3 часа*).
- **Определённая структура ввода/вывода**: корпус документов и файл вопросов на входе; JSON с ответами — на выходе.
- **Единая точка входа**: скрипт `scripts/run_all.py` (или модуль `python -m scripts.run_all …`).

---

## Структура проекта
```
ragkit_baseline/
  answer/               # генерация ответа, валидация, проверка источников
  configs/              # конфиги (YAML)
  data/                 # сюда пишутся промежуточные артефакты и финальные ответы
  index/                # индексация: FAISS + BM25
  ingest/               # парсинг PDF (PyMuPDF), fallback для OCR можно добавить
  orchestrator/         # пайплайн и асинхронный бэкенд
  retriever/            # роутер, гибридный поиск, parent->page, rerank-плейсхолдер
  scripts/
    run_all.py          # ЕДИНАЯ ТОЧКА ВХОДА
  requirements.txt
  README.md / README_CodeX.md
```

---

## Входные и выходные данные

### Вход
- **Корпус PDF/сканов**: `./corpus/*.pdf` (имя файла = `doc_id.pdf`).
- **Вопросы**: `./data/questions.jsonl`, одна строка на вопрос:
```json
{"question_id": 1, "question_text": "Текст вопроса"}
{"question_id": 2, "question_text": "Ещё вопрос"}
```
> *Типы ответов выводятся эвристически; при наличии поля `"expected_answer_type"` оно будет использовано.*

### Выход
- **Ответы**: `./answers.json` — список объектов в **строгой** схеме:
```json
[
  {
    "question_id": "<string|int>",
    "answer": "<number|boolean|string|array[string]|"N/A">",
    "sources": [{"document": "<doc_id>", "page": <int>}]
  }
]
```
- `sources` обязателен; страница должна подтверждать ответ.

---

## Команды (универсальные)

### 1) Ингест
```bash
python scripts/run_all.py ingest --input ./corpus --out ./data/pages.jsonl
```
- Парсит PDF страницами (PyMuPDF), сохраняет текст.
- Можно добавить OCR-fallback (PaddleOCR/Tesseract) в `ingest/parse_pdf.py` (см. TODO в коде).

### 2) Индексация
```bash
python scripts/run_all.py index --pages ./data/pages.jsonl   --out_dense ./data/faiss.index   --out_bm25   ./data/bm25.json
```
- **FAISS**: сейчас используется `dummy_embed` (заглушка); точку замены на SentenceTransformers помечена в коде.
- **BM25**: строится простой сериализуемый индекс.

### 3) Ответы
```bash
python scripts/run_all.py answer --pages ./data/pages.jsonl   --faiss ./data/faiss.index --bm25 ./data/bm25.json   --questions ./data/questions.jsonl --out ./answers.json
```
- Гибридный поиск (dense+BM25) → parent→page → rerank-плейсхолдер → генерация ответа по типу.
- Включены валидатор JSON и базовая проверка источников.

---

## Особенности и ограничения базлайна
- **Без интернета**: никаких внешних вызовов/моделей по сети.
- **Эмбеддинги-заглушка**: `index/build_dense.py::dummy_embed`. Для реального качества замените на SentenceTransformers (см. ниже).
- **LLM-rerank — плейсхолдер**: `retriever/rerank.py` (RapidFuzz). Здесь точка для замен на cross-encoder/LLM.
- **Strict JSON**: `answer/validate.py` гарантирует схему; `answer/source_check.py` — наличие подтверждения на странице.

---

## Быстрый запуск на Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Настроить PYTHONPATH (если нужно)
$env:PYTHONPATH = (Get-Location).Path

python scripts/run_all.py ingest --input .\corpus --out .\data\pages.jsonl
python scripts/run_all.py index  --pages .\data\pages.jsonl --out_dense .\dataaiss.index --out_bm25 .\datam25.json
python scripts/run_all.py answer --pages .\data\pages.jsonl --faiss .\dataaiss.index --bm25 .\datam25.json --questions .\data\questions.jsonl --out .nswers.json
```

> **Важно:** в PowerShell не используйте `\` для переноса строк (это Bash-способ). Переносы в PS — обратной кавычкой `` ` `` или запускайте каждая команда в одну строку.

---

## Конфигурация (configs/default.yaml)
Ключевые параметры (можно менять без правки кода):
```yaml
chunk_size_tokens: 300
chunk_overlap_tokens: 50
top_k_dense: 12
top_k_bm25: 6
rerank_in: 24
rerank_out: 2
answer_max_pages: 2
timeout_per_q_seconds: 90
cot_enabled_types: ["number", "compare", "aggregate"]
number_tolerance_pct: 0.5
language_mode: "auto"
```

---

## Как адаптировать под реальный прогон (более точный Retrieval)

### A) Подключить SentenceTransformers вместо заглушки
1. Установить пакет (если правила CodeX позволяют):
   ```bash
   pip install sentence-transformers
   ```
2. В `index/build_dense.py` заменить функцию `dummy_embed` на реальную, например:
   ```python
   from sentence_transformers import SentenceTransformer
   _model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
   def embed(texts): return _model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
   ```
3. Пересобрать индекс: шаг **2)** (индексация).

### B) Включить rerank (cross-encoder или LLM)
- Точка входа: `retriever/rerank.py::llm_like_rerank`
- Можно заменить на cross-encoder (MiniLM), ранжировать top-N страниц и оставить top-M=2–3.

---

## Формат вопросов и ожидаемых типов
Эвристики по типам (если нет `expected_answer_type`):
- `boolean` — если есть «да/нет/yes/no/является ли…»
- `number` — «сколько/сумма/процент/%/млн/тыс»
- `list[string]` — «перечисл/список/назовите все»
- иначе `string`

> Тип ответа влияет на промпт/шаблон и постобработку (нормализация чисел, проверка отрицаний и т.п.).

---

## Ограничения и SLA
- Целевой бюджет: **200 вопросов ≤ 120 мин wall-clock** (оставляем буфер до лимита).
- Средняя задержка на вопрос: ≤ 30–40с (в базлайне без тяжёлых моделей).

---

## Типичные ошибки и фиксы
- **PowerShell ругается на переносы** — запускайте команды одной строкой (см. выше).
- **`ModuleNotFoundError: ingest`** — добавьте `__init__.py` в папки и/или используйте `python -m scripts.run_all …`.
- **FAISS на Windows**: если установка нестабильна, попробуйте
  ```powershell
  pip install "faiss-cpu==1.8.0.post1"
  ```
  или временно замените FAISS на `hnswlib` (потребуется небольшой патч).
- **Пустой `pages.jsonl`** → индексация упадёт. Убедитесь, что `./corpus` содержит PDF.

---

## Интерфейсы для CodeX (entrypoint)
Единственная точка входа:
```bash
python scripts/run_all.py <ingest|index|answer> [параметры]
```
Альтернатива (если требуется модульный запуск):
```bash
python -m scripts.run_all <ingest|index|answer> [параметры]
```

---

## Где менять логику
- **Парсинг/OCR**: `ingest/parse_pdf.py`
- **Индексы**: `index/build_dense.py`, `index/build_bm25.py`
- **Retrieval**: `retriever/hybrid.py`, `retriever/rerank.py`, `retriever/parent_page.py`
- **Генерация/валидация**: `answer/generator.py`, `answer/validate.py`, `answer/source_check.py`
- **Оркестрация**: `orchestrator/pipeline.py`

---

## Локальный минимальный тест (без реального корпуса)
Создать 2 PDF с текстом (PyMuPDF должен быть установлен):
```bash
python - << 'PY'
import fitz, os
os.makedirs('corpus', exist_ok=True)
d=fitz.open(); p=d.new_page(); p.insert_text((72,72),'Отчёт doc1. Выручка: 1 234,5 USD в 2024. Сертификация: да.'); d.save('corpus/doc1.pdf')
d=fitz.open(); p=d.new_page(); p.insert_text((72,72),'Отчёт doc2. Параметр A = 42. Сертификация: нет.'); d.save('corpus/doc2.pdf')
PY
```
Далее — обычные шаги ingest/index/answer.

---

## Примечание о повторяемости
- Все шаги детерминированы (без случайных сидов).
- Логи времени и статусов можно расширить в `orchestrator/pipeline.py` (таймеры этапов, перцентиль задержек).

Удачи на хакатоне! 🚀
