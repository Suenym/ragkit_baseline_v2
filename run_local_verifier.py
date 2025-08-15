#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_local_verifier.py
Запускает набор локальных проверок из папки local_verifier/tests и печатает сводку.
Предполагается запуск из корня репозитория (где лежит scripts/run_all.py).
"""

import os
import sys
import argparse
import subprocess
from shutil import which

PY = sys.executable or "python"

TESTS = [
    ("answers_schema_check", [PY, "local_verifier/tests/answers_schema_check.py", "answers.json"]),
    ("evidence_probe",       [PY, "local_verifier/tests/evidence_probe.py", "answers.json", "--pages", "./data/pages.jsonl"]),
    ("dense_norm_check",     [PY, "local_verifier/tests/dense_norm_check.py", "./data/faiss.index"]),
    ("hybrid_invariants",    [PY, "local_verifier/tests/hybrid_invariants.py"]),
    ("rerank_invariants",    [PY, "local_verifier/tests/rerank_invariants.py"]),
    ("cache_same_process",   [PY, "local_verifier/tests/cache_same_process.py"]),
]

def run(cmd, env=None):
    print(f"\n$ {' '.join(cmd)}")
    p = subprocess.run(cmd, env=env, text=True, capture_output=True)
    sys.stdout.write(p.stdout or "")
    sys.stderr.write(p.stderr or "")
    return p.returncode == 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true",
        help="Перед тестами запустить ingest→index→answer (если нужны свежие артефакты).")
    ap.add_argument("--corpus", default="./corpus", help="Путь к корпусу PDF/сканов")
    ap.add_argument("--pages", default="./data/pages.jsonl", help="Файл pages.jsonl (выход ingest)")
    ap.add_argument("--faiss", default="./data/faiss.index", help="FAISS индекс")
    ap.add_argument("--bm25",  default="./data/bm25.json", help="BM25 индекс/словарь")
    ap.add_argument("--questions", default="./data/questions.jsonl", help="Вопросы")
    ap.add_argument("--answers", default="./answers.json", help="Файл ответов")
    args = ap.parse_args()

    # Проверка, что мы в корне репозитория
    if not os.path.exists("scripts/run_all.py"):
        print("WARN: scripts/run_all.py не найден. Убедитесь, что запускаете из корня репозитория.")
    env = os.environ.copy()
    sep = os.pathsep
    env["PYTHONPATH"] = env.get("PYTHONPATH", "") + (sep if env.get("PYTHONPATH") else "") + "."

    ok_prepare = True
    if args.prepare:
        print("\n=== PREPARE: ingest → index → answer ===")
        ok_prepare &= run([PY, "scripts/run_all.py", "ingest", "--input", args.corpus, "--out", args.pages], env)
        ok_prepare &= run([PY, "scripts/run_all.py", "index", "--pages", args.pages, "--out_dense", args.faiss, "--out_bm25", args.bm25], env)
        ok_prepare &= run([PY, "scripts/run_all.py", "answer", "--pages", args.pages, "--faiss", args.faiss, "--bm25", args.bm25, "--questions", args.questions, "--out", args.answers], env)
        if not ok_prepare:
            print("\nERROR: Подготовительный прогон завершился с ошибками — проверьте сообщения выше.")
            sys.exit(2)

    # Обновляем пути в тестах согласно аргументам
    global TESTS
    TESTS = [
        ("answers_schema_check", [PY, "local_verifier/tests/answers_schema_check.py", args.answers]),
        ("evidence_probe",       [PY, "local_verifier/tests/evidence_probe.py", args.answers, "--pages", args.pages]),
        ("dense_norm_check",     [PY, "local_verifier/tests/dense_norm_check.py", args.faiss]),
        ("hybrid_invariants",    [PY, "local_verifier/tests/hybrid_invariants.py"]),
        ("rerank_invariants",    [PY, "local_verifier/tests/rerank_invariants.py"]),
        ("cache_same_process",   [PY, "local_verifier/tests/cache_same_process.py"]),
    ]

    print("\n=== RUN TESTS ===")
    results = []
    all_ok = True
    for name, cmd in TESTS:
        ok = run(cmd, env)
        results.append((name, ok))
        all_ok &= ok

    print("\n=== SUMMARY ===")
    width = max(len(n) for n, _ in results) + 2
    for name, ok in results:
        print(f"{name.ljust(width)} {'✅ PASS' if ok else '❌ FAIL'}")
    if all_ok:
        print("\nALL CHECKS PASSED ✅")
        sys.exit(0)
    else:
        print("\nSOME CHECKS FAILED ❌  — см. логи выше.")
        sys.exit(1)

if __name__ == "__main__":
    main()
